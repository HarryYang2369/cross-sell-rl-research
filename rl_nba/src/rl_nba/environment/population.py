"""Population panel environment: a 48-month sequential cross-sell simulation.

Unlike the single-customer bandit env (:mod:`rl_nba.environment.gym_env`), this
steps the **whole customer base** forward one month at a time and optimises
**long-term cumulative value** over the horizon. Each month:

1. the agent proposes a product (+priority) for every customer from the frame
   ``(n_customers x state)``;
2. governance gates it — **consent**, **eligibility** (not already owned), and a
   **contact-frequency cooldown**;
3. the monthly **capacity** keeps only the highest-priority offers; the rest
   become **no-offer**;
4. executed offers may convert (**issuance**); every owned policy may **lapse**
   or **surrender**; contacted customers may **complain**; engagement drifts;
5. a **long-term, discounted reward** nets issuance value against
   lapse / surrender / complaint / operational costs.

An episode is ``population.n_months`` steps; everything is vectorised over the
population (numpy only), so a full episode runs in well under a minute. Every
hazard rate and reward weight is a config knob — the exact reward formula is
still being researched, so the defaults are documented placeholders.

    from rl_nba.config import load_config
    from rl_nba.environment.population import run_episode, plot_episode

    trace = run_episode(load_config("config/rl_nba_config.yml"), agent="linucb")
    print(trace.summary)
    plot_episode(trace, "rl_nba/results/episode.png")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_nba.config import AppConfig
from rl_nba.run import prepare_experiment

# ---------------------------------------------------------------------------
# value tables
# ---------------------------------------------------------------------------


def _value_table(config: AppConfig, kind: str) -> np.ndarray:
    """Per-product value array for ``kind`` (conversion/revenue/ape/vnb)."""
    products = config.products
    if kind == "conversion":
        return np.ones(len(products.catalog))
    source = {"revenue": products.premiums, "ape": products.ape, "vnb": products.vnb}[kind]
    return np.array([source[p] for p in products.catalog], dtype=float)


# ---------------------------------------------------------------------------
# the environment
# ---------------------------------------------------------------------------


class PopulationEnv:
    """Vectorised 48-month population panel env (see module docstring)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        pop = config.population
        prepared = prepare_experiment(config)
        sim = prepared.simulator

        # The whole offerable pool — scale is set by synthetic.n_customers (or real data).
        self.n = sim.n_customers
        self._contexts = sim.contexts.astype(np.float64)
        self._probs = sim.probabilities
        self._owned0 = sim.owned.copy()
        self.n_actions = sim.n_actions
        self.context_dim = self._contexts.shape[1]

        # value tables: one drives the reward, both APE/VNB are tracked for metrics.
        self._value = _value_table(config, pop.issuance_value)   # reward currency
        self._ape = _value_table(config, "ape")
        self._vnb = _value_table(config, "vnb")

        self.n_months = pop.n_months
        self._cap = self.n if pop.monthly_capacity in (-1, None) else pop.monthly_capacity
        self._pop = pop

    # -- state ----------------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        pop = self._pop
        self._rng = np.random.default_rng(pop.seed if seed is None else seed)
        self.owned = self._owned0.copy()
        self._consent = self._rng.random(self.n) < pop.consent_rate
        self._do_not_contact = np.zeros(self.n, dtype=bool)
        self._since_contact = np.full(self.n, pop.contact_cooldown + 1, dtype=int)
        self._engagement = self._rng.uniform(0.3, 0.7, self.n)
        self._complaints = np.zeros(self.n, dtype=int)
        self.month = 0
        self.discounted_return = 0.0
        return self.frame()

    def frame(self) -> np.ndarray:
        """Observation: the ``(n_customers x state)`` context frame."""
        return self._contexts

    # -- the monthly step -----------------------------------------------------

    def step(
        self, recommended: np.ndarray, priority: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance one month. ``recommended[i]`` is a product index or -1 (no
        proposal); ``priority[i]`` ranks customers when capacity is scarce."""
        pop = self._pop
        rng = self._rng
        n, values = self.n, self._value
        recommended = np.asarray(recommended, dtype=int)
        priority = np.asarray(priority, dtype=float)

        # --- governance: who is even contactable this month ---
        has_rec = recommended >= 0
        rec_safe = np.where(has_rec, recommended, 0)
        owns_rec = self.owned[np.arange(n), rec_safe] & has_rec
        contactable = (
            self._consent
            & ~self._do_not_contact
            & (self._since_contact >= pop.contact_cooldown)
            & has_rec
            & ~owns_rec
        )

        # --- capacity: execute only the top-priority offers ---
        idx = np.flatnonzero(contactable)
        order = idx[np.argsort(-priority[idx], kind="stable")]
        executed = order[: self._cap]
        exec_prod = recommended[executed]

        # --- executed offers: draw conversions / issuances ---
        conv = rng.random(executed.size) < self._probs[executed, exec_prod]
        issued = executed[conv]
        issued_prod = exec_prod[conv]
        self.owned[issued, issued_prod] = True
        self._engagement[issued] = np.clip(self._engagement[issued] + 0.1, 0.0, 1.0)
        self._since_contact += 1
        self._since_contact[executed] = 0

        issue_value = float(values[issued_prod].sum())
        op_cost = pop.operational_cost * executed.size
        # bandit learning signal: realised conversion value per executed offer.
        exp_reward = np.where(conv, values[exec_prod], 0.0)

        # --- portfolio dynamics: lapse then surrender (a policy can't do both) ---
        lapse = self.owned & (rng.random(self.owned.shape) < pop.lapse_rate)
        surr = self.owned & ~lapse & (rng.random(self.owned.shape) < pop.surrender_rate)
        self.owned &= ~(lapse | surr)
        lapse_cost = pop.lapse_penalty * float((lapse * values[None, :]).sum())
        surr_cost = pop.surrender_penalty * float((surr * values[None, :]).sum())

        # --- complaints: baseline + fatigue for anyone contacted this month ---
        contacted = np.zeros(n, dtype=bool)
        contacted[executed] = True
        comp_prob = pop.complaint_rate + pop.fatigue_penalty * contacted
        complained = rng.random(n) < comp_prob
        self._complaints += complained
        self._do_not_contact |= complained  # a complaint opts the customer out
        self._engagement[complained] = np.clip(self._engagement[complained] - 0.2, 0.0, 1.0)
        complaint_cost = pop.complaint_cost * int(complained.sum())

        # --- engagement decay ---
        self._engagement *= 1.0 - pop.engagement_decay

        # --- reward + discounting ---
        reward = issue_value - op_cost - lapse_cost - surr_cost - complaint_cost
        self.discounted_return += (pop.discount**self.month) * reward
        self.month += 1
        done = self.month >= self.n_months

        info: dict[str, Any] = {
            "month": self.month,
            "reward": reward,
            "discounted_return": self.discounted_return,
            "offers_made": int(executed.size),
            "capacity": int(self._cap),
            "issuances": int(issued.size),
            "lapses": int(lapse.sum()),
            "surrenders": int(surr.sum()),
            "complaints": int(complained.sum()),
            "ape_issued": float(self._ape[issued_prod].sum()),
            "vnb_issued": float(self._vnb[issued_prod].sum()),
            "active_policies": int(self.owned.sum()),
            "mean_engagement": float(self._engagement.mean()),
            "do_not_contact": int(self._do_not_contact.sum()),
            "relationship_buckets": self._relationship_buckets(),
            # learning signal for a bandit agent (executed offers only):
            "experience": (self._contexts[executed], exec_prod, exp_reward),
        }
        return self.frame(), reward, done, info

    def _relationship_buckets(self) -> dict[str, int]:
        """Distribution of customers by number of products held (lifecycle depth)."""
        held = self.owned.sum(axis=1)
        return {
            "0": int((held == 0).sum()),
            "1": int((held == 1).sum()),
            "2": int((held == 2).sum()),
            "3+": int((held >= 3).sum()),
        }


# ---------------------------------------------------------------------------
# agents (per-customer policies applied to the frame)
# ---------------------------------------------------------------------------


class PopulationLinUCB:
    """LinUCB applied row-wise to the frame: proposes the best *eligible* product
    (and its UCB score for ranking) for every customer, and learns from the
    realised conversions of the offers that were executed."""

    name = "linucb"

    def __init__(self, n_actions: int, context_dim: int, alpha: float = 1.0, ridge: float = 1.0):
        self.n_actions, self.context_dim, self.alpha = n_actions, context_dim, alpha
        eye = np.eye(context_dim)
        self._A = np.stack([eye * ridge for _ in range(n_actions)])
        self._b = np.zeros((n_actions, context_dim))

    def propose(self, frame: np.ndarray, owned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = np.empty((frame.shape[0], self.n_actions))
        for a in range(self.n_actions):
            a_inv = np.linalg.inv(self._A[a])
            theta = a_inv @ self._b[a]
            bonus = np.sqrt(np.einsum("ij,jk,ik->i", frame, a_inv, frame, optimize=True))
            scores[:, a] = frame @ theta + self.alpha * bonus
        scores[owned] = -np.inf
        best = scores.argmax(axis=1)
        priority = scores.max(axis=1)
        all_owned = owned.all(axis=1)
        best[all_owned] = -1
        priority[all_owned] = -np.inf
        return best, priority

    def update(self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        for a in np.unique(actions):
            mask = actions == a
            x = contexts[mask]
            self._A[a] += x.T @ x
            self._b[a] += (rewards[mask, None] * x).sum(axis=0)


class PopulationRandom:
    """Baseline: propose a random eligible product with a random priority."""

    name = "random"

    def __init__(self, n_actions: int, context_dim: int, seed: int = 0):
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    def propose(self, frame: np.ndarray, owned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = frame.shape[0]
        weights = (~owned).astype(float)
        weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-9, None)
        noise = self._rng.random((n, self.n_actions))
        noise[owned] = -1.0
        best = noise.argmax(axis=1)
        best[owned.all(axis=1)] = -1
        return best, self._rng.random(n)

    def update(self, *args: Any) -> None:  # no learning
        pass


def _make_agent(name: str, env: PopulationEnv, config: AppConfig) -> Any:
    if name == "random":
        return PopulationRandom(env.n_actions, env.context_dim, seed=config.population.seed)
    if name == "linucb":
        return PopulationLinUCB(env.n_actions, env.context_dim, alpha=config.agent.alpha)
    raise ValueError(f"Unknown population agent '{name}' (use 'linucb' or 'random')")


# ---------------------------------------------------------------------------
# runner + trace
# ---------------------------------------------------------------------------


@dataclass
class EpisodeTrace:
    """One episode: the config, agent, per-month metrics, and a rollup summary."""

    agent: str
    months: list[dict[str, Any]]
    summary: dict[str, float]

    def to_frame(self) -> pd.DataFrame:
        cols = [k for k in self.months[0] if k not in ("relationship_buckets", "experience")]
        return pd.DataFrame([{c: m[c] for c in cols} for m in self.months])


def run_episode(config: AppConfig, agent: str = "linucb", seed: int | None = None) -> EpisodeTrace:
    """Run one full episode of ``agent`` on the population env; return its trace."""
    env = PopulationEnv(config)
    frame = env.reset(seed)
    policy = _make_agent(agent, env, config)

    months: list[dict[str, Any]] = []
    done = False
    while not done:
        best, priority = policy.propose(frame, env.owned)
        frame, _reward, done, info = env.step(best, priority)
        policy.update(*info["experience"])
        months.append(info)

    totals = {
        "discounted_return": env.discounted_return,
        "total_reward": float(sum(m["reward"] for m in months)),
        "issuances": int(sum(m["issuances"] for m in months)),
        "lapses": int(sum(m["lapses"] for m in months)),
        "surrenders": int(sum(m["surrenders"] for m in months)),
        "complaints": int(sum(m["complaints"] for m in months)),
        "total_ape": float(sum(m["ape_issued"] for m in months)),
        "total_vnb": float(sum(m["vnb_issued"] for m in months)),
        "mean_offers_per_month": float(np.mean([m["offers_made"] for m in months])),
    }
    return EpisodeTrace(agent=agent, months=months, summary=totals)


def compare(config: AppConfig, agents: tuple[str, ...] = ("random", "linucb")) -> pd.DataFrame:
    """Run one episode per agent and return a summary table (evaluation view)."""
    rows = []
    for name in agents:
        trace = run_episode(config, agent=name)
        rows.append({"agent": name, **trace.summary})
    return pd.DataFrame(rows).sort_values("discounted_return", ascending=False)


# ---------------------------------------------------------------------------
# episode figure (the tested stand-in for the interactive dashboard)
# ---------------------------------------------------------------------------


def plot_episode(trace: EpisodeTrace, path: str | Path) -> Path:
    """Multi-panel PNG of one episode: reward, APE/VNB, outcomes, capacity,
    relationship movement, engagement. (The interactive dashboard is Phase 3.)"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = trace.to_frame()
    m = df["month"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Population episode — agent: {trace.agent}", fontweight="bold")

    ax[0, 0].plot(m, df["reward"], label="monthly reward")
    ax[0, 0].plot(m, df["discounted_return"], label="cumulative (discounted)")
    ax[0, 0].axhline(0, lw=0.6, color="k")
    ax[0, 0].set_title("Reward trajectory")
    ax[0, 0].legend()

    ax[0, 1].plot(m, df["ape_issued"], label="APE issued")
    ax[0, 1].plot(m, df["vnb_issued"], label="VNB issued")
    ax[0, 1].plot(m, df["ape_issued"].rolling(6, min_periods=1).mean(), "--", label="APE 6m avg")
    ax[0, 1].set_title("APE / VNB per month")
    ax[0, 1].legend()

    for col in ("issuances", "lapses", "surrenders", "complaints"):
        ax[0, 2].plot(m, df[col], label=col)
    ax[0, 2].set_title("Outcomes per month")
    ax[0, 2].legend()

    ax[1, 0].plot(m, df["offers_made"], label="offers executed")
    ax[1, 0].plot(m, df["capacity"], "--", label="capacity")
    ax[1, 0].set_title("Capacity usage (agent/telesales)")
    ax[1, 0].legend()

    buckets = pd.DataFrame([mo["relationship_buckets"] for mo in trace.months])
    ax[1, 1].stackplot(
        m, [buckets[c] for c in ("0", "1", "2", "3+")], labels=["0", "1", "2", "3+"]
    )
    ax[1, 1].set_title("Relationship depth (products held)")
    ax[1, 1].legend(loc="upper left")

    active_rel = df["active_policies"] / df["active_policies"].iloc[0]
    ax[1, 2].plot(m, df["mean_engagement"], label="mean engagement")
    ax[1, 2].plot(m, active_rel, label="active policies (rel.)")
    ax[1, 2].set_title("Engagement & portfolio")
    ax[1, 2].legend()

    for a in ax.flat:
        a.set_xlabel("month")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    import argparse

    from rl_nba.config import load_config

    parser = argparse.ArgumentParser(description="Run the population panel episode.")
    parser.add_argument("--config", default="config/rl_nba_config.yml")
    parser.add_argument("--agent", default="linucb", help="linucb | random")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(compare(config).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    trace = run_episode(config, agent=args.agent)
    out = plot_episode(trace, Path(config.experiment.output_dir) / "episode.png")
    print(f"\nSaved episode figure for '{args.agent}' to {out}")


if __name__ == "__main__":
    main()
