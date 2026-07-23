"""Record a human-readable playback trace of one agent's episode.

The trace is what the playback dashboard replays: for every decision it captures
the customer, the agent's value estimate and exploration bonus for each eligible
product, the offer it made, and what happened — plus running totals against the
random and oracle baselines. See :func:`write_dashboard` to render it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_nba.agents import create_agent
from rl_nba.config import AppConfig
from rl_nba.data import ownership_matrix
from rl_nba.run import prepare_experiment, run_experiment

_TEMPLATE = Path(__file__).with_name("_playback_template.html")

# Friendly display names for the model shown on the dashboard.
_MODEL_LABELS: dict[str, str] = {
    "random": "Random (floor)",
    "epsilon_greedy": "ε-greedy",
    "linucb": "LinUCB (linear)",
    "lin_ts": "Thompson sampling",
    "rff_ucb": "RFF-UCB (non-linear)",
}

# Human-readable labels for the customer attributes worth showing, in display
# order. Only columns actually present in the data are shown, so this is safe
# across schemas.
_ATTRIBUTE_LABELS: dict[str, str] = {
    "customer_age": "Age",
    "age": "Age",
    "customer_gender": "Gender",
    "customer_marital_status": "Marital status",
    "customer_income_range": "Monthly income",
    "wealth_segment": "Wealth segment",
    "region": "Region",
}


def _pretty(value: Any) -> str:
    """Format a cell value for display (whole floats as ints, else as-is)."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:,.0f}"
    return str(value)


def _pretty_product(product: str) -> str:
    return product.replace("_", " ")


def champion_type(config: AppConfig) -> str:
    """Identify the champion: the highest mean-reward model type the config runs.

    A single ``agent.type`` is trivially its own champion — returned without a
    comparison. ``agent.type: all`` runs the full comparison and returns the
    winner (``summarize_results`` orders best mean reward first). It uses the
    config's own ``experiment.n_rounds`` — *not* a shortened run — so the
    dashboard's champion matches the one in ``metrics.csv``: the model ranking
    can cross over with horizon (a fast learner leads early, a richer model wins
    later), and picking at a different round count would name a different model.
    """
    types = config.agent.types_to_run
    if len(types) == 1:
        return types[0]
    print(
        f"Identifying champion among {list(types)} "
        f"({config.experiment.n_rounds:,} rounds each; this takes a moment)..."
    )
    outcome = run_experiment(config)
    winner = str(outcome.summary.iloc[0]["agent"])
    print(f"  champion: {winner}")
    return winner


def record_playback(
    config: AppConfig,
    model: str = "champion",
    n_steps: int = 150,
    seed: int = 20260721,
) -> dict[str, Any]:
    """Record ``n_steps`` decisions of one model for the playback dashboard.

    ``model`` chooses which model plays back: ``"champion"`` (the best-performing
    type the config runs — see :func:`champion_type`) or an explicit type name
    (e.g. ``"linucb"`` / ``"rff_ucb"``). The model is built with the config's
    ``agent`` settings (``alpha``, ``n_features``) and the same journey masking it
    trains under, so the trace matches how that model actually learns.

    Returns a JSON-serializable dict with ``meta`` and a list of ``steps``.
    """
    is_champion = model in (None, "champion")
    model_type = champion_type(config) if is_champion else str(model)

    prepared = prepare_experiment(config)
    simulator = prepared.simulator
    schema, catalog = config.data.schema, config.products.catalog

    owned = ownership_matrix(prepared.customers, schema, catalog)
    offerable_mask = ~owned.all(axis=1)
    offerable = prepared.customers[offerable_mask].reset_index(drop=True)
    offerable_owned = owned[offerable_mask]

    # Journey masking, identical to training: drop the journey block when this
    # model runs with agent.journey: false (never for random, which ignores it).
    feature_indices = None
    if (
        prepared.builder is not None
        and not config.agent.journey
        and model_type != "random"
    ):
        feature_indices = prepared.builder.columns_for(include_journey=False)

    def agent_view(context: np.ndarray) -> np.ndarray:
        return context if feature_indices is None else context[feature_indices]

    params: dict[str, Any] = {}
    if model_type in ("linucb", "rff_ucb"):
        params["alpha"] = config.agent.alpha
    if model_type == "rff_ucb":
        params["n_features"] = config.agent.n_features
    agent_dim = prepared.context_dim if feature_indices is None else len(feature_indices)
    agent = create_agent(
        model_type,
        n_actions=simulator.n_actions,
        context_dim=agent_dim,
        rng=np.random.default_rng(seed),
        **params,
    )
    # Only value-based agents expose an estimate/bonus split; random does not.
    can_explain = hasattr(agent, "explain")
    env_rng = np.random.default_rng(seed + 1)

    n_steps = min(n_steps, len(prepared.customer_sequence))
    cumulative = {"agent": 0.0, "random": 0.0, "oracle": 0.0}
    conversions = 0
    value_earned = 0.0
    steps: list[dict[str, Any]] = []

    for step in range(n_steps):
        customer = int(prepared.customer_sequence[step])
        context = simulator.context(customer)
        view = agent_view(context)
        eligible = simulator.eligible_actions(customer)

        if can_explain:
            breakdown = agent.explain(view, eligible)
            chosen = max(breakdown, key=lambda item: item["score"])["action"]
        else:
            chosen = agent.select_action(view, eligible)
            breakdown = [
                {"action": int(action), "estimate": 0.0, "bonus": 0.0, "score": 0.0}
                for action in eligible
            ]
        reward = simulator.step(customer, chosen, env_rng)
        agent.update(view, chosen, reward)

        converted = reward > 0.0
        conversions += int(converted)
        value_earned += reward
        cumulative["agent"] += simulator.expected_reward(customer, chosen)
        cumulative["oracle"] += simulator.best_expected_reward(customer)
        cumulative["random"] += float(
            np.mean([simulator.expected_reward(customer, action) for action in eligible])
        )

        options = [
            {
                "product": _pretty_product(catalog[item["action"]]),
                # Clamp for display: a negative expected value reads as "~0 to us".
                "estimate": round(max(item["estimate"], 0.0), 1),
                "bonus": round(max(item["bonus"], 0.0), 1),
                "chosen": item["action"] == chosen,
            }
            for item in sorted(breakdown, key=lambda item: item["score"], reverse=True)
        ]
        row = offerable.iloc[customer]
        steps.append(
            {
                "step": step,
                "customer_id": str(row[schema.customer_id]),
                "attributes": _customer_attributes(row),
                "owns": [
                    _pretty_product(product)
                    for product, held in zip(catalog, offerable_owned[customer], strict=True)
                    if held
                ],
                "options": options,
                "chosen": _pretty_product(catalog[chosen]),
                "converted": bool(converted),
                "reward": round(float(reward), 1),
                "cumulative": {name: round(value, 1) for name, value in cumulative.items()},
                "value_earned": round(value_earned, 1),
                "conversion_rate": round(conversions / (step + 1), 4),
            }
        )

    return {
        "meta": {
            "reward_type": config.reward.type,
            "currency": "HK$" if config.reward.type in ("ape", "vnb") else "",
            "value_word": {"vnb": "business value", "ape": "premium", "revenue": "premium"}.get(
                config.reward.type, "conversions"
            ),
            "products": [_pretty_product(product) for product in catalog],
            "n_steps": n_steps,
            "model": model_type,
            "model_label": _MODEL_LABELS.get(model_type, model_type),
            "is_champion": bool(is_champion),
            "journey": bool(config.agent.journey and feature_indices is None),
            "alpha": config.agent.alpha,
        },
        "steps": steps,
    }


def _customer_attributes(row: pd.Series) -> list[list[str]]:
    return [
        [label, _pretty(row[column])]
        for column, label in _ATTRIBUTE_LABELS.items()
        if column in row.index
    ]


def render_dashboard(trace: dict[str, Any]) -> str:
    """Inject a recorded trace into the dashboard template, returning full HTML."""
    template = _TEMPLATE.read_text(encoding="utf-8")
    # Escape "<" so the JSON can never terminate the <script> block early.
    payload = json.dumps(trace, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("/*__PLAYBACK_TRACE__*/", payload)


def write_dashboard(
    config: AppConfig,
    out_path: str | Path,
    model: str = "champion",
    n_steps: int = 150,
    seed: int = 20260721,
) -> Path:
    """Record an episode of the chosen model and write the playback dashboard."""
    trace = record_playback(config, model=model, n_steps=n_steps, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_dashboard(trace), encoding="utf-8")
    return out_path
