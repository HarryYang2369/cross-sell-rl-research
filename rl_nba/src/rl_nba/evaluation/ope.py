"""Off-policy evaluation (OPE): estimate a new policy's value from logged data.

This is the bridge to real data. Once historical offer logs exist (who was
offered what, with what probability, and whether they bought), these
estimators score a candidate policy without running it on live customers —
the practice popularized by Open Bandit Pipeline. Until then,
:func:`collect_logged_feedback` produces logs from the simulator under a
uniform-random logging policy, mimicking a randomized pilot campaign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_nba.agents.base import BanditAgent
from rl_nba.environment.simulator import CrossSellSimulator


@dataclass(frozen=True)
class LoggedFeedback:
    """Logged bandit feedback: one row per historical offer decision.

    ``propensities`` holds the logging policy's probability of the action it
    actually took; ``eligible`` marks which actions were available per row.
    """

    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    propensities: np.ndarray
    eligible: np.ndarray
    customer_indices: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.actions)
        shapes_ok = (
            self.contexts.shape[0] == n
            and self.rewards.shape == (n,)
            and self.propensities.shape == (n,)
            and self.eligible.shape[0] == n
            and self.customer_indices.shape == (n,)
        )
        if not shapes_ok:
            raise ValueError("LoggedFeedback arrays disagree on the number of rows")
        if np.any(self.propensities <= 0.0):
            raise ValueError("propensities must be strictly positive")

    @property
    def n_rounds(self) -> int:
        return len(self.actions)


def ips_value(logs: LoggedFeedback, target_action_probs: np.ndarray) -> float:
    """Inverse Propensity Scoring estimate of the target policy's mean reward.

    ``target_action_probs[i, a]`` is the probability the target policy would
    pick action ``a`` in logged context ``i``. IPS reweights each logged reward
    by how much more (or less) the target policy likes the logged action than
    the logging policy did. Unbiased, but high-variance when policies differ a
    lot.
    """
    weights = _importance_weights(logs, target_action_probs)
    return float(np.mean(weights * logs.rewards))


def snips_value(logs: LoggedFeedback, target_action_probs: np.ndarray) -> float:
    """Self-normalized IPS: biased but much lower variance than plain IPS."""
    weights = _importance_weights(logs, target_action_probs)
    total = weights.sum()
    if total <= 0.0:
        return 0.0
    return float((weights * logs.rewards).sum() / total)


def _importance_weights(logs: LoggedFeedback, target_action_probs: np.ndarray) -> np.ndarray:
    if target_action_probs.shape[0] != logs.n_rounds:
        raise ValueError("target_action_probs must have one row per logged round")
    taken = target_action_probs[np.arange(logs.n_rounds), logs.actions]
    return taken / logs.propensities


def collect_logged_feedback(
    simulator: CrossSellSimulator,
    n_rounds: int,
    rng: np.random.Generator,
) -> LoggedFeedback:
    """Simulate a randomized pilot: uniform-random offers with known propensities."""
    context_dim = simulator.context(0).shape[0]
    contexts = np.zeros((n_rounds, context_dim))
    actions = np.zeros(n_rounds, dtype=int)
    rewards = np.zeros(n_rounds)
    propensities = np.zeros(n_rounds)
    eligible = np.zeros((n_rounds, simulator.n_actions), dtype=bool)
    customers = np.zeros(n_rounds, dtype=int)
    for row in range(n_rounds):
        customer = int(rng.integers(0, simulator.n_customers))
        eligible_actions = simulator.eligible_actions(customer)
        action = int(rng.choice(eligible_actions))
        contexts[row] = simulator.context(customer)
        actions[row] = action
        rewards[row] = simulator.step(customer, action, rng)
        propensities[row] = 1.0 / len(eligible_actions)
        eligible[row, eligible_actions] = True
        customers[row] = customer
    return LoggedFeedback(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        propensities=propensities,
        eligible=eligible,
        customer_indices=customers,
    )


def policy_action_probs(agent: BanditAgent, logs: LoggedFeedback) -> np.ndarray:
    """Target-policy action probabilities on the logged contexts.

    Asks the (frozen, already-trained) agent to act on each logged context and
    one-hot encodes the choice. For stochastic agents such as Thompson
    Sampling this is a single sampled draw per row, which is a reasonable
    Monte Carlo approximation for evaluation purposes.
    """
    probs = np.zeros((logs.n_rounds, agent.n_actions))
    for row in range(logs.n_rounds):
        eligible_actions = np.flatnonzero(logs.eligible[row])
        action = agent.select_action(logs.contexts[row], eligible_actions)
        probs[row, action] = 1.0
    return probs
