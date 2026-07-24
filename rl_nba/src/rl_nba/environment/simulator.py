"""Simulated customer-response environment.

Since no real outcome logs exist yet, conversions are drawn from a hidden
ground-truth model: each product has a random weight vector over the context
features, and conversion probability is a logistic function of the weighted
context. Agents never see these weights — they only observe (context, action,
reward) triples, exactly as they would in production.

Nothing outside this module (and the evaluation code that reads expected
rewards for regret) may touch the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


@dataclass(frozen=True)
class ConversionModel:
    """Hidden ground truth: per-product weights over the context features.

    ``weights`` has shape ``(n_actions, context_dim)``; column 0 multiplies the
    intercept feature and therefore sets each product's base conversion rate.
    """

    weights: np.ndarray

    @classmethod
    def sample(
        cls,
        context_dim: int,
        n_actions: int,
        base_conversion_rate: float,
        context_influence: float,
        rng: np.random.Generator,
        base_dim: int | None = None,
        journey_influence: float = 1.0,
    ) -> ConversionModel:
        """Draw a random ground-truth model with a calibrated base rate.

        ``base_dim`` (default: ``context_dim``) is the number of leading columns
        drawn as the "base" world. Columns beyond it (e.g. Digital-Twin journey
        one-hots) are drawn in a *separate* rng call scaled by
        ``journey_influence`` — so the base weights are identical whether or not
        those extra columns are present (enabled/disabled worlds stay comparable
        on their common features), and ``journey_influence`` sets how strongly the
        journey block drives behaviour (1.0 = like any feature; higher = a world
        where journey stage genuinely matters).
        """
        base_dim = context_dim if base_dim is None else base_dim
        scale = context_influence / np.sqrt(max(base_dim - 1, 1))
        weights = rng.normal(0.0, scale, size=(n_actions, base_dim))
        weights[:, 0] = _logit(base_conversion_rate) + rng.normal(0.0, 0.35, size=n_actions)
        if base_dim < context_dim:
            extra = rng.normal(
                0.0, scale * journey_influence, size=(n_actions, context_dim - base_dim)
            )
            weights = np.hstack([weights, extra])
        return cls(weights=weights)

    def probabilities(self, contexts: np.ndarray) -> np.ndarray:
        """Conversion probability matrix of shape ``(n_rows, n_actions)``."""
        return _sigmoid(contexts @ self.weights.T)


class CrossSellSimulator:
    """Bandit environment over a fixed customer pool.

    Each round the runner picks a customer, the agent picks an eligible
    product (never one already owned), and the simulator draws a Bernoulli
    conversion from the hidden model. ``action_values`` converts a conversion
    into reward: all ones for ``conversion`` reward, per-product premiums for
    ``revenue`` reward.

    Customers who already own every product are dropped at construction.
    """

    def __init__(
        self,
        contexts: np.ndarray,
        owned: np.ndarray,
        model: ConversionModel,
        action_values: np.ndarray,
    ) -> None:
        if contexts.shape[0] != owned.shape[0]:
            raise ValueError("contexts and owned must have the same number of rows")
        if owned.shape[1] != model.weights.shape[0]:
            raise ValueError("owned and model must agree on the number of actions")
        if len(action_values) != owned.shape[1]:
            raise ValueError("action_values must have one entry per action")

        offerable = ~owned.all(axis=1)
        if not offerable.any():
            raise ValueError("Every customer already owns every product; nothing to offer.")
        self._contexts = contexts[offerable]
        self._owned = owned[offerable]
        self._values = np.asarray(action_values, dtype=float)
        self._probabilities = model.probabilities(self._contexts)
        self._expected = self._probabilities * self._values[np.newaxis, :]
        masked = np.where(self._owned, -np.inf, self._expected)
        self._best_expected = masked.max(axis=1)
        self._best_actions = masked.argmax(axis=1)

    @property
    def n_customers(self) -> int:
        return self._contexts.shape[0]

    @property
    def n_actions(self) -> int:
        return self._owned.shape[1]

    @property
    def contexts(self) -> np.ndarray:
        """Context matrix of the offerable customers ``(n_customers, context_dim)``."""
        return self._contexts

    @property
    def probabilities(self) -> np.ndarray:
        """Conversion probability per customer and action ``(n_customers, n_actions)``."""
        return self._probabilities

    @property
    def owned(self) -> np.ndarray:
        """Initial ownership matrix ``(n_customers, n_actions)`` (bool)."""
        return self._owned

    def context(self, customer: int) -> np.ndarray:
        return self._contexts[customer]

    def eligible_actions(self, customer: int) -> np.ndarray:
        """Indices of products the customer does not own yet."""
        return np.flatnonzero(~self._owned[customer])

    def step(self, customer: int, action: int, rng: np.random.Generator) -> float:
        """Draw the reward for offering ``action`` to ``customer``."""
        if self._owned[customer, action]:
            raise ValueError(
                f"Action {action} is not eligible for customer {customer} (already owned)."
            )
        converted = rng.random() < self._probabilities[customer, action]
        return float(self._values[action]) if converted else 0.0

    def expected_reward(self, customer: int, action: int) -> float:
        return float(self._expected[customer, action])

    def best_expected_reward(self, customer: int) -> float:
        """Expected reward of the best eligible action (the oracle choice)."""
        return float(self._best_expected[customer])

    def best_actions(self, customers: np.ndarray) -> np.ndarray:
        """Oracle-best eligible action for each customer index given."""
        return self._best_actions[np.asarray(customers, dtype=int)]

    def oracle_value(self) -> float:
        """Mean per-round expected reward of always offering the best product."""
        return float(self._best_expected.mean())

    def mean_conversion_probability(self) -> float:
        """Average conversion probability across all eligible (customer, product) pairs."""
        return float(self._probabilities[~self._owned].mean())

    def personalization_rate(self) -> float:
        """Fraction of customers whose best product is not the most common best.

        Near 0 means one product dominates (context barely matters); higher
        values mean personalization has room to pay off.
        """
        counts = np.bincount(self._best_actions, minlength=self.n_actions)
        return float(1.0 - counts.max() / self.n_customers)
