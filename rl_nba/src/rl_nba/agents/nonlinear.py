"""Non-linear contextual bandit: Random Fourier Features + LinUCB.

LinUCB is *linear* in the context, so it cannot represent interactions between
features (e.g. journey stage × age) or other non-linear structure. This agent
maps the context through random Fourier features —
``z(x) = sqrt(2/D)·cos(Ω x + b)`` with ``Ω ~ N(0, 2γ)``, ``b ~ U(0, 2π)`` —
which approximate an RBF kernel, then runs LinUCB in that non-linear feature
space. It keeps LinUCB's optimism-based (UCB) exploration while modelling a
non-linear value function, and needs only numpy.

Because the journey label is a non-linear (threshold) function of the raw
features, a non-linear agent can *recover the journey effect from the base
features alone* — which is exactly where a non-linear model earns its keep.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from rl_nba.agents.base import BanditAgent, argmax_random_tiebreak
from rl_nba.agents.linear import _PerActionRidge


class RFFLinUCBAgent(BanditAgent):
    """LinUCB on a random-Fourier-feature map of the context (a kernel bandit).

    ``n_features`` random features approximate an RBF kernel of bandwidth
    ``gamma`` (default ``1/context_dim``); ``alpha`` scales the exploration
    bonus, ``ridge`` the regulariser.
    """

    name: ClassVar[str] = "rff_ucb"

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        rng: np.random.Generator,
        alpha: float = 1.0,
        ridge: float = 1.0,
        n_features: int = 128,
        gamma: float | None = None,
    ) -> None:
        super().__init__(n_actions, context_dim, rng)
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        self.alpha = alpha
        self.n_features = int(n_features)
        bandwidth = 1.0 / context_dim if gamma is None else float(gamma)
        self._omega = rng.normal(
            0.0, np.sqrt(2.0 * bandwidth), size=(self.n_features, context_dim)
        )
        self._bias = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)
        self._norm = np.sqrt(2.0 / self.n_features)
        self._state = _PerActionRidge(n_actions, self.n_features, ridge)

    def _features(self, context: np.ndarray) -> np.ndarray:
        """Random Fourier feature map of the context (approximates an RBF kernel)."""
        return self._norm * np.cos(self._omega @ context + self._bias)

    def select_action(self, context: np.ndarray, eligible_actions: np.ndarray) -> int:
        self._check_eligible(eligible_actions)
        mapped = self._features(context)
        scores = np.full(self.n_actions, -np.inf)
        for action in eligible_actions:
            estimate = float(self._state.theta(action) @ mapped)
            variance = float(mapped @ self._state.a_inv(action) @ mapped)
            scores[action] = estimate + self.alpha * np.sqrt(max(variance, 0.0))
        return argmax_random_tiebreak(scores, self.rng)

    def explain(
        self, context: np.ndarray, eligible_actions: np.ndarray
    ) -> list[dict[str, float]]:
        """Per-action score breakdown: value ``estimate`` + exploration ``bonus``.

        Same introspection hook as :meth:`LinUCBAgent.explain`, but computed in
        the random-Fourier-feature space this agent actually learns in — so the
        playback dashboard can visualise a non-linear champion the same way.
        """
        self._check_eligible(eligible_actions)
        mapped = self._features(context)
        breakdown = []
        for action in eligible_actions:
            estimate = float(self._state.theta(action) @ mapped)
            variance = float(mapped @ self._state.a_inv(action) @ mapped)
            bonus = self.alpha * float(np.sqrt(max(variance, 0.0)))
            breakdown.append(
                {
                    "action": int(action),
                    "estimate": estimate,
                    "bonus": bonus,
                    "score": estimate + bonus,
                }
            )
        return breakdown

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self._state.update(action, self._features(context), reward)
