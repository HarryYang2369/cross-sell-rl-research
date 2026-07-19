"""Contextual linear agents: LinUCB and Linear Thompson Sampling.

Both maintain, per product, a ridge regression of reward on context
(``A = ridge*I + sum(x xᵀ)``, ``b = sum(r x)``). They differ only in how they
turn the fitted model into exploration: LinUCB adds an upper confidence bound
(Li et al. 2010, "A Contextual-Bandit Approach to Personalized News Article
Recommendation"); Thompson Sampling draws the coefficients from their
posterior and acts greedily on the draw.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from cross_sell_rl.agents.base import BanditAgent, argmax_random_tiebreak


class _PerActionRidge:
    """Ridge sufficient statistics per action, with lazily refreshed caches."""

    def __init__(self, n_actions: int, context_dim: int, ridge: float) -> None:
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        identity = np.eye(context_dim)
        self._A = np.tile(identity * ridge, (n_actions, 1, 1))
        self._b = np.zeros((n_actions, context_dim))
        self._A_inv = np.tile(identity / ridge, (n_actions, 1, 1))
        self._theta = np.zeros((n_actions, context_dim))
        self._chol = np.tile(identity / np.sqrt(ridge), (n_actions, 1, 1))
        self._stale = np.zeros(n_actions, dtype=bool)

    def update(self, action: int, context: np.ndarray, reward: float) -> None:
        self._A[action] += np.outer(context, context)
        self._b[action] += reward * context
        self._stale[action] = True

    def theta(self, action: int) -> np.ndarray:
        self._refresh(action)
        return self._theta[action]

    def a_inv(self, action: int) -> np.ndarray:
        self._refresh(action)
        return self._A_inv[action]

    def chol_a_inv(self, action: int) -> np.ndarray:
        """Cholesky factor of A⁻¹, i.e. the posterior covariance square root."""
        self._refresh(action)
        return self._chol[action]

    def _refresh(self, action: int) -> None:
        if not self._stale[action]:
            return
        a_inv = np.linalg.inv(self._A[action])
        a_inv = (a_inv + a_inv.T) / 2.0  # keep symmetric against float drift
        self._A_inv[action] = a_inv
        self._theta[action] = a_inv @ self._b[action]
        self._chol[action] = np.linalg.cholesky(a_inv)
        self._stale[action] = False


class LinUCBAgent(BanditAgent):
    """Disjoint LinUCB: score = θₐ·x + alpha * sqrt(xᵀ Aₐ⁻¹ x).

    ``alpha`` scales the exploration bonus; higher explores more.
    """

    name: ClassVar[str] = "linucb"

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        rng: np.random.Generator,
        alpha: float = 1.0,
        ridge: float = 1.0,
    ) -> None:
        super().__init__(n_actions, context_dim, rng)
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self._state = _PerActionRidge(n_actions, context_dim, ridge)

    def select_action(self, context: np.ndarray, eligible_actions: np.ndarray) -> int:
        self._check_eligible(eligible_actions)
        scores = np.full(self.n_actions, -np.inf)
        for action in eligible_actions:
            estimate = float(self._state.theta(action) @ context)
            variance = float(context @ self._state.a_inv(action) @ context)
            scores[action] = estimate + self.alpha * np.sqrt(max(variance, 0.0))
        return argmax_random_tiebreak(scores, self.rng)

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self._state.update(action, context, reward)


class LinTSAgent(BanditAgent):
    """Linear Thompson Sampling: sample θ̃ₐ ~ N(θₐ, scale² Aₐ⁻¹), act greedily.

    ``scale`` widens the posterior; higher explores more.
    """

    name: ClassVar[str] = "lin_ts"

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        rng: np.random.Generator,
        scale: float = 1.0,
        ridge: float = 1.0,
    ) -> None:
        super().__init__(n_actions, context_dim, rng)
        if scale < 0.0:
            raise ValueError("scale must be non-negative")
        self.scale = scale
        self._state = _PerActionRidge(n_actions, context_dim, ridge)

    def select_action(self, context: np.ndarray, eligible_actions: np.ndarray) -> int:
        self._check_eligible(eligible_actions)
        scores = np.full(self.n_actions, -np.inf)
        for action in eligible_actions:
            noise = self.rng.standard_normal(self.context_dim)
            theta_sample = (
                self._state.theta(action) + self.scale * self._state.chol_a_inv(action) @ noise
            )
            scores[action] = float(theta_sample @ context)
        return argmax_random_tiebreak(scores, self.rng)

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self._state.update(action, context, reward)
