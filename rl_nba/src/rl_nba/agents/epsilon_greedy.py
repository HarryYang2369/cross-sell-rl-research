"""Context-free epsilon-greedy agent."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from rl_nba.agents.base import BanditAgent, argmax_random_tiebreak


class EpsilonGreedyAgent(BanditAgent):
    """Tracks each product's average reward, ignoring customer context.

    This is the classic multi-armed bandit: with probability ``epsilon`` it
    explores a random eligible product, otherwise it exploits the best average
    so far (trying untried products first). It exists to quantify the value of
    personalization — the contextual agents should beat it.
    """

    name: ClassVar[str] = "epsilon_greedy"

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        rng: np.random.Generator,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(n_actions, context_dim, rng)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self._counts = np.zeros(n_actions)
        self._reward_sums = np.zeros(n_actions)

    def select_action(self, context: np.ndarray, eligible_actions: np.ndarray) -> int:
        self._check_eligible(eligible_actions)
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(eligible_actions))
        means = np.divide(
            self._reward_sums,
            self._counts,
            out=np.zeros(self.n_actions),
            where=self._counts > 0,
        )
        scores = np.full(self.n_actions, -np.inf)
        scores[eligible_actions] = np.where(
            self._counts[eligible_actions] > 0, means[eligible_actions], np.inf
        )
        return argmax_random_tiebreak(scores, self.rng)

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self._counts[action] += 1
        self._reward_sums[action] += reward
