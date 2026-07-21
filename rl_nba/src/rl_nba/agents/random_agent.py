"""Uniform-random baseline agent."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from rl_nba.agents.base import BanditAgent


class RandomAgent(BanditAgent):
    """Offers a uniformly random eligible product; the floor every learner must beat."""

    name: ClassVar[str] = "random"

    def select_action(self, context: np.ndarray, eligible_actions: np.ndarray) -> int:
        self._check_eligible(eligible_actions)
        return int(self.rng.choice(eligible_actions))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass
