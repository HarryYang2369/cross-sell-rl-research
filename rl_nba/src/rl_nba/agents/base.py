"""Common interface for contextual bandit agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


def argmax_random_tiebreak(scores: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the highest score, breaking ties uniformly at random.

    ``+inf`` scores (used to force exploration of untried actions) win outright;
    ``-inf`` marks ineligible actions and never wins as long as at least one
    finite or ``+inf`` score exists.
    """
    best = np.max(scores)
    if np.isposinf(best):
        candidates = np.flatnonzero(np.isposinf(scores))
    else:
        candidates = np.flatnonzero(scores >= best - 1e-12)
    return int(rng.choice(candidates))


class BanditAgent(ABC):
    """A policy that picks one eligible product per round and learns from reward.

    Contract: :meth:`select_action` must return an element of
    ``eligible_actions``; :meth:`update` is then called once with the observed
    reward for that action. Agents must draw all randomness from ``self.rng``
    so experiments stay reproducible.
    """

    name: ClassVar[str]

    def __init__(self, n_actions: int, context_dim: int, rng: np.random.Generator) -> None:
        if n_actions < 1:
            raise ValueError("n_actions must be at least 1")
        if context_dim < 1:
            raise ValueError("context_dim must be at least 1")
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.rng = rng

    @abstractmethod
    def select_action(self, context: np.ndarray, eligible_actions: np.ndarray) -> int:
        """Choose one action index from ``eligible_actions`` for this context."""

    @abstractmethod
    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Learn from the observed reward of the chosen action."""

    @staticmethod
    def _check_eligible(eligible_actions: np.ndarray) -> None:
        if len(eligible_actions) == 0:
            raise ValueError("eligible_actions is empty; nothing can be offered")
