"""Bandit agents and the registry used to build them from config."""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_nba.agents.base import BanditAgent
from rl_nba.agents.epsilon_greedy import EpsilonGreedyAgent
from rl_nba.agents.linear import LinTSAgent, LinUCBAgent
from rl_nba.agents.random_agent import RandomAgent

_REGISTRY: dict[str, type[BanditAgent]] = {
    agent_class.name: agent_class
    for agent_class in (RandomAgent, EpsilonGreedyAgent, LinUCBAgent, LinTSAgent)
}


def available_agents() -> list[str]:
    """Registry names accepted in the ``experiment.agents`` config section."""
    return sorted(_REGISTRY)


def create_agent(
    name: str,
    *,
    n_actions: int,
    context_dim: int,
    rng: np.random.Generator,
    **params: Any,
) -> BanditAgent:
    """Instantiate a registered agent by name with config-supplied parameters."""
    try:
        agent_class = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown agent '{name}'. Available agents: {available_agents()}"
        ) from None
    try:
        return agent_class(n_actions=n_actions, context_dim=context_dim, rng=rng, **params)
    except TypeError as error:
        raise ValueError(f"Invalid parameters {params} for agent '{name}': {error}") from error


__all__ = [
    "BanditAgent",
    "EpsilonGreedyAgent",
    "LinTSAgent",
    "LinUCBAgent",
    "RandomAgent",
    "available_agents",
    "create_agent",
]
