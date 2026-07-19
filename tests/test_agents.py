"""Agent behavior: factory, masking, and learning sanity checks."""

from __future__ import annotations

import numpy as np
import pytest

from cross_sell_rl.agents import available_agents, create_agent
from cross_sell_rl.agents.base import BanditAgent

CONTEXT_DIM = 2


def _make(name: str, seed: int = 0, **params) -> BanditAgent:
    return create_agent(
        name,
        n_actions=2,
        context_dim=CONTEXT_DIM,
        rng=np.random.default_rng(seed),
        **params,
    )


def _context(rng: np.random.Generator) -> np.ndarray:
    return np.array([1.0, rng.uniform(-1.0, 1.0)])


def _reward(action: int, context: np.ndarray) -> float:
    """Action 0 pays when the feature is positive, action 1 when negative."""
    return 1.0 if (action == 0) == (context[1] > 0) else 0.0


def _train(agent: BanditAgent, rounds: int = 500, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    both = np.array([0, 1])
    for _ in range(rounds):
        context = _context(rng)
        action = agent.select_action(context, both)
        agent.update(context, action, _reward(action, context))


def _accuracy(agent: BanditAgent, seed: int = 2, samples: int = 400) -> float:
    """Correct-action rate on fresh contexts with a clear margin from zero."""
    rng = np.random.default_rng(seed)
    both = np.array([0, 1])
    correct = 0
    counted = 0
    while counted < samples:
        context = _context(rng)
        if abs(context[1]) < 0.2:
            continue
        counted += 1
        action = agent.select_action(context, both)
        correct += int(_reward(action, context) == 1.0)
    return correct / samples


def test_registry_lists_all_agents() -> None:
    assert available_agents() == ["epsilon_greedy", "lin_ts", "linucb", "random"]


def test_unknown_agent_name_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown agent"):
        _make("deep_q_network")


def test_bad_agent_params_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid parameters"):
        _make("linucb", learning_rate=0.5)


@pytest.mark.parametrize("name", ["random", "epsilon_greedy", "linucb", "lin_ts"])
def test_agents_respect_eligibility_mask(name: str) -> None:
    agent = _make(name)
    only_action_one = np.array([1])
    context = np.array([1.0, 0.5])
    for _ in range(20):
        assert agent.select_action(context, only_action_one) == 1


@pytest.mark.parametrize("name", ["linucb", "lin_ts"])
def test_contextual_agents_learn_context_dependent_rule(name: str) -> None:
    agent = _make(name)
    _train(agent)
    assert _accuracy(agent) >= 0.8


def test_context_free_agent_cannot_personalize() -> None:
    agent = _make("epsilon_greedy", epsilon=0.05)
    _train(agent)
    # Both actions pay ~50% on average, so a context-blind policy stays near chance.
    assert _accuracy(agent) <= 0.7


def test_agents_are_reproducible() -> None:
    first = _make("lin_ts", seed=7)
    second = _make("lin_ts", seed=7)
    _train(first, rounds=100)
    _train(second, rounds=100)
    rng = np.random.default_rng(3)
    context = _context(rng)
    assert first.select_action(context, np.array([0, 1])) == second.select_action(
        context, np.array([0, 1])
    )
