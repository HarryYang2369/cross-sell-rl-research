"""Simulated environment invariants."""

from __future__ import annotations

import numpy as np
import pytest

from cross_sell_rl.env import ConversionModel, CrossSellSimulator


def _tiny_simulator(action_values: np.ndarray) -> CrossSellSimulator:
    contexts = np.array([[1.0, 0.2], [1.0, -0.4], [1.0, 1.0]])
    owned = np.array([[True, False], [False, False], [True, True]])  # last row: nothing to offer
    model = ConversionModel.sample(
        context_dim=2,
        n_actions=2,
        base_conversion_rate=0.2,
        context_influence=1.0,
        rng=np.random.default_rng(0),
    )
    return CrossSellSimulator(contexts, owned, model, action_values)


def test_customers_with_nothing_to_offer_are_dropped() -> None:
    simulator = _tiny_simulator(np.ones(2))
    assert simulator.n_customers == 2


def test_eligible_actions_exclude_owned_products() -> None:
    simulator = _tiny_simulator(np.ones(2))
    assert simulator.eligible_actions(0).tolist() == [1]
    assert simulator.eligible_actions(1).tolist() == [0, 1]


def test_step_rejects_owned_product() -> None:
    simulator = _tiny_simulator(np.ones(2))
    with pytest.raises(ValueError, match="not eligible"):
        simulator.step(0, 0, np.random.default_rng(1))


def test_conversion_rewards_are_binary() -> None:
    simulator = _tiny_simulator(np.ones(2))
    rng = np.random.default_rng(2)
    rewards = {simulator.step(1, 0, rng) for _ in range(200)}
    assert rewards <= {0.0, 1.0}
    assert len(rewards) == 2  # both outcomes occur at these probabilities


def test_revenue_rewards_use_action_values() -> None:
    simulator = _tiny_simulator(np.array([10.0, 250.0]))
    rng = np.random.default_rng(3)
    rewards = {simulator.step(1, 1, rng) for _ in range(200)}
    assert rewards <= {0.0, 250.0}


def test_best_expected_reward_bounds_every_eligible_action() -> None:
    simulator = _tiny_simulator(np.ones(2))
    for customer in range(simulator.n_customers):
        best = simulator.best_expected_reward(customer)
        for action in simulator.eligible_actions(customer):
            assert best >= simulator.expected_reward(customer, action) - 1e-12


def test_step_frequency_matches_expected_reward() -> None:
    simulator = _tiny_simulator(np.ones(2))
    rng = np.random.default_rng(4)
    draws = np.mean([simulator.step(1, 1, rng) for _ in range(4000)])
    assert draws == pytest.approx(simulator.expected_reward(1, 1), abs=0.03)


def test_summary_statistics_are_sane(small_simulator) -> None:
    assert 0.0 < small_simulator.mean_conversion_probability() < 1.0
    assert 0.0 <= small_simulator.personalization_rate() <= 1.0
    assert small_simulator.oracle_value() > 0.0
