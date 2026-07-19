"""Off-policy evaluation estimators."""

from __future__ import annotations

import numpy as np
import pytest

from cross_sell_rl.evaluation.ope import (
    LoggedFeedback,
    collect_logged_feedback,
    ips_value,
    snips_value,
)


def _handmade_logs() -> LoggedFeedback:
    n, n_actions = 4, 2
    return LoggedFeedback(
        contexts=np.zeros((n, 1)),
        actions=np.array([0, 1, 0, 1]),
        rewards=np.array([1.0, 0.0, 1.0, 1.0]),
        propensities=np.array([0.5, 0.25, 0.5, 0.25]),
        eligible=np.ones((n, n_actions), dtype=bool),
        customer_indices=np.zeros(n, dtype=int),
    )


def test_ips_matches_hand_computation() -> None:
    logs = _handmade_logs()
    target_probs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 1.0]])
    # weights: 1/.5=2, 1/.25=4, .5/.5=1, 1/.25=4 -> mean(w*r) = (2+0+1+4)/4
    assert ips_value(logs, target_probs) == pytest.approx(7.0 / 4.0)
    # snips: sum(w*r)/sum(w) = 7 / 11
    assert snips_value(logs, target_probs) == pytest.approx(7.0 / 11.0)


def test_non_positive_propensities_rejected() -> None:
    with pytest.raises(ValueError, match="propensities"):
        LoggedFeedback(
            contexts=np.zeros((1, 1)),
            actions=np.array([0]),
            rewards=np.array([1.0]),
            propensities=np.array([0.0]),
            eligible=np.ones((1, 2), dtype=bool),
            customer_indices=np.zeros(1, dtype=int),
        )


def test_collected_logs_have_valid_propensities(small_simulator) -> None:
    logs = collect_logged_feedback(small_simulator, n_rounds=500, rng=np.random.default_rng(0))
    assert logs.n_rounds == 500
    assert np.all(logs.propensities > 0.0)
    assert np.all(logs.propensities <= 1.0)
    # every logged action was eligible in its own row
    assert logs.eligible[np.arange(500), logs.actions].all()


def test_ips_recovers_true_value_of_oracle_policy(small_simulator) -> None:
    """Statistical check: IPS on a randomized pilot ≈ the true policy value."""
    rng = np.random.default_rng(1)
    logs = collect_logged_feedback(small_simulator, n_rounds=20000, rng=rng)
    best = small_simulator.best_actions(logs.customer_indices)
    target_probs = np.zeros((logs.n_rounds, small_simulator.n_actions))
    target_probs[np.arange(logs.n_rounds), best] = 1.0
    truth = small_simulator.oracle_value()
    assert ips_value(logs, target_probs) == pytest.approx(truth, abs=0.03)
    assert snips_value(logs, target_probs) == pytest.approx(truth, abs=0.03)
