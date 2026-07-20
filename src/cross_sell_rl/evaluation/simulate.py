"""Run agents against the simulator and summarize how well they learned."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cross_sell_rl.agents.base import BanditAgent
from cross_sell_rl.env.simulator import CrossSellSimulator


@dataclass(frozen=True)
class SimulationResult:
    """Per-round trajectories of one agent's simulation run.

    ``rewards`` are the noisy observed rewards; ``expected_rewards`` are the
    true expected reward of each chosen action (known only to the simulator),
    which give denoised learning curves. ``regrets`` measure, per round, how
    much expected reward was lost versus the oracle's best eligible product.
    """

    agent_name: str
    rewards: np.ndarray
    expected_rewards: np.ndarray
    regrets: np.ndarray

    @property
    def n_rounds(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())

    @property
    def mean_reward(self) -> float:
        return float(self.rewards.mean())

    @property
    def cumulative_regret(self) -> float:
        return float(self.regrets.sum())


def run_simulation(
    agent: BanditAgent,
    simulator: CrossSellSimulator,
    customer_sequence: np.ndarray,
    rng: np.random.Generator,
    feature_indices: np.ndarray | None = None,
    label: str | None = None,
) -> SimulationResult:
    """Play ``agent`` through the given customer sequence, learning as it goes.

    Passing the same ``customer_sequence`` to every agent makes comparisons
    fair: all agents face the same customers in the same order.
    ``feature_indices`` restricts which context dimensions the agent sees
    (the simulator itself always acts on the full state); ``label`` names the
    run in results, defaulting to the agent's registry name.
    """
    n_rounds = len(customer_sequence)
    rewards = np.zeros(n_rounds)
    expected = np.zeros(n_rounds)
    regrets = np.zeros(n_rounds)
    for round_index, customer in enumerate(customer_sequence):
        customer = int(customer)
        full_context = simulator.context(customer)
        context = full_context if feature_indices is None else full_context[feature_indices]
        eligible = simulator.eligible_actions(customer)
        action = agent.select_action(context, eligible)
        reward = simulator.step(customer, action, rng)
        agent.update(context, action, reward)
        rewards[round_index] = reward
        expected[round_index] = simulator.expected_reward(customer, action)
        regrets[round_index] = simulator.best_expected_reward(customer) - expected[round_index]
    return SimulationResult(
        agent_name=label or agent.name,
        rewards=rewards,
        expected_rewards=expected,
        regrets=regrets,
    )


def summarize_results(results: list[SimulationResult]) -> pd.DataFrame:
    """One row per agent: reward, regret, and uplift over the random baseline."""
    baseline = next((r.mean_reward for r in results if r.agent_name == "random"), None)
    rows = []
    for result in results:
        uplift = (
            (result.mean_reward / baseline - 1.0) * 100.0
            if baseline is not None and baseline > 0
            else float("nan")
        )
        rows.append(
            {
                "agent": result.agent_name,
                "mean_reward": result.mean_reward,
                "total_reward": result.total_reward,
                "cumulative_regret": result.cumulative_regret,
                "uplift_vs_random_pct": uplift,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("mean_reward", ascending=False, ignore_index=True)
