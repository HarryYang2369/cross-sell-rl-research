"""Run the configured cross-sell experiment.

The pipeline is exposed as reusable functions so the command line, notebooks,
and tests all share one code path:

    prepare_experiment(config) -> PreparedExperiment   # data + simulator
    run_agents(prepared)       -> list[SimulationResult]
    run_experiment(config)     -> ExperimentOutcome     # the whole thing
    save_outputs(outcome)      -> (metrics.csv, learning_curves.png)

Command line::

    python -m rl_nba.run --config config/rl_nba_config.yml
    rl-nba --config config/rl_nba_config.yml        # console script
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rl_nba.agents import create_agent
from rl_nba.config import AppConfig, load_config
from rl_nba.data import load_customers, ownership_matrix
from rl_nba.environment import ConversionModel, CrossSellSimulator
from rl_nba.evaluation import SimulationResult, run_simulation, summarize_results
from rl_nba.evaluation.plots import plot_learning_curves
from rl_nba.features import FeatureEncoder
from rl_nba.state import StateBuilder

# Per-agent progress callback: (label, algorithm, feature_count, result, seconds).
AgentCallback = Callable[[str, str, int, SimulationResult, float], None]


def build_simulator(
    config: AppConfig, contexts: np.ndarray, owned: np.ndarray
) -> CrossSellSimulator:
    """Assemble the simulated environment described by the config."""
    n_actions = len(config.products.catalog)
    if config.reward.type == "conversion":
        action_values = np.ones(n_actions)
    else:
        value_sources = {
            "revenue": config.products.premiums,
            "ape": config.products.ape,
            "vnb": config.products.vnb,
        }
        mapping = value_sources[config.reward.type]
        action_values = np.array(
            [mapping[product] for product in config.products.catalog], dtype=float
        )
    model = ConversionModel.sample(
        context_dim=contexts.shape[1],
        n_actions=n_actions,
        base_conversion_rate=config.environment.base_conversion_rate,
        context_influence=config.environment.context_influence,
        rng=np.random.default_rng(config.environment.seed),
    )
    return CrossSellSimulator(contexts, owned, model, action_values)


def build_contexts(
    config: AppConfig, customers: pd.DataFrame
) -> tuple[np.ndarray, int, StateBuilder | None]:
    """Encode customers into context vectors via grouped state or a flat schema.

    Returns the context matrix, its dimension, and the fitted
    :class:`StateBuilder` when grouped state is configured (``None`` for the
    flat schema, which has no feature groups to mask on).
    """
    if config.state is not None:
        builder = StateBuilder(config.data.schema, config.products.catalog, config.state)
        return builder.fit_transform(customers), builder.context_dim, builder
    encoder = FeatureEncoder(config.data.schema, config.products.catalog)
    contexts = encoder.fit_transform(customers)
    return contexts, encoder.context_dim, None


@dataclass
class PreparedExperiment:
    """The world an experiment runs in: customers, the simulator, and the
    shared customer sequence every agent will face."""

    config: AppConfig
    customers: pd.DataFrame
    simulator: CrossSellSimulator
    context_dim: int
    builder: StateBuilder | None
    customer_sequence: np.ndarray


@dataclass
class ExperimentOutcome:
    """Everything a run produces. Returned by :func:`run_experiment` so a
    notebook or test can inspect the data, the results, and the summary."""

    prepared: PreparedExperiment
    results: list[SimulationResult]
    summary: pd.DataFrame

    @property
    def config(self) -> AppConfig:
        return self.prepared.config

    @property
    def customers(self) -> pd.DataFrame:
        return self.prepared.customers

    @property
    def simulator(self) -> CrossSellSimulator:
        return self.prepared.simulator

    @property
    def context_dim(self) -> int:
        return self.prepared.context_dim

    @property
    def builder(self) -> StateBuilder | None:
        return self.prepared.builder

    @property
    def subtitle(self) -> str:
        """One-line caption describing the run, used on the plot."""
        experiment = self.config.experiment
        return (
            f"{experiment.n_rounds:,} simulated offers per agent | "
            f"reward: {self.config.reward.type} | "
            "oracle = always offering the best eligible product"
        )


def prepare_experiment(config: AppConfig) -> PreparedExperiment:
    """Load customers, encode context, and build the simulated environment."""
    customers = load_customers(config)
    contexts, context_dim, builder = build_contexts(config, customers)
    owned = ownership_matrix(customers, config.data.schema, config.products.catalog)
    simulator = build_simulator(config, contexts, owned)
    customer_sequence = np.random.default_rng(config.experiment.seed).integers(
        0, simulator.n_customers, size=config.experiment.n_rounds
    )
    return PreparedExperiment(
        config=config,
        customers=customers,
        simulator=simulator,
        context_dim=context_dim,
        builder=builder,
        customer_sequence=customer_sequence,
    )


def run_agents(
    prepared: PreparedExperiment, on_agent: AgentCallback | None = None
) -> list[SimulationResult]:
    """Run every configured agent against the same customer sequence.

    Each agent may restrict itself to a subset of feature groups (``features``)
    and drop the derived trend/coverage-gap columns (``derived: false``); this
    is how baseline and enhanced state designs compete fairly. Pass ``on_agent``
    to receive live per-agent progress.
    """
    config = prepared.config
    experiment = config.experiment
    builder = prepared.builder
    results: list[SimulationResult] = []
    for index, (label, raw_params) in enumerate(experiment.agents.items()):
        params = dict(raw_params)
        agent_type = str(params.pop("type", label))
        feature_groups = params.pop("features", None)
        include_derived = bool(params.pop("derived", True))
        feature_indices = None
        if builder is not None and (feature_groups is not None or not include_derived):
            feature_indices = builder.columns_for(feature_groups, include_derived)
        agent_dim = prepared.context_dim if feature_indices is None else len(feature_indices)
        agent = create_agent(
            agent_type,
            n_actions=prepared.simulator.n_actions,
            context_dim=agent_dim,
            rng=np.random.default_rng([experiment.seed, index]),
            **params,
        )
        started = time.perf_counter()
        result = run_simulation(
            agent,
            prepared.simulator,
            prepared.customer_sequence,
            rng=np.random.default_rng([experiment.seed, index, 1]),
            feature_indices=feature_indices,
            label=label,
        )
        if on_agent is not None:
            on_agent(label, agent_type, agent_dim, result, time.perf_counter() - started)
        results.append(result)
    return results


def run_experiment(
    config: AppConfig, on_agent: AgentCallback | None = None
) -> ExperimentOutcome:
    """Run the full experiment end to end and return everything it produced."""
    prepared = prepare_experiment(config)
    results = run_agents(prepared, on_agent=on_agent)
    return ExperimentOutcome(
        prepared=prepared, results=results, summary=summarize_results(results)
    )


def save_outputs(
    outcome: ExperimentOutcome, output_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Write ``metrics.csv`` and ``learning_curves.png``; return their paths."""
    directory = Path(output_dir or outcome.config.experiment.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.csv"
    outcome.summary.to_csv(metrics_path, index=False)
    plot_path = plot_learning_curves(
        outcome.results,
        outcome.simulator.oracle_value(),
        directory / "learning_curves.png",
        subtitle=outcome.subtitle,
    )
    return metrics_path, plot_path


def describe(prepared: PreparedExperiment) -> None:
    """Print the human-readable summary of the prepared environment."""
    config, simulator = prepared.config, prepared.simulator
    print(
        f"  data source: {config.data.source} | "
        f"products: {', '.join(config.products.catalog)} | reward: {config.reward.type}"
    )
    print(
        f"  customers: {len(prepared.customers)} loaded, "
        f"{simulator.n_customers} with at least one product to offer"
    )
    print(f"  context vector: {prepared.context_dim} dimensions")
    if prepared.builder is not None:
        state = config.state
        n_gaps = len(config.products.catalog) if state.coverage_gaps.segment_by else 0
        print(
            f"  state: groups [{', '.join(state.active_group_names)}] "
            f"(delivery: {state.delivery}) + {len(state.trends)} trend "
            f"+ {n_gaps} coverage-gap features"
        )
    print(
        f"  simulated environment: mean conversion probability "
        f"{simulator.mean_conversion_probability():.1%}, "
        f"personalization rate {simulator.personalization_rate():.0%} "
        f"(share of customers whose best product is not the most common best)"
    )
    print(f"  oracle expected reward per offer: {simulator.oracle_value():.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the cross-sell bandit experiment described by a YAML config."
    )
    parser.add_argument(
        "--config",
        default="config/rl_nba_config.yml",
        help="Path to the YAML config (default: config/rl_nba_config.yml)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"Config: {args.config}")

    prepared = prepare_experiment(config)
    describe(prepared)
    print(
        f"\nRunning {len(config.experiment.agents)} agents for "
        f"{config.experiment.n_rounds:,} rounds each (same customer sequence for all)..."
    )

    def print_agent(
        label: str, agent_type: str, agent_dim: int, result: SimulationResult, elapsed: float
    ) -> None:
        print(
            f"  {label:<12} ({agent_type}, {agent_dim} features)  "
            f"mean reward {result.mean_reward:.4f}   "
            f"cumulative regret {result.cumulative_regret:9.1f}   ({elapsed:.1f}s)"
        )

    results = run_agents(prepared, on_agent=print_agent)
    outcome = ExperimentOutcome(
        prepared=prepared, results=results, summary=summarize_results(results)
    )

    print("\nSummary (sorted by mean reward):")
    print(outcome.summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    metrics_path, plot_path = save_outputs(outcome)
    print(f"\nSaved {metrics_path} and {plot_path}")
    print(
        "Next: set data.source to csv/parquet (with data.path) in the config "
        "to run the same pipeline on a Databricks export."
    )


if __name__ == "__main__":
    main()
