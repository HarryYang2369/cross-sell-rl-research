"""Command-line entry point: run the configured cross-sell experiment.

Usage::

    python -m cross_sell_rl.run --config configs/default.yaml
    cross-sell-rl --config configs/default.yaml        # console script
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cross_sell_rl.agents import create_agent
from cross_sell_rl.config import AppConfig, load_config
from cross_sell_rl.data import load_customers, ownership_matrix
from cross_sell_rl.env import ConversionModel, CrossSellSimulator
from cross_sell_rl.evaluation import run_simulation, summarize_results
from cross_sell_rl.evaluation.plots import plot_learning_curves
from cross_sell_rl.features import FeatureEncoder
from cross_sell_rl.state import StateBuilder


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the cross-sell bandit experiment described by a YAML config."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print(
        f"  data source: {config.data.source} | "
        f"products: {', '.join(config.products.catalog)} | reward: {config.reward.type}"
    )

    customers = load_customers(config)
    builder: StateBuilder | None = None
    if config.state is not None:
        builder = StateBuilder(config.data.schema, config.products.catalog, config.state)
        contexts = builder.fit_transform(customers)
        context_dim = builder.context_dim
    else:
        encoder = FeatureEncoder(config.data.schema, config.products.catalog)
        contexts = encoder.fit_transform(customers)
        context_dim = encoder.context_dim
    owned = ownership_matrix(customers, config.data.schema, config.products.catalog)
    simulator = build_simulator(config, contexts, owned)

    print(
        f"  customers: {len(customers)} loaded, "
        f"{simulator.n_customers} with at least one product to offer"
    )
    print(f"  context vector: {context_dim} dimensions")
    if builder is not None:
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

    experiment = config.experiment
    sequence_rng = np.random.default_rng(experiment.seed)
    customer_sequence = sequence_rng.integers(
        0, simulator.n_customers, size=experiment.n_rounds
    )

    print(
        f"\nRunning {len(experiment.agents)} agents for "
        f"{experiment.n_rounds:,} rounds each (same customer sequence for all)..."
    )
    results = []
    for index, (label, raw_params) in enumerate(experiment.agents.items()):
        params = dict(raw_params)
        agent_type = str(params.pop("type", label))
        feature_groups = params.pop("features", None)
        include_derived = bool(params.pop("derived", True))
        feature_indices = None
        if builder is not None and (feature_groups is not None or not include_derived):
            feature_indices = builder.columns_for(feature_groups, include_derived)
        agent_dim = context_dim if feature_indices is None else len(feature_indices)
        agent = create_agent(
            agent_type,
            n_actions=simulator.n_actions,
            context_dim=agent_dim,
            rng=np.random.default_rng([experiment.seed, index]),
            **params,
        )
        started = time.perf_counter()
        result = run_simulation(
            agent,
            simulator,
            customer_sequence,
            rng=np.random.default_rng([experiment.seed, index, 1]),
            feature_indices=feature_indices,
            label=label,
        )
        elapsed = time.perf_counter() - started
        print(
            f"  {label:<12} ({agent_type}, {agent_dim} features)  "
            f"mean reward {result.mean_reward:.4f}   "
            f"cumulative regret {result.cumulative_regret:9.1f}   ({elapsed:.1f}s)"
        )
        results.append(result)

    summary = summarize_results(results)
    print("\nSummary (sorted by mean reward):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    output_dir = Path(experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    summary.to_csv(metrics_path, index=False)
    subtitle = (
        f"{experiment.n_rounds:,} simulated offers per agent | reward: {config.reward.type} | "
        "oracle = always offering the best eligible product"
    )
    plot_path = plot_learning_curves(
        results, simulator.oracle_value(), output_dir / "learning_curves.png", subtitle=subtitle
    )
    print(f"\nSaved {metrics_path} and {plot_path}")
    print(
        "Next: set data.source to csv/parquet (with data.path) in the config "
        "to run the same pipeline on a Databricks export."
    )


if __name__ == "__main__":
    main()
