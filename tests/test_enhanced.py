"""The enhanced production-schema config: reward sources and labeled models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cross_sell_rl.config import ConfigError, config_from_dict, load_config
from cross_sell_rl.run import build_simulator, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_enhanced_yaml_loads_and_validates() -> None:
    config = load_config(REPO_ROOT / "configs" / "enhanced.yaml")
    assert config.reward.type == "vnb"
    assert config.state is not None
    assert "agent_context" in config.state.active_group_names  # delivery: mixed
    assert set(config.experiment.agents) == {"random", "baseline", "enhanced"}
    assert config.experiment.agents["baseline"]["type"] == "linucb"


def test_vnb_reward_requires_value_table() -> None:
    with pytest.raises(ConfigError, match="products.vnb"):
        config_from_dict({"reward": {"type": "vnb"}})


def test_ape_reward_requires_value_table() -> None:
    with pytest.raises(ConfigError, match="products.ape"):
        config_from_dict({"reward": {"type": "ape"}})


def test_vnb_rewards_use_configured_values() -> None:
    config = config_from_dict(
        {
            "products": {"catalog": ["a", "b"], "vnb": {"a": 10.0, "b": 20.0}},
            "reward": {"type": "vnb"},
        }
    )
    contexts = np.hstack([np.ones((6, 1)), np.linspace(-1, 1, 6)[:, np.newaxis]])
    owned = np.zeros((6, 2), dtype=bool)
    simulator = build_simulator(config, contexts, owned)
    rng = np.random.default_rng(0)
    rewards = {simulator.step(0, 0, rng) for _ in range(300)}
    assert rewards <= {0.0, 10.0}
    assert 10.0 in rewards


def _small_grouped_experiment(tmp_path: Path) -> dict:
    return {
        "state": {
            "delivery": "mixed",
            "feature_groups": {
                "profile": {
                    "numeric": ["customer_age"],
                    "categorical": ["wealth_segment"],
                },
                "behaviour": {
                    "numeric": [
                        "customer_purchase_count_past_3m",
                        "customer_purchase_count_past_12m",
                    ]
                },
            },
            "trends": [
                {
                    "short": "customer_purchase_count_past_3m",
                    "long": "customer_purchase_count_past_12m",
                }
            ],
            "coverage_gaps": {"segment_by": ["wealth_segment"]},
        },
        "products": {"catalog": ["medical", "saving", "annuity"]},
        "synthetic": {"n_customers": 250, "seed": 1},
        "experiment": {
            "n_rounds": 300,
            "seed": 2,
            "output_dir": str(tmp_path / "results"),
            "agents": {
                "random": {},
                "baseline": {"type": "linucb", "features": ["profile"], "derived": False},
                "enhanced": {"type": "linucb"},
            },
        },
    }


def test_labeled_models_run_end_to_end(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_small_grouped_experiment(tmp_path)))

    main(["--config", str(config_path)])

    metrics = pd.read_csv(tmp_path / "results" / "metrics.csv")
    assert set(metrics["agent"]) == {"random", "baseline", "enhanced"}
    assert (tmp_path / "results" / "learning_curves.png").exists()


def test_unknown_model_type_rejected(tmp_path: Path) -> None:
    raw = _small_grouped_experiment(tmp_path)
    raw["experiment"]["agents"] = {"fancy": {"type": "deep_q_network"}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Unknown agent"):
        main(["--config", str(config_path)])
