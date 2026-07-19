"""Config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cross_sell_rl.config import ConfigError, config_from_dict, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_code_defaults() -> None:
    from_file = load_config(REPO_ROOT / "configs" / "default.yaml")
    from_code = config_from_dict({})
    assert from_file == from_code


def test_empty_config_is_valid() -> None:
    config = config_from_dict({})
    assert config.data.source == "synthetic"
    assert config.products.catalog == ("home", "life", "health", "travel", "pet")
    assert config.reward.type == "conversion"
    assert "linucb" in config.experiment.agents


def test_partial_override_keeps_other_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"experiment": {"n_rounds": 50}}))
    config = load_config(path)
    assert config.experiment.n_rounds == 50
    assert config.experiment.seed == 42  # untouched default


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("no/such/config.yaml")


def test_unknown_source_rejected() -> None:
    with pytest.raises(ConfigError, match="data.source"):
        config_from_dict({"data": {"source": "excel"}})


def test_csv_source_requires_path() -> None:
    with pytest.raises(ConfigError, match="data.path"):
        config_from_dict({"data": {"source": "csv"}})


def test_unknown_reward_type_rejected() -> None:
    with pytest.raises(ConfigError, match="reward.type"):
        config_from_dict({"reward": {"type": "clicks"}})


def test_revenue_reward_requires_premiums() -> None:
    with pytest.raises(ConfigError, match="premium"):
        config_from_dict(
            {
                "reward": {"type": "revenue"},
                "products": {"catalog": ["home", "boat"], "premiums": {"home": 950.0}},
            }
        )


def test_duplicate_products_rejected() -> None:
    with pytest.raises(ConfigError, match="duplicate"):
        config_from_dict({"products": {"catalog": ["home", "home"]}})


def test_databricks_source_requires_connection_fields() -> None:
    with pytest.raises(ConfigError, match="databricks"):
        config_from_dict({"data": {"source": "databricks"}})
