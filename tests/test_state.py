"""Grouped state config, derived features, and per-model masks."""

from __future__ import annotations

import pandas as pd
import pytest

from cross_sell_rl.config import ConfigError, config_from_dict
from cross_sell_rl.data import generate_customers
from cross_sell_rl.state import StateBuilder


def _grouped_config_dict() -> dict:
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
                "agent_context": {"numeric": ["agent_sales_count_past_12m"]},
            },
            "trends": [
                {
                    "short": "customer_purchase_count_past_3m",
                    "long": "customer_purchase_count_past_12m",
                }
            ],
            "coverage_gaps": {"segment_by": ["wealth_segment"]},
        },
        "products": {"catalog": ["medical", "saving"]},
        "synthetic": {"n_customers": 300, "seed": 4},
    }


@pytest.fixture()
def grouped_builder() -> StateBuilder:
    config = config_from_dict(_grouped_config_dict())
    frame = generate_customers(
        schema=config.data.schema,
        products=config.products.catalog,
        n_customers=300,
        seed=4,
    )
    builder = StateBuilder(
        config.data.schema, config.products.catalog, config.state, min_segment_size=5
    )
    return builder.fit(frame)


def test_schema_is_assembled_from_active_groups() -> None:
    config = config_from_dict(_grouped_config_dict())
    assert config.data.schema.numeric_features == (
        "customer_age",
        "customer_purchase_count_past_3m",
        "customer_purchase_count_past_12m",
        "agent_sales_count_past_12m",
    )
    assert config.data.schema.categorical_features == ("wealth_segment",)


def test_flat_schema_lists_conflict_with_groups() -> None:
    raw = _grouped_config_dict()
    raw["data"] = {"schema": {"numeric_features": ["customer_age"]}}
    with pytest.raises(ConfigError, match="feature_groups"):
        config_from_dict(raw)


def test_direct_delivery_drops_agent_context() -> None:
    raw = _grouped_config_dict()
    raw["state"]["delivery"] = "direct"
    config = config_from_dict(raw)
    assert "agent_context" not in config.state.active_group_names
    assert "agent_sales_count_past_12m" not in config.data.schema.numeric_features


def test_explicit_enabled_overrides_delivery() -> None:
    raw = _grouped_config_dict()
    raw["state"]["delivery"] = "direct"
    raw["state"]["feature_groups"]["agent_context"]["enabled"] = True
    config = config_from_dict(raw)
    assert "agent_context" in config.state.active_group_names


def test_trend_columns_must_be_active_numeric_features() -> None:
    raw = _grouped_config_dict()
    raw["state"]["trends"] = [{"short": "customer_age", "long": "no_such_column"}]
    with pytest.raises(ConfigError, match="no_such_column"):
        config_from_dict(raw)


def test_segment_by_must_be_categorical() -> None:
    raw = _grouped_config_dict()
    raw["state"]["coverage_gaps"] = {"segment_by": ["customer_age"]}
    with pytest.raises(ConfigError, match="segment_by"):
        config_from_dict(raw)


def test_duplicate_columns_across_groups_rejected() -> None:
    raw = _grouped_config_dict()
    raw["state"]["feature_groups"]["extra"] = {"numeric": ["customer_age"]}
    with pytest.raises(ConfigError, match="more than one"):
        config_from_dict(raw)


def test_agent_features_must_reference_active_groups() -> None:
    raw = _grouped_config_dict()
    raw["experiment"] = {"agents": {"model": {"type": "linucb", "features": ["nope"]}}}
    with pytest.raises(ConfigError, match="nope"):
        config_from_dict(raw)


def test_agent_features_require_grouped_state() -> None:
    with pytest.raises(ConfigError, match="features"):
        config_from_dict(
            {"experiment": {"agents": {"model": {"type": "linucb", "features": ["profile"]}}}}
        )


def test_feature_names_include_derived_features(grouped_builder: StateBuilder) -> None:
    names = grouped_builder.feature_names
    trend = "trend_customer_purchase_count_past_3m_over_customer_purchase_count_past_12m"
    assert trend in names
    assert "coverage_gap_medical" in names
    assert "coverage_gap_saving" in names
    assert grouped_builder.context_dim == len(names)


def test_trend_feature_is_standardized(grouped_builder: StateBuilder) -> None:
    config = config_from_dict(_grouped_config_dict())
    frame = generate_customers(
        schema=config.data.schema,
        products=config.products.catalog,
        n_customers=300,
        seed=4,
    )
    matrix = grouped_builder.transform(frame)
    trend_column = matrix[:, grouped_builder.feature_names.index(
        "trend_customer_purchase_count_past_3m_over_customer_purchase_count_past_12m"
    )]
    assert abs(trend_column.mean()) < 1e-9
    assert abs(trend_column.std() - 1.0) < 1e-9


def test_columns_for_restricts_to_selected_groups(grouped_builder: StateBuilder) -> None:
    indices = grouped_builder.columns_for(["profile"], include_derived=False)
    selected = [grouped_builder.feature_names[i] for i in indices]
    assert "intercept" in selected
    assert "owns_medical" in selected and "owns_saving" in selected
    assert "customer_age" in selected
    assert any(name.startswith("wealth_segment=") for name in selected)
    assert "customer_purchase_count_past_3m" not in selected
    assert not any(name.startswith(("trend_", "coverage_gap_")) for name in selected)

    full = grouped_builder.columns_for(None, include_derived=True)
    assert full.tolist() == list(range(grouped_builder.context_dim))


def test_columns_for_unknown_group_raises(grouped_builder: StateBuilder) -> None:
    with pytest.raises(ValueError, match="Unknown or disabled"):
        grouped_builder.columns_for(["holdings"])


def test_coverage_gap_values_are_segment_rate_minus_ownership() -> None:
    raw = {
        "state": {
            "feature_groups": {"g": {"numeric": ["x"], "categorical": ["seg"]}},
            "coverage_gaps": {"segment_by": ["seg"]},
        },
        "products": {"catalog": ["a", "b"]},
    }
    config = config_from_dict(raw)
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(8)],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "seg": ["s1"] * 4 + ["s2"] * 4,
            "has_a": [1, 1, 1, 0, 0, 0, 0, 0],
            "has_b": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    builder = StateBuilder(
        config.data.schema, config.products.catalog, config.state, min_segment_size=2
    ).fit(frame)
    matrix = builder.transform(frame)
    gap_a = matrix[:, builder.feature_names.index("coverage_gap_a")]
    # segment s1 owns product a at rate 0.75: owners sit below, the one gap at +0.75
    assert gap_a[:4] == pytest.approx([-0.25, -0.25, -0.25, 0.75])
    assert gap_a[4:] == pytest.approx([0.0, 0.0, 0.0, 0.0])

    # unseen segment falls back to the population-wide ownership rate (3/8)
    unseen = frame.iloc[[0]].assign(seg="s3", has_a=0)
    gap_unseen = builder.transform(unseen)[0, builder.feature_names.index("coverage_gap_a")]
    assert gap_unseen == pytest.approx(3 / 8)
