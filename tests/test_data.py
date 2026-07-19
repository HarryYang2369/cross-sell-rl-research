"""Synthetic generation, schema validation, and source loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cross_sell_rl.config import SchemaConfig, config_from_dict
from cross_sell_rl.data import (
    SchemaValidationError,
    generate_customers,
    load_customers,
    ownership_matrix,
    required_columns,
    validate_customer_frame,
)

PRODUCTS = ("home", "life", "pet")


def test_generated_frame_matches_schema() -> None:
    schema = SchemaConfig()
    frame = generate_customers(schema, PRODUCTS, n_customers=200, seed=1)
    assert len(frame) == 200
    assert list(frame.columns) == required_columns(schema, PRODUCTS)
    validate_customer_frame(frame, schema, PRODUCTS)  # should not raise
    for product in PRODUCTS:
        assert set(frame[schema.owned_column(product)].unique()) <= {0, 1}


def test_generation_is_deterministic() -> None:
    schema = SchemaConfig()
    first = generate_customers(schema, PRODUCTS, n_customers=100, seed=9)
    second = generate_customers(schema, PRODUCTS, n_customers=100, seed=9)
    different = generate_customers(schema, PRODUCTS, n_customers=100, seed=10)
    pd.testing.assert_frame_equal(first, second)
    assert not first.drop(columns=[schema.customer_id]).equals(
        different.drop(columns=[schema.customer_id])
    )


def test_unknown_columns_fall_back_to_generic_distributions() -> None:
    schema = SchemaConfig(
        numeric_features=("wibble_score",), categorical_features=("wobble_type",)
    )
    frame = generate_customers(schema, PRODUCTS, n_customers=50, seed=2)
    assert frame["wibble_score"].dtype.kind == "f"
    assert frame["wobble_type"].str.startswith("wobble_type_").all()


def test_validation_reports_missing_columns() -> None:
    schema = SchemaConfig()
    frame = generate_customers(schema, PRODUCTS, n_customers=10, seed=3)
    broken = frame.drop(columns=["age", schema.owned_column("pet")])
    with pytest.raises(SchemaValidationError, match="age"):
        validate_customer_frame(broken, schema, PRODUCTS)


def test_ownership_matrix_shape_and_values() -> None:
    schema = SchemaConfig()
    frame = generate_customers(schema, PRODUCTS, n_customers=40, seed=4)
    owned = ownership_matrix(frame, schema, PRODUCTS)
    assert owned.shape == (40, len(PRODUCTS))
    assert owned.dtype == bool


def test_load_customers_from_csv_roundtrip(tmp_path: Path) -> None:
    base = config_from_dict({"synthetic": {"n_customers": 30, "seed": 5}})
    frame = generate_customers(
        base.data.schema, base.products.catalog, n_customers=30, seed=5
    )
    csv_path = tmp_path / "customers.csv"
    frame.to_csv(csv_path, index=False)
    config = config_from_dict({"data": {"source": "csv", "path": str(csv_path)}})
    loaded = load_customers(config)
    assert len(loaded) == 30
    assert list(loaded.columns) == list(frame.columns)


def test_load_customers_csv_with_wrong_schema_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "wrong.csv"
    pd.DataFrame({"customer": [1, 2], "age": [30, 40]}).to_csv(csv_path, index=False)
    config = config_from_dict({"data": {"source": "csv", "path": str(csv_path)}})
    with pytest.raises(SchemaValidationError):
        load_customers(config)


def test_databricks_source_is_a_clear_stub() -> None:
    config = config_from_dict(
        {
            "data": {
                "source": "databricks",
                "databricks": {"catalog": "main", "schema": "crm", "table": "customers"},
            }
        }
    )
    with pytest.raises(NotImplementedError, match="csv/parquet"):
        load_customers(config)
