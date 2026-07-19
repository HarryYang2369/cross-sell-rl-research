"""Feature encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cross_sell_rl.config import SchemaConfig
from cross_sell_rl.features import FeatureEncoder, NotFittedError

SCHEMA = SchemaConfig(
    numeric_features=("age", "income"),
    categorical_features=("region",),
    owned_product_prefix="has_",
)
PRODUCTS = ("home", "pet")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d"],
            "age": [20.0, 30.0, 40.0, 50.0],
            "income": [1000.0, 2000.0, 3000.0, 4000.0],
            "region": ["north", "south", "north", "east"],
            "has_home": [1, 0, 0, 1],
            "has_pet": [0, 0, 1, 0],
        }
    )


def test_layout_and_shape() -> None:
    encoder = FeatureEncoder(SCHEMA, PRODUCTS)
    matrix = encoder.fit_transform(_frame())
    # 1 intercept + 2 numeric + 3 region categories + 2 ownership flags
    assert matrix.shape == (4, 8)
    assert encoder.context_dim == 8
    assert encoder.feature_names == [
        "intercept", "age", "income",
        "region=east", "region=north", "region=south",
        "owns_home", "owns_pet",
    ]
    assert np.allclose(matrix[:, 0], 1.0)


def test_numeric_standardization() -> None:
    encoder = FeatureEncoder(SCHEMA, PRODUCTS)
    matrix = encoder.fit_transform(_frame())
    assert abs(matrix[:, 1].mean()) < 1e-9
    assert abs(matrix[:, 1].std(ddof=1) - 1.0) < 1e-9


def test_one_hot_rows_sum_to_one_for_known_categories() -> None:
    encoder = FeatureEncoder(SCHEMA, PRODUCTS)
    matrix = encoder.fit_transform(_frame())
    onehot_block = matrix[:, 3:6]
    assert np.allclose(onehot_block.sum(axis=1), 1.0)


def test_unseen_category_encodes_as_zeros() -> None:
    encoder = FeatureEncoder(SCHEMA, PRODUCTS).fit(_frame())
    new_row = _frame().iloc[[0]].assign(region="atlantis")
    matrix = encoder.transform(new_row)
    assert np.allclose(matrix[0, 3:6], 0.0)


def test_missing_numeric_imputed_with_fitted_mean() -> None:
    encoder = FeatureEncoder(SCHEMA, PRODUCTS).fit(_frame())
    new_row = _frame().iloc[[0]].assign(age=np.nan)
    matrix = encoder.transform(new_row)
    assert matrix[0, 1] == pytest.approx(0.0)  # fitted mean standardizes to 0


def test_transform_before_fit_raises() -> None:
    encoder = FeatureEncoder(SCHEMA, PRODUCTS)
    with pytest.raises(NotFittedError):
        encoder.transform(_frame())
