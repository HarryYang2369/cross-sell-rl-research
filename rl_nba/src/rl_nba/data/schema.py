"""Schema checks and ownership helpers shared by all data sources."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rl_nba.config import SchemaConfig


class SchemaValidationError(ValueError):
    """Raised when a customer table does not match the configured schema."""


def required_columns(schema: SchemaConfig, catalog: Sequence[str]) -> list[str]:
    """All column names the configured schema expects in a customer table."""
    return [
        schema.customer_id,
        *schema.numeric_features,
        *schema.categorical_features,
        *(schema.owned_column(product) for product in catalog),
    ]


def validate_customer_frame(
    frame: pd.DataFrame, schema: SchemaConfig, catalog: Sequence[str]
) -> None:
    """Raise :class:`SchemaValidationError` if required columns are missing."""
    missing = [
        column for column in required_columns(schema, catalog) if column not in frame.columns
    ]
    if missing:
        found = list(frame.columns)
        raise SchemaValidationError(
            f"Customer data is missing {len(missing)} column(s) required by the config: "
            f"{missing}. Either rename the data columns or update data.schema / "
            f"products.catalog in the config to match. Columns found: {found}"
        )


def ownership_matrix(
    frame: pd.DataFrame, schema: SchemaConfig, catalog: Sequence[str]
) -> np.ndarray:
    """Boolean matrix of shape ``(n_customers, n_products)``: True where owned."""
    columns = [schema.owned_column(product) for product in catalog]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return values > 0.5
