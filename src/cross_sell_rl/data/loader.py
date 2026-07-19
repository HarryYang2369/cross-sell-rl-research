"""Load customer data from whichever source the config selects."""

from __future__ import annotations

import pandas as pd

from cross_sell_rl.config import AppConfig
from cross_sell_rl.data.schema import validate_customer_frame
from cross_sell_rl.data.synthetic import generate_customers


def load_customers(config: AppConfig) -> pd.DataFrame:
    """Return the customer table for ``config.data.source``, schema-validated."""
    data = config.data
    if data.source == "synthetic":
        frame = generate_customers(
            schema=data.schema,
            products=config.products.catalog,
            n_customers=config.synthetic.n_customers,
            seed=config.synthetic.seed,
        )
    elif data.source == "csv":
        frame = pd.read_csv(data.path)
    elif data.source == "parquet":
        frame = pd.read_parquet(data.path)
    elif data.source == "databricks":
        raise NotImplementedError(
            "Direct Databricks access is scaffolded but needs workspace credentials. "
            "Today, either export the table and use data.source csv/parquet with "
            "data.path, or implement this branch with databricks-sql-connector "
            "reading data.databricks.{catalog,schema,table}."
        )
    else:  # pragma: no cover - blocked earlier by config validation
        raise ValueError(f"Unknown data source: {data.source!r}")
    validate_customer_frame(frame, data.schema, config.products.catalog)
    return frame
