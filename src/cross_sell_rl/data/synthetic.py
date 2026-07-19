"""Synthetic insurance-customer generation.

The generator reads the same schema config as every other data source, so the
frame it produces has exactly the shape real data will have — switching
``data.source`` from ``synthetic`` to ``csv``/``parquet`` later requires no
code changes.

Column names the generator recognises (``age``, ``region``, …) get realistic
insurance-flavoured distributions; any other configured column falls back to a
generic distribution, so custom schemas still work out of the box.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from cross_sell_rl.config import SchemaConfig

_KNOWN_NUMERIC: dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "age": lambda rng, n: np.clip(rng.normal(42.0, 14.0, n), 18, 85).round(0),
    "tenure_years": lambda rng, n: np.clip(rng.exponential(6.0, n), 0, 40).round(1),
    "annual_premium": lambda rng, n: rng.lognormal(7.0, 0.45, n).round(2),
    "num_claims": lambda rng, n: rng.poisson(0.4, n).astype(float),
    "credit_score": lambda rng, n: np.clip(rng.normal(680.0, 75.0, n), 300, 850).round(0),
}

_KNOWN_CATEGORICAL: dict[str, tuple[list[str], list[float]]] = {
    "region": (["north", "south", "east", "west", "central"], [0.24, 0.22, 0.20, 0.18, 0.16]),
    "acquisition_channel": (["agent", "online", "phone", "partner"], [0.40, 0.30, 0.15, 0.15]),
}

# Ownership probability of each product is drawn once per product from this range.
_OWNERSHIP_PROB_RANGE = (0.08, 0.35)


def _generic_numeric(rng: np.random.Generator, n: int) -> np.ndarray:
    location = rng.uniform(0.0, 100.0)
    scale = rng.uniform(5.0, 25.0)
    return rng.normal(location, scale, n).round(2)


def _generic_categorical(rng: np.random.Generator, n: int, column: str) -> np.ndarray:
    categories = [f"{column}_{suffix}" for suffix in "abcd"]
    return rng.choice(categories, size=n)


def generate_customers(
    schema: SchemaConfig,
    products: Sequence[str],
    n_customers: int,
    seed: int,
) -> pd.DataFrame:
    """Generate a deterministic synthetic customer table matching ``schema``."""
    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray | list[str]] = {
        schema.customer_id: [f"C{i:06d}" for i in range(n_customers)]
    }

    for name in schema.numeric_features:
        generator = _KNOWN_NUMERIC.get(name)
        if generator is not None:
            columns[name] = generator(rng, n_customers)
        else:
            columns[name] = _generic_numeric(rng, n_customers)

    for name in schema.categorical_features:
        if name in _KNOWN_CATEGORICAL:
            categories, weights = _KNOWN_CATEGORICAL[name]
            columns[name] = rng.choice(categories, size=n_customers, p=weights)
        else:
            columns[name] = _generic_categorical(rng, n_customers, name)

    low, high = _OWNERSHIP_PROB_RANGE
    for product in products:
        ownership_rate = rng.uniform(low, high)
        columns[schema.owned_column(product)] = (
            rng.random(n_customers) < ownership_rate
        ).astype(int)

    return pd.DataFrame(columns)
