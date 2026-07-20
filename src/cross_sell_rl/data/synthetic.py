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


def _zero_inflated_lognormal(
    rng: np.random.Generator, n: int, mean: float, sigma: float, nonzero_share: float
) -> np.ndarray:
    return (rng.lognormal(mean, sigma, n) * (rng.random(n) < nonzero_share)).round(2)


_KNOWN_NUMERIC: dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    # --- toy default schema ---
    "age": lambda rng, n: np.clip(rng.normal(42.0, 14.0, n), 18, 85).round(0),
    "tenure_years": lambda rng, n: np.clip(rng.exponential(6.0, n), 0, 40).round(1),
    "annual_premium": lambda rng, n: rng.lognormal(7.0, 0.45, n).round(2),
    "num_claims": lambda rng, n: rng.poisson(0.4, n).astype(float),
    "credit_score": lambda rng, n: np.clip(rng.normal(680.0, 75.0, n), 300, 850).round(0),
    # --- production-schema profile & holdings (HKD-flavoured magnitudes) ---
    "customer_age": lambda rng, n: np.clip(rng.normal(41.0, 12.0, n), 21, 80).round(0),
    "customer_holdings_count": lambda rng, n: (1 + rng.poisson(1.2, n)).astype(float),
    "customer_holdings_ap": lambda rng, n: rng.lognormal(9.6, 0.8, n).round(0),
    "customer_holdings_sum_assured": lambda rng, n: rng.lognormal(13.0, 0.9, n).round(0),
    "customer_inforce_policy_holding_ap": lambda rng, n: rng.lognormal(9.5, 0.8, n).round(0),
    "customer_all_policy_holding_count": lambda rng, n: (1 + rng.poisson(1.5, n)).astype(float),
    # --- recent behaviour (window counts drawn independently per column: real
    # exports will have consistent nested windows; these match marginals only) ---
    "customer_purchase_count_past_1m": lambda rng, n: rng.poisson(0.06, n).astype(float),
    "customer_purchase_count_past_3m": lambda rng, n: rng.poisson(0.18, n).astype(float),
    "customer_purchase_count_past_6m": lambda rng, n: rng.poisson(0.35, n).astype(float),
    "customer_purchase_count_past_12m": lambda rng, n: rng.poisson(0.70, n).astype(float),
    "customer_purchased_ap_past_12m": lambda rng, n: _zero_inflated_lognormal(
        rng, n, 9.3, 1.0, 0.35
    ),
    "customer_lapsed_policy_count_past_12m": lambda rng, n: rng.poisson(0.08, n).astype(float),
    "customer_surrender_policy_count_past_12m": lambda rng, n: rng.poisson(0.05, n).astype(
        float
    ),
    # --- channel flags ---
    "tied_agency_customer": lambda rng, n: (rng.random(n) < 0.55).astype(int),
    "bancassured_customer": lambda rng, n: (rng.random(n) < 0.30).astype(int),
    "brokers_customer": lambda rng, n: (rng.random(n) < 0.15).astype(int),
    # --- high-value segment proxies ---
    "customer_inforce_private_bmu_count": lambda rng, n: rng.poisson(0.05, n).astype(float),
    "customer_inforce_maxfocus_bmu_count": lambda rng, n: rng.poisson(0.08, n).astype(float),
    "customer_inforced_wealthicon_value_usd": lambda rng, n: _zero_inflated_lognormal(
        rng, n, 11.0, 1.0, 0.07
    ),
    # --- servicing-agent performance & quality ---
    "agent_sales_count_past_12m": lambda rng, n: rng.poisson(16.0, n).astype(float),
    "agent_sales_ap_past_12m": lambda rng, n: rng.lognormal(12.2, 0.7, n).round(0),
    "agent_avg_sales_policy_count_past_12m": lambda rng, n: np.clip(
        rng.normal(1.4, 0.4, n), 0.3, 3.0
    ).round(2),
    "agent_policy_13m_lapse_rate": lambda rng, n: rng.uniform(0.0, 0.35, n).round(3),
    "agent_repurchase_count_past_6m": lambda rng, n: rng.poisson(2.0, n).astype(float),
}

_KNOWN_CATEGORICAL: dict[str, tuple[list[str], list[float]]] = {
    # --- toy default schema ---
    "region": (["north", "south", "east", "west", "central"], [0.24, 0.22, 0.20, 0.18, 0.16]),
    "acquisition_channel": (["agent", "online", "phone", "partner"], [0.40, 0.30, 0.15, 0.15]),
    # --- production-schema profile ---
    "customer_gender": (["F", "M"], [0.52, 0.48]),
    "customer_marital_status": (
        ["married", "single", "divorced", "widowed"],
        [0.48, 0.38, 0.09, 0.05],
    ),
    "customer_income_range": (["0-20k", "20-40k", "40-80k", "80k+"], [0.30, 0.35, 0.22, 0.13]),
    "wealth_segment": (
        ["mass", "emerging_affluent", "affluent", "high_net_worth"],
        [0.55, 0.25, 0.15, 0.05],
    ),
    # --- engagement (voice-of-customer satisfaction segments) ---
    "voc_purchase": (
        ["satisfied", "neutral", "dissatisfied", "not_surveyed"],
        [0.30, 0.15, 0.05, 0.50],
    ),
    "voc_service": (
        ["satisfied", "neutral", "dissatisfied", "not_surveyed"],
        [0.25, 0.15, 0.06, 0.54],
    ),
    "voc_claim": (
        ["satisfied", "neutral", "dissatisfied", "not_surveyed"],
        [0.12, 0.08, 0.05, 0.75],
    ),
    # --- servicing-agent quality tier ---
    "mdrt_label": (["none", "mdrt", "cot", "tot"], [0.80, 0.14, 0.045, 0.015]),
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
