"""Standardized customer journey model.

Maps a customer's **config** feature row onto two canonical journey axes:

* **life stage** — where they are in life (young single … retirement / legacy),
* **relationship stage** — where they are with us (new … multi-product … at-risk).

Both are derived only from columns that already exist in ``config.yml``; any
column that is absent degrades gracefully to ``"unknown"`` rather than erroring,
so the model is safe across schemas. The rules are intentionally simple and
transparent — they are a labelling convention for training, visualization, and
explainability, not a predictive model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

LIFE_STAGES = (
    "young_single",
    "newly_married",
    "new_parents",
    "established_family",
    "affluent_accumulator",
    "pre_retirement",
    "retirement",
    "legacy_estate",
    "unknown",
)

RELATIONSHIP_STAGES = (
    "new_or_prospect",
    "single_product",
    "multi_product",
    "active_growing",
    "dormant",
    "at_risk",
    "unknown",
)


@dataclass(frozen=True)
class JourneyState:
    """Where a customer sits on both journey axes at one point in time."""

    life_stage: str
    relationship_stage: str

    @property
    def label(self) -> str:
        return f"{self.life_stage} / {self.relationship_stage}"


def _num(row: Any, column: str) -> float | None:
    if column not in row:
        return None
    try:
        value = float(row[column])
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def _cat(row: Any, column: str) -> str | None:
    if column not in row:
        return None
    value = row[column]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def owned_count(row: Any, products: Sequence[str], prefix: str = "has_") -> int:
    """How many catalog products the customer already holds (config ownership flags)."""
    return sum(1 for p in products if (_num(row, f"{prefix}{p}") or 0.0) > 0.5)


def infer_life_stage(row: Any) -> str:
    """Life stage from age, refined by marital status, wealth, and holdings."""
    age = _num(row, "customer_age")
    if age is None:
        age = _num(row, "age")
    if age is None:
        return "unknown"
    wealth = _cat(row, "wealth_segment")
    affluent = wealth in ("affluent", "high_net_worth")
    hnw = wealth == "high_net_worth"
    marital = _cat(row, "customer_marital_status")
    holdings = _num(row, "customer_holdings_count") or 0.0

    if age >= 63:
        return "retirement"
    if age >= 58 and hnw:
        return "legacy_estate"
    if age >= 52:
        return "pre_retirement"
    if 40 <= age <= 58 and affluent:
        return "affluent_accumulator"
    if age >= 38:
        return "established_family"
    if marital == "married":
        # newly married vs. new parents: proxy family formation by portfolio build-up
        return "new_parents" if holdings >= 2 else "newly_married"
    if age < 30:
        return "young_single"
    return "established_family"


def infer_relationship_stage(row: Any, products: Sequence[str], prefix: str = "has_") -> str:
    """Relationship stage from attrition signals, recent activity, and breadth."""
    lapsed = _num(row, "customer_lapsed_policy_count_past_12m") or 0.0
    surrendered = _num(row, "customer_surrender_policy_count_past_12m") or 0.0
    if lapsed > 0 or surrendered > 0:
        return "at_risk"
    recent = _num(row, "customer_purchase_count_past_3m") or 0.0
    past_year = _num(row, "customer_purchase_count_past_12m") or 0.0
    held = owned_count(row, products, prefix)
    if recent > 0:
        return "active_growing"
    if held == 0:
        return "new_or_prospect"
    if held == 1:
        return "single_product"
    if past_year == 0:
        return "dormant"
    return "multi_product"


def infer_journey(row: Any, products: Sequence[str], prefix: str = "has_") -> JourneyState:
    """Full journey state (both axes) for a customer feature row."""
    return JourneyState(
        life_stage=infer_life_stage(row),
        relationship_stage=infer_relationship_stage(row, products, prefix),
    )
