"""Digital Twin of Customer (DToC) layer.

A ``DigitalTwin`` is the per-customer abstraction that ties the whole system
together. It holds a customer's timeline of states — **historical**, **current**,
and (optionally) **projected future** — and exposes four capabilities:

* **RL training** — every state carries the exact context vector the policy is
  trained on, so the twin and the (vectorized) training pipeline share one
  world model (encoder + conversion model + reward). The twin is the readable,
  per-customer view; bulk training stays on the array pipeline for scale.
* **Scenario testing** — :meth:`DigitalTwin.project` rolls the customer forward
  under a chosen policy (or a fixed offer), producing a plausible future
  trajectory and its cumulative value. Enabled by ``dtoc.future_mode: simulate``.
* **Journey visualization** — :meth:`DigitalTwin.records` returns the full
  timeline with the standardized journey state at each step; :func:`plot_journey`
  draws it.
* **Policy explainability** — :meth:`DigitalTwin.explain` returns the policy's
  per-product value estimate and exploration bonus at the current state.

**Honesty:** projected future states are *model roll-outs*, not predictions of
the real person, and the transition model below is a transparent simplification
(a real feature-evolution/forecast model is future work). Set
``dtoc.future_mode: placeholder`` to represent only observed history + current.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from rl_nba.config import AppConfig
from rl_nba.data import load_customers
from rl_nba.environment import ConversionModel
from rl_nba.features import FeatureEncoder
from rl_nba.journey import JourneyState, infer_journey
from rl_nba.state import StateBuilder

# Policy signature used by projection: (context, eligible_action_indices) -> action index.
Policy = Callable[[np.ndarray, np.ndarray], int]

# Transition constants — deliberately simple and documented (see module docstring).
_SUM_ASSURED_PER_PREMIUM = 50.0  # rough: sum assured ≈ 50× annual premium


class DToCDisabledError(RuntimeError):
    """Raised when the DToC layer is used while ``dtoc.enabled`` is false."""


def dtoc_enabled(config: AppConfig) -> bool:
    """Whether the DToC (twin) layer is active — see ``dtoc.enabled`` in the config.

    When ``False`` the project runs in plain feature-vector mode (as before the
    DToC existed); the training pipeline is identical either way.
    """
    return config.dtoc.enabled


@dataclass
class TwinState:
    """One snapshot in a customer's timeline."""

    step: int  # 0 = current, < 0 = history, > 0 = projected future
    label: str  # "history" | "current" | "future"
    features: dict[str, Any]
    journey: JourneyState
    context: np.ndarray
    offered: str | None = None  # product offered that led to this state (future only)
    converted: bool | None = None
    reward: float | None = None


class DToCWorld:
    """Shared world model behind every twin: encoder, conversion model, reward,
    and the transition model. Built to match the training environment exactly."""

    def __init__(
        self,
        config: AppConfig,
        encoder: StateBuilder | FeatureEncoder,
        model: ConversionModel,
        action_values: np.ndarray,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.model = model
        self.action_values = action_values
        self.products = tuple(config.products.catalog)
        self.schema = config.data.schema
        self._feature_columns = [
            schema_col
            for schema_col in (
                self.schema.customer_id,
                *self.schema.numeric_features,
                *self.schema.categorical_features,
                *(self.schema.owned_column(p) for p in self.products),
            )
        ]

    @classmethod
    def from_config(cls, config: AppConfig, frame: pd.DataFrame | None = None) -> DToCWorld:
        """Build the world (fitting the encoder on ``frame`` or freshly loaded data).

        Raises :class:`DToCDisabledError` if ``dtoc.enabled`` is false — the
        project is then in plain feature-vector mode and twins are not used.
        """
        if not config.dtoc.enabled:
            raise DToCDisabledError(
                "DToC is disabled (dtoc.enabled: false) — the project is running in "
                "plain feature-vector mode. Set dtoc.enabled: true in the config to "
                "use the Digital Twin of Customer layer."
            )
        if frame is None:
            frame = load_customers(config)
        if config.state is not None:
            encoder: StateBuilder | FeatureEncoder = StateBuilder(
                config.data.schema,
                config.products.catalog,
                config.state,
                include_journey=config.dtoc.enabled,  # keep twin context == training context
            ).fit(frame)
        else:
            encoder = FeatureEncoder(config.data.schema, config.products.catalog).fit(frame)
        model = ConversionModel.sample(
            context_dim=encoder.context_dim,
            n_actions=len(config.products.catalog),
            base_conversion_rate=config.environment.base_conversion_rate,
            context_influence=config.environment.context_influence,
            rng=np.random.default_rng(config.environment.seed),
            base_dim=encoder.context_dim - getattr(encoder, "n_journey", 0),
            journey_influence=config.environment.journey_influence,
        )
        return cls(config, encoder, model, _action_values(config))

    # --- encoding & environment queries -------------------------------------
    def encode(self, features: dict[str, Any]) -> np.ndarray:
        """Encode a feature dict into the context vector the policy sees."""
        return self.encoder.transform(pd.DataFrame([features]))[0]

    def eligible_actions(self, features: dict[str, Any]) -> np.ndarray:
        """Indices of products the customer does not already own."""
        held = np.array(
            [(float(features.get(self.schema.owned_column(p), 0.0)) > 0.5) for p in self.products]
        )
        return np.flatnonzero(~held)

    def conversion_probability(self, context: np.ndarray, action: int) -> float:
        return float(self.model.probabilities(context[np.newaxis, :])[0, action])

    # --- transition model (simplified, transparent) -------------------------
    def _premium(self, product: str) -> float:
        return float(
            self.config.products.premiums.get(product)
            or self.config.products.ape.get(product)
            or 1000.0
        )

    def advance_time(self, features: dict[str, Any]) -> dict[str, Any]:
        """One time step passes: age up, reset the 1-month activity window."""
        out = dict(features)
        months = self.config.dtoc.time_step_months
        for age_col in ("customer_age", "age"):
            if age_col in out and out[age_col] is not None:
                out[age_col] = float(out[age_col]) + months / 12.0
        if "customer_purchase_count_past_1m" in out:
            out["customer_purchase_count_past_1m"] = 0.0
        return out

    def apply_purchase(self, features: dict[str, Any], product: str) -> dict[str, Any]:
        """Customer buys ``product``: update ownership, portfolio, and activity."""
        out = dict(features)
        premium = self._premium(product)
        owned_col = self.schema.owned_column(product)
        if owned_col in out:
            out[owned_col] = 1
        _increment(out, "customer_holdings_count", 1)
        _increment(out, "customer_all_policy_holding_count", 1)
        _increment(out, "customer_holdings_ap", premium)
        _increment(out, "customer_inforce_policy_holding_ap", premium)
        _increment(out, "customer_holdings_sum_assured", premium * _SUM_ASSURED_PER_PREMIUM)
        _increment(out, "customer_purchased_ap_past_12m", premium)
        for window in ("1m", "3m", "6m", "12m"):
            _increment(out, f"customer_purchase_count_past_{window}", 1)
        return out

    # --- state construction -------------------------------------------------
    def make_state(
        self,
        features: dict[str, Any],
        step: int,
        label: str,
        offered: str | None = None,
        converted: bool | None = None,
        reward: float | None = None,
    ) -> TwinState:
        return TwinState(
            step=step,
            label=label,
            features=dict(features),
            journey=infer_journey(
                pd.Series(features), self.products, self.schema.owned_product_prefix
            ),
            context=self.encode(features),
            offered=offered,
            converted=converted,
            reward=reward,
        )


@dataclass
class DigitalTwin:
    """A customer's timeline: observed history + current, plus projected future."""

    customer_id: str
    history: list[TwinState]  # oldest → newest; the last entry is "current"
    world: DToCWorld
    future: list[TwinState] = field(default_factory=list)

    @property
    def current(self) -> TwinState:
        return self.history[-1]

    def project(
        self, policy: Any, horizon: int | None = None, seed: int = 0
    ) -> list[TwinState]:
        """Roll the customer forward under ``policy``; store and return future states.

        ``policy`` may be an agent (with ``select_action``), a plain callable
        ``(context, eligible) -> action``, or a product name / index for a fixed
        offer. No projection is done when ``dtoc.future_mode == "placeholder"``.
        """
        if self.world.config.dtoc.future_mode == "placeholder":
            self.future = []
            return self.future
        steps = self.world.config.dtoc.horizon if horizon is None else horizon
        act = _as_policy(policy, self.world)
        rng = np.random.default_rng([seed, zlib.crc32(str(self.customer_id).encode())])
        features = dict(self.current.features)
        future: list[TwinState] = []
        for step in range(1, steps + 1):
            context = self.world.encode(features)
            eligible = self.world.eligible_actions(features)
            if len(eligible) == 0:
                break
            action = int(act(context, eligible))
            prob = self.world.conversion_probability(context, action)
            converted = bool(rng.random() < prob)
            reward = float(self.world.action_values[action]) if converted else 0.0
            features = self.world.advance_time(features)
            if converted:
                features = self.world.apply_purchase(features, self.world.products[action])
            future.append(
                self.world.make_state(
                    features,
                    step=step,
                    label="future",
                    offered=self.world.products[action],
                    converted=converted,
                    reward=reward,
                )
            )
        self.future = future
        return future

    def timeline(self) -> list[TwinState]:
        """Full ordered trajectory: history (incl. current) + projected future."""
        return [*self.history, *self.future]

    def scenario_value(self) -> float:
        """Cumulative projected reward over the future trajectory."""
        return float(sum(s.reward or 0.0 for s in self.future))

    def explain(self, agent: Any) -> list[dict[str, Any]]:
        """Policy's per-product scoring at the current state (needs ``agent.explain``)."""
        if not hasattr(agent, "explain"):
            raise TypeError(
                f"{type(agent).__name__} has no explain(); use LinUCB for explainability."
            )
        eligible = self.world.eligible_actions(self.current.features)
        ranked = sorted(agent.explain(self.current.context, eligible),
                        key=lambda item: item["score"], reverse=True)
        return [
            {"product": self.world.products[item["action"]], "chosen": index == 0, **item}
            for index, item in enumerate(ranked)
        ]

    def records(self) -> list[dict[str, Any]]:
        """Flat, viz-friendly rows for the whole timeline."""
        rows = []
        for state in self.timeline():
            rows.append(
                {
                    "step": state.step,
                    "label": state.label,
                    "life_stage": state.journey.life_stage,
                    "relationship_stage": state.journey.relationship_stage,
                    "holdings_count": state.features.get("customer_holdings_count"),
                    "holdings_ap": state.features.get("customer_holdings_ap"),
                    "offered": state.offered,
                    "converted": state.converted,
                    "reward": state.reward,
                }
            )
        return rows


def twin_from_row(world: DToCWorld, row: pd.Series) -> DigitalTwin:
    """Build a twin from a single customer snapshot (history = [current])."""
    features = {col: row[col] for col in world._feature_columns if col in row}
    current = world.make_state(features, step=0, label="current")
    return DigitalTwin(str(row[world.schema.customer_id]), history=[current], world=world)


def twin_from_panel(world: DToCWorld, rows: Sequence[pd.Series]) -> DigitalTwin:
    """Build a twin from a customer's ordered panel (oldest → newest snapshots)."""
    if not rows:
        raise ValueError("panel is empty")
    history: list[TwinState] = []
    for index, row in enumerate(rows):
        features = {col: row[col] for col in world._feature_columns if col in row}
        is_current = index == len(rows) - 1
        history.append(
            world.make_state(
                features, step=index - (len(rows) - 1), label="current" if is_current else "history"
            )
        )
    return DigitalTwin(str(rows[-1][world.schema.customer_id]), history=history, world=world)


def fixed_policy(product_index: int) -> Policy:
    """A reusable scenario policy that always offers a given product *index*.

    To offer by name, pass the product name straight to
    :meth:`DigitalTwin.project` (e.g. ``twin.project("whole_life")``), which
    resolves the name against the catalog.
    """
    index = int(product_index)

    def _policy(context: np.ndarray, eligible: np.ndarray) -> int:
        return index if index in eligible else int(eligible[0])

    return _policy


# --- internals ---------------------------------------------------------------
def _action_values(config: AppConfig) -> np.ndarray:
    catalog = config.products.catalog
    if config.reward.type == "conversion":
        return np.ones(len(catalog))
    source = {
        "revenue": config.products.premiums,
        "ape": config.products.ape,
        "vnb": config.products.vnb,
    }[config.reward.type]
    return np.array([source[p] for p in catalog], dtype=float)


def _increment(features: dict[str, Any], column: str, amount: float) -> None:
    if column in features and features[column] is not None:
        features[column] = float(features[column]) + amount


def _as_policy(policy: Any, world: DToCWorld) -> Policy:
    if hasattr(policy, "select_action"):
        return lambda context, eligible: policy.select_action(context, eligible)
    if isinstance(policy, (str, int, np.integer)):
        return lambda context, eligible: _resolve_product(policy, eligible, world)
    if callable(policy):
        return policy
    raise TypeError(f"Unsupported policy: {type(policy).__name__}")


def _resolve_product(
    product: str | int, eligible: np.ndarray, world: DToCWorld | None = None
) -> int:
    if isinstance(product, str):
        if world is None:
            raise ValueError("world required to resolve a product name")
        product = world.products.index(product)
    product = int(product)
    return product if product in eligible else int(eligible[0])


def plot_journey(twin: DigitalTwin, output_path: str, metric: str = "customer_holdings_ap") -> str:
    """Draw the twin's timeline: a portfolio metric over history + future, with
    purchases marked and journey stage annotated. Returns the saved path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rl_nba.evaluation.plots import (
        _BASELINE,
        _GRIDLINE,
        _INK_MUTED,
        _INK_PRIMARY,
        _INK_SECONDARY,
        _PAGE,
        _SERIES_COLORS,
        _SURFACE,
    )

    states = twin.timeline()
    xs = [s.step for s in states]
    ys = [float(s.features.get(metric) or 0.0) for s in states]

    figure, ax = plt.subplots(figsize=(10, 4.6), dpi=150, facecolor=_PAGE, constrained_layout=True)
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
    ax.tick_params(colors=_INK_MUTED, labelsize=9)
    ax.grid(True, color=_GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.axvline(0, color=_INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.annotate("now", xy=(0, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", color=_INK_MUTED, fontsize=9)
    ax.plot(xs, ys, color=_SERIES_COLORS[0], linewidth=1.8, marker="o", markersize=4)

    for state in states:
        if state.converted:
            ax.annotate(
                f"+{state.offered}", xy=(state.step, float(state.features.get(metric) or 0.0)),
                xytext=(0, 8), textcoords="offset points", ha="center",
                color=_SERIES_COLORS[1], fontsize=8, fontweight="bold",
            )

    ax.set_xlabel("time step (months from now)", color=_INK_MUTED, fontsize=9)
    ax.set_ylabel(metric, color=_INK_MUTED, fontsize=9)
    figure.suptitle(
        f"Customer {twin.customer_id} — journey timeline", x=0.01, ha="left",
        color=_INK_PRIMARY, fontsize=13, fontweight="bold",
    )
    figure.text(
        0.01, 0.925,
        f"now: {twin.current.journey.label}  |  projected value: "
        f"{twin.scenario_value():,.0f}",
        ha="left", color=_INK_SECONDARY, fontsize=9.5,
    )
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output_path
