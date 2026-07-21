"""Record a human-readable playback trace of one agent's episode.

The trace is what the playback dashboard replays: for every decision it captures
the customer, the agent's value estimate and exploration bonus for each eligible
product, the offer it made, and what happened — plus running totals against the
random and oracle baselines. See :func:`write_dashboard` to render it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_nba.agents.linear import LinUCBAgent
from rl_nba.config import AppConfig
from rl_nba.data import ownership_matrix
from rl_nba.run import prepare_experiment

_TEMPLATE = Path(__file__).with_name("_playback_template.html")

# Human-readable labels for the customer attributes worth showing, in display
# order. Only columns actually present in the data are shown, so this is safe
# across schemas.
_ATTRIBUTE_LABELS: dict[str, str] = {
    "customer_age": "Age",
    "age": "Age",
    "customer_gender": "Gender",
    "customer_marital_status": "Marital status",
    "customer_income_range": "Monthly income",
    "wealth_segment": "Wealth segment",
    "region": "Region",
}


def _pretty(value: Any) -> str:
    """Format a cell value for display (whole floats as ints, else as-is)."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:,.0f}"
    return str(value)


def _pretty_product(product: str) -> str:
    return product.replace("_", " ")


def record_playback(
    config: AppConfig,
    n_steps: int = 150,
    alpha: float = 1.0,
    seed: int = 20260721,
) -> dict[str, Any]:
    """Run the enhanced (full-state LinUCB) agent and record ``n_steps`` decisions.

    Returns a JSON-serializable dict with ``meta`` and a list of ``steps``.
    """
    prepared = prepare_experiment(config)
    simulator = prepared.simulator
    schema, catalog = config.data.schema, config.products.catalog

    owned = ownership_matrix(prepared.customers, schema, catalog)
    offerable_mask = ~owned.all(axis=1)
    offerable = prepared.customers[offerable_mask].reset_index(drop=True)
    offerable_owned = owned[offerable_mask]

    agent = LinUCBAgent(
        n_actions=simulator.n_actions,
        context_dim=prepared.context_dim,
        rng=np.random.default_rng(seed),
        alpha=alpha,
    )
    env_rng = np.random.default_rng(seed + 1)

    n_steps = min(n_steps, len(prepared.customer_sequence))
    cumulative = {"agent": 0.0, "random": 0.0, "oracle": 0.0}
    conversions = 0
    value_earned = 0.0
    steps: list[dict[str, Any]] = []

    for step in range(n_steps):
        customer = int(prepared.customer_sequence[step])
        context = simulator.context(customer)
        eligible = simulator.eligible_actions(customer)

        breakdown = agent.explain(context, eligible)
        chosen = max(breakdown, key=lambda item: item["score"])["action"]
        reward = simulator.step(customer, chosen, env_rng)
        agent.update(context, chosen, reward)

        converted = reward > 0.0
        conversions += int(converted)
        value_earned += reward
        cumulative["agent"] += simulator.expected_reward(customer, chosen)
        cumulative["oracle"] += simulator.best_expected_reward(customer)
        cumulative["random"] += float(
            np.mean([simulator.expected_reward(customer, action) for action in eligible])
        )

        options = [
            {
                "product": _pretty_product(catalog[item["action"]]),
                # Clamp for display: a negative expected value reads as "~0 to us".
                "estimate": round(max(item["estimate"], 0.0), 1),
                "bonus": round(max(item["bonus"], 0.0), 1),
                "chosen": item["action"] == chosen,
            }
            for item in sorted(breakdown, key=lambda item: item["score"], reverse=True)
        ]
        row = offerable.iloc[customer]
        steps.append(
            {
                "step": step,
                "customer_id": str(row[schema.customer_id]),
                "attributes": _customer_attributes(row),
                "owns": [
                    _pretty_product(product)
                    for product, held in zip(catalog, offerable_owned[customer], strict=True)
                    if held
                ],
                "options": options,
                "chosen": _pretty_product(catalog[chosen]),
                "converted": bool(converted),
                "reward": round(float(reward), 1),
                "cumulative": {name: round(value, 1) for name, value in cumulative.items()},
                "value_earned": round(value_earned, 1),
                "conversion_rate": round(conversions / (step + 1), 4),
            }
        )

    return {
        "meta": {
            "reward_type": config.reward.type,
            "currency": "HK$" if config.reward.type in ("ape", "vnb") else "",
            "value_word": {"vnb": "business value", "ape": "premium", "revenue": "premium"}.get(
                config.reward.type, "conversions"
            ),
            "products": [_pretty_product(product) for product in catalog],
            "n_steps": n_steps,
            "alpha": alpha,
        },
        "steps": steps,
    }


def _customer_attributes(row: pd.Series) -> list[list[str]]:
    return [
        [label, _pretty(row[column])]
        for column, label in _ATTRIBUTE_LABELS.items()
        if column in row.index
    ]


def render_dashboard(trace: dict[str, Any]) -> str:
    """Inject a recorded trace into the dashboard template, returning full HTML."""
    template = _TEMPLATE.read_text(encoding="utf-8")
    # Escape "<" so the JSON can never terminate the <script> block early.
    payload = json.dumps(trace, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("/*__PLAYBACK_TRACE__*/", payload)


def write_dashboard(
    config: AppConfig,
    out_path: str | Path,
    n_steps: int = 150,
    alpha: float = 1.0,
    seed: int = 20260721,
) -> Path:
    """Record an episode and write a self-contained playback dashboard."""
    trace = record_playback(config, n_steps=n_steps, alpha=alpha, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_dashboard(trace), encoding="utf-8")
    return out_path
