"""Learning-curve and regret plots for simulation runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; never open a window

import matplotlib.pyplot as plt
import numpy as np

from rl_nba.evaluation.simulate import SimulationResult

# Validated categorical palette (light mode), assigned to agents in fixed slot
# order — never cycled. Chrome colors are the matching surface/ink roles.
_SERIES_COLORS = ("#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834")
_SURFACE = "#fcfcfb"
_PAGE = "#f9f9f7"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    return (cumulative[window:] - cumulative[:-window]) / window


def _spread_positions(positions: list[float], min_gap: float) -> list[float]:
    """Nudge y-positions apart so stacked end-of-line labels stay readable."""
    order = np.argsort(positions)
    adjusted = np.asarray(positions, dtype=float)
    for previous, current in zip(order[:-1], order[1:], strict=True):
        if adjusted[current] - adjusted[previous] < min_gap:
            adjusted[current] = adjusted[previous] + min_gap
    return adjusted.tolist()


def _annotate_line_ends(ax: plt.Axes, labels: list[tuple[str, float, float]]) -> None:
    """Direct labels at line ends; identity text stays in ink, not series color."""
    y_low, y_high = ax.get_ylim()
    spread = _spread_positions([y for _, _, y in labels], min_gap=0.04 * (y_high - y_low))
    for (name, x, _), y in zip(labels, spread, strict=True):
        ax.annotate(
            name, xy=(x, y), xytext=(5, 0), textcoords="offset points",
            color=_INK_SECONDARY, fontsize=8.5, va="center",
        )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
    ax.tick_params(colors=_INK_MUTED, labelsize=9)
    ax.grid(True, color=_GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_learning_curves(
    results: Sequence[SimulationResult],
    oracle_value: float,
    output_path: str | Path,
    subtitle: str | None = None,
) -> Path:
    """Save a two-panel figure: rolling expected reward and cumulative regret.

    The reward panel uses the simulator's *expected* reward of each chosen
    action rather than the noisy Bernoulli draws, so curves show learning
    rather than luck. The oracle line marks the best achievable value.
    """
    if len(results) > len(_SERIES_COLORS):
        raise ValueError(
            f"Cannot color {len(results)} agents distinctly; "
            f"plot at most {len(_SERIES_COLORS)} or fold some together."
        )
    colors = {result.agent_name: _SERIES_COLORS[i] for i, result in enumerate(results)}
    n_rounds = max(result.n_rounds for result in results)
    window = max(1, min(2000, n_rounds // 30))

    figure, (reward_ax, regret_ax) = plt.subplots(
        1, 2, figsize=(12.5, 5.2), dpi=150, facecolor=_PAGE, constrained_layout=True
    )
    figure.get_layout_engine().set(rect=(0, 0, 1, 0.90))

    for ax in (reward_ax, regret_ax):
        _style_axis(ax)
        ax.set_xlabel("round", color=_INK_MUTED, fontsize=9)
        ax.set_xlim(0, n_rounds * 1.17)  # inner room for direct labels

    reward_ax.set_title(
        "Average expected reward per offer (rolling mean)",
        loc="left", color=_INK_PRIMARY, fontsize=11,
    )
    regret_ax.set_title(
        "Cumulative regret (lower is better)",
        loc="left", color=_INK_PRIMARY, fontsize=11,
    )

    reward_ax.axhline(oracle_value, color=_INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.1)
    reward_ax.annotate(
        "oracle", xy=(0.99, oracle_value), xycoords=("axes fraction", "data"),
        xytext=(0, 4), textcoords="offset points", ha="right",
        color=_INK_MUTED, fontsize=9,
    )

    reward_labels: list[tuple[str, float, float]] = []
    regret_labels: list[tuple[str, float, float]] = []
    for result in results:
        color = colors[result.agent_name]
        smoothed = _rolling_mean(result.expected_rewards, window)
        rounds = np.arange(len(smoothed)) + window
        reward_ax.plot(rounds, smoothed, color=color, linewidth=1.7, label=result.agent_name)
        cumulative = np.cumsum(result.regrets)
        regret_ax.plot(
            np.arange(1, result.n_rounds + 1), cumulative,
            color=color, linewidth=1.7, label=result.agent_name,
        )
        reward_labels.append((result.agent_name, float(rounds[-1]), float(smoothed[-1])))
        regret_labels.append((result.agent_name, float(result.n_rounds), float(cumulative[-1])))

    _annotate_line_ends(reward_ax, reward_labels)
    _annotate_line_ends(regret_ax, regret_labels)

    reward_ax.legend(
        loc="upper left", frameon=False, fontsize=9,
        labelcolor=_INK_SECONDARY, handlelength=1.6,
    )

    figure.suptitle(
        "Cross-sell bandit agents — simulated learning",
        x=0.01, ha="left", color=_INK_PRIMARY, fontsize=13, fontweight="bold",
    )
    if subtitle:
        figure.text(0.01, 0.925, subtitle, ha="left", color=_INK_SECONDARY, fontsize=9.5)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output_path
