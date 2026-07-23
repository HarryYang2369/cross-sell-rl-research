"""Grouped state construction: feature groups, trend ratios, coverage gaps.

This maps the experiment's design language — a state space made of named
feature groups, temporal trend signals, and coverage-gap detection against
segment-typical portfolios — onto the flat encoding machinery in
``features.py``. Models can be restricted to any subset of groups via column
masks, which is how a *baseline* state design and an *enhanced* one compete
fairly within a single experiment.

Layout of the full context matrix::

    [ base encoding (intercept | numerics | one-hots | ownership flags)
      | trend ratios | coverage gaps | journey one-hots (Digital Twin, optional) ]

The journey block (``life_stage`` + ``relationship_stage`` one-hots) is added
only when ``include_journey`` is set — i.e. when ``dtoc.enabled`` is true — so
the Digital Twin's journey model becomes part of the state the policy learns on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rl_nba.config import SchemaConfig, StateConfig, TrendConfig
from rl_nba.data.schema import ownership_matrix
from rl_nba.features import FeatureEncoder, NotFittedError
from rl_nba.journey import LIFE_STAGES, RELATIONSHIP_STAGES, infer_journey

_LIFE_IDX = {stage: index for index, stage in enumerate(LIFE_STAGES)}
_REL_IDX = {stage: index for index, stage in enumerate(RELATIONSHIP_STAGES)}
_N_JOURNEY = len(LIFE_STAGES) + len(RELATIONSHIP_STAGES)


class StateBuilder:
    """Builds full context vectors and per-model feature masks.

    ``min_segment_size`` guards the coverage-gap features: segments smaller
    than this fall back to the population-wide ownership rates, so tiny
    segments don't produce noisy "typical portfolios".
    """

    def __init__(
        self,
        schema: SchemaConfig,
        products: Sequence[str],
        state: StateConfig,
        min_segment_size: int = 20,
        include_journey: bool = False,
    ) -> None:
        self.schema = schema
        self.products = tuple(products)
        self.state = state
        self.min_segment_size = min_segment_size
        self.include_journey = include_journey
        self._encoder = FeatureEncoder(schema, products)
        self._trend_means: dict[str, float] = {}
        self._trend_stds: dict[str, float] = {}
        self._segment_rates: dict[str, np.ndarray] = {}
        self._global_rates: np.ndarray | None = None
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> StateBuilder:
        """Fit the base encoder, trend statistics, and segment portfolios."""
        self._encoder.fit(frame)
        for trend in self.state.trends:
            ratio = self._trend_ratio(frame, trend)
            mean = float(np.mean(ratio))
            std = float(np.std(ratio))
            name = _trend_name(trend)
            self._trend_means[name] = mean if np.isfinite(mean) else 0.0
            self._trend_stds[name] = std if np.isfinite(std) and std > 1e-9 else 1.0
        if self.state.coverage_gaps.segment_by:
            owned = ownership_matrix(frame, self.schema, self.products).astype(float)
            self._global_rates = owned.mean(axis=0)
            keys = self._segment_keys(frame)
            grouped = pd.DataFrame(owned).groupby(keys)
            sizes = grouped.size()
            rates = grouped.mean()
            self._segment_rates = {
                str(key): rates.loc[key].to_numpy()
                for key in sizes.index[sizes >= self.min_segment_size]
            }
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Encode ``frame`` into the full ``(n_rows, context_dim)`` matrix."""
        self._check_fitted()
        blocks = [self._encoder.transform(frame)]
        for trend in self.state.trends:
            name = _trend_name(trend)
            ratio = self._trend_ratio(frame, trend)
            standardized = (ratio - self._trend_means[name]) / self._trend_stds[name]
            blocks.append(standardized[:, np.newaxis])
        if self.state.coverage_gaps.segment_by:
            owned = ownership_matrix(frame, self.schema, self.products).astype(float)
            keys = self._segment_keys(frame)
            rates = np.vstack(
                [self._segment_rates.get(key, self._global_rates) for key in keys]
            )
            blocks.append(rates - owned)
        if self.include_journey:
            blocks.append(self._journey_block(frame))
        return np.hstack(blocks)

    def _journey_block(self, frame: pd.DataFrame) -> np.ndarray:
        """One-hot of the standardized journey state (life stage + relationship stage)."""
        block = np.zeros((len(frame), _N_JOURNEY))
        prefix = self.schema.owned_product_prefix
        for row, record in enumerate(frame.to_dict("records")):
            journey = infer_journey(record, self.products, prefix)
            block[row, _LIFE_IDX[journey.life_stage]] = 1.0
            block[row, len(LIFE_STAGES) + _REL_IDX[journey.relationship_stage]] = 1.0
        return block

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    @property
    def feature_names(self) -> list[str]:
        """Names of every context dimension: base encoding, then derived."""
        self._check_fitted()
        names = list(self._encoder.feature_names)
        names.extend(_trend_name(trend) for trend in self.state.trends)
        if self.state.coverage_gaps.segment_by:
            names.extend(f"coverage_gap_{product}" for product in self.products)
        if self.include_journey:
            names.extend(f"life_stage={stage}" for stage in LIFE_STAGES)
            names.extend(f"relationship_stage={stage}" for stage in RELATIONSHIP_STAGES)
        return names

    @property
    def context_dim(self) -> int:
        return len(self.feature_names)

    @property
    def n_journey(self) -> int:
        """Number of trailing journey one-hot columns (0 unless include_journey)."""
        return _N_JOURNEY if self.include_journey else 0

    def columns_for(
        self,
        groups: Sequence[str] | None = None,
        include_derived: bool = True,
        include_journey: bool = True,
    ) -> np.ndarray:
        """Column indices visible to a model limited to the given feature groups.

        ``None`` groups means all active groups. The intercept and the product
        ownership flags are always included. ``include_derived`` toggles the
        trend + coverage-gap columns; ``include_journey`` toggles the Digital
        Twin journey one-hots (present only when the builder was created with
        ``include_journey=True``) — so a model can be run with or without the
        DToC's journey view for a clean comparison.
        """
        self._check_fitted()
        active = set(self.state.active_group_names)
        if groups is None:
            group_names = active
        else:
            group_names = {str(name) for name in groups}
            unknown = group_names - active
            if unknown:
                raise ValueError(
                    f"Unknown or disabled feature groups {sorted(unknown)}; "
                    f"active groups: {sorted(active)}"
                )
        numeric = {
            column
            for name in group_names
            for column in self.state.feature_groups[name].numeric
        }
        categorical = {
            column
            for name in group_names
            for column in self.state.feature_groups[name].categorical
        }
        base_names = self._encoder.feature_names
        indices = []
        for index, name in enumerate(base_names):
            always = name == "intercept" or name.startswith("owns_")
            in_numeric = name in numeric
            in_categorical = any(name.startswith(f"{column}=") for column in categorical)
            if always or in_numeric or in_categorical:
                indices.append(index)

        n_base = len(base_names)
        n_cov = len(self.products) if self.state.coverage_gaps.segment_by else 0
        derived_end = n_base + len(self.state.trends) + n_cov
        if include_derived:
            indices.extend(range(n_base, derived_end))
        if include_journey and self.include_journey:
            indices.extend(range(derived_end, derived_end + _N_JOURNEY))
        return np.asarray(indices, dtype=int)

    def _trend_ratio(self, frame: pd.DataFrame, trend: TrendConfig) -> np.ndarray:
        short = pd.to_numeric(frame[trend.short], errors="coerce").fillna(0.0)
        long = pd.to_numeric(frame[trend.long], errors="coerce").fillna(0.0)
        # +1 smoothing keeps customers with no long-window activity finite.
        return (short / (long + 1.0)).to_numpy(dtype=float)

    def _segment_keys(self, frame: pd.DataFrame) -> pd.Series:
        columns = list(self.state.coverage_gaps.segment_by)
        return frame[columns].astype(str).agg("|".join, axis=1)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("StateBuilder must be fitted before use; call fit() first.")


def _trend_name(trend: TrendConfig) -> str:
    return f"trend_{trend.short}_over_{trend.long}"
