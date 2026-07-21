"""Feature encoding: customer rows -> dense context vectors for the agents."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rl_nba.config import SchemaConfig


class NotFittedError(RuntimeError):
    """Raised when :meth:`FeatureEncoder.transform` is called before ``fit``."""


class FeatureEncoder:
    """Encodes customer rows into fixed-length context vectors.

    Vector layout: ``[intercept, standardized numerics, one-hot categoricals,
    ownership flags]``. Means, standard deviations, and category vocabularies
    are learned in :meth:`fit`. The encoder is safe to reuse on new batches:
    missing numeric values are imputed with the fitted mean and categories
    unseen at fit time encode as all-zeros.
    """

    def __init__(self, schema: SchemaConfig, products: Sequence[str]) -> None:
        self.schema = schema
        self.products = tuple(products)
        self._numeric_means: dict[str, float] = {}
        self._numeric_stds: dict[str, float] = {}
        self._vocabularies: dict[str, dict[str, int]] = {}
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> FeatureEncoder:
        """Learn standardization statistics and category vocabularies."""
        for name in self.schema.numeric_features:
            values = pd.to_numeric(frame[name], errors="coerce")
            mean = float(values.mean())
            std = float(values.std())
            self._numeric_means[name] = mean if np.isfinite(mean) else 0.0
            self._numeric_stds[name] = std if np.isfinite(std) and std > 1e-9 else 1.0
        for name in self.schema.categorical_features:
            categories = sorted(frame[name].astype(str).unique())
            self._vocabularies[name] = {category: i for i, category in enumerate(categories)}
        self._fitted = True
        return self

    @property
    def context_dim(self) -> int:
        self._check_fitted()
        n_onehot = sum(len(vocab) for vocab in self._vocabularies.values())
        return 1 + len(self.schema.numeric_features) + n_onehot + len(self.products)

    @property
    def feature_names(self) -> list[str]:
        """Human-readable name of each context dimension, in vector order."""
        self._check_fitted()
        names = ["intercept", *self.schema.numeric_features]
        for column, vocab in self._vocabularies.items():
            names.extend(f"{column}={category}" for category in vocab)
        names.extend(f"owns_{product}" for product in self.products)
        return names

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Encode ``frame`` into a ``(n_rows, context_dim)`` float matrix."""
        self._check_fitted()
        n_rows = len(frame)
        matrix = np.zeros((n_rows, self.context_dim))
        matrix[:, 0] = 1.0
        column = 1
        for name in self.schema.numeric_features:
            values = pd.to_numeric(frame[name], errors="coerce")
            filled = values.fillna(self._numeric_means[name]).to_numpy(dtype=float)
            matrix[:, column] = (filled - self._numeric_means[name]) / self._numeric_stds[name]
            column += 1
        for name in self.schema.categorical_features:
            vocab = self._vocabularies[name]
            mapped = frame[name].astype(str).map(vocab).to_numpy()
            known = ~pd.isna(mapped)
            matrix[np.flatnonzero(known), column + mapped[known].astype(int)] = 1.0
            column += len(vocab)
        for product in self.products:
            owned = pd.to_numeric(frame[self.schema.owned_column(product)], errors="coerce")
            matrix[:, column] = (owned.fillna(0.0).to_numpy(dtype=float) > 0.5).astype(float)
            column += 1
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("FeatureEncoder must be fitted before use; call fit() first.")
