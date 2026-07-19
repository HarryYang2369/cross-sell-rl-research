"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from cross_sell_rl.config import AppConfig, config_from_dict
from cross_sell_rl.data import generate_customers, ownership_matrix
from cross_sell_rl.env import ConversionModel, CrossSellSimulator
from cross_sell_rl.features import FeatureEncoder


@pytest.fixture()
def small_config() -> AppConfig:
    """A fast, fully-defaulted config with a small synthetic pool."""
    return config_from_dict({"synthetic": {"n_customers": 400, "seed": 3}})


@pytest.fixture()
def small_simulator(small_config: AppConfig) -> CrossSellSimulator:
    frame = generate_customers(
        schema=small_config.data.schema,
        products=small_config.products.catalog,
        n_customers=small_config.synthetic.n_customers,
        seed=small_config.synthetic.seed,
    )
    encoder = FeatureEncoder(small_config.data.schema, small_config.products.catalog)
    contexts = encoder.fit_transform(frame)
    owned = ownership_matrix(frame, small_config.data.schema, small_config.products.catalog)
    model = ConversionModel.sample(
        context_dim=contexts.shape[1],
        n_actions=len(small_config.products.catalog),
        base_conversion_rate=0.08,
        context_influence=1.5,
        rng=np.random.default_rng(11),
    )
    return CrossSellSimulator(
        contexts, owned, model, np.ones(len(small_config.products.catalog))
    )
