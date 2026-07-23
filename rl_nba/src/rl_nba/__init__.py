"""Config-driven contextual-bandit / RL pipeline for insurance cross-sell.

The package mirrors the module layout of Open Bandit Pipeline:
``data`` (sources + synthetic generation), ``agents`` (policies),
``env`` (simulator), and ``evaluation`` (simulation studies + off-policy
evaluation). ``run`` is the command-line entry point.
"""

from rl_nba.config import AppConfig, load_config
from rl_nba.dtoc import (
    DigitalTwin,
    DToCDisabledError,
    DToCWorld,
    dtoc_enabled,
    fixed_policy,
    twin_from_panel,
    twin_from_row,
)
from rl_nba.journey import JourneyState, infer_journey

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "DToCDisabledError",
    "DToCWorld",
    "DigitalTwin",
    "JourneyState",
    "dtoc_enabled",
    "fixed_policy",
    "infer_journey",
    "load_config",
    "twin_from_panel",
    "twin_from_row",
    "__version__",
]
