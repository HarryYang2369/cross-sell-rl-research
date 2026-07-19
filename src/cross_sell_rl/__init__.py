"""Config-driven contextual-bandit / RL pipeline for insurance cross-sell.

The package mirrors the module layout of Open Bandit Pipeline:
``data`` (sources + synthetic generation), ``agents`` (policies),
``env`` (simulator), and ``evaluation`` (simulation studies + off-policy
evaluation). ``run`` is the command-line entry point.
"""

from cross_sell_rl.config import AppConfig, load_config

__version__ = "0.1.0"

__all__ = ["AppConfig", "load_config", "__version__"]
