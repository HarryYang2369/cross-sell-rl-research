"""Typed configuration loading.

The YAML config file is the single source of truth for where data comes from,
what the columns are called, which products can be offered, how reward is
defined, and how experiments run. Every key is optional: omitted keys fall
back to the defaults below, which match ``configs/default.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

VALID_SOURCES = ("synthetic", "csv", "parquet", "databricks")
VALID_REWARD_TYPES = ("conversion", "revenue", "ape", "vnb")
VALID_DELIVERY_MODES = ("assigned_agent", "mixed", "direct")

# The feature group that describes the selling agent rather than the customer.
# state.delivery controls whether it enters the state (direct sales have no
# human intermediary, so agent quality cannot predict offer success there).
AGENT_CONTEXT_GROUP = "agent_context"

_DEFAULT_NUMERIC_FEATURES = ("age", "tenure_years", "annual_premium", "num_claims", "credit_score")
_DEFAULT_CATEGORICAL_FEATURES = ("region", "acquisition_channel")
_DEFAULT_CATALOG = ("home", "life", "health", "travel", "pet")
_DEFAULT_PREMIUMS: Mapping[str, float] = MappingProxyType(
    {"home": 950.0, "life": 620.0, "health": 1400.0, "travel": 180.0, "pet": 320.0}
)
_DEFAULT_AGENTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "random": MappingProxyType({}),
        "epsilon_greedy": MappingProxyType({"epsilon": 0.1}),
        "linucb": MappingProxyType({"alpha": 1.0}),
        "lin_ts": MappingProxyType({"scale": 0.3}),
    }
)


class ConfigError(ValueError):
    """Raised when the config file is missing, malformed, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class SchemaConfig:
    """Column mapping between the config and the customer data."""

    customer_id: str = "customer_id"
    numeric_features: tuple[str, ...] = _DEFAULT_NUMERIC_FEATURES
    categorical_features: tuple[str, ...] = _DEFAULT_CATEGORICAL_FEATURES
    owned_product_prefix: str = "has_"

    def owned_column(self, product: str) -> str:
        """Name of the 0/1 column that marks ownership of ``product``."""
        return f"{self.owned_product_prefix}{product}"


@dataclass(frozen=True)
class DatabricksConfig:
    catalog: str | None = None
    schema: str | None = None
    table: str | None = None


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    path: str | None = None
    databricks: DatabricksConfig = field(default_factory=DatabricksConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)


@dataclass(frozen=True)
class ProductsConfig:
    catalog: tuple[str, ...] = _DEFAULT_CATALOG
    premiums: Mapping[str, float] = _DEFAULT_PREMIUMS
    ape: Mapping[str, float] = MappingProxyType({})
    vnb: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True)
class RewardConfig:
    type: str = "conversion"


@dataclass(frozen=True)
class FeatureGroupConfig:
    """One named block of the state space (e.g. profile, holdings)."""

    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    enabled: bool | None = None  # None -> on, except agent_context follows state.delivery


@dataclass(frozen=True)
class TrendConfig:
    """A temporal trend feature: short-window activity relative to a long window."""

    short: str
    long: str


@dataclass(frozen=True)
class CoverageGapsConfig:
    """Coverage-gap features: customer holdings vs. their segment's typical portfolio."""

    segment_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class StateConfig:
    """Grouped state-space definition (supersedes flat data.schema feature lists)."""

    delivery: str = "mixed"
    feature_groups: Mapping[str, FeatureGroupConfig] = MappingProxyType({})
    trends: tuple[TrendConfig, ...] = ()
    coverage_gaps: CoverageGapsConfig = field(default_factory=CoverageGapsConfig)

    def group_enabled(self, name: str) -> bool:
        group = self.feature_groups[name]
        if group.enabled is not None:
            return group.enabled
        if name == AGENT_CONTEXT_GROUP:
            return self.delivery in ("assigned_agent", "mixed")
        return True

    @property
    def active_group_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.feature_groups if self.group_enabled(name))

    @property
    def active_numeric(self) -> tuple[str, ...]:
        return tuple(
            column
            for name in self.active_group_names
            for column in self.feature_groups[name].numeric
        )

    @property
    def active_categorical(self) -> tuple[str, ...]:
        return tuple(
            column
            for name in self.active_group_names
            for column in self.feature_groups[name].categorical
        )


@dataclass(frozen=True)
class SyntheticConfig:
    n_customers: int = 5000
    seed: int = 7


@dataclass(frozen=True)
class EnvironmentConfig:
    base_conversion_rate: float = 0.06
    context_influence: float = 1.4
    seed: int = 2026


@dataclass(frozen=True)
class ExperimentConfig:
    n_rounds: int = 30000
    seed: int = 42
    output_dir: str = "results"
    agents: Mapping[str, Mapping[str, Any]] = _DEFAULT_AGENTS


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    products: ProductsConfig = field(default_factory=ProductsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    state: StateConfig | None = None
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an :class:`AppConfig` from a YAML file."""
    path = Path(path)
    _require(path.is_file(), f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    _require(raw is None or isinstance(raw, dict), f"Top level of {path} must be a mapping")
    return config_from_dict(raw or {})


def config_from_dict(raw: Mapping[str, Any]) -> AppConfig:
    """Build and validate a config from a YAML-shaped dictionary."""
    state = _parse_state(_section(raw, "state"))
    data = _parse_data(_section(raw, "data"))
    if state is not None:
        data = _apply_feature_groups(data, state)
    config = AppConfig(
        data=data,
        products=_parse_products(_section(raw, "products")),
        reward=_parse_reward(_section(raw, "reward")),
        state=state,
        synthetic=_parse_synthetic(_section(raw, "synthetic")),
        environment=_parse_environment(_section(raw, "environment")),
        experiment=_parse_experiment(_section(raw, "experiment")),
    )
    _validate(config)
    return config


def _apply_feature_groups(data: DataConfig, state: StateConfig) -> DataConfig:
    """Assemble the flat schema feature lists from the active state groups."""
    defaults = SchemaConfig()
    _require(
        data.schema.numeric_features == defaults.numeric_features
        and data.schema.categorical_features == defaults.categorical_features,
        "Feature columns are defined by state.feature_groups; leave "
        "data.schema.numeric_features / categorical_features unset when using them",
    )
    schema = replace(
        data.schema,
        numeric_features=state.active_numeric,
        categorical_features=state.active_categorical,
    )
    return replace(data, schema=schema)


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key) or {}
    _require(isinstance(value, Mapping), f"Config section '{key}' must be a mapping")
    return value


def _str_tuple(value: Any, name: str) -> tuple[str, ...]:
    _require(isinstance(value, (list, tuple)), f"'{name}' must be a list of column names")
    return tuple(str(item) for item in value)


def _parse_data(section: Mapping[str, Any]) -> DataConfig:
    defaults = SchemaConfig()
    schema_raw = section.get("schema") or {}
    _require(isinstance(schema_raw, Mapping), "'data.schema' must be a mapping")
    schema = SchemaConfig(
        customer_id=str(schema_raw.get("customer_id", defaults.customer_id)),
        numeric_features=_str_tuple(
            schema_raw.get("numeric_features", defaults.numeric_features),
            "data.schema.numeric_features",
        ),
        categorical_features=_str_tuple(
            schema_raw.get("categorical_features", defaults.categorical_features),
            "data.schema.categorical_features",
        ),
        owned_product_prefix=str(
            schema_raw.get("owned_product_prefix", defaults.owned_product_prefix)
        ),
    )
    databricks_raw = section.get("databricks") or {}
    _require(isinstance(databricks_raw, Mapping), "'data.databricks' must be a mapping")
    databricks = DatabricksConfig(
        catalog=databricks_raw.get("catalog"),
        schema=databricks_raw.get("schema"),
        table=databricks_raw.get("table"),
    )
    path = section.get("path")
    return DataConfig(
        source=str(section.get("source", "synthetic")),
        path=None if path is None else str(path),
        databricks=databricks,
        schema=schema,
    )


def _parse_products(section: Mapping[str, Any]) -> ProductsConfig:
    defaults = ProductsConfig()
    catalog = _str_tuple(section.get("catalog", defaults.catalog), "products.catalog")
    return ProductsConfig(
        catalog=catalog,
        premiums=_value_map(section.get("premiums", defaults.premiums), "products.premiums"),
        ape=_value_map(section.get("ape", {}), "products.ape"),
        vnb=_value_map(section.get("vnb", {}), "products.vnb"),
    )


def _value_map(raw: Any, name: str) -> dict[str, float]:
    _require(isinstance(raw, Mapping), f"'{name}' must be a mapping of product -> value")
    return {str(product): float(value) for product, value in raw.items()}


def _parse_state(section: Mapping[str, Any]) -> StateConfig | None:
    if not section:
        return None
    groups_raw = section.get("feature_groups") or {}
    _require(isinstance(groups_raw, Mapping), "'state.feature_groups' must be a mapping")
    groups: dict[str, FeatureGroupConfig] = {}
    for name, group_raw in groups_raw.items():
        group_raw = group_raw or {}
        _require(
            isinstance(group_raw, Mapping),
            f"Feature group '{name}' must be a mapping with numeric/categorical lists",
        )
        enabled = group_raw.get("enabled")
        groups[str(name)] = FeatureGroupConfig(
            numeric=_str_tuple(
                group_raw.get("numeric", ()), f"state.feature_groups.{name}.numeric"
            ),
            categorical=_str_tuple(
                group_raw.get("categorical", ()), f"state.feature_groups.{name}.categorical"
            ),
            enabled=None if enabled is None else bool(enabled),
        )
    trends_raw = section.get("trends") or []
    _require(isinstance(trends_raw, (list, tuple)), "'state.trends' must be a list")
    trends = []
    for index, entry in enumerate(trends_raw):
        _require(
            isinstance(entry, Mapping) and "short" in entry and "long" in entry,
            f"state.trends[{index}] must be a mapping with 'short' and 'long' column names",
        )
        trends.append(TrendConfig(short=str(entry["short"]), long=str(entry["long"])))
    gaps_raw = section.get("coverage_gaps") or {}
    _require(isinstance(gaps_raw, Mapping), "'state.coverage_gaps' must be a mapping")
    coverage_gaps = CoverageGapsConfig(
        segment_by=_str_tuple(gaps_raw.get("segment_by", ()), "state.coverage_gaps.segment_by")
    )
    return StateConfig(
        delivery=str(section.get("delivery", "mixed")),
        feature_groups=groups,
        trends=tuple(trends),
        coverage_gaps=coverage_gaps,
    )


def _parse_reward(section: Mapping[str, Any]) -> RewardConfig:
    return RewardConfig(type=str(section.get("type", "conversion")))


def _parse_synthetic(section: Mapping[str, Any]) -> SyntheticConfig:
    defaults = SyntheticConfig()
    return SyntheticConfig(
        n_customers=int(section.get("n_customers", defaults.n_customers)),
        seed=int(section.get("seed", defaults.seed)),
    )


def _parse_environment(section: Mapping[str, Any]) -> EnvironmentConfig:
    defaults = EnvironmentConfig()
    return EnvironmentConfig(
        base_conversion_rate=float(
            section.get("base_conversion_rate", defaults.base_conversion_rate)
        ),
        context_influence=float(section.get("context_influence", defaults.context_influence)),
        seed=int(section.get("seed", defaults.seed)),
    )


def _parse_experiment(section: Mapping[str, Any]) -> ExperimentConfig:
    defaults = ExperimentConfig()
    agents_raw = section.get("agents", defaults.agents)
    _require(isinstance(agents_raw, Mapping), "'experiment.agents' must be a mapping")
    agents: dict[str, dict[str, Any]] = {}
    for name, params in agents_raw.items():
        params = params or {}
        _require(
            isinstance(params, Mapping),
            f"Parameters for agent '{name}' must be a mapping, got {type(params).__name__}",
        )
        agents[str(name)] = dict(params)
    return ExperimentConfig(
        n_rounds=int(section.get("n_rounds", defaults.n_rounds)),
        seed=int(section.get("seed", defaults.seed)),
        output_dir=str(section.get("output_dir", defaults.output_dir)),
        agents=agents,
    )


def _validate(config: AppConfig) -> None:
    data, products = config.data, config.products
    _require(
        data.source in VALID_SOURCES,
        f"data.source must be one of {VALID_SOURCES}, got '{data.source}'",
    )
    if data.source in ("csv", "parquet"):
        _require(bool(data.path), f"data.path is required when data.source is '{data.source}'")
    if data.source == "databricks":
        db = data.databricks
        _require(
            all([db.catalog, db.schema, db.table]),
            "data.databricks.catalog/schema/table are all required when "
            "data.source is 'databricks'",
        )

    _require(len(products.catalog) > 0, "products.catalog must list at least one product")
    _require(
        len(set(products.catalog)) == len(products.catalog),
        "products.catalog contains duplicate product names",
    )
    overlap = set(data.schema.numeric_features) & set(data.schema.categorical_features)
    _require(
        not overlap,
        f"Columns listed as both numeric and categorical: {sorted(overlap)}",
    )

    _require(
        config.reward.type in VALID_REWARD_TYPES,
        f"reward.type must be one of {VALID_REWARD_TYPES}, got '{config.reward.type}'",
    )
    value_sources = {
        "revenue": ("products.premiums", products.premiums),
        "ape": ("products.ape", products.ape),
        "vnb": ("products.vnb", products.vnb),
    }
    if config.reward.type in value_sources:
        source_name, mapping = value_sources[config.reward.type]
        missing = [p for p in products.catalog if p not in mapping]
        _require(
            not missing,
            f"reward.type '{config.reward.type}' needs a {source_name} value for "
            f"every product; missing: {missing}",
        )

    _validate_state(config)

    _require(config.synthetic.n_customers > 0, "synthetic.n_customers must be positive")
    _require(
        0.0 < config.environment.base_conversion_rate < 1.0,
        "environment.base_conversion_rate must be strictly between 0 and 1",
    )
    _require(
        config.environment.context_influence >= 0.0,
        "environment.context_influence must be non-negative",
    )
    _require(config.experiment.n_rounds > 0, "experiment.n_rounds must be positive")
    _require(len(config.experiment.agents) > 0, "experiment.agents must list at least one agent")


def _validate_state(config: AppConfig) -> None:
    state = config.state
    if state is None:
        for label, params in config.experiment.agents.items():
            _require(
                "features" not in params,
                f"Agent '{label}' sets 'features' but no state.feature_groups are configured",
            )
        return

    _require(len(state.feature_groups) > 0, "state.feature_groups must define at least one group")
    _require(
        state.delivery in VALID_DELIVERY_MODES,
        f"state.delivery must be one of {VALID_DELIVERY_MODES}, got '{state.delivery}'",
    )
    all_columns = [*state.active_numeric, *state.active_categorical]
    duplicates = sorted({column for column in all_columns if all_columns.count(column) > 1})
    _require(
        not duplicates,
        f"Columns listed in more than one active feature group (or as both numeric "
        f"and categorical): {duplicates}",
    )
    numeric = set(state.active_numeric)
    categorical = set(state.active_categorical)
    for index, trend in enumerate(state.trends):
        _require(
            trend.short != trend.long,
            f"state.trends[{index}] uses the same column for short and long windows",
        )
        for column in (trend.short, trend.long):
            _require(
                column in numeric,
                f"state.trends[{index}] column '{column}' is not a numeric feature "
                f"of any active group",
            )
    for column in state.coverage_gaps.segment_by:
        _require(
            column in categorical,
            f"state.coverage_gaps.segment_by column '{column}' is not a categorical "
            f"feature of any active group",
        )
    active_names = set(state.active_group_names)
    for label, params in config.experiment.agents.items():
        features = params.get("features")
        if features is None:
            continue
        _require(
            isinstance(features, (list, tuple)),
            f"Agent '{label}': 'features' must be a list of feature-group names",
        )
        unknown = {str(name) for name in features} - active_names
        _require(
            not unknown,
            f"Agent '{label}' references unknown or disabled feature groups "
            f"{sorted(unknown)}; active groups: {sorted(active_names)}",
        )
