"""Typed configuration loading.

The YAML config file is the single source of truth for where data comes from,
what the columns are called, which products can be offered, how reward is
defined, and how experiments run. Every key is optional: omitted keys fall
back to the in-code defaults below. ``config/rl_nba_config.yml`` is the shipped
example that exercises these settings (it overrides most of them).
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
VALID_FUTURE_MODES = ("simulate", "placeholder")
VALID_AGENT_TYPES = ("random", "linucb", "rff_ucb", "all")
VALID_GYM_MODES = ("none", "bandit", "rollout", "masked")

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
    # Strength of the journey block's effect in the (simulated) ground truth, as a
    # multiple of the base per-feature weight scale. 1.0 = journey matters like any
    # feature (largely redundant); higher = a world where journey stage genuinely
    # drives behaviour (only meaningful when dtoc.enabled adds the journey block).
    journey_influence: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    n_rounds: int = 30000
    seed: int = 42
    output_dir: str = "results"


@dataclass(frozen=True)
class AgentConfig:
    """The model to train. ``type: all`` runs every model type; any other value
    runs that single model. ``alpha`` (exploration strength) applies to
    ``linucb`` / ``rff_ucb``; ``journey`` toggles the Digital-Twin journey
    features; ``n_features`` is required for ``rff_ucb`` (and ``all``) and
    ignored by the other types.
    """

    type: str = "linucb"
    alpha: float = 1.0
    journey: bool = True
    n_features: int | None = None

    @property
    def types_to_run(self) -> tuple[str, ...]:
        return ("random", "linucb", "rff_ucb") if self.type == "all" else (self.type,)


@dataclass(frozen=True)
class DToCConfig:
    """Digital Twin of Customer layer.

    ``enabled``: ``True`` makes the DToC layer available (per-customer twins for
    scenario testing, journey visualization, and explainability); ``False`` runs
    the project in plain **feature-vector** mode (as before the DToC existed) and
    building a twin raises a clear error. Training uses the feature vector either
    way — this flag only gates the twin layer on top.

    ``future_mode`` (only when enabled): ``simulate`` projects the customer
    forward through the environment (scenario what-ifs); ``placeholder`` defines
    the future slot but performs no projection.
    """

    enabled: bool = True
    future_mode: str = "simulate"
    horizon: int = 6  # number of future steps to project when future_mode == "simulate"
    time_step_months: int = 1  # months advanced per projected step


@dataclass(frozen=True)
class GymConfig:
    """The Gymnasium environment (``rl_nba.environment.gym_env``).

    ``mode`` picks the flavour (or turns the env off):
      * ``none``    — the Gym env is disabled; constructing ``CrossSellEnv``
        raises ``GymDisabledError``. Use it to declare "we're not using Gym".
      * ``bandit``  — every offer is its own one-step episode (``terminated``
        each step); the faithful contextual-bandit view.
      * ``rollout`` — ``steps_per_episode`` offers per episode (``truncated`` at
        the limit, a fresh customer each step); the shape most deep-RL trainers
        (Stable-Baselines3, RLlib) expect. Eligibility is a *soft* mask — an
        agent may offer an owned product and earns ``illegal_reward``.
      * ``masked``  — like ``rollout``, but eligibility is *hard*-enforced: an
        illegal (owned-product) offer is remapped to a legal one, so it can
        never happen. Pairs with sb3-contrib ``MaskablePPO``.

    The eligibility mask is always exposed — ``info['action_mask']`` and
    ``action_masks()`` — regardless of mode.
    """

    mode: str = "rollout"
    steps_per_episode: int = 256
    illegal_reward: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def single_step(self) -> bool:
        """bandit mode: one offer per episode (terminated), vs a rollout batch."""
        return self.mode == "bandit"

    @property
    def hard_mask(self) -> bool:
        """masked mode: eligibility is enforced, not just penalized."""
        return self.mode == "masked"


@dataclass(frozen=True)
class PopulationConfig:
    """The 48-month population panel environment (``rl_nba.environment.population``).

    A *sequential* env: it steps the whole customer base forward one month at a
    time and optimises long-term cumulative value. Every hazard rate and reward
    weight here is a knob — the exact reward formula is still being researched,
    so these are documented placeholders, not final numbers.
    """

    # -- horizon --
    # (The number of customers is the whole offerable pool — set by
    # synthetic.n_customers, or the real data. There is no separate knob here.)
    n_months: int = 48         # episode length (monthly cut-offs)
    seed: int = 7

    # -- capacity & governance --
    monthly_capacity: int = 500  # max offers executed per month (-1 = unlimited)
    consent_rate: float = 0.9    # fraction of customers who consent to contact
    contact_cooldown: int = 2    # months before a contacted customer may be contacted again
    operational_cost: float = 0.0  # cost charged per executed offer (agent/telesales resource)

    # -- long-term reward --
    discount: float = 0.999      # per-month discount on the cumulative reward
    issuance_value: str = "vnb"  # value credited on issuance: conversion | revenue | ape | vnb
    lapse_penalty: float = 0.5      # fraction of a policy's value lost on lapse
    surrender_penalty: float = 0.8  # fraction lost on surrender
    complaint_cost: float = 100.0   # flat cost charged per complaint

    # -- monthly dynamics (hazards / drift) --
    lapse_rate: float = 0.004      # monthly per-policy lapse hazard
    surrender_rate: float = 0.002  # monthly per-policy surrender hazard
    complaint_rate: float = 0.001  # baseline monthly complaint probability
    fatigue_penalty: float = 0.05  # extra complaint prob when contacted inside the cooldown
    engagement_decay: float = 0.01  # monthly engagement decay


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    products: ProductsConfig = field(default_factory=ProductsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    state: StateConfig | None = None
    agent: AgentConfig = field(default_factory=AgentConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dtoc: DToCConfig = field(default_factory=DToCConfig)
    gym: GymConfig = field(default_factory=GymConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)


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
        agent=_parse_agent(_section(raw, "agent")),
        synthetic=_parse_synthetic(_section(raw, "synthetic")),
        environment=_parse_environment(_section(raw, "environment")),
        experiment=_parse_experiment(_section(raw, "experiment")),
        dtoc=_parse_dtoc(_section(raw, "dtoc")),
        gym=_parse_gym(_section(raw, "gym")),
        population=_parse_population(_section(raw, "population")),
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


def _parse_dtoc(section: Mapping[str, Any]) -> DToCConfig:
    defaults = DToCConfig()
    return DToCConfig(
        enabled=bool(section.get("enabled", defaults.enabled)),
        future_mode=str(section.get("future_mode", defaults.future_mode)),
        horizon=int(section.get("horizon", defaults.horizon)),
        time_step_months=int(section.get("time_step_months", defaults.time_step_months)),
    )


def _parse_gym(section: Mapping[str, Any]) -> GymConfig:
    defaults = GymConfig()
    return GymConfig(
        mode=str(section.get("mode", defaults.mode)),
        steps_per_episode=int(section.get("steps_per_episode", defaults.steps_per_episode)),
        illegal_reward=float(section.get("illegal_reward", defaults.illegal_reward)),
    )


def _parse_population(section: Mapping[str, Any]) -> PopulationConfig:
    d = PopulationConfig()
    g = section.get
    return PopulationConfig(
        n_months=int(g("n_months", d.n_months)),
        seed=int(g("seed", d.seed)),
        monthly_capacity=int(g("monthly_capacity", d.monthly_capacity)),
        consent_rate=float(g("consent_rate", d.consent_rate)),
        contact_cooldown=int(g("contact_cooldown", d.contact_cooldown)),
        operational_cost=float(g("operational_cost", d.operational_cost)),
        discount=float(g("discount", d.discount)),
        issuance_value=str(g("issuance_value", d.issuance_value)),
        lapse_penalty=float(g("lapse_penalty", d.lapse_penalty)),
        surrender_penalty=float(g("surrender_penalty", d.surrender_penalty)),
        complaint_cost=float(g("complaint_cost", d.complaint_cost)),
        lapse_rate=float(g("lapse_rate", d.lapse_rate)),
        surrender_rate=float(g("surrender_rate", d.surrender_rate)),
        complaint_rate=float(g("complaint_rate", d.complaint_rate)),
        fatigue_penalty=float(g("fatigue_penalty", d.fatigue_penalty)),
        engagement_decay=float(g("engagement_decay", d.engagement_decay)),
    )


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
        journey_influence=float(section.get("journey_influence", defaults.journey_influence)),
    )


def _parse_experiment(section: Mapping[str, Any]) -> ExperimentConfig:
    defaults = ExperimentConfig()
    return ExperimentConfig(
        n_rounds=int(section.get("n_rounds", defaults.n_rounds)),
        seed=int(section.get("seed", defaults.seed)),
        output_dir=str(section.get("output_dir", defaults.output_dir)),
    )


def _parse_agent(section: Mapping[str, Any]) -> AgentConfig:
    defaults = AgentConfig()
    n_features = section.get("n_features")
    return AgentConfig(
        type=str(section.get("type", defaults.type)),
        alpha=float(section.get("alpha", defaults.alpha)),
        journey=bool(section.get("journey", defaults.journey)),
        n_features=None if n_features is None else int(n_features),
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
    _require(
        config.environment.journey_influence >= 0.0,
        "environment.journey_influence must be non-negative",
    )
    agent = config.agent
    _require(
        agent.type in VALID_AGENT_TYPES,
        f"agent.type must be one of {VALID_AGENT_TYPES}, got '{agent.type}'",
    )
    _require(agent.alpha >= 0.0, "agent.alpha must be non-negative")
    if agent.type in ("rff_ucb", "all"):
        _require(
            agent.n_features is not None and agent.n_features > 0,
            f"agent.n_features is required and must be positive when agent.type is '{agent.type}'",
        )

    _require(
        config.dtoc.future_mode in VALID_FUTURE_MODES,
        f"dtoc.future_mode must be one of {VALID_FUTURE_MODES}, got '{config.dtoc.future_mode}'",
    )
    _require(config.dtoc.horizon >= 0, "dtoc.horizon must be non-negative")
    _require(config.dtoc.time_step_months >= 1, "dtoc.time_step_months must be at least 1")

    gym = config.gym
    _require(
        gym.mode in VALID_GYM_MODES,
        f"gym.mode must be one of {VALID_GYM_MODES}, got '{gym.mode}'",
    )
    _require(gym.steps_per_episode >= 1, "gym.steps_per_episode must be at least 1")

    pop = config.population
    _require(pop.n_months >= 1, "population.n_months must be at least 1")
    _require(
        pop.issuance_value in VALID_REWARD_TYPES,
        f"population.issuance_value must be one of {VALID_REWARD_TYPES}, "
        f"got '{pop.issuance_value}'",
    )
    _require(0.0 < pop.discount <= 1.0, "population.discount must be in (0, 1]")
    _require(pop.contact_cooldown >= 0, "population.contact_cooldown must be non-negative")
    for name in ("consent_rate", "lapse_rate", "surrender_rate", "complaint_rate"):
        value = getattr(pop, name)
        _require(0.0 <= value <= 1.0, f"population.{name} must be in [0, 1]")


def _validate_state(config: AppConfig) -> None:
    state = config.state
    if state is None:
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
