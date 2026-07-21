# rl_nba — reinforcement-learning next-best-action

A config-driven reinforcement-learning pipeline for insurance cross-sell:
given a customer, decide **which additional product to offer** so that
conversions are maximized.

Following industry practice (see [References](#references)), the first phase
uses **contextual bandits** — reinforcement learning with single-step
episodes, the standard starting point for next-best-offer systems — with a
clear upgrade path to off-policy evaluation on real logs and, later, full
sequential RL. The module layout mirrors
[Open Bandit Pipeline](https://github.com/st-tech/zr-obp): data / agents
(policies) / environment / evaluation.

## Quickstart

```bash
# install the package from the project directory
cd rl_nba
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# run from the monorepo root so the relative config path resolves
cd ..
rl_nba/.venv/bin/python -m rl_nba.run --config config/rl_nba_config.yml
```

The run prints a comparison table and writes `rl_nba/results/metrics.csv` and
`rl_nba/results/learning_curves.png`. The shipped config compares three models —
`random` (the floor), `baseline` (LinUCB on profile + holdings), and
`enhanced` (LinUCB on the full state). You should see `enhanced` beat
`baseline` beat `random`: the first gap is the value of personalization, the
second is the value of the richer customer view (trend and coverage-gap
signals).

## How it works

Each round is one offer decision:

1. A customer is drawn from the pool; their features are encoded into a
   **context vector** (standardized numerics + one-hot categoricals +
   product-ownership flags).
2. The agent picks one **eligible product** (never one the customer already
   owns) — this is the *action*.
3. The environment draws a conversion; the agent observes the **reward**
   (1/0 conversion by default, or premium revenue) and updates itself.

Agents implemented behind one `BanditAgent` interface:

| Agent | What it does |
|---|---|
| `random` | Uniform over eligible products — the floor. |
| `epsilon_greedy` | Classic context-free bandit — measures what you get *without* personalization. |
| `linucb` | Per-product ridge regression + optimism bonus (Li et al. 2010). |
| `lin_ts` | Linear Thompson Sampling — posterior sampling, often best in practice. |

**What is real vs. simulated:** there are no real outcome logs yet, so
customer *responses* come from a hidden ground-truth model (random per-product
weights over the context features, logistic link) that agents can only learn
about through interaction. Customer *features* can already be real (see
below). Regret and "oracle" metrics use the hidden model and are only
available in simulation.

## Configuration (`config/rl_nba_config.yml`)

Everything is driven by one YAML file; every key is optional and falls back to
an in-code default, so you only spell out what you override. The config lives
in the monorepo-root `config/` folder as `rl_nba_config.yml`; it models the
real production schema (synthetic until an export lands) and maps
the five design considerations onto config sections:

| Design consideration | Config section |
|---|---|
| 1. State space | `state.feature_groups` — named groups (profile, holdings, behaviour, channel, engagement, value_segment, agent_context), each toggleable |
| 2. Action space | `products.catalog` — product categories |
| 3. Reward | `reward.type: conversion \| ape \| vnb` + `products.ape` / `products.vnb` value tables |
| 4. Temporal trends | `state.trends` — short-vs-long window activity ratios |
| 5. Coverage gaps | `state.coverage_gaps` — holdings vs. the segment-typical portfolio |

`state.delivery: assigned_agent | mixed | direct` controls whether
servicing-agent quality features enter the state (direct sales have no human
intermediary, so they are excluded automatically).

Model entries carry a `type:` (algorithm) so the same algorithm can run under
several labels with different state designs. `features:` restricts a model to
selected groups and `derived: false` hides the trend/coverage-gap features —
that is how the `baseline` and `enhanced` state designs compete fairly on
identical customer sequences:

```yaml
agents:
  baseline: {type: linucb, alpha: 1.0, features: [profile, holdings], derived: false}
  enhanced: {type: linucb, alpha: 1.0}   # all active groups + derived features
```

**Simpler flat schema.** For quick experiments you can drop the `state:` block
entirely and list the columns directly instead — the grouping, trends, and
coverage gaps are optional:

```yaml
data:
  schema:
    numeric_features: [age, tenure_years, annual_premium]
    categorical_features: [region, acquisition_channel]
```

**Using real data.** Because the synthetic generator reads the *same* schema,
generated data always has exactly the shape your config declares. When you get
a real export out of Databricks, the swap is config-only:

```yaml
data:
  source: parquet              # synthetic | csv | parquet | databricks
  path: /path/to/customers_export.parquet
```

If a required column is missing you get one clear error listing exactly which
columns the config expects versus what the file contains. Columns the
generator doesn't recognize fall back to generic distributions, so custom
schemas work end-to-end before you know your real column names.

`data.source: databricks` is scaffolded (config fields are parsed and
validated) but intentionally unimplemented until workspace credentials exist —
the loader raises a clear message pointing at the csv/parquet route.

## Project layout

```
config/
  rl_nba_config.yml             this project's config (in the shared monorepo config folder)
rl_nba/                         this project
  pyproject.toml                packaging — installs as `rl-nba`, entry point `rl_nba.run:main`
  src/rl_nba/
    config.py                   typed config loading + validation
    features.py                 FeatureEncoder: rows -> context vectors
    data/                       schema checks, synthetic generator, source-dispatching loader
    agents/                     BanditAgent interface, registry, 4 agents
    env/simulator.py            hidden ground-truth conversion model + bandit environment
    evaluation/
      simulate.py               run agents, learning curves, regret, summary table
      ope.py                    IPS / SNIPS off-policy evaluation (ready for real logs)
      plots.py                  results figure
    run.py                      CLI entry point
```

## Roadmap

1. **Now — simulation:** prove the loop end-to-end; compare agents; tune the
   config to your product catalog.
2. **Real customer features:** export a customer table from Databricks to
   CSV/Parquet, map columns in `data.schema`, rerun. Responses stay simulated.
3. **Real outcomes:** once historical offer logs exist (who was offered what,
   and whether they bought), use `evaluation/ope.py` (IPS/SNIPS) to score
   policies on real data without touching live customers. If offers were
   never randomized, logged propensities won't exist — plan a small
   randomized pilot to collect them.
4. **Sequential RL:** when the goal shifts from per-offer conversion to
   long-term customer value, graduate to multi-step methods (e.g. fitted Q
   iteration / offline RL à la [d3rlpy](https://github.com/takuseno/d3rlpy)).

## References

- [Grid Dynamics — Building a next-best-action model using reinforcement learning](https://www.griddynamics.com/blog/building-a-next-best-action-model-using-reinforcement-learning):
  the staged bandits→RL blueprint this project follows.
- [Open Bandit Pipeline (ZOZO)](https://github.com/st-tech/zr-obp): reference
  architecture for bandit pipelines and off-policy evaluation.
- Li et al. 2010, [A Contextual-Bandit Approach to Personalized News Article
  Recommendation](https://arxiv.org/abs/1003.0146): the LinUCB algorithm.
- [Next Best Action in insurance](https://www.altexsoft.com/blog/next-best-action-insurance/):
  domain context and constraints.
- [Reinforcement Learning applied to Insurance Portfolio Pursuit](https://arxiv.org/abs/2408.00713):
  where this can go once large-scale logs exist.
