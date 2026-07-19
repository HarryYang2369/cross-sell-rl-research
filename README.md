# cross-sell-rl-research

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
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# run the experiment described by the config
.venv/bin/python -m cross_sell_rl.run --config configs/default.yaml

# run the tests
.venv/bin/python -m pytest
```

The run prints a comparison table and writes `results/metrics.csv` and
`results/learning_curves.png`. You should see the contextual agents (`linucb`,
`lin_ts`) clearly beat the `random` and context-free `epsilon_greedy`
baselines — that gap is the value of personalization.

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

## Configuration (`configs/default.yaml`)

Everything is driven by one YAML file; every key is optional and falls back to
the defaults shown in `configs/default.yaml`. The parts you will touch first:

```yaml
data:
  source: synthetic          # synthetic | csv | parquet | databricks
  path: null                 # file path when source is csv/parquet
  schema:                    # column mapping — edit to match your real tables
    customer_id: customer_id
    numeric_features: [age, tenure_years, annual_premium, num_claims, credit_score]
    categorical_features: [region, acquisition_channel]
    owned_product_prefix: has_   # expects 0/1 columns like has_home, has_pet

products:
  catalog: [home, life, health, travel, pet]   # the action space

reward:
  type: conversion           # conversion | revenue
```

Because the synthetic generator reads the *same* schema section, generated
data always has exactly the shape your config declares. When you get a real
export out of Databricks, the swap is config-only:

```yaml
data:
  source: parquet
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
configs/default.yaml            single source of truth (data, schema, products, reward, experiment)
src/cross_sell_rl/
  config.py                     typed config loading + validation
  features.py                   FeatureEncoder: rows -> context vectors
  data/                         schema checks, synthetic generator, source-dispatching loader
  agents/                       BanditAgent interface, registry, 4 agents
  env/simulator.py              hidden ground-truth conversion model + bandit environment
  evaluation/
    simulate.py                 run agents, learning curves, regret, summary table
    ope.py                      IPS / SNIPS off-policy evaluation (ready for real logs)
    plots.py                    results figure
  run.py                        CLI entry point
tests/                          pytest suite (config, data, features, agents, env, OPE, end-to-end)
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
