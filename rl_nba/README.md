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
`rl_nba/results/learning_curves.png`. The shipped config uses `agent.type: all`, so it
runs three models — `random` (the floor), `linucb` (linear contextual bandit), and
`rff_ucb` (non-linear kernel bandit) — on identical customer sequences. You should see
both learners beat `random` by a wide margin (the value of personalization).

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
| `rff_ucb` | **Non-linear** kernel bandit: LinUCB on Random Fourier Features (captures interactions). |

One `agent:` block selects the model: `type` (`random` / `linucb` / `rff_ucb` / `all`),
`alpha` (exploration), `journey` (use the Digital-Twin journey features), and `n_features`
(required for `rff_ucb` / `all`). `type: all` runs every model on identical draws (the harness
uses common random numbers). To compare with vs without journey, run once with `journey: true`
and once with `journey: false`.

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

The model is chosen with a single `agent:` block — `type: all` runs every model type on
identical customer sequences:

```yaml
agent:
  type: all         # random | linucb | rff_ucb | all
  alpha: 1.0        # exploration (linucb / rff_ucb)
  journey: true     # use the Digital-Twin journey features (needs dtoc.enabled)
  n_features: 256   # RFF features — required for rff_ucb and 'all'
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

## Examples

- **`examples/run_simulation.ipynb`** — runs the whole simulation cell by cell.
  It finds the project and config on its own and installs its own dependencies,
  so you can just open it and Run All (needs a Python 3.11+ kernel).
- **Playback dashboard** — an interactive, self-contained page for non-technical
  viewers: step through one episode and watch the model score each product (value
  estimate + exploration bonus), make an offer, and learn. By default it plays back
  the **champion** — the best-performing model the config runs, chosen on the same
  `metrics.csv` ranking (a header badge names it). Serve it locally:

  ```bash
  python -m rl_nba.serve                 # champion of the config's models, on :8000
  python -m rl_nba.serve --model linucb  # or pin a specific model to display
  # rl-nba-dashboard --port 9000         # console script, if the package is installed
  ```

  The command regenerates the page from the current config, serves it on
  localhost (opening your browser), and stops on Ctrl+C. With `agent.type: all` it
  runs the comparison first to identify the champion (a few seconds); a single
  `agent.type` is played directly. Regenerate the file yourself with
  `rl_nba.playback.write_dashboard(config, path, model="champion")`.

## Project layout

```
config/
  rl_nba_config.yml             this project's config (in the shared monorepo config folder)
rl_nba/                         this project
  pyproject.toml                packaging — installs as `rl-nba`, entry point `rl_nba.run:main`
  requirements.txt              runtime deps (mirrors pyproject), for pip / the notebook
  examples/
    run_simulation.ipynb        run the whole simulation cell by cell
    playback_dashboard.html     interactive step-by-step episode playback (open in a browser)
  docs/
    customer_state_and_journey_model.md   customer states, journey, and full feature catalog
    digital_twin_of_customer.md           DToC layer + journey model + Databricks table linking
    rl_project_spec_draft.md              initial project spec / requirements (draft)
    RL_for_cross_sell_overview.pptx       concept slide deck
  src/rl_nba/
    config.py                   typed config loading + validation
    features.py                 FeatureEncoder: rows -> context vectors
    data/                       schema checks, synthetic generator, source-dispatching loader
    agents/                     BanditAgent interface, registry, 4 agents
    environment/simulator.py    hidden ground-truth conversion model + bandit environment
    evaluation/
      simulate.py               run agents, learning curves, regret, summary table
      ope.py                    IPS / SNIPS off-policy evaluation (ready for real logs)
      plots.py                  results figure
    journey.py                  standardized journey model (life-stage + relationship-stage)
    dtoc.py                     Digital Twin of Customer: per-customer timeline + scenarios
    playback.py                 record an episode + render the playback dashboard
    serve.py                    serve the playback dashboard on localhost
    run.py                      CLI entry point (prepare / run / save pipeline)
```

## Digital Twin of Customer (DToC)

Each customer can be wrapped in a `DigitalTwin` — a timeline of states (historical / current /
projected-future) that serves as one abstraction for RL training, scenario testing, journey
visualization, and policy explainability. It's built only on config features and reuses the training
world model. See `docs/digital_twin_of_customer.md`.

```python
from rl_nba import DToCWorld, twin_from_row
world = DToCWorld.from_config(config)
twin  = twin_from_row(world, customer_row)
twin.current.journey.label        # e.g. "new_parents / single_product"
twin.project(trained_agent)       # future_mode: simulate -> scenario roll-out
twin.explain(trained_agent)       # per-product value + exploration bonus (why this offer)
```

**`dtoc.enabled`** actually changes training: `true` folds the journey model into the state (models train
on `life_stage` + `relationship_stage`) and makes twins available; `false` is the plain feature-vector
pipeline (building a twin then errors). Run `agent.journey: true` vs `false` to isolate the journey
features' value — ≈ neutral at the default `environment.journey_influence`, but a clear win (+14% for
`linucb`) when journey genuinely drives behaviour (see `docs/digital_twin_of_customer.md` §5).
`dtoc.future_mode` switches future-state projection on (`simulate`) or off (`placeholder`).

## Gym environment (for RL libraries)

`rl_nba.environment.gym_env.CrossSellEnv` wraps the simulator in a standard **Gymnasium** `Env`, so
Stable-Baselines3 / RLlib / CleanRL can train on the same world the bandits use (journey features
included when `dtoc.enabled`). Observation = the customer context (`Box`); action = which product to
offer (`Discrete`); reward = your configured `conversion`/`ape`/`vnb`.

```python
from rl_nba.config import load_config
from rl_nba.environment.gym_env import CrossSellEnv

env = CrossSellEnv(config=load_config("config/rl_nba_config.yml"))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)   # info["action_mask"] = eligibility
```

The `gym:` config block sets the RL-loop shape (everything else is reused). One `mode` switch picks the
flavour — or turns the env off:

- **`mode`** — the single "which flavour" switch:
  - `none` — Gym disabled; building `CrossSellEnv` raises `GymDisabledError` (declare "not using Gym").
  - `bandit` — one offer = one episode (`terminated` each step); the faithful contextual-bandit view.
  - `rollout` — a batch of `steps_per_episode` offers per episode (`truncated` at the limit, a fresh
    customer each step); what most deep-RL trainers expect. Eligibility is a *soft* mask.
  - `masked` — like `rollout` but eligibility is *hard*-enforced (an illegal offer is remapped to a
    legal one); pairs with sb3-contrib's `MaskablePPO`.
- **`steps_per_episode`** — episode length for `rollout` / `masked`.
- **`illegal_reward`** — the eligibility mask is always exposed (`info["action_mask"]` and
  `action_masks()`); this is only what an *unmasked* agent earns for offering an owned product in the
  soft modes (`bandit` / `rollout`).

`gymnasium` is a core dependency; the deep-RL trainers are the optional `rl` extra (`pip install -e .[rl]`,
which pulls in PyTorch). See `examples/gym_quickstart.py` for a random rollout plus an optional
MaskablePPO run. Note: on a single-step problem, deep RL is usually *outperformed* by the contextual
bandits here — the Gym env's value is ecosystem interoperability and the on-ramp to sequential RL.

## Population panel environment (48-month sequential)

`rl_nba.environment.population` is a **sequential** env that steps the *whole customer base* forward one
month at a time and optimises **long-term cumulative value** (not just immediate conversion). Each month:
the agent proposes a `(product, priority)` per customer from the frame `(n_customers × state)`; governance
gates it (**consent**, **eligibility**, **contact-frequency cooldown**); the monthly **capacity** keeps
only the highest-priority offers and the rest become **no-offer**; executed offers may convert
(**issuance**), every policy may **lapse**/**surrender**, contacted customers may **complain**; and the
reward nets issuance value against lapse/surrender/complaint/operational costs, discounted over the horizon.

```bash
python -m rl_nba.environment.population --agent linucb   # runs an episode, saves results/episode.png
```

```python
from rl_nba.config import load_config
from rl_nba.environment.population import run_episode, compare, plot_episode
cfg = load_config("config/rl_nba_config.yml")
print(compare(cfg))                    # linucb vs random on long-term value
plot_episode(run_episode(cfg), "episode.png")
```

`LinUCB` applies row-wise to the frame (a per-customer policy, fully vectorised — a 5,000-customer, 48-month
episode runs in ~1s), and already beats random on long-term value. It uses the whole customer pool (scale
is set by `synthetic.n_customers`, or the real data — there's no separate population size knob). Everything
else is a config knob under `population:` — `n_months`, `monthly_capacity`, `consent_rate`,
`contact_cooldown`, `discount`, `issuance_value` (ape/vnb/…), and every hazard/penalty (`lapse_rate`,
`surrender_rate`,
`complaint_rate`, `fatigue_penalty`, `lapse_penalty`, …). The reward weights are **documented placeholders**
until the exact long-term-value formula is researched. A bandit learns immediate offer value here; a true
*sequential* optimiser (PPO/DQN via RLlib) is the planned next step, along with interactive episode/evaluation
dashboards (the `EpisodeTrace` already captures every metric they need; `plot_episode` renders a PNG for now).

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
