# Digital Twin of Customer (DToC) & Standardized Journey Model

**Status:** Draft v0.1 · **Date:** 2026-07-23 · **Owner:** Harry
**Code:** `rl_nba/journey.py`, `rl_nba/dtoc.py` · **Config:** `dtoc:` block in `config/rl_nba_config.yml`
**Related:** [`customer_state_and_journey_model.md`](customer_state_and_journey_model.md) (feature catalog),
[`rl_project_spec_draft.md`](rl_project_spec_draft.md)

---

## 1. Purpose & scope

The **Digital Twin of Customer (DToC)** is the per-customer abstraction that ties the whole system
together. Each twin holds a customer's **timeline of states** — historical, current, and (optionally)
projected future — and serves as the single core object for **RL training, scenario testing, journey
visualization, and policy explainability**.

It is built **only on the features already in `config.yml`** (no dependency on the "New" features in the
feature catalog), and it reuses the *same* world model as training (encoder + conversion model + reward),
so the twin view and the training pipeline are always consistent.

> **Twin, not oracle.** Historical and current states are *observed data*. Future states are *model
> roll-outs* — plausible trajectories under a policy, not predictions of the real person. The forward
> transition model is a transparent simplification; a proper feature-evolution/forecast model is future
> work. Set `dtoc.future_mode: placeholder` to represent only observed history + current.

---

## 2. Standardized customer journey model

Every state carries a **journey label** on two orthogonal axes, derived from config columns only
(`rl_nba/journey.py`). Any missing column degrades to `unknown` rather than erroring.

**Life stage** (where they are in life) — from `customer_age`, refined by `customer_marital_status`,
`wealth_segment`, `customer_holdings_count`:

`young_single` · `newly_married` · `new_parents` · `established_family` · `affluent_accumulator` ·
`pre_retirement` · `retirement` · `legacy_estate` · `unknown`

**Relationship stage** (where they are with us) — from attrition, recent activity, and breadth
(`customer_lapsed/surrender_policy_count_past_12m`, `customer_purchase_count_past_3m/12m`, ownership flags):

`new_or_prospect` · `single_product` · `multi_product` · `active_growing` · `dormant` · `at_risk` ·
`unknown`

A customer's full journey state is the pair, e.g. **`new_parents / single_product`**. The rules are a
transparent *labelling convention* (for training signals, filtering, and explanation) — not a predictive
model. `at_risk` and `dormant`, for instance, are the states where the governance layer would *suppress*
cross-sell.

---

## 3. The DToC layer

Three objects (`rl_nba/dtoc.py`):

- **`TwinState`** — one snapshot: `step` (0 = current, <0 history, >0 future), `label`, the config
  `features`, the derived `journey` state, the encoded `context` vector, and — for projected steps —
  the `offered` product, `converted`, and `reward`.
- **`DigitalTwin`** — a customer's ordered `history` (last entry = current) plus a projected `future`.
- **`DToCWorld`** — the shared world model: the fitted **encoder** (context vectors), the **conversion
  model**, the **reward** values, and the **transition model**. `DToCWorld.from_config(config)` builds it
  to match the training environment exactly (same seed, same encoding), so a twin's `context` is
  byte-for-byte what the policy trains on.

**Building twins**

```python
world = DToCWorld.from_config(config)         # shared world (encoder + conversion + reward)
twin  = twin_from_row(world, customer_row)     # single snapshot  -> history = [current]
twin  = twin_from_panel(world, monthly_rows)   # real panel        -> full history + current
```

**Future states — configurable.** `DigitalTwin.project(policy)` rolls the customer forward under a policy;
each step it encodes the state, the policy offers an eligible product, a conversion is drawn from the
world's conversion model, and — if it converts — the **transition model** updates the customer
(ownership, holdings count/AP/sum-assured, recent-purchase counts) before time advances. Governed by:

```yaml
dtoc:
  enabled: true           # false = plain feature-vector mode (no twins); see §5
  future_mode: simulate   # simulate = project forward; placeholder = no projection
  horizon: 6              # steps to project (months)
  time_step_months: 1
```

The transition model is deliberately simple and readable (`DToCWorld.apply_purchase` / `advance_time`):
a purchase sets the ownership flag, increments holdings count and all-policy count, adds the product's
premium to the AP/in-force/sum-assured totals, and bumps the recent-purchase windows. It is a stand-in
for a future learned feature-evolution model.

---

## 4. The four capabilities

**A. RL training.** Every `TwinState.context` is the exact vector the policy is trained on — the twin and
the (vectorized) training pipeline share one `DToCWorld`. The twin is the *readable, per-customer* view;
bulk training stays on the array pipeline for scale (2 M customers as Python objects is infeasible, but
the two are guaranteed consistent because they use the same encoder/conversion/reward).

**B. Scenario testing.** Project the same customer under different policies and compare:

```python
a = twin_from_row(world, row); a.project(trained_agent)     # smart policy
b = twin_from_row(world, row); b.project("whole_life")        # always offer whole life
a.scenario_value(), b.scenario_value()   # e.g. 6,000  vs  0
```

(In our demo, forcing `whole_life` on a *new-parent* customer returned 0, while the trained policy —
offering `term_life` then `accident` — returned 6,000. That contrast *is* the scenario test.)

**C. Journey visualization.** `twin.records()` returns the full timeline (per-step journey label, holdings,
offer, outcome) for any UI; `plot_journey(twin, path)` draws it — a portfolio metric over history +
future with purchases marked and the journey stage annotated.

**D. Policy explainability.** `twin.explain(agent)` returns the policy's per-product **value estimate +
exploration bonus** at the current state (needs a LinUCB-style agent), sorted, with the chosen product
flagged — so "why did it recommend term life here?" has a concrete, per-customer answer.

```
term_life    value= 710  +bonus=0  = 710   <- would offer
medical      value= 607  +bonus=0  = 607
saving       value= 371  +bonus=0  = 371
```

---

## 5. Configuration & honesty

- **`enabled` — use the DToC, or just feature vectors like before.** `dtoc.enabled: true` makes the twin
  layer available; `dtoc.enabled: false` runs the project in **plain feature-vector mode** (the behaviour
  before the DToC existed) and `DToCWorld.from_config` raises a clear `DToCDisabledError`. **Crucially, the
  training pipeline is identical in both modes** — it always encodes customers into the feature (context)
  vector and never instantiates twins (2 M+ customers as Python objects would not scale). The twin is a
  *layer on top* of that same representation for per-customer scenario/visualization/explainability work,
  so the toggle gates the layer, not the training.
- **Config features only.** The twin, the journey model, and the transition model use exclusively the
  columns declared in `config.yml`. The ~132-feature catalog remains a *future menu*, not a dependency.
- **`future_mode`** (only when enabled) switches projection on (`simulate`) or off (`placeholder`).
- **Limitations to state plainly:** future states are model roll-outs (not forecasts of the real person);
  the transition model is a simplification; and `DToCWorld.from_config` currently draws its conversion
  model from the *simulated* environment — when real data + real outcomes arrive, that same slot is
  replaced by a model fit on reality, and everything above is unchanged.

---

## 6. Linking Databricks tables for training

This is the format for pointing training (and the DToC) at **real** Databricks tables.

### 6.1 The two tables you need

| Table | Grain (one row per…) | Key columns | Feeds |
|---|---|---|---|
| **Customer feature panel** | `customer_id` × `cut_off_date` | `customer_id`, `cut_off_date`, + every column in `state.feature_groups` and the `has_<product>` flags | the **state** (context) and the twin's history/current |
| **Offer & outcome log** | `customer_id` × `offer_date` × `product` | `customer_id`, `offer_date`, `product`, `converted`, `purchase_ap`/`vnb`, (ideally) `offer_propensity` | the **reward** + off-policy evaluation + attribution |

The panel is the **43 cut-off dates × 2.13 M customers** ≈ 91.7 M rows. The offer log is the **gating
dependency** for learning on real data (see the spec, §6): without it, reward and `prior_offer_*`
features can't be computed.

### 6.2 Naming — Unity Catalog three-level namespace

Reference tables as **`catalog.schema.table`** (e.g. `main.crm.customer_feature_panel`). Put this in the
config:

```yaml
data:
  source: databricks
  databricks:
    catalog: main
    schema: crm
    table: customer_feature_panel     # the feature panel
  schema:
    customer_id: customer_id          # must match the table's column names exactly
    owned_product_prefix: has_
# state.feature_groups column names must equal the panel's column names
# (rename in a SQL view if they differ).
```

### 6.3 Connection patterns (pick per environment)

1. **Run inside Databricks** (recommended for the full 91.7 M-row backtest): a Databricks notebook/job
   reads with Spark — `spark.read.table("main.crm.customer_feature_panel")` — filters to one
   `cut_off_date` at a time (per the 16 GB memory constraint), and processes each slice. Spark handles
   the scale and the streaming-per-cut-off requirement naturally.
2. **External connection** (local dev, smaller extracts): the `databricks-sql-connector` Python package —
   connect with *server hostname*, *HTTP path*, and a *token*, then `SELECT … FROM
   main.crm.customer_feature_panel WHERE cut_off_date = :d`.
3. **Export to Parquet** (works **today**, zero new code): export the table(s) to Parquet and use
   `data.source: parquet` with `data.path`. This is the current supported path — `data.source:
   databricks` is scaffolded (the config fields parse and validate) but the loader still raises a clear
   "export to parquet" message until the connector branch is implemented.

### 6.4 Schema-mapping rules

- Column names in `state.feature_groups` / `data.schema` must **equal** the table's column names — or wrap
  the table in a Databricks **view** that renames them (cheapest way to adapt without touching config
  semantics). E.g. `wealth_segment` ← `"Tag-Final"`.
- The panel must carry a `cut_off_date` column; training/replay iterates cut-off dates as the time axis,
  and the DToC's `twin_from_panel` consumes a customer's rows ordered by that date.
- Ownership must be present as `has_<product>` 0/1 columns (or derived in the view).
- Keep **one grain per table** — don't mix customer-level and policy-level rows; join in a view first.

---

## 7. Where it fits & next steps

- **Now:** the DToC runs on synthetic data (config schema), exercising all four capabilities. Use it for
  design reviews, per-customer sanity checks, and the explainability/scenario demos.
- **With real data:** point `twin_from_panel` at the Databricks feature panel (§6); the twin's history
  becomes the real 43-date trajectory. The conversion model behind `DToCWorld` is swapped for one fit on
  real outcomes once the offer log exists.
- **Later:** replace the simplified transition model with a learned feature-evolution model (unlocks
  higher-fidelity `simulate` future states), and surface the twin timeline in the interactive dashboard.
