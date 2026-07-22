# Customer State & Journey Model for Cross-Sell RL

**Status:** Draft v0.1 · **Date:** 2026-07-21 · **Owner:** Harry
**Related:** `config/rl_nba_config.yml` (the current state definition), `rl_project_spec_draft.md`

---

## 1. Purpose & scope

This document defines **what an insurance customer *is* and *does*** from the point of view of the
cross-sell recommendation engine, so we can agree on:

1. the **journey states** a customer moves through (life-stage and relationship),
2. the **situations/events** they get involved with (the triggers a decision must react to),
3. the **action space** the engine chooses from, and
4. the **customer-state feature catalog** — everything we could capture, how much of it we already
   have in `config.yml`, and what is worth adding.

It is deliberately broader than what we will build in Phase 1. Treat Section 6 as a menu: Section 9
proposes the subset to start with.

---

## 2. How this feeds the RL system

The engine makes one decision per customer at a point in time. In reinforcement-learning terms:

| RL element | In cross-sell | Where it is defined |
|---|---|---|
| **State** | Everything we know about the customer *now* (Sections 3 & 6) | `config.yml` → `state.feature_groups` |
| **Action** | Which product to recommend — or to hold (Section 5) | `config.yml` → `products.catalog` |
| **Reward** | Value created by the resulting purchase (conversion / APE / VNB) | `config.yml` → `reward.type` |

The **state is the input to every decision** — richer, more predictive state means sharper
personalization, *provided the signal is real*. The point of this document is to make the state
definition explicit and reviewable rather than implicit in a column list.

---

## 3. The customer journey — two lenses

A customer sits in **two overlapping journeys at once**. The engine benefits from knowing where they
are in each, because the "next logical product" depends on both.

### 3.1 Life-stage states (the need-creating journey)

Classic life-stage map, but the **"Signals we could read" column is restricted to columns that actually
exist in `config.yml`** — and lists *all* of them that contribute, so the column doubles as "what to
read/record to place a customer in a stage."

_Shorthand:_ `holdings_*` = the five size columns (`customer_holdings_count`, `_holdings_ap`,
`_holdings_sum_assured`, `_all_policy_holding_count`, `customer_inforce_policy_holding_ap`);
`purchase_count_*` = the four windows (`customer_purchase_count_past_1m/3m/6m/12m`);
`wealth_flags` = `customer_inforce_private_bmu_count`, `_maxfocus_bmu_count`,
`customer_inforced_wealthicon_value_usd`.

| Life-stage state | Age band (`customer_age`) | Needs it opens (`products.catalog`) | Signals we could read (config columns only) |
|---|---|---|---|
| Young single / early career | 21–29 | accident, medical, first saving | `customer_marital_status`=single · `customer_income_range` 0–40k · `wealth_segment`=mass · `has_*`: none/accident/medical · `holdings_*` all low · `purchase_count_*` & `customer_purchased_ap_past_12m` low · lapse/surrender ≈ 0 · `wealth_flags` = 0 |
| Newly married / partnering | 25–35 | term_life, medical, saving | `customer_marital_status`=married · `customer_income_range` 20–80k · `wealth_segment` mass→emerging · `has_accident/medical` · `holdings_*` low, rising (1–2) · `purchase_count_3m/6m` rising · `wealth_flags` = 0 |
| New parents / young family | 28–40 | term_life, whole_life, saving, medical | `customer_marital_status`=married · `customer_income_range` 40–80k · `has_medical/term_life/saving` · `customer_holdings_count` 2–3, rising `customer_holdings_sum_assured` · **high `purchase_count_1m/3m/6m`** · `wealth_flags` = 0 |
| Established family / peak earning | 38–52 | whole_life, investment, annuity, medical | `customer_income_range` 40–80k+ · `wealth_segment` emerging→affluent · `has_term_life/medical/saving/whole_life` · `holdings_*` all high (3–4) · steady `purchase_count_*`, high `customer_purchased_ap_past_12m` · lapse/surrender low · affluent: `customer_inforce_private_bmu_count` > 0 |
| Affluent accumulator | 40–58 | investment, whole_life, annuity | `customer_income_range`=80k+ · `wealth_segment` affluent→HNW · `has_whole_life/investment/saving/medical` · high `customer_holdings_ap` / `customer_inforce_policy_holding_ap` (4–6) · active, high `customer_purchased_ap_past_12m` · **`wealth_flags` all > 0** |
| Pre-retirement | 52–63 | annuity, saving, whole_life, medical | `customer_age` 52–63 · `wealth_segment` affluent→HNW · `has_whole_life/saving/investment/medical` · `holdings_*` high, `customer_holdings_sum_assured` plateauing · **`customer_purchase_count_past_12m` slowing** · **`customer_surrender_policy_count_past_12m` may rise** · `wealth_flags` present |
| Retirement / decumulation | 63+ | annuity, medical, whole_life | `customer_income_range` drops to 0–40k · `has_annuity/whole_life/medical` · high `customer_inforce_policy_holding_ap` · low `purchase_count_*` · **`customer_surrender_policy_count_past_12m` elevated (drawdown)** · legacy holders show `wealth_flags` |
| Legacy / estate (HNW) | 58+ | whole_life, investment, annuity | `wealth_segment`=high_net_worth · `customer_income_range`=80k+ · `has_whole_life/investment/annuity` · very high `holdings_*` · selective large `customer_purchased_ap_past_12m` · **`wealth_flags` all high** |

**Every config column is accounted for.** The 26 columns in the Signals column above are the ones that
*discriminate* life stage. The other 13 are read for **every** customer but **do not vary by life stage**,
so they shape *how/whether* to act rather than *which* stage someone is in — record them, just not as
stage signals:

- `customer_gender` — compliance-sensitive, weak stage signal.
- `tied_agency_customer`, `bancassured_customer`, `brokers_customer` — the delivery channel.
- `voc_purchase`, `voc_service`, `voc_claim` — satisfaction (drives timing / suppression).
- `agent_sales_count_past_12m`, `agent_sales_ap_past_12m`, `agent_avg_sales_policy_count_past_12m`, `agent_policy_13m_lapse_rate`, `agent_repurchase_count_past_6m`, `mdrt_label` — the servicing agent.

> Genuine stages we **can't** separate with today's config — homeowner/mortgage, business owner, health
> event — need the "New" features in Section 6 (Groups A, E, K): `mortgage_flag`, `employment_status`,
> claims columns.

### 3.2 Relationship-lifecycle states (with the insurer)

These describe the *commercial relationship* and are largely derivable from data we already touch.

| Relationship state | Definition | Why it matters for cross-sell |
|---|---|---|
| **Prospect / lead** | No policy yet | Out of scope for cross-sell (acquisition problem) |
| **New customer** | First policy < ~6 months old | High openness, but avoid over-contacting early |
| **Single-product holder** | Exactly one category held | Biggest cross-sell headroom |
| **Multi-product / growing** | 2+ categories, recent additions | Deepen relationship; watch for saturation |
| **Engaged / active** | Recent logins, payments on time, interactions | Best moment to offer |
| **Dormant / inactive** | No interaction in N months | Re-engage before offering |
| **At-risk** | Lapse/surrender signals, missed payments, complaints | **Retention first, cross-sell later** |
| **In-claim** | Open claim in progress | Usually *do not* cross-sell; sensitive moment |
| **Lapsed / surrendered** | Policy terminated | Win-back candidate, different playbook |
| **Win-back** | Previously lapsed, re-approachable | Separate treatment |

**Design implication:** the relationship state is itself a valuable *derived* feature (Group B) and
also drives **eligibility** — e.g. suppress offers while `In-claim` or `At-risk`.

---

## 4. Situations & events the customer experiences (triggers)

These are the discrete things that happen *to* or *around* a customer. Each is both a **potential
trigger** for a decision and a **signal** to fold into state. This is also the list of event streams
we would need access to.

| Event / situation | What it signals | Trigger a decision? | Data source (likely) |
|---|---|---|---|
| Policy purchase / application | Openness, need just met | Yes (cool-off, then next-best) | Policy admin |
| Underwriting decision / rating | Health/risk class, eligibility | Feeds eligibility | Underwriting |
| Premium payment (on-time) | Healthy relationship | Weak positive | Billing |
| Late / missed payment, arrears | Affordability stress, lapse risk | Yes (retention) | Billing |
| Policy renewal / anniversary | Natural review moment | Yes | Policy admin |
| Rider / coverage change | Evolving need | Yes | Policy admin |
| Beneficiary change | Life event (marriage, birth, divorce) | Yes (life-stage) | Policy admin |
| Address / contact / job update | Life event, mobility | Weak | CRM |
| Claim submitted | Need realized, sensitive moment | Usually suppress | Claims |
| Claim outcome (approved/denied) | Satisfaction, trust shift | After settlement | Claims |
| Complaint / service escalation | Dissatisfaction, churn risk | Suppress, then recover | Service / CRM |
| Surrender / partial withdrawal | Disengagement, cash need | Yes (retention) | Policy admin |
| Policy loan taken | Liquidity need | Context | Policy admin |
| Lapse (grace period) | Imminent loss | Yes (save) | Billing |
| Maturity / payout | Cash in hand, re-invest need | Yes (re-invest offer) | Policy admin |
| Quote requested / app abandoned | Active shopping intent | Yes (strong) | Digital / quoting |
| App login / web visit / email open | Digital engagement | Context / timing | Digital analytics |
| Agent meeting logged | High-intent channel touch | Yes | Agency / CRM |
| Marketing/offer sent & response | Prior-offer history (for learning) | Feeds reward & state | **Campaign logs (gap today)** |
| VoC / NPS survey response | Satisfaction segment | Context | Survey |

> **The most important row is "Marketing/offer sent & response."** Without a log linking an offer to
> the purchase that followed it, we cannot compute reward on real data (see the spec, §6).

---

## 5. The action space

### 5.1 System actions (what the RL agent decides)

- **Phase 1 (now):** *which product category to recommend* — one of `accident, medical, saving,
  term_life, whole_life, investment, annuity, …`. This is `products.catalog`.
- **Natural extensions (later):**
  - **"Hold / no offer"** as an explicit action (respect fatigue; sometimes the best move is silence).
  - **Coverage amount / tier** (upsell vs. cross-sell).
  - **Channel** (tied agent, banca, app push, email, SMS).
  - **Timing** (now vs. wait for a better moment).
  - **Message / framing** (protection vs. savings angle).

Each added dimension multiplies the decision space and the data needed, so they are sequenced, not
launched together.

### 5.2 Customer actions (what they do — observed, not chosen by us)

Accept · decline · ignore · request more info · buy a *different* product · lapse · surrender ·
complain · refer. These are **outcomes/signals**, captured into behaviour features (Group D) and, for
the offered product, into the **reward**.

---

## 6. Customer state — the feature catalog

Below is the full menu, grouped. **Status** marks whether it already exists in `config.yml`
(`In config`) or is a **new** suggestion. "Example values" are illustrative.

### A. Demographics & profile

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `customer_age` | Life-stage anchor | 21–85 | In config |
| `customer_gender` | Demographic (compliance-sensitive) | F / M | In config |
| `customer_marital_status` | Life-stage, need for dependents' cover | single/married/divorced/widowed | In config |
| `customer_income_range` | Affordability, product tier | 0-20k / 20-40k / 40-80k / 80k+ | In config |
| `wealth_segment` (Tag-Final) | Value tier, product suitability | mass / affluent / HNW | In config |
| `occupation_class` | Risk class, affordability, needs | professional / manual / … | New |
| `industry_sector` | Employment stability, keyman needs | finance / trade / gov | New |
| `education_level` | Proxy for product sophistication | secondary / tertiary | New |
| `number_of_dependents` | Protection need size | 0–5+ | New |
| `has_children` / `children_age_bands` | Education-savings trigger | none / 0-6 / 7-12 / teen | New |
| `homeowner_flag` / `housing_status` | Mortgage-protection need | own / mortgage / rent | New |
| `employment_status` | Life stage, annuity relevance | employed / self-emp / retired | New |
| `residency_status` / `nationality` | Eligibility, product set | resident / expat | New |
| `preferred_language` | Channel/message fit | EN / ZH | New |
| `smoker_status` | Underwriting, medical/CI relevance | Y / N | New |
| `underwriting_risk_class` | Eligibility & pricing | standard / rated / declined | New |

### B. Relationship & tenure (mostly derived)

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `customer_tenure_months` | Trust, maturity of relationship | 0–360 | New |
| `months_since_onboarding` | New-customer cool-off window | 0–∞ | New |
| `months_since_last_purchase` | Cross-sell recency | 0–∞ | New |
| `relationship_stage` | The lifecycle state (§3.2) | single / multi / at-risk | New |
| `life_stage` | The life-stage state (§3.1), inferred | new_parent / pre_retire | New |
| `distinct_product_categories_held` | Breadth of relationship | 1–7 | New |
| `first_product_category` | Entry product (pathway analysis) | medical | New |
| `primary_product_category` | Largest holding by AP | whole_life | New |

### C. Portfolio & holdings

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `customer_holdings_count` | Number of policies | 1–10 | In config |
| `customer_holdings_ap` | Total annual premium | HK$ | In config |
| `customer_holdings_sum_assured` | Total cover | HK$ | In config |
| `customer_inforce_policy_holding_ap` | In-force AP (CLV proxy) | HK$ | In config |
| `customer_all_policy_holding_count` | All policies incl. lapsed | 1–15 | In config |
| `has_<category>` ownership flags | What they already hold (eligibility) | 0/1 per category | In config |
| `sum_assured_per_category` | Where cover is concentrated | HK$ per cat | New |
| `ap_per_category` | Spend mix | HK$ per cat | New |
| `protection_gap_estimate` | Need minus current cover — the core cross-sell signal | HK$ | New |
| `coverage_to_income_ratio` | Under/over-insured | ratio | New |
| `product_mix_diversity` | Concentration (HHI/entropy) | 0–1 | New |
| `policy_riders_count` | Attachment depth | 0–8 | New |
| `policy_coverage_count` | Breadth within policies | 0–10 | New |
| `average_policy_age` / `oldest` / `newest` | Portfolio maturity | months | New |
| `autopay_enrollment_flag` | Stickiness, lapse risk | Y/N | New |
| `share_of_wallet_estimate` | Room to grow this customer | 0–1 | New |

### D. Behaviour & digital engagement

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `customer_purchase_count_past_{1,3,6,12}m` | Buying momentum (4 windows) | counts | In config |
| `customer_purchased_ap_past_12m` | Recent spend | HK$ | In config |
| `customer_lapsed_policy_count_past_12m` | Disengagement | counts | In config |
| `customer_surrender_policy_count_past_12m` | Cash-out behaviour | counts | In config |
| `quote_requests_count_12m` | Shopping intent | counts | New |
| `application_abandonment_count` | Friction / hesitation | counts | New |
| `app_logins_90d` / `days_since_last_login` | Digital engagement & timing | counts / days | New |
| `website_visits_90d` | Interest signal | counts | New |
| `email_open_rate` / `email_click_rate` | Reachability, channel fit | 0–1 | New |
| `call_center_contacts_12m` | Service load, needs | counts | New |
| `service_requests_12m` | Engagement / friction | counts | New |
| `complaints_count_12m` | Dissatisfaction (suppress offers) | counts | New |
| `agent_meetings_12m` | High-intent touches | counts | New |
| `content_engagement_score` | Nurture readiness | 0–100 | New |
| `prior_offers_count` / `prior_offer_accept_rate` | Offer fatigue & responsiveness | counts / 0–1 | New (needs offer logs) |
| `days_since_last_offer` | Fatigue governance | days | New (needs offer logs) |

### E. Claims history

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `claims_count_lifetime` / `_12m` | Realized risk, engagement | counts | New |
| `total_claim_amount` | Severity | HK$ | New |
| `months_since_last_claim` | Recency (sensitivity window) | months | New |
| `claim_type_mix` | Which risks materialized | medical / accident / CI | New |
| `claim_approval_rate` | Experience quality | 0–1 | New |
| `open_claim_flag` | **Hard suppressor** for offers | Y/N | New |
| `claim_experience_satisfaction` | Trust shift post-claim | 1–5 | New |

### F. Payment & financial signals

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `payment_method` | Stickiness (auto vs manual) | auto / manual | New |
| `payment_frequency` | Cash-flow pattern | monthly / annual | New |
| `premium_to_income_ratio` | Affordability headroom | 0–1 | New |
| `late_payment_count_12m` | Lapse-risk signal | counts | New |
| `arrears_flag` | Immediate risk | Y/N | New |
| `reinstatement_count` | Recovered lapses | counts | New |
| `policy_loan_flag` / `balance` | Liquidity need | Y/N / HK$ | New |
| `partial_withdrawal_count` | Disengagement / cash need | counts | New |

### G. Channel & servicing

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `tied_agency_customer` | Channel flag | 0/1 | In config |
| `bancassured_customer` | Channel flag | 0/1 | In config |
| `brokers_customer` | Channel flag | 0/1 | In config |
| `preferred_channel` | Where to deliver the offer | agent / app / email | New |
| `digital_engagement_tier` | Digital reachability | low / med / high | New |
| `servicing_agent_id` | Join key to agent context | id | New |
| `orphan_policy_flag` | No active servicing agent | Y/N | New |
| `agent_change_recent_flag` | Relationship disruption | Y/N | New |
| `servicing_branch` / `district` | Geography | code | New |

### H. Servicing-agent context

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `agent_sales_count_past_{3,6,12}m` | Agent productivity | counts | In config (12m) |
| `agent_sales_ap_past_12m` | Agent value throughput | HK$ | In config |
| `agent_avg_sales_policy_count_past_12m` | Multi-policy skill | ratio | In config |
| `agent_policy_13m_lapse_rate` | Agent quality / persistency | 0–1 | In config |
| `agent_repurchase_count_past_6m` | Ability to re-sell | counts | In config |
| `mdrt_label` | Agent tier | none / MDRT / COT / TOT | In config |
| `agent_tenure_months` | Experience | months | New |
| `agent_product_specialization` | Fit to recommended product | category mix | New |
| `agent_persistency_rate` | Book quality | 0–1 | New |
| `agent_active_capacity` | Can they act on the lead? | load index | New |

> **Only relevant when a human delivers the offer.** `state.delivery: direct` already drops this
> whole group automatically.

### I. Predictive scores & value segments

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `customer_inforce_private_bmu_count` | High-value segment | counts | In config |
| `customer_inforce_maxfocus_bmu_count` | Premium product holder | counts | In config |
| `customer_inforced_wealthicon_value_usd` | Wealth-tier value | US$ | In config |
| `predicted_lifetime_value` (CLV) | Long-term worth (reward shaping) | HK$ | New |
| `lapse_propensity_score` | Churn risk (retention gate) | 0–1 | New |
| `claim_risk_score` | Expected loss (profit view) | 0–1 | New |
| `affordability_score` | Can they sustain more premium | 0–1 | New |
| `product_propensity_scores` | Existing model scores per product | 0–1 each | New |
| `fraud_risk_flag` | Suppress / route | Y/N | New |

> Existing propensity/CLV models are **valuable inputs to the RL state**, not competitors to it — the
> engine can use their scores as features.

### J. Voice of customer & satisfaction

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `voc_purchase` | Satisfaction with buying | satisfied / neutral / dissatisfied | In config |
| `voc_service` | Satisfaction with service | " | In config |
| `voc_claim` | Satisfaction with claims | " | In config |
| `nps_score` | Advocacy / churn signal | 0–10 | New |
| `last_survey_recency` | Freshness of the signal | months | New |
| `interaction_sentiment` | From calls/chat (NLP) | −1…+1 | New |
| `complaint_resolution_satisfaction` | Recovery quality | 1–5 | New |

### K. Life events & external context (highest value, hardest to source)

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| `recent_life_event_flag` | Marriage/birth/home/job — prime trigger | type + recency | New |
| `mortgage_flag` | Mortgage-protection need | Y/N | New |
| `business_owner_flag` | Keyman / liability need | Y/N | New |
| `policy_anniversary_proximity` | Natural review window | days | New |
| `renewal_window_flag` | Timing | Y/N | New |
| `age_band_transition` | Crossing a pricing/need threshold | Y/N | New |
| `calendar_month` / `seasonality` | Campaign & tax cycles | month | New |
| `macro_regime` | Rates/market — savings vs annuity appeal | context | New |
| `tax_season_flag` | Savings/retirement timing | Y/N | New |

### L. Derived trends & RFM

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| Purchase-pace **trends** | Short vs long window ratios | ratio | In config (`state.trends`) |
| **Coverage gaps** vs. segment | Next-logical-product signal | Δ per category | In config (`state.coverage_gaps`) |
| `engagement_trend` | Warming / cooling | slope | New |
| `ap_growth_trend` | Spend trajectory | slope | New |
| `rfm_recency` / `frequency` / `monetary` | Classic value triplet | tiers | New |
| `claim_frequency_trend` | Rising risk | slope | New |

### M. Eligibility, consent & contact governance (constraints, *not* learned state)

These are **not fed to the learner** — they filter the option set *before* it decides. Capturing them
is mandatory for a compliant system.

| Feature | What it captures / why it matters | Example values | Status |
|---|---|---|---|
| Per-product `eligibility_flags` | Age/suitability/underwriting gates | Y/N per product | New |
| `marketing_consent` / opt-in | Legal basis to contact | Y/N | New |
| `channel_consent` (email/SMS/call) | Per-channel permission | Y/N each | New |
| `do_not_contact_flag` | Hard suppressor | Y/N | New |
| `contact_frequency_counter` | Offers in last N days (fatigue cap) | counts | New |
| `last_contact_recency` | Spacing | days | New |
| `vulnerable_customer_flag` | Regulatory care | Y/N | New |

---

## 7. How many things do we need to capture?

Counting **line items** in Section 6 (several expand into multiple columns — e.g. the purchase-count
line is 4 windows, ownership flags are one per product):

| Dimension | In `config.yml` now | Suggested additions | Candidate total |
|---|---:|---:|---:|
| A. Demographics & profile | 5 | 11 | 16 |
| B. Relationship & tenure | 0 | 8 | 8 |
| C. Portfolio & holdings | 6 | 11 | 17 |
| D. Behaviour & digital engagement | 4 | 13 | 17 |
| E. Claims history | 0 | 7 | 7 |
| F. Payment & financial | 0 | 8 | 8 |
| G. Channel & servicing | 3 | 6 | 9 |
| H. Agent context | 6 | 4 | 10 |
| I. Predictive scores & value | 3 | 6 | 9 |
| J. Voice of customer | 3 | 4 | 7 |
| K. Life events & external | 0 | 9 | 9 |
| L. Derived trends & RFM | 2 | 6 | 8 |
| M. Eligibility & consent *(constraints)* | 0 | 7 | 7 |
| **Total** | **≈ 32** | **≈ 100** | **≈ 132** |

**Read of this:** the state today (~32 line items, expanding to ~45 raw columns after windows and
per-category flags) is a solid **profile + holdings + recent-behaviour** core. The largest untapped
value is in four areas: **claims history, payment/affordability signals, digital-engagement/offer
history, and life-event triggers** — none of which we capture yet, and all of which are highly
predictive of *when* and *what* to cross-sell.

Note: more features is not automatically better. Each column must (a) be reliably available, (b) plausibly
predict response, and (c) survive compliance review. The engine can already switch any group on/off,
so we add features and *measure* whether they earn their place (as we did going baseline → enhanced).

---

## 8. Data readiness & sourcing

| Tier | Groups | Availability | Effort |
|---|---|---|---|
| **Have (in Databricks, in config)** | A(core), C(core), D(core), G, H, I(segments), J | Ready | Low — map columns |
| **Likely available, not yet wired** | B (derivable), C(extended), E (claims), F (billing), L | Needs joins across policy/claims/billing | Medium |
| **Partially available / needs capture** | D(digital, offer history), I(model scores), K | Digital analytics + campaign logs + external | Medium–High |
| **Governance (must have before go-live)** | M | Consent/eligibility systems | Medium |

The single biggest data dependency is **offer-and-outcome logging** (Section 4): the record that a
recommendation was made and what followed. It powers both the reward and the `prior_offer_*` features.

---

## 9. Recommended Phase-1 state (prioritization)

Don't wait for all 132. Start with a **compact, high-signal, low-risk** subset and grow it by measured
uplift:

- **Keep (already live):** A-core, C-core, D-core, G, H, I-segments, J, plus `trends` and
  `coverage_gaps`.
- **Add first (high value, mostly derivable):** `customer_tenure_months`, `relationship_stage`,
  `months_since_last_purchase` (B); `protection_gap_estimate`, `coverage_to_income_ratio` (C);
  `claims_count_12m`, `open_claim_flag` (E); `late_payment_count_12m`, `payment_method` (F).
- **Add when the pipe exists:** digital-engagement + `prior_offer_*` (D), `lapse_propensity_score` /
  `predicted_lifetime_value` (I), life-event triggers (K).
- **Mandatory before any live contact:** the whole of Group M (consent, eligibility, fatigue caps).

Everything here is expressed the same way it already is in `config.yml` — add a column to a feature
group and the pipeline picks it up; toggle a group off to test its contribution.
