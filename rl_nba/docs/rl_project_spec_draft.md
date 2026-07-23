# Cross-Sell Reinforcement Learning — Initial Spec & Requirements

> **Status:** 🟡 DRAFT v0.1 — for review · **Date:** 2026-07-21 · **Owner:** Harry
> **Reviewers:** _(data team, actuarial, distribution/agency, compliance — TBC)_
> **Companion doc:** [`customer_state_and_journey_model.md`](customer_state_and_journey_model.md)
>
> _This is a starting draft to circulate and mark up later this week. Sections marked_ **[DECIDE]**
> _need an owner's call; sections marked_ **[CONFIRM]** _need a fact from the data/actuarial teams._

---

## 1. Objective

Build a **self-learning next-best-offer engine** that, for a given existing customer, recommends the
**single insurance product most worth offering** — optimizing the **value the business actually earns**,
not just raw conversions — and that **keeps improving from outcomes** rather than being retrained by hand.

**One-line success:** the engine's recommendations measurably beat our current targeting on a live A/B
test, within agreed guardrails.

---

## 2. Business context & why RL

- Cross-sell is our highest-leverage growth lever: customers are already acquired, trusted, and known.
- Today's targeting (segments / static propensity models) is **trained once, then frozen**, personalizes
  only to the segment, and never tests its own blind spots.
- Cross-sell is a **repeated decision under uncertainty that generates its own feedback** — exactly the
  shape reinforcement learning is built for. We start with **contextual bandits** (single-step RL), the
  proven industry entry point, with a clear path to sequential RL later.
- A working, config-driven prototype already exists (`rl_nba`) and demonstrates the mechanism in
  simulation. This spec is about taking it toward real data and a pilot.

---

## 3. Scope

**In scope (this initiative)**
- Which **product category** to recommend to an existing customer.
- A **value-weighted** objective (conversion → APE → VNB).
- Offline evaluation, then a small **randomized pilot**, then a **live A/B test**.
- A reusable, config-driven pipeline (data → state → policy → reward → evaluation).

**Out of scope (for now)**
- New-customer acquisition / lead generation.
- Choosing **channel, timing, message, or coverage amount** (later phases).
- Pricing / underwriting decisions.
- Full sequential / lifetime-value RL (Phase 5+).
- Automated sending of offers without a human/compliance gate (pilot stays supervised).

---

## 4. Problem formulation

| Element | Definition | Config location |
|---|---|---|
| **Decision cadence** | One recommendation per customer at a trigger/scoring point | — |
| **State** | Customer profile + holdings + behaviour + engagement + segment (see companion doc) | `state.feature_groups` |
| **Action** | One product category from the catalog (Phase 1); "hold" added later | `products.catalog` |
| **Reward** | Value of the resulting purchase | `reward.type` |
| **Constraints** | Eligibility, suitability, consent, contact-fatigue caps — applied *before* the policy chooses | (governance layer) |
| **Method** | Contextual bandit (LinUCB / RFF kernel / Thompson) → sequential RL later | `agent.type` |

**[DECIDE] Reward definition.** Options: `conversion` (1/0), `APE` (premium volume), `VNB` (economic
value). Recommendation: **VNB** so the engine optimizes profit, not volume — pending real VNB figures
per category. A retention term (subtract lapses) is a Phase-4+ enhancement.

**[DECIDE] "Hold / no-offer" action.** Include an explicit do-nothing action so the engine can protect
contact budget? Recommended yes, but as a fast-follow after the basic loop works.

---

## 5. Success metrics

| Metric | Definition | Target |
|---|---|---|
| **Primary — value uplift** | VNB (or APE) per offered customer vs. current approach | **[CONFIRM]** with business |
| Conversion uplift | Offer→purchase rate vs. control | Positive, significant |
| Regret / % of achievable value | How close to best-in-hindsight | Trending up over time |
| Retention effect | Lapse/surrender among targeted customers | No worse than control |
| Guardrail: complaint / opt-out rate | Customer-experience cost | ≤ control |
| Operational: coverage & latency | % customers scorable; scoring time | Meets SLA **[CONFIRM]** |

---

## 6. Data requirements & gaps

**State features** — per the companion catalog. Phase-1 subset is a mappable set of columns we largely
already have in Databricks.

**The three load-bearing data questions** (in priority order):

1. **[CONFIRM] Offer-to-outcome attribution.** Can we link *a specific recommendation/campaign* to the
   *purchase that followed it*? **This is the gating dependency** — without it, reward cannot be computed
   on real data and nothing is learned from reality.
2. **[CONFIRM] Historical offer logs.** Do logs of past offers/campaigns and their responses exist, and
   how far back? These enable *offline* evaluation before any live test. If offers were never randomized,
   their selection probabilities are unknown → we plan a small **randomized pilot** to collect clean data.
3. **[CONFIRM] Real product economics.** Actual **APE and VNB per category** (currently placeholders in
   the config) to make the reward economically correct.

**Also needed:** exact production **column names/definitions** to map into `state.feature_groups`, and
the **consent/eligibility** data for the governance layer (Group M in the companion doc).

---

## 7. Solution approach & architecture

- **Config-driven pipeline (exists):** one YAML file defines the data source, the customer schema, the
  product catalog, the reward, and which models to compare. Swapping synthetic → real data is a
  config-only change (`data.source: parquet|databricks`).
- **Modules (exists):** data/loader · feature/state builder · agents (LinUCB, Thompson sampling, +
  baselines) · simulated environment · evaluation (learning curves, regret) + **off-policy evaluation
  (IPS/SNIPS)** ready for real logs.
- **Environment target:** Databricks (customer data lives there). **[CONFIRM]** access & the read path
  (direct connector vs. export to Parquet).
- **Serving (later):** batch scoring to start (score → hand to agents/channels), streaming/real-time as a
  later option. **[DECIDE]** batch vs. real-time for the pilot.
- **Guardrails layer:** eligibility/consent/fatigue filters sit *in front of* the policy; exploration is
  budgeted and monitored; every recommendation is logged with its estimated value and uncertainty (auditable).

---

## 8. Delivery phases (evidence ladder)

Each rung de-risks the next; no customer is contacted before Phase 4; Phases 3–5 carry a go/no-go review.

| Phase | Goal | Key dependency | Rough effort | Status |
|---|---|---|---|---|
| **1. Simulation** | Prove the loop; compare models | none | done | ✅ complete |
| **2. Real customer features** | Run the pipeline on a real customer export (responses still simulated) | one Databricks export + column mapping | S | Next |
| **3. Offline / historical replay** | Score candidate policies against past outcomes (off-policy eval) | offer logs + **attribution** | M | Blocked on §6.1–6.2 |
| **4. Randomized pilot** | Small, capped live test to collect clean data & first real evidence | pilot design + compliance sign-off | M–L | — |
| **5. Live A/B test** | Prove business value vs. current practice | pilot success | L | — |
| **6+. Extend** | Channel/timing/hold actions; sequential RL for lifetime value | scale + trust | L+ | Later |

**[DECIDE] Milestone dates / who owns each phase.** Fill in at review.

---

## 9. Guardrails, compliance & ethics

- **Compliance filters first:** suitability, age/product eligibility, and **sensitive-attribute
  exclusions** (e.g. handling of `gender`) sit ahead of the engine. **[CONFIRM]** rules with compliance.
- **Suppression states:** no cross-sell while `in-claim`, `at-risk`, or under complaint; respect
  `do_not_contact` and marketing consent.
- **Contact fatigue:** frequency caps live *outside* the learning loop and are non-negotiable.
- **Exploration budget:** the share of "benefit-of-the-doubt" offers is capped and monitored.
- **Auditability & fairness:** every decision logs its inputs, estimated value, and uncertainty; monitor
  outcome disparities across demographic groups. **[DECIDE]** fairness metric & threshold.
- **Data privacy:** PII handling per policy; features derived within governed environment.

---

## 10. Risks & assumptions

| Risk / assumption | Impact | Mitigation |
|---|---|---|
| No offer-to-purchase attribution | Can't learn from real data | Randomized pilot to generate clean logs (Phase 4) |
| Features carry weak signal | Modest uplift | Measure per-group contribution; start compact, grow by evidence |
| Real behaviour ≠ simulation | Sim results don't transfer | Treat sim as machinery proof only; gate on Phase 4–5 evidence |
| Placeholder VNB/APE values | Wrong optimization target | Get actuarial figures before Phase 3 |
| Low decision volume | Slow learning | RL needs scale; confirm monthly offer volume **[CONFIRM]** |
| Compliance constraints tighten action space | Fewer eligible offers | Bake constraints in from day one |
| Agent/channel adoption | Recommendations ignored | Involve distribution early; explainable outputs |

---

## 11. Open questions / decisions needed

**For the data team**
1. Can a purchase be attributed to a preceding offer/campaign? _(gating)_
2. Do historical offer/campaign logs exist, with response and — ideally — selection probability?
3. Exact production column names/definitions for the Phase-1 state features.
4. Databricks access & preferred read path (connector vs. export).

**For actuarial / finance**
5. APE and VNB per product category (to replace config placeholders).
6. Any retention/persistency value to fold into reward later?

**For distribution / agency**
7. How are offers delivered, and by whom (tied agent / banca / broker / direct)? _(sets whether agent
   features enter the state)_
8. Contact-frequency limits and current campaign cadence.

**For compliance / risk**
9. Suitability & eligibility rules; treatment of sensitive attributes; fairness expectations.
10. Vulnerable-customer and consent handling.

**Product / sponsor**
11. Primary success metric & target uplift; batch vs. real-time; pilot size & timeline; budget/ownership.

---

## 12. Stakeholders (lightweight RACI) — **[DECIDE]**

| Area | Name / team | Role |
|---|---|---|
| Project lead / DS | Harry | R / A |
| Data engineering (Databricks) | _TBC_ | R |
| Actuarial (VNB/APE) | _TBC_ | C |
| Distribution / agency | _TBC_ | C |
| Compliance / risk | _TBC_ | C / sign-off |
| Business sponsor | _TBC_ | A |

---

## 13. Next steps (this week)

- [ ] Circulate this draft + the companion state/journey doc to the reviewers above.
- [ ] Get answers to the four **data-team** questions in §11 (esp. attribution).
- [ ] Pull one **real customer export** from Databricks to unblock Phase 2.
- [ ] Ask actuarial for **APE/VNB per category**.
- [ ] Confirm **who delivers offers** (sets the agent-feature decision) and **monthly offer volume**.
- [ ] Book a 45-min review to walk this doc and agree Phase-2 scope, owners, and dates.

---

_Appendix: the working prototype (`rl_nba`) already implements the config-driven pipeline, four bandit
agents, off-policy evaluation scaffolding, a run notebook, and an interactive playback dashboard
(`python -m rl_nba.serve`). It runs today on synthetic data shaped to the intended production schema._
