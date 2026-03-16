# EVN HyperLogistics Pitch Deck (PowerPoint-Ready Paste Format)

## Venture Thesis (Use verbatim)
EVN delivers safety-compliant, explainable AI rerouting for middle-mile logistics.

## PPT Paste Instructions
- Paste each slide's `Title` into the slide title placeholder.
- Paste `On-Slide Text` bullets into the body placeholder.
- Paste `Speaker Notes` text into PowerPoint Notes pane.
- Keep body text to 3 bullets max per slide.

---

## Slide 1
**Title:** Problem

**On-Slide Text:**
- Middle-mile teams face real-time disruption decisions without trustworthy AI.
- Current tools detect risk but do not produce safe, explainable reroutes.
- The core gap is prediction-to-action under compliance constraints.

**Visual Cue:** Split panel: disruption alert vs no trusted reroute.

**Proof Metric (show on slide):** 7.7M US accident records.

**Speaker Notes (paste into PPT Notes):**
Middle-mile managers need to make decisions quickly during weather or accident disruptions, but existing systems often stop at risk detection. They do not consistently produce safe, explainable next actions. We frame this as the prediction-action gap. The operating pressure is large and persistent, reflected in 7.7 million US accident records that capture disruption reality at national scale.

---

## Slide 2
**Title:** Solution

**On-Slide Text:**
- ReMindRAG retrieves grounded context for reroute reasoning.
- SRSNet forecasts short-horizon disruption propagation.
- Hard safety veto blocks physically invalid routes.

**Visual Cue:** 3-step flow: Retrieve -> Forecast -> Safety Veto.

**Proof Metric (show on slide):** 100% bridge-constraint enforcement.

**Speaker Notes (paste into PPT Notes):**
HyperLogistics combines learning and rules in one decision loop. ReMindRAG grounds recommendations in evidence, SRSNet predicts near-term risk spread, and the symbolic safety layer enforces non-negotiable constraints like bridge clearance and load limits. This means unsafe routes are rejected before dispatch decisions are made, which converts AI output into compliance-ready action.

---

## Slide 3
**Title:** Product/Technology

**On-Slide Text:**
- Snowflake-native medallion pipeline: Bronze, Silver, Gold.
- Core stack: ReMindRAG + SRSNet + Cortex + Spatial SQL guardrails.
- Streamlit interface provides explainable route comparison.

**Visual Cue:** 4-layer architecture diagram.

**Proof Metric (show on slide):** 4-8 hour forecasting window.

**Speaker Notes (paste into PPT Notes):**
This is built as a deployable platform, not just an experiment. Data ingestion and transformation follow a medallion structure, with retrieval and forecasting layered into the decision engine. Cortex handles inference close to the data, and Spatial SQL applies deterministic safety checks. The front end lets dispatch users see options and rationale clearly, with forecasting tuned to practical 4-to-8-hour middle-mile horizons.

---

## Slide 4
**Title:** Target Customer

**On-Slide Text:**
- Primary users: Logistics Network Managers and Area Managers.
- Core jobs: disruption triage, reroute approval, compliance confirmation.
- Value: faster decisions with auditable justification.

**Visual Cue:** Two-persona card (Network Manager, Area Manager).

**Proof Metric (show on slide):** 100 synthetic manager query-response pairs.

**Speaker Notes (paste into PPT Notes):**
We target the exact operators accountable for middle-mile outcomes: logistics network managers and area managers. Their workflow is time-sensitive and risk-sensitive, so recommendations must be both useful and defensible. To align the system with real user language and scenarios, the Week 8 adaptation set includes 100 manager-style query-response pairs.

---

## Slide 5
**Title:** Market Opportunity

**On-Slide Text:**
- Freight routing operates in a disruption-heavy environment.
- Most AI tools still lack deterministic safety compliance.
- EVN sits at the intersection of operations, trust, and compliance.

**Visual Cue:** Triad graphic: accidents, weather, bridge constraints.

**Proof Metric (show on slide):** 7.7M accidents, multi-TB NOAA, 600K+ bridges.

**Speaker Notes (paste into PPT Notes):**
The opportunity is driven by operational complexity and data availability. Freight teams contend with high disruption frequency from traffic and weather, while still needing strict physical feasibility checks. HyperLogistics addresses this gap directly by combining broad data coverage with explainable and safety-enforced recommendations in one system.

---

## Slide 6
**Title:** Progress/Traction

**On-Slide Text:**
- Live HyperLogistics deployment is available.
- Domain adaptation improved accuracy from 40% to 95%.
- ReMindRAG reproducibility audit passed 17/17 tests.

**Visual Cue:** Traction bars: 40% -> 95%, 17/17 reproducibility.

**Proof Metric (show on slide):** +55 percentage-point accuracy lift.

**Speaker Notes (paste into PPT Notes):**
We have evidence across product, model quality, and engineering reliability. The app is deployed, adaptation performance improves from 40 to 95 percent, and reproducibility is validated by a full 17 out of 17 test pass with all identified issues fixed in audit reporting. This shows momentum and execution quality, not just conceptual progress.

---

## Slide 7
**Title:** Business Model

**On-Slide Text:**
- Core: enterprise SaaS licensing for explainable rerouting.
- Services: implementation and workflow integration support.
- Expansion: premium analytics modules for ops intelligence.

**Visual Cue:** Tier ladder: Core Platform, Integration, Analytics Add-ons.

**Proof Metric (show on slide):** 9-tool analytics agent foundation.

**Speaker Notes (paste into PPT Notes):**
Our model is a land-and-expand enterprise structure. The base subscription delivers safety-compliant rerouting, implementation services accelerate adoption, and analytics add-ons increase account value over time. Week 6 already demonstrates a 9-tool analytics foundation that supports this expansion path.

---

## Slide 8
**Title:** Team

**On-Slide Text:**
- Joel Vinas: internal data engineering and preprocessing.
- Daniel Evans: external data integration and app delivery.
- Tony Nguyen: automation architecture and reproducibility.

**Visual Cue:** Role-to-component matrix with key file ownership.

**Proof Metric (show on slide):** 33.3% contribution per member.

**Speaker Notes (paste into PPT Notes):**
Team execution is organized by clear ownership across the stack. Internal pipelines, external data integration, app delivery, automation, and reproducibility are each assigned and delivered. That balanced distribution, reflected at 33.3 percent each in reporting, enabled consistent progress across multiple weeks.

---

## Slide 9
**Title:** Next Steps/Vision

**On-Slide Text:**
- Close remaining eval hardening and benchmark lock.
- Scale adaptation data and model coverage.
- Add OAuth/RBAC and expand enterprise rollout.

**Visual Cue:** 3-phase roadmap: Hardening -> Controls -> Rollout.

**Proof Metric (show on slide):** 15 eval queries plus 5 metamorphic pairs.

**Speaker Notes (paste into PPT Notes):**
Next steps are execution-focused. We finalize remaining evaluation hardening, then expand training and test coverage beyond current adaptation scope. In parallel, we add enterprise controls like OAuth and role-based access so the system can move from pilot readiness to broader deployment in operational environments.

---

## Claim-to-Source Verification Map

| Slide | Number/Claim Used | Source File |
|---|---|---|
| 1 | 7.7M accidents | `CS5542_SmartSC_Optimization_System/README.md` |
| 2 | 100% safety compliance via hard veto | `CS5542_SmartSC_Optimization_System/README.md` |
| 3 | 4-8 hour forecasting windows | `CS5542_SmartSC_Optimization_System/README.md` |
| 4 | 100 query-response pairs for manager scenarios | `COMP_SCI_5542/Week_8/reports/GROUP_REPORT.md` |
| 5 | 7.7M accidents, multi-terabyte NOAA, 600K+ bridges | `CS5542_SmartSC_Optimization_System/README.md` |
| 6 | 40% to 95% (+55 pts), 17/17 pass, 14/14 fixed | `COMP_SCI_5542/Week_8/reports/GROUP_REPORT.md`, `ReMindRAG_Week7/REPRO_AUDIT.md` |
| 7 | 9-tool agent capability | `COMP_SCI_5542/Week_6/README.md` |
| 8 | 33.3% each + role ownership | `CS5542_SmartSC_Optimization_System/CONTRIBUTIONS.md`, `COMP_SCI_5542/Week_8/reports/GROUP_REPORT.md` |
| 9 | 15 queries + 5 metamorphic pairs | `COMP_SCI_5542/Week_8/reports/GROUP_REPORT.md` |

---

## Final Timing Check
- 9 slides x ~30-40 seconds each = ~4.5 to 6.0 minutes.
