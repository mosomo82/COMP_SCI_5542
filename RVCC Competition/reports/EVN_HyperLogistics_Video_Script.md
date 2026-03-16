# EVN HyperLogistics — RVCC Video Pitch Script
**Total Target Runtime: 4:45 – 5:00**
**Presenter: Tony Nguyen (lead), Joel Vinas, Daniel Evans**
**Slide deck: EVN_Deterministic_Freight_AI.pptx (13 slides)**

> **Recording notes:**
> - Speak at a natural, measured pace — no rushing.
> - Advance the slide at every `[SLIDE X]` marker.
> - Timing targets per segment are in parentheses — practice to those, not shorter.
> - Metrics and proper nouns should be pronounced clearly and not sped through.

---

## RVCC Criteria & Slide-to-Time Map

| Criterion | Weight | Slides | Time |
|---|---|---|---|
| Opening | — | 1 | ~10 sec |
| Problem & Value Proposition | 25% | 2–5 | ~75 sec |
| Customer & Market Opportunity | 20% | 6–7 | ~50 sec |
| Progress, Action & Learning | 25% | 8–10 | ~75 sec |
| Team & Commitment | 20% | 11–12 | ~50 sec |
| Coachability & RVCC Fit | 10% | 11 + 13 | ~30 sec |
| **Total** | | **13 slides** | **~290 sec** |

---

## OPENING

**[SLIDE 1 — Title]**
*(~10 seconds)*

Hi, I'm Tony, and with my teammates Daniel and Joel, we're building **EVN HyperLogistics** — a safety-compliant, explainable AI system for middle-mile logistics.

---

## SECTION 1: Problem & Value Proposition
*(Target: ~75 seconds — Slides 2–5)*

---

**[SLIDE 2 — Problem]**
*(~30 seconds)*

Every day, freight dispatchers face a version of this scenario: a disruption hits — a highway accident, an ice storm, a road closure — and they need a reroute *right now*.

Today's tools can tell them *something is wrong*. They cannot tell them *what to do next* in a way that is safe, explainable, and compliant with DOT bridge regulations.

We call this the **prediction-action gap.** It is the difference between detecting risk and acting on it — and right now, logistics teams are stuck in that gap.

The scale is real: the US freight network operates under **7.7 million recorded accident conditions**. This is not a niche edge case. It is a daily operational reality.

---

**[SLIDE 3 — Solution]**
*(~25 seconds)*

HyperLogistics closes this gap with three components working in sequence.

First, **ReMindRAG** — it retrieves grounded evidence from historical disruption knowledge to support the reasoning behind each reroute suggestion.

Second, **SRSNet** — it forecasts how disruptions will spread over the next 4 to 8 hours, which is exactly the operational window for middle-mile shipments.

Third, and most important: a **hard symbolic safety veto**. Every recommended route is automatically checked against the National Bridge Inventory — load limits, vertical clearances — before any decision reaches a dispatcher. If a bridge on the route cannot handle the vehicle, the route is blocked. Not flagged. Blocked.

That is **100% physical constraint enforcement** — something no general-purpose AI tool currently provides.

---

**[SLIDE 4 — Product / Platform Architecture]**
*(~20 seconds)*

Under the hood, this runs on a **Snowflake-native medallion pipeline** — Bronze, Silver, and Gold layers — where data is ingested, transformed, and served close to where inference happens.

The front end is a live Streamlit application that gives dispatchers side-by-side route comparisons with full reasoning traces. They can see *why* a route was chosen, not just *what* was chosen.

This is a deployable platform, not a notebook. It is live today.

---

**[SLIDE 5 — Technology Stack]**
*(~15 seconds)*

Three data sources power the safety layer: **7.7 million accident records**, real-time **NOAA weather data**, and **600,000-plus National Bridge Inventory records** — all live in Snowflake and checked at the moment of route generation.

The safety veto is not a post-processing step. It runs inline, before any recommendation surfaces. No physically non-compliant route ever reaches a dispatcher.

---

## SECTION 2: Customer & Market Opportunity
*(Target: ~50 seconds — Slides 6–7)*

---

**[SLIDE 6 — Target Customer]**
*(~28 seconds)*

Our primary users are **Logistics Network Managers and Area Managers** at trucking carriers and third-party logistics providers.

These are the people who own middle-mile outcomes. When a disruption hits, they have minutes — not hours — to approve a reroute that is operationally viable *and* legally defensible. A bad decision can violate bridge weight limits, delay an entire lane, or trigger a DOT compliance event.

Their current tools were not built for this. They're using general routing tools that optimize for time and cost but have no concept of physical constraint compliance.

To make sure our system actually speaks their language, we built **100 domain-adapted instruction scenarios** directly from real manager decision patterns — disruption queries, constraint veto reasoning, and reroute approvals. That is product-market fit grounded in operational reality, not user assumption.

---

**[SLIDE 7 — Market Opportunity]**
*(~22 seconds)*

The opportunity is defined by the operating environment itself.

Freight routing in the US is disruption-heavy. We operate across **7.7 million accident records**, **multi-terabyte NOAA weather data**, and **600,000-plus bridge constraint records** — all live in our pipeline.

Most routing AI optimizes for efficiency. None enforce hard physical compliance with DOT constraints. HyperLogistics sits at the intersection of operations, trust, and compliance — and that intersection is currently unaddressed.

---

## SECTION 3: Progress, Action & Learning
*(Target: ~75 seconds — Slides 8–10)*

---

**[SLIDE 8 — Progress Overview]**
*(~20 seconds)*

Let me be concrete about what we have built and what we have learned.

The application is live. Our **Gemini 2.5 Flash agent**, running **9 specialized analytics tools**, passed **4 out of 5 evaluation scenarios** with **86% tool selection accuracy** across fleet, safety, route, fuel, and maintenance domains. The full stack — ingestion to safety veto to dispatcher UI — is operational on Snowflake today.

---

**[SLIDE 9 — Reproducibility]**
*(~25 seconds)*

On reproducibility — the engineering discipline enterprise customers actually require — we ran a structured audit of the ReMindRAG component. We identified **14 issues** across three severity levels: 6 critical, 4 high, 4 medium. We fixed all 14 and verified with an automated test suite. **17 out of 17 tests pass.**

The lesson was foundational: **reliability is a product feature, not a bonus.** Passing a structured audit is what converts a demo into a procurement conversation.

---

**[SLIDE 10 — Model Lift]**
*(~30 seconds)*

Here is the clearest measure of learning through action.

The baseline model — no specialization — achieved **40% accuracy** on logistics constraint reasoning tasks. After QLoRA fine-tuning on Phi-2 with our 100-pair instruction dataset, combined with few-shot prompting, accuracy reached **95%**. That is a **plus-55 percentage-point lift**.

But here is what we actually learned: **accuracy alone does not drive adoption. Explainability and compliance enforcement do.**

Early versions produced correct answers that managers still could not act on — because there was no reasoning trace and no guarantee the route was physically valid. Adding the safety veto and the ReMindRAG rationale panel changed that immediately. That insight now shapes every design decision we make.

---

## SECTION 4: Team & Commitment
*(Target: ~50 seconds — Slides 11–12)*

---

**[SLIDE 11 — Business Model]**
*(~18 seconds)*

Our go-to-market is enterprise SaaS: an annual license for safety-compliant rerouting, implementation services for onboarding, and premium analytics add-ons for account expansion. Our **9-tool analytics agent foundation is already operational** and ready to serve as that expansion layer.

---

**[SLIDE 12 — Team]**
*(~32 seconds)*

There are three of us, and we each own a distinct layer of this stack.

**Joel Vinas** owns internal data engineering — ingestion, preprocessing, and the silver-layer pipelines.

**Daniel Evans** owns external data integration — NOAA weather, app delivery, and evaluation infrastructure.

I'm **Tony Nguyen**, and I own pipeline automation, model adaptation, Snowflake-S3 orchestration, and reproducibility controls.

We are MS Data Science and Analytics students, and we built every component in this system ourselves. We identified the prediction-action gap while building real logistics AI, we kept building because the problem is real, and the hardest technical work is already done.

---

## SECTION 5: Coachability & RVCC Fit
*(Target: ~30 seconds — Slides 11 + 13)*

*(Note: Slide 11 Business Model was already introduced above. This section advances to Slide 13 for the closing.)*

---

**[SLIDE 13 — Next Steps / Vision]**
*(~30 seconds)*

We are applying to the RVCC because we want to learn how to transition this technology from pilot readiness to broader operational deployment.

Our immediate next steps are closing our evaluation hardening and implementing OAuth and Role-Based Access Controls for the enterprise. We have our core enterprise SaaS licensing model defined, and our 9-tool analytics agent foundation is already operational to drive account expansion.

We are looking for RVCC's mentorship to refine our go-to-market motion and scale this into a real-world enterprise solution.

**EVN HyperLogistics. The reroute that is already safe before it reaches dispatch.**

Thank you.

---

## Post-Recording Checklist

- [ ] Runtime is between 4:30 and 5:00 — not one second over 5:00
- [ ] Slide advances match every `[SLIDE X]` cue marker exactly (13 total)
- [ ] Metrics spoken clearly: "7.7 million", "95 percent", "17 out of 17", "plus 55 points"
- [ ] Opening names all three team members by name
- [ ] Closing tagline is the last thing judges hear before cut
- [ ] Video is publicly accessible — no login required — before submitting the link
- [ ] Accompanying slide file uploaded alongside the video link
