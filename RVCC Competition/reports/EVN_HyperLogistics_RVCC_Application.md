# EVN HyperLogistics — RVCC Venture Competition Application

**Venture:** EVN HyperLogistics
**Product:** HyperLogistics — Safety-compliant, explainable AI rerouting for middle-mile logistics
**Date:** March 15, 2026
**Team:** Tony Nguyen · Daniel Evans · Joel Vinas

---

## 1. Problem & Value Proposition
*What problem are you solving and why does your solution improve the situation?*

Middle-mile logistics managers face a critical daily problem: when disruptions hit — weather events, traffic accidents, road closures — they must make immediate rerouting decisions that are both operationally sound and physically compliant with DOT regulations. Existing tools can flag risk, but they stop short of producing a safe, explainable reroute recommendation.

We call this the **prediction-action gap**.

The scale of this problem is real. Our system is grounded in 7.7 million US accident records, multi-terabyte NOAA weather data, and 600,000+ DOT bridge constraint records. These are not model assumptions — they define the actual operating environment freight teams work in every day.

**HyperLogistics closes this gap** by combining three components into a single decision flow:

| Component | Role |
|---|---|
| **ReMindRAG** | Retrieves grounded evidence for reroute reasoning from historical disruption knowledge |
| **SRSNet** | Forecasts short-horizon (4–8 hour) disruption propagation aligned to transit windows |
| **Symbolic Safety Veto** | Hard-blocks any route that violates DOT bridge clearance or load limits |

The result: a dispatch recommendation that is not just intelligent, but **compliance-ready and explainable**. Every output cites its reasoning and has passed a deterministic safety check — something no general-purpose AI tool currently provides for this workflow.

---

## 2. Customer & Market Opportunity
*Who is the target customer and why is this opportunity meaningful?*

**Primary users:** Logistics Network Managers and Area Managers at trucking carriers and third-party logistics providers (3PLs).

These operators own middle-mile execution outcomes. They are accountable for on-time performance, cost efficiency, and safety compliance under constant disruption pressure. Their core job during a disruption is to triage, reroute, and confirm DOT compliance — often in minutes.

**Why this opportunity is meaningful:**

- Disruptions are high-frequency and consequential. A single bad reroute can violate bridge load limits, incur DOT liability, or delay a full shipment lane.
- Most routing AI tools today optimize for efficiency but **lack deterministic safety enforcement**. They are not built for compliance-critical decisions.
- HyperLogistics sits at the exact intersection where this is unaddressed: **operations + explainability + hard compliance enforcement**.

To validate product-market fit, we built 100 domain-adapted instruction scenarios drawn directly from real manager decision patterns — disruption queries, bridge constraint vetoes, and reroute approval flows. These are not synthetic edge cases. They reflect the actual language and decisions logistics managers face daily.

**Market context:**

| Data Signal | Scale |
|---|---|
| US accident records (2016–2023) | 7.7 million records |
| NOAA weather data (disruption triggers) | Multi-terabyte |
| National Bridge Inventory (DOT) | 600,000+ bridges |

---

## 3. Progress & Learning
*What steps have you taken so far and what have you learned?*

We have moved from architecture to working product across multiple development phases.

### What We Built

| Milestone | Evidence |
|---|---|
| **Live deployment** | HyperLogistics Streamlit app is publicly accessible at https://cs5542hyperlogistics.streamlit.app/ |
| **Snowflake-native pipeline** | Medallion architecture (Bronze/Silver/Gold) with Snowpipe, External Tables, and Spatial SQL |
| **Domain adaptation (Week 8)** | Accuracy improved from **40% to 95%** using QLoRA fine-tuning (Phi-2, LoRA rank=16) and few-shot prompting |
| **Agent execution (Week 6)** | Gemini 2.5 Flash agent with 9 analytics tools — **4/5 scenarios passed (80%), 6/7 tool selections correct (86%)** |
| **Reproducibility (Week 7)** | ReMindRAG audit: **17/17 automated tests passed**, 14/14 identified issues resolved |

### What We Learned

**The biggest barrier to enterprise trust is not model accuracy — it is explainability and compliance enforcement.**

Early iterations produced accurate recommendations that managers still could not use, because the system could not cite why a route was chosen or confirm it was physically valid. Once we added the symbolic bridge-constraint veto and the ReMindRAG reasoning trace, confidence in the outputs changed significantly.

The +55 percentage-point accuracy improvement from baseline to adapted model confirmed that domain-specific instruction tuning on logistics jargon and DOT constraint reasoning is not optional — it is the core product differentiator.

---

## 4. Team & Commitment
*Who is on the founding team and why are you working on this venture?*

| Member | Role | Stack Ownership |
|---|---|---|
| **Tony Nguyen** | ML / Full-Stack Engineer | Pipeline automation, model adaptation (QLoRA), S3/Snowflake orchestration, reproducibility |
| **Daniel Evans** | Data / Backend Engineer | External data engineering (NOAA weather), Streamlit app delivery, evaluation support |
| **Joel Vinas** | Data / ML Engineer | Internal data engineering, ingestion scripts, silver-layer preprocessing |

All three are MS Data Science & Analytics students with equal equity and full-stack ownership. We are not advisors with slide decks — we are the engineers who built every component in this system. Responsibilities map directly to code and architecture ownership across the full stack.

**Why we are working on this:**

We identified the prediction-action gap firsthand while building logistics AI systems for coursework. The constraint-compliance problem is not academic — it is a live operational risk for every freight company running middle-mile operations. We kept building because the technical path was clear and the operating need was real.

We are committed to this venture because the hardest parts are already built: the data pipeline, the retrieval architecture, the safety veto, the adapted model, and the live application. What remains is go-to-market execution.

---

## 5. RVCC Fit
*How will RVCC help you move the venture forward?*

RVCC fills the precise gaps that technical founders face after building a working product but before achieving first commercial traction.

### What We Are Asking For

**Mentorship — go-to-market for logistics operations buyers**
We understand the technology. We need guidance on how to sell into operations leadership at carriers and 3PLs: who champions this internally, what procurement looks like at mid-size carriers, and how to structure a paid pilot agreement.

**Network — freight operator and ecosystem introductions**
A warm introduction to one logistics operator, regional carrier, or Snowflake ecosystem partner would compress our sales cycle significantly. The product is live and demonstrable today.

**Funding — runway for hardening and first pilot**
Specific near-term needs:
- Complete evaluation hardening and lock benchmark reporting
- Scale adaptation dataset beyond 100 examples for broader coverage
- Add enterprise access controls (OAuth / RBAC) required for production deployment
- Support a first paid pilot engagement

### Why RVCC Resources Convert Directly to Traction

| Current Asset | RVCC Unlock |
|---|---|
| Live deployed application | Pilot-ready with one enterprise introduction |
| 95% adapted accuracy | Publishable benchmark with mentorship on framing |
| 17/17 reproducibility tests | Engineering credibility for enterprise procurement evaluation |
| 9-tool analytics foundation | Expansion path for enterprise analytics add-on revenue |

RVCC at this stage would convert proven technical capability into a go-to-market motion — which is exactly the gap the competition is designed to close.

---

*EVN HyperLogistics — RVCC 2026 Submission*
