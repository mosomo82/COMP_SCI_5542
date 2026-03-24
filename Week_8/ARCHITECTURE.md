# ARCHITECTURE.md — HyperLogistics Domain Adaptation System

> **CS 5542 — Big Data and Analytics**
> Team: Tony Nguyen · Daniel Evans · Joel Vinas
> Lab 8: Domain Adaptation | Lab 9 Documentation Expansion
> Repository: [mosomo82/COMP_SCI_5542](https://github.com/mosomo82/COMP_SCI_5542/tree/main/Week_8)

---

## 1. System Overview

HyperLogistics is a Snowflake-native supply chain resilience system that bridges the prediction-action gap for middle-mile logistics. The system uses a domain-adapted language model (Phi-2 + QLoRA) as a Safety Validation Agent to generate explainable, constraint-compliant rerouting justifications when real-time disruptions occur. No LLM reasoning can override a physical constraint veto — hard DOT bridge limits are enforced deterministically by the Consensus Planning Protocol before any response is returned.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  EXTERNAL INPUTS                                                             │
│  US Accidents (Kaggle) · NOAA Weather Alerts · NBI Bridge Inventory (CSV)   │
│           │                       │                        │                 │
│           ▼                       ▼                        ▼                 │
│     Snowpipe (stream)       Snowpipe (stream)       Snowflake Stage          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — DATA PERCEPTION                                                   │
│                                                                              │
│  BRONZE tables (raw)  ──►  SILVER tables (cleaned, typed, indexed)           │
│  · SILVER.WEATHER_ALERTS          · SILVER.BRIDGE_INVENTORY_GEO             │
│  · SILVER.ACCIDENT_BLACKSPOTS     · SILVER.ROUTE_SEGMENTS                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — INTELLIGENCE & FORECASTING                                        │
│                                                                              │
│  ┌──────────────────────────┐    ┌──────────────────────────────────────┐   │
│  │  ReMindRAG               │    │  SRSNet                              │   │
│  │  LLM-guided KG traversal │    │  Adaptive risk propagation forecast  │   │
│  │  Retrieves relevant       │    │  4–8h ahead, route-level risk score  │   │
│  │  disruption history and   │    │  drives route candidate generation   │   │
│  │  constraint precedents    │    │                                      │   │
│  └──────────┬───────────────┘    └──────────────────┬───────────────────┘   │
│             │                                        │                       │
│             └──────────────────┬─────────────────────┘                      │
│                                │                                             │
│                    Route candidates + retrieved context                      │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — VALIDATION & SAFETY  (Consensus Planning Protocol)               │
│                                                                              │
│  Step 3a — DETERMINISTIC CONSTRAINT CHECK (Spatial SQL)                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  SELECT bridge_id, height_limit_ft, weight_limit_tons                │   │
│  │  FROM SILVER.BRIDGE_INVENTORY_GEO                                    │   │
│  │  WHERE ST_INTERSECTS(bridge_geom, route_corridor_geom)               │   │
│  │  ORDER BY weight_limit_tons ASC                                      │   │
│  │                                                                      │   │
│  │  IF vehicle_height > MIN(height_limit_ft)                            │   │
│  │     OR vehicle_weight > MIN(weight_limit_tons)  →  HARD VETO        │   │
│  │  (No LLM can override this gate)                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                 │                                            │
│                    ┌────────────┴────────────┐                               │
│                    │ HARD VETO               │ PASS                          │
│                    ▼                         ▼                               │
│            Return VETO with       Step 3b — LLM SAFETY VALIDATION           │
│            constraint detail      ┌──────────────────────────────────────┐  │
│                                   │  Phi-2 + PEFT adapter (QLoRA)        │  │
│                                   │  Input: disruption + route +          │  │
│                                   │         bridge facts (injected)       │  │
│                                   │  Output: structured CoT justification │  │
│                                   │  (APPROVE or VETO + reasoning chain)  │  │
│                                   └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — APPLICATION                                                       │
│                                                                              │
│  demo_dashboard.py  (Streamlit)                                              │
│  · Dispatcher enters a natural-language reroute query                        │
│  · Side-by-side comparison: baseline Phi-2 vs adapted strategies             │
│  · Displays APPROVE/VETO decision, CoT reasoning trace, metric scores        │
│  · Live deployment: https://cs5542lab8.streamlit.app/                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Descriptions

### 2.1 Layer 1 — Data Perception

| Table | Source | Content |
|---|---|---|
| `SILVER.WEATHER_ALERTS` | NOAA (Snowpipe) | Real-time alerts with severity, affected route segments, timestamps |
| `SILVER.ACCIDENT_BLACKSPOTS` | US Accidents dataset (Snowpipe) | Geocoded accident locations with severity and lane-impact flags |
| `SILVER.BRIDGE_INVENTORY_GEO` | NBI bridge database (Stage) | Bridge geom, height clearance (ft), weight limit (tons), permit flags |
| `SILVER.ROUTE_SEGMENTS` | Internal | Candidate route geometries for spatial intersection queries |

### 2.2 Layer 2 — Intelligence & Forecasting

**ReMindRAG** (from Lab 7) acts as the retrieval backbone. When a dispatcher submits a query, ReMindRAG traverses the knowledge graph to surface relevant precedents — prior disruptions of the same type on the same corridor, previously approved or vetoed routing decisions, and constraint records for the bridges on the candidate route. This context is injected into the Phi-2 prompt alongside the live disruption data.

**SRSNet** generates a 4–8 hour risk propagation forecast for each route candidate. The forecast scores are used to rank candidate routes before the CPP validation step, ensuring the model evaluates lower-risk routes first.

### 2.3 Layer 3 — Consensus Planning Protocol (CPP)

The CPP is the core safety innovation of HyperLogistics. It enforces a strict two-step validation:

**Step 3a — Deterministic constraint check (Spatial SQL):**
All bridges on the proposed route are queried from `SILVER.BRIDGE_INVENTORY_GEO` using a spatial intersection. If the vehicle's height or weight exceeds *any* bridge limit on the route, a HARD VETO is issued immediately. This gate is deterministic and cannot be overridden by the LLM. The specific bridge ID, limit, and exceedance value are included in the veto response.

**Step 3b — LLM safety validation (Phi-2 + PEFT):**
If the route clears Step 3a, the domain-adapted model generates a structured CoT justification. The prompt explicitly injects the bridge constraints retrieved in Step 3a (height, weight limits) so the model never needs to recall physical limits from parametric memory. This injection is the primary fix for the hallucination failure mode observed in Lab 8 evaluation.

The reasoning chain follows the pattern:
```
Disruption → affected_segments → candidate_route → bridge_constraints (injected) → decision
```

### 2.4 Domain Adaptation Components

| Component | File | Description |
|---|---|---|
| Dataset generation | `data/generate_dataset.py` | Synthesizes 100+ query-response pairs from Snowflake SILVER tables |
| Training dataset | `data/instruction_dataset.json` | 100 instruction/input/output pairs covering weather + accident disruptions |
| QLoRA fine-tuning | `notebook/peft_finetuning.py` | Phi-2, rank=16, alpha=32, 4-bit NF4, 3 epochs on T4/Colab |
| PEFT adapter | `adapted_model/` | Lightweight adapter weights loaded on top of base Phi-2 at inference |
| Prompt strategies | `adaption_method/prompt_adaptation.py` | Few-Shot, SC-CoT, ReAct builders |
| Evaluation | `app/evaluation.py` | 15-query gold set + 5-metric scoring + metamorphic test suite |
| Demo dashboard | `app/demo_dashboard.py` | Streamlit side-by-side comparison interface |

### 2.5 Prompt Adaptation Strategies

Three strategies are implemented in `prompt_adaptation.py`:

| Strategy | Mechanism | Best For |
|---|---|---|
| Few-Shot | Injects 3 expert-validated examples selected by disruption-type similarity | Consistent output format; low variance |
| SC-CoT | Generates 3 independent CoT chains, aggregates by majority vote | Highest CoT quality (100%); strong constraint coverage |
| ReAct | Interleaves Thought/Action/Observation steps | Grounded step-by-step traces; good for auditability |

PEFT fine-tuning achieves the highest accuracy (95%) but trails on CoT quality (67%) — the training data's output format prioritized final decisions over intermediate reasoning steps.

---

## 3. Identified Failure Modes and Mitigations

### 3.1 Monotonicity Failure (Q13→Q14)

**Failure:** Adding a bridge weight violation to an otherwise approvable query did not reliably flip the decision from APPROVE to VETO.

**Root cause:** The LLM was performing the constraint check rather than Spatial SQL. When the constraint was introduced in Q14, the model sometimes failed to recognize it as disqualifying.

**Mitigation:** The CPP Step 3a hard gate (Spatial SQL) is designed to prevent this entirely. Any route where the vehicle weight exceeds `MIN(weight_limit_tons)` across all intersecting bridges is vetoed before the LLM is invoked. The monotonicity invariant becomes a property of the Spatial SQL query, not of the LLM.

### 3.2 Hallucinated Bridge Constraints

**Failure:** On the RTX 3060 evaluation, the PEFT model fabricated weight/height limits not present in the evidence, then correctly vetoed based on the invented values.

**Root cause:** Bridge constraints were not injected into the prompt context. The model relied on parametric memory to recall physical limits.

**Mitigation:** All bridge constraints on the candidate route are retrieved via Step 3a and injected verbatim into the LLM prompt before Step 3b runs. The prompt template explicitly marks injected constraints as `[RETRIEVED FROM BRIDGE_INVENTORY_GEO]` so the model is never asked to recall physical facts from memory.

### 3.3 CoT Quality Regression After Fine-Tuning

**Failure:** PEFT achieved 95% decision accuracy but only 67% CoT quality, vs SC-CoT which achieved 100% on both.

**Root cause:** The `instruction_dataset.json` output fields contained the final APPROVE/VETO decisions with brief justifications, but did not include full structured reasoning chains. The model learned the correct answer format but not the full Disruption → Route → Constraint → Decision chain.

**Mitigation (Lab 9):** The dataset generation script (`generate_dataset.py`) is updated to produce outputs that always include the full 4-step CoT chain. Retraining with 300+ pairs (expanded dataset) using CoT-formatted outputs is planned.

---

## 4. File Structure

```
Week_8/
├── adaption_method/
│   └── prompt_adaptation.py      # Few-Shot, SC-CoT, ReAct prompt builders
├── app/
│   ├── demo_dashboard.py         # Streamlit comparison dashboard
│   └── evaluation.py             # 5-metric scoring + metamorphic testing
├── data/
│   ├── instruction_dataset.json  # 100 training pairs (instruction/input/output)
│   ├── evaluation_queries.json   # 15 gold-standard test queries
│   └── generate_dataset.py       # Dataset generation from Snowflake SILVER tables
├── adapted_model/                # PEFT adapter weights (QLoRA, Phi-2)
├── notebook/
│   └── peft_finetuning.py        # QLoRA training script (run on Colab T4)
├── ARCHITECTURE.md               # This file
└── EVALUATION.md                 # Expanded evaluation methodology
```

---

## 5. Environment and Dependencies

| Component | Specification |
|---|---|
| Base model | `microsoft/phi-2` (2.7B parameters) |
| Quantization | 4-bit NF4 via `bitsandbytes`, double quantization enabled |
| LoRA config | Rank=16, Alpha=32, target layers: `q_proj`, `v_proj` |
| Training | 3 epochs, batch=4, lr=2e-4, `paged_adamw_8bit` optimizer |
| Training platform | Google Colab (T4 GPU, free tier) |
| Inference (local) | RTX 3060 12GB, 4-bit NF4 (matching training config) |
| Deployment | Streamlit Community Cloud (`cs5542lab8.streamlit.app`) |
| Data warehouse | Snowflake (`CS5542_WEEK8`, SILVER schema) |
| Key libraries | `transformers`, `peft`, `bitsandbytes`, `trl`, `streamlit`, `snowflake-connector-python` |
