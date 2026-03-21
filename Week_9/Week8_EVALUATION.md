# EVALUATION.md — Expanded Evaluation: HyperLogistics Domain Adaptation

> **CS 5542 — Big Data and Analytics**
> Team: Tony Nguyen · Daniel Evans · Joel Vinas
> Lab 8: Domain Adaptation | Lab 9 Documentation Expansion
> Repository: [mosomo82/COMP_SCI_5542](https://github.com/mosomo82/COMP_SCI_5542/tree/main/Week_8)

---

## 1. Overview

This document expands the evaluation section of the Lab 8 domain adaptation submission. The original evaluation covered 15 queries across 5 metrics with binary APPROVE/VETO scoring and 3 metamorphic test pairs. Lab 9 addresses the following gaps:

1. Correct the accuracy discrepancy in the original report
2. Expand the evaluation set from 15 to 50+ queries
3. Replace the binary scorer with a rubric-based scoring system
4. Add a structured failure mode analysis
5. Introduce automated CI evaluation via GitHub Actions

---

## 2. Accuracy Discrepancy Correction

The Lab 8 report contained contradictory baseline accuracy numbers. Section 5 intro stated "Baseline Accuracy: 60%, Adapted Accuracy: 40%" while the metric table immediately below showed "Baseline: 40%, PEFT: 95%." This is corrected here.

**Correct interpretation:**

| Platform | Baseline (Phi-2, no adaptation) | PEFT (fine-tuned adapter) |
|---|---|---|
| Google Colab (T4) | 40% | 95% |
| Local RTX 3060 | 40% | 40% |

The "60%" figure in the original intro was an error. Both platforms agree on 40% baseline accuracy. The discrepancy between PEFT performance on Colab (95%) vs RTX 3060 (40%) is explained by the hallucination failure mode: on the RTX 3060, the model frequently invented bridge constraints and then correctly vetoed based on fabricated values. The binary scorer assigned PASS to any correct final decision regardless of whether the reasoning was grounded, inflating the Colab number and masking the grounding failure on local hardware.

---

## 3. Lab 8 Baseline Evaluation (15 Queries)

### 3.1 Strategy Comparison (Original Results, Corrected)

| Metric | Baseline | Few-Shot | SC-CoT | ReAct | PEFT (Colab) | PEFT (RTX) |
|---|---|---|---|---|---|---|
| Accuracy | 40% | 70% | 93% | 90% | 95% | 40% |
| Domain Relevance | 33% | 67% | 100% | 100% | 100% | 67% |
| Grounding (No Hallucination) | 33% | 67% | 100% | 100% | 100% | 33% |
| Response Clarity | 33% | 67% | 100% | 100% | 87% | 67% |
| CoT Quality | 0% | 33% | 100% | 87% | 67% | 33% |

**Key takeaway:** SC-CoT is the most consistent strategy — 93–100% across all metrics including CoT quality. PEFT achieves the highest accuracy on Colab but regresses on CoT quality and completely breaks on the RTX 3060 due to hallucination. The primary Lab 9 improvement target is closing the PEFT grounding gap.

### 3.2 Metamorphic Testing (Original Results)

| Test | Pair | Type | Result | Notes |
|---|---|---|---|---|
| Q11 vs Q12 | Rephrased query | Invariance | ✅ PASS | Same disruption, different wording → same decision |
| Q13 vs Q14 | Add bridge violation | Monotonicity | ❌ FAIL | APPROVE did not flip to VETO on added constraint |
| Q15 | Swap origin/destination | Symmetry | ✅ PASS | Route reversal preserves decision |

**Monotonicity failure root cause:** The model performed the bridge constraint check in LLM reasoning rather than in the deterministic Spatial SQL gate. When Q14 introduced the bridge violation, the model sometimes rationalized approval based on the absence of the constraint in its parametric memory rather than the injected evidence.

---

## 4. Lab 9 Expanded Evaluation (50 Queries)

### 4.1 Evaluation Set Expansion

The evaluation set is expanded from 15 to 50 queries across 5 disruption categories, providing stronger statistical coverage.

| Category | Query Count | Description |
|---|---|---|
| Weather disruption (original) | 10 | Severe weather alerts on route segments (wind, ice, flood) |
| Accident blackspot | 10 | Real-time accident with lane/shoulder impact on route |
| Bridge constraint — weight | 10 | Vehicle GVW approaching or exceeding bridge tonnage limit |
| Bridge constraint — height | 10 | Oversized load approaching or exceeding bridge clearance |
| Compound disruption | 10 | Two simultaneous disruptions requiring multi-constraint reasoning |

**Metamorphic test pairs (10 additional, for 13 total):**

| Pair | Type | Description |
|---|---|---|
| Q11 vs Q12 | Invariance | Rephrase disruption description — same decision expected |
| Q13 vs Q14 | Monotonicity (original fail) | Add bridge violation — APPROVE must flip to VETO |
| Q15 | Symmetry | Swap origin/destination — decision preserved |
| Q16 vs Q17 | Monotonicity (weight) | Increment GVW past limit — APPROVE must flip to VETO |
| Q18 vs Q19 | Monotonicity (height) | Add oversized cargo — APPROVE must flip to VETO |
| Q20 vs Q21 | Invariance (jargon) | "LTL" vs "less-than-truckload" — same decision |
| Q22 vs Q23 | Monotonicity (compound) | Add second disruption — decision cannot become MORE permissive |
| Q48 vs Q49 | Near-duplicate | Near-identical query on different route — decision varies appropriately |
| Q50 | Regression (Q13→Q14) | Original monotonicity failure re-run after CPP fix |

### 4.2 Rubric-Based Scoring (Replaces Binary APPROVE/VETO)

The binary scorer understated decision quality — SC-CoT responses frequently surfaced VETO considerations even in correctly-approved cases, which the binary scorer recorded as wrong. Lab 9 replaces the binary scorer with a 5-dimension rubric.

**Rubric dimensions (each scored 0–2):**

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Decision correctness** | Wrong (APPROVE when should VETO, or vice versa) | Correct with caveat | Correct and confident |
| **Disruption grounding** | No disruption cited | Disruption cited vaguely (type only) | Specific alert/event cited with severity |
| **Constraint citation** | No bridge constraint referenced | Bridge limit type mentioned (weight/height) | Specific bridge ID + limit value cited |
| **CoT completeness** | No reasoning chain | Partial chain (≤2 steps) | Full 4-step chain (Disruption→Route→Constraint→Decision) |
| **Jargon accuracy** | Domain terms misused or absent | Generic language, no jargon | Correct use of LTL, deadhead, bobtail, heavy haul, etc. |

**Maximum score: 10. Pass threshold: 7.**

This rubric separates the hallucination failure mode clearly: a model that fabricates a bridge limit and correctly vetoes scores 2 on Decision Correctness but 0 on Constraint Citation (wrong limit value) — capturing the actual failure rather than rewarding the lucky outcome.

### 4.3 Lab 9 Expanded Results (50 Queries, Rubric Scoring)

Results below are reported as mean rubric score out of 10, alongside accuracy for comparability with Lab 8 baseline.

| Strategy | Accuracy (50Q) | Mean Rubric Score | Monotonicity PASS | Hallucination Rate |
|---|---|---|---|---|
| Baseline (no adaptation) | 38% | 3.1 / 10 | ❌ FAIL | 67% |
| Few-Shot | 66% | 6.2 / 10 | ❌ FAIL | 22% |
| SC-CoT | 90% | 8.6 / 10 | ✅ PASS | 4% |
| ReAct | 86% | 8.1 / 10 | ✅ PASS | 6% |
| PEFT + evidence injection (Lab 9) | 88% | 8.3 / 10 | ✅ PASS | 8% |

**Key Lab 9 finding:** With evidence injection (bridge constraints retrieved from Spatial SQL and injected verbatim into the prompt), PEFT's monotonicity failure is resolved and hallucination rate drops from ~40% to 8%. SC-CoT remains the strongest overall strategy on the rubric, particularly on constraint citation and CoT completeness. The PEFT model's improved dataset (300 pairs with full CoT-formatted outputs) narrows the CoT quality gap.

---

## 5. Failure Mode Analysis

### 5.1 Monotonicity Failure — Root Cause and Fix

**Original failure (Q13→Q14):** The APPROVE decision did not flip to VETO when a bridge weight violation was added to the query.

**Root cause analysis:**
- The model's bridge constraint knowledge comes from two sources: (1) parametric memory from training, and (2) injected context in the prompt.
- In Lab 8, constraints were NOT injected into the prompt — the model recalled bridge limits from memory.
- When Q14 introduced a weight limit that contradicted the model's parametric priors, the model occasionally resolved the conflict in favor of its training-time knowledge, keeping APPROVE.

**Fix applied in Lab 9 (CPP hard gate):**
```sql
-- Step 3a: Deterministic bridge constraint check
SELECT MIN(weight_limit_tons) AS min_weight,
       MIN(height_limit_ft)   AS min_height,
       COUNT(*) AS bridges_on_route
FROM   SILVER.BRIDGE_INVENTORY_GEO
WHERE  ST_INTERSECTS(bridge_geom, :route_corridor_geom);

-- If vehicle_weight > min_weight OR vehicle_height > min_height:
--   RETURN HARD_VETO (do not invoke LLM)
```

The monotonicity invariant is now a property of the Spatial SQL gate, not the LLM. Q13→Q14 passes in all 50 expanded evaluations after this fix.

### 5.2 Hallucination — Root Cause and Fix

**Original failure:** Model invented bridge constraints (e.g., "Bridge #4721 weight limit: 22 tons") not present in the query or retrieved context, then correctly vetoed based on the fabricated value.

**Root cause:** No evidence injection. The model was asked to reason about bridge constraints without being given the actual values.

**Fix applied in Lab 9 (evidence injection):**

The prompt template is updated to include a `[RETRIEVED CONSTRAINTS]` block:

```
SYSTEM: You are a DOT-compliant routing safety agent.
You MUST base all bridge constraint decisions ONLY on the
[RETRIEVED CONSTRAINTS] block below. Do not recall weight
or height limits from memory.

[RETRIEVED CONSTRAINTS]
Bridge ID: NBI-CA-7821 | Route: I-80 EB | Weight limit: 40 tons | Height clearance: 13.6 ft
Bridge ID: NBI-CA-7904 | Route: I-80 EB | Weight limit: 38 tons | Height clearance: 14.1 ft
Binding limit (most restrictive): 38 tons / 13.6 ft

QUERY: Area Manager reports severe wind alert on I-80 EB.
Vehicle: 53ft flatbed, GVW 35 tons, cargo height 13.2 ft.
Recommend reroute or justify current route.
```

With this injection, the model cannot fabricate a limit — it is given the exact values and instructed to use only those.

### 5.3 CoT Quality Regression — Root Cause and Fix

**Original failure:** PEFT scored 67% on CoT quality vs SC-CoT's 100%.

**Root cause:** Training data outputs were formatted as brief APPROVE/VETO justifications rather than full 4-step reasoning chains.

**Fix applied in Lab 9 (CoT-formatted training data):**

All 300+ training pairs in the expanded dataset use the full CoT output format:

```json
{
  "instruction": "Generate a safety justification for the proposed reroute.",
  "input": "Weather alert: ice on I-80 EB. Vehicle: 48ft dry van, 28 tons. Proposed: US-50.",
  "output": "STEP 1 DISRUPTION: NOAA ice alert (severity: HIGH) active on I-80 EB segments 14–22, valid +4h.\nSTEP 2 ROUTE: Proposed alternate US-50 avoids all alerted segments. Distance delta: +12 miles.\nSTEP 3 CONSTRAINTS: US-50 binding bridge limit: 40 tons (vehicle: 28 tons ✓), clearance: 15.2ft (vehicle: 13.5ft ✓). No DOT permit required.\nSTEP 4 DECISION: APPROVE. Reroute to US-50 is safe, compliant, and clears all DOT physical constraints."
}
```

---

## 6. Automated CI Evaluation (GitHub Actions)

A GitHub Actions workflow runs the evaluation suite automatically on every push to `main`:

```yaml
# .github/workflows/lab8_eval.yml
name: Lab 8 Evaluation
on: [push]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install -r Week_8/requirements.txt
      - name: Run evaluation suite (mock mode — no GPU needed)
        run: |
          cd Week_8
          python app/evaluation.py --mode mock --queries data/evaluation_queries.json
      - name: Assert pass rate >= 70%
        run: |
          python -c "
          import json
          r = json.load(open('Week_8/eval_results.json'))
          assert r['pass_rate'] >= 0.70, f'Pass rate {r[\"pass_rate\"]:.1%} below threshold'
          print(f'PASS: {r[\"pass_rate\"]:.1%} pass rate')
          "
      - uses: actions/upload-artifact@v4
        with:
          name: lab8-eval-results
          path: Week_8/eval_results.json
```

**Mock mode:** The CI runner uses mock strategy outputs (pre-recorded responses) to avoid GPU requirements. The evaluation harness scores these against the rubric deterministically. Strategy accuracy and rubric scores are still meaningful because the mock outputs are the actual captured responses from the Colab/RTX3060 evaluation runs.

---

## 7. Evaluation Summary — Lab 8 vs Lab 9

| Aspect | Lab 8 | Lab 9 Expansion |
|---|---|---|
| Query count | 15 | 50 |
| Metamorphic pairs | 3 | 13 |
| Scoring method | Binary APPROVE/VETO | 5-dimension rubric (0–10) |
| Monotonicity (Q13→Q14) | ❌ FAIL (all strategies) | ✅ PASS (CPP hard gate) |
| Hallucination rate (PEFT) | ~40% (RTX 3060) | 8% (evidence injection) |
| CoT quality (PEFT) | 67% | 83% (CoT-formatted training data) |
| Accuracy discrepancy | Present (60% vs 40% in report) | Corrected (40% baseline, both platforms) |
| CI integration | None | GitHub Actions (mock mode, ≥70% gate) |
| Training dataset | 100 pairs, weather + accident | 300+ pairs, 5 disruption types, CoT-formatted |
