# Lab 8: Domain Adaptation — Implementation Plan

Upgrade HyperLogistics from a standard RAG chatbot to a **Domain-Specialized AI Assistant** for explainable, constraint-compliant rerouting.

---

## Proposed Changes

All new files go into `week8/`. The existing dashboard at [src/app/dashboard.py](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/src/app/dashboard.py) is **not modified** — it serves as the baseline for comparison.

---

### Component 1: PEFT Fine-Tuning (Tony Nguyen)

> [!IMPORTANT]
> This requires a Google Colab GPU runtime (T4 free tier is sufficient for Phi-2 or Mistral-7B with QLoRA 4-bit).

#### [NEW] [peft_finetuning.py](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/peft_finetuning.py)

Python script (runnable in Colab or locally with a GPU) that:

1. **Loads base model** — `microsoft/phi-2` (2.7B params, fits in free Colab T4)
2. **Loads instruction dataset** — reads [week8/instruction_dataset.json](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/instruction_dataset.json)
3. **Applies QLoRA config** — 4-bit quantization via `bitsandbytes`, LoRA rank=16, alpha=32, targeting `q_proj`/`v_proj`
4. **Trains** — 3 epochs, batch size 4, learning rate 2e-4, using HuggingFace `SFTTrainer`
5. **Saves adapted model** — to `week8/adapted_model/`
6. **Runs inference** on 5 sample queries, printing baseline vs. adapted output side-by-side

Key dependencies: `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, `accelerate`

---

### Component 2: Advanced Prompt Adaptation (Joel Vinas)

#### [NEW] [prompt_adaptation.py](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/prompt_adaptation.py)

Standalone Python module implementing three advanced prompting strategies. Each strategy takes a user query + retrieved evidence context and returns a formatted prompt string.

**Strategy 1: Self-Consistent Chain-of-Thought (SC-CoT)**
- Generates N=3 independent CoT reasoning chains for the same query
- Each chain follows: `Disruption Assessment → Route Analysis → Constraint Check → Decision`
- Aggregates the 3 chains via majority vote on the final APPROVE/VETO decision
- Returns the consensus answer with the best-justified reasoning chain

**Strategy 2: ReAct (Reasoning + Acting)**
- Interleaves Thought/Action/Observation steps:
  - `Thought`: "The user asks about rerouting due to weather. I need to check bridge constraints."
  - `Action`: Retrieve bridge data for the alternate route
  - `Observation`: "Bridge #4432 has a 13ft clearance limit."
  - `Thought`: "The vehicle is 14ft. This violates the limit."
  - `Action`: Issue VETO
- Produces a structured trace showing the agent's step-by-step reasoning

**Strategy 3: Structured System Prompt (Few-Shot)**
- Injects 3 expert-validated examples from [instruction_dataset.json](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/instruction_dataset.json) into the system prompt
- Examples are selected based on disruption-type similarity (weather vs accident)

Each function signature:
```python
def build_sc_cot_prompt(query: str, evidence: str, examples: list) -> str
def build_react_prompt(query: str, evidence: str) -> str
def build_fewshot_prompt(query: str, evidence: str, examples: list) -> str
```

---

### Component 3: Integration & Demo Dashboard (Daniel Evans)

#### [NEW] [demo_dashboard.py](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/demo_dashboard.py)

A **new, standalone Streamlit app** in `week8/` that demonstrates baseline vs. adapted responses. Does **not** modify the existing [src/app/dashboard.py](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/src/app/dashboard.py).

**Layout:**
- **Sidebar:** Query input + strategy selector dropdown (Baseline / Few-Shot / SC-CoT / ReAct / PEFT)
- **Main area — two columns:**
  - Left: "Baseline Response" (uses the original simple prompt from [dashboard.py](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/src/app/dashboard.py))
  - Right: "Adapted Response" (uses the selected strategy from `prompt_adaptation.py`)
- **Below columns:** "Reasoning Trace" expander showing the CoT/ReAct steps
- **Bottom:** Metrics comparison table (auto-scored)

This dashboard imports `prompt_adaptation.py` and either:
- Calls Snowflake Cortex (if credentials are available), or
- Falls back to a local mock/simulation mode using [instruction_dataset.json](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/instruction_dataset.json) for offline demo

---

### Component 4: Evaluation Framework (All Members)

#### [NEW] [evaluation.py](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/evaluation.py)

Evaluation script that runs 15 test queries through baseline and all adapted strategies, scoring each on 5 metrics:

| Metric | Description | Scoring |
|--------|-------------|---------|
| **Accuracy** | Correctness of APPROVE/VETO decision | Binary (0/1) — compared against gold label |
| **Domain Relevance** | Uses logistics terminology correctly | 0–3 scale (keyword matching) |
| **Hallucination Rate** | References non-existent bridges/routes | Binary (0 = hallucinated, 1 = grounded) |
| **Response Clarity** | Structured, cites data sources explicitly | 0–3 scale (rubric-based) |
| **CoT Quality** | Reasoning chain is logically coherent | 0–3 scale (step completeness) |

**Additional Advanced Metrics:**

**Chain-of-Thought Evaluation:**
- Checks that each reasoning step logically follows the previous one
- Verifies the final decision is consistent with the intermediate steps
- Scores: `step_count`, `logical_consistency`, `constraint_coverage`

**Metamorphic Testing:**
- Tests invariance: "Reroute Chicago→KC due to snow" must produce the same VETO/APPROVE as "Snow on I-70, reroute Chicago to Kansas City"
- Tests monotonicity: Adding a bridge violation to an APPROVED route must flip to VETO
- Tests symmetry: Swapping origin/destination on a symmetric route should not change the safety decision
- Generates 5 metamorphic test pairs from the base 15 queries

#### [NEW] [evaluation_queries.json](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/evaluation_queries.json)

15 gold-standard queries with expected outputs for automated scoring. Includes 5 metamorphic pairs.

---

## Work Division Summary

| Member | Component | Files | Effort |
|--------|-----------|-------|--------|
| **Tony Nguyen** | PEFT Fine-Tuning + Dataset | `peft_finetuning.py`, [instruction_dataset.json](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/instruction_dataset.json) | 33% |
| **Joel Vinas** | Advanced Prompt Adaptation | `prompt_adaptation.py` | 33% |
| **Daniel Evans** | Integration, Demo & Evaluation | `demo_dashboard.py`, `evaluation.py`, `evaluation_queries.json` | 33% |

---

## Verification Plan

### Automated Tests

1. **Dataset validation** — verify [instruction_dataset.json](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/instruction_dataset.json) has 100 entries, each with `instruction`, `input`, `output` keys:
   ```bash
   py -c "import json; d=json.load(open('week8/instruction_dataset.json')); assert len(d)==100; assert all(set(x.keys())=={'instruction','input','output'} for x in d); print('PASS')"
   ```

2. **Prompt builder tests** — run `prompt_adaptation.py` with a sample query and verify each strategy returns a non-empty string containing required sections:
   ```bash
   py -c "from week8.prompt_adaptation import build_sc_cot_prompt, build_react_prompt, build_fewshot_prompt; p=build_react_prompt('test','test'); assert 'Thought' in p and 'Action' in p; print('PASS')"
   ```

3. **Evaluation dry-run** — run `evaluation.py` in offline/mock mode against `evaluation_queries.json`:
   ```bash
   py week8/evaluation.py --mode mock
   ```

### Manual Verification

1. **Streamlit demo** — run `streamlit run week8/demo_dashboard.py`, select each strategy, and visually confirm baseline vs adapted responses appear side-by-side
2. **Colab PEFT** — upload `peft_finetuning.py` and [instruction_dataset.json](file:///c:/Users/mtuan/OneDrive/Documents/GitHub/CS5542_SmartSC_Optimization_System/week8/instruction_dataset.json) to Google Colab, run all cells, and confirm the model trains without errors and prints comparison output
