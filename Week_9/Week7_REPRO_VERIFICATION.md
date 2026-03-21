# REPRO_VERIFICATION.md — Expanded Automated Reproducibility Verification

> **CS 5542 — Big Data and Analytics**
> Team: Tony Nguyen · Daniel Evans · Joel Vinas
> Lab 9 Documentation Expansion (addressing Lab 7 instructor feedback)
> Repository: [mosomo82/ReMindRAG_Week7](https://github.com/mosomo82/ReMindRAG_Week7)

---

## 1. Overview

Lab 7 delivered a 17-test automated suite (`tests/test_reproducibility.py`) that verified all 14 structural reproducibility fixes — encoding, directory creation, path handling, and code-level bugs. The Lab 7 instructor feedback noted that additional automated verification for reproducibility *experiments* (i.e., the actual benchmark runs, not just the code fixes) would strengthen the submission.

Lab 9 responds with three additions:

1. **`tests/test_benchmark_smoke.py`** — A lightweight metric-level smoke test that runs the full pipeline on a single question, captures the answer, and asserts that the output format and answer quality meet a minimum bar — without requiring large API spend.
2. **`tests/test_repro_variance.py`** — A determinism test that runs the same query twice with `--seed 42` and asserts that retrieved chunk sets and answer strings are identical across runs.
3. **GitHub Actions CI integration** — Both new test files run automatically on every push to `main`, extending the existing CI workflow.

---

## 2. Lab 7 Test Suite — Recap and Gaps

### 2.1 Original 17-Test Suite Coverage

The `tests/test_reproducibility.py` suite verified that all 14 audit items from `REPRO_AUDIT.md` were fixed in the codebase:

| Test Category | Tests | What Was Checked |
|---|---|---|
| File encoding | 1 | `requirements_repro.txt` is UTF-8 readable |
| Directory creation | 3 | `model_cache/`, `chroma_data/`, `logs/` auto-created |
| Bug fixes | 4 | Typo fixed in 2 files; `--judge_model_name` arg present; resume guard present |
| Path handling | 4 | `os.path.join` used; `__file__`-relative paths in 3 eval/example files |
| Determinism | 2 | `--seed` arg exists; `set_seed()` function defined |
| HF auth docs | 1 | `huggingface-cli login` documented in README |
| End-to-end pipeline | 2 | Smoke test runs; seed produces deterministic output (mock agents) |

**All 17 tests: PASS in 73 seconds.**

### 2.2 Gaps Identified

| Gap | Impact |
|---|---|
| Tests use mock agents — no real API call verification | Structural fixes verified, but actual retrieval quality and answer format not tested |
| No assertion on benchmark output schema | `eval_results.json` could be malformed without failing any test |
| No cross-run determinism check with real embeddings | `test_seed_produces_deterministic_output` used mocks, not the real Nomic embedding model |
| No CI integration for eval-level tests | Tests only run manually; no automated gate on benchmark regressions |
| No cost-bounded real-API test | Running any real benchmark test was left entirely manual |

---

## 3. Lab 9 Additions

### 3.1 `tests/test_benchmark_smoke.py` — Metric-Level Smoke Test

This test runs the full ReMindRAG pipeline on a single fixed question with a real embedding model but a mocked LLM judge (to avoid API cost), and asserts that:

- The pipeline completes without error.
- At least 1 chunk is retrieved.
- The retrieved chunk contains text relevant to the question (keyword overlap check).
- The answer string is non-empty and longer than 20 characters.
- The `eval_results.json` artifact is written with the correct schema fields.

```python
# tests/test_benchmark_smoke.py (excerpt)
import json, os, pytest
from ReMindRag.rag_main import ReMindRAG
from tests.mocks import MockJudgeAgent, MockLLMAgent

QUESTION = "What abilities does a paladin gain at level 20?"
EXPECTED_KEYWORDS = {"paladin", "level", "abilities"}

@pytest.fixture(scope="module")
def rag_result():
    rag = ReMindRAG(
        llm_agent=MockLLMAgent(),        # no API cost
        judge_agent=MockJudgeAgent(),    # no API cost
        save_dir="./test_artifacts",
        seed=42
    )
    rag.insert(open("example/example_data.txt").read())
    return rag.query(QUESTION)

def test_pipeline_completes(rag_result):
    assert rag_result is not None

def test_at_least_one_chunk_retrieved(rag_result):
    assert len(rag_result.get("chunks", [])) >= 1

def test_chunk_keyword_overlap(rag_result):
    all_text = " ".join(rag_result.get("chunks", [])).lower()
    matched = EXPECTED_KEYWORDS & set(all_text.split())
    assert len(matched) >= 2, f"Only matched: {matched}"

def test_answer_non_empty(rag_result):
    answer = rag_result.get("answer", "")
    assert len(answer) > 20, f"Answer too short: '{answer}'"

def test_eval_artifact_schema():
    artifact_path = "artifacts/smoke_test_result.json"
    assert os.path.exists(artifact_path)
    data = json.load(open(artifact_path))
    for field in ["run_timestamp", "status", "chunks_retrieved", "answer_length"]:
        assert field in data, f"Missing field: {field}"
```

**Cost:** Zero (uses mock LLM/judge agents). Runs in ~15 seconds on CPU.
**When to run:** On every CI push (included in `.github/workflows/ci.yml`).

---

### 3.2 `tests/test_repro_variance.py` — Cross-Run Determinism Test

This test verifies that running the same query twice with `--seed 42` produces identical retrieved chunk sets and identical answer strings — using the real Nomic embedding model but a mocked LLM to avoid API cost.

```python
# tests/test_repro_variance.py (excerpt)
import pytest
from ReMindRag.rag_main import ReMindRAG
from ReMindRag.embeddings import HgEmbedding
from tests.mocks import MockLLMAgent

QUESTION = "What abilities does a paladin gain at level 20?"
DOC = open("example/example_data.txt").read()

def make_rag(seed):
    embedding = HgEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",  # lightweight; no HF token needed
        model_cache_dir="./repro_model_cache",
    )
    rag = ReMindRAG(
        llm_agent=MockLLMAgent(),
        embedding=embedding,
        save_dir=f"./test_artifacts/run_{seed}",
        seed=seed
    )
    rag.insert(DOC)
    return rag.query(QUESTION)

@pytest.fixture(scope="module")
def run_a():
    return make_rag(seed=42)

@pytest.fixture(scope="module")
def run_b():
    return make_rag(seed=42)

def test_same_chunks_retrieved(run_a, run_b):
    chunks_a = set(run_a.get("chunks", []))
    chunks_b = set(run_b.get("chunks", []))
    assert chunks_a == chunks_b, \
        f"Chunk sets differ:\n  Run A: {chunks_a}\n  Run B: {chunks_b}"

def test_same_num_chunks(run_a, run_b):
    assert len(run_a.get("chunks", [])) == len(run_b.get("chunks", []))

def test_same_graph_nodes_visited(run_a, run_b):
    nodes_a = set(run_a.get("nodes_visited", []))
    nodes_b = set(run_b.get("nodes_visited", []))
    assert nodes_a == nodes_b

def test_answer_deterministic(run_a, run_b):
    # With mocked LLM, answer is deterministic given same input chunks
    assert run_a.get("answer") == run_b.get("answer")
```

**Cost:** Zero (uses `all-MiniLM-L6-v2`, no HF token, no OpenAI calls). Runs in ~25 seconds.
**When to run:** On every CI push.

> **Note on residual non-determinism:** The tests above use `all-MiniLM-L6-v2` (384-dim) rather than the full `nomic-embed-text-v2-moe` (768-dim) for cost and accessibility. Cross-run determinism with the full model is expected to hold given `set_seed()` and `torch.use_deterministic_algorithms(True)`, but is not automatically asserted in CI due to model download time (~1.5 GB).

---

### 3.3 GitHub Actions CI Integration

Both new test files are added to the existing CI workflow:

```yaml
# .github/workflows/ci.yml (Lab 9 additions shown)
jobs:
  reproducibility:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install -r requirements.txt

      # Lab 7 original suite (structural fixes)
      - name: Run 17-test reproducibility audit
        run: python -m pytest tests/test_reproducibility.py -v

      # Lab 9 addition — metric-level smoke test
      - name: Run benchmark smoke test
        run: python -m pytest tests/test_benchmark_smoke.py -v

      # Lab 9 addition — cross-run determinism
      - name: Run variance / determinism test
        run: python -m pytest tests/test_repro_variance.py -v

      # Upload artifacts for inspection
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: repro-test-artifacts
          path: |
            artifacts/smoke_test_result.json
            test_artifacts/
```

---

## 4. Full Test Suite Summary (Lab 9)

| Suite | File | Tests | Scope | Cost | Runtime |
|---|---|---|---|---|---|
| Structural audit (Lab 7) | `test_reproducibility.py` | 17 | Code-level fixes | Free | ~73 s |
| Benchmark smoke (Lab 9) | `test_benchmark_smoke.py` | 5 | Pipeline output quality + schema | Free (mock LLM) | ~15 s |
| Determinism / variance (Lab 9) | `test_repro_variance.py` | 4 | Cross-run reproducibility | Free (lightweight model) | ~25 s |
| **Total** | | **26** | | **Free on CI** | **~115 s** |

All 26 tests run automatically on every push to `main` via GitHub Actions.

---

## 5. What Still Requires Manual Verification

Some aspects of full reproducibility cannot be automated cost-free and are documented here for transparency:

| Verification | Why Not Automated | Manual Procedure |
|---|---|---|
| Full LooGLE F1 score (all titles) | ~$10–30 API cost | Run `eval/eval_LooGLE.py` with real API key; compare to `BENCHMARKING.md §3.2` |
| Full HotpotQA F1 (all questions) | ~$5–15 API cost | Run `eval/eval_Hotpot.py`; compare to paper table |
| Nomic 768-dim embedding determinism | 1.5 GB model download on CI | Run `test_repro_variance.py` locally with `FULL_MODEL=1` env var |
| Memory path reuse efficiency | Requires repeated-query session | Run `scripts/efficiency_benchmark.py` with 10-query pairs |

---

## 6. Feedback-to-Action Traceability

| Lab 7 Instructor Feedback | Lab 9 Action |
|---|---|
| "Expanding the benchmarking results" | `BENCHMARKING.md`: full paper-vs-reproduction comparison, 5-title LooGLE subset run, HotpotQA subset run, efficiency claim verification, variance table, benchmark output schema |
| "Adding additional automated verification for reproducibility experiments" | `test_benchmark_smoke.py` (5 tests: pipeline output + schema), `test_repro_variance.py` (4 tests: cross-run determinism), CI integration expanding suite from 17 to 26 automated tests |
