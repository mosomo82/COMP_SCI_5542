"""
Notebook Patcher — CS5542 Lab 4
================================
Run this script to programmatically fix issues in CS5542_Lab4_Notebook.ipynb.

Usage:
    python src/patch_notebook.py
"""

import json
import re
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "CS5542_Lab4_Notebook.ipynb"
BACKUP = NOTEBOOK.with_suffix(".ipynb.bak")


def load_notebook(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb: dict, path: Path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        f.write("\n")


def get_source(cell: dict) -> str:
    """Join source lines into a single string."""
    return "".join(cell.get("source", []))


def set_source(cell: dict, text: str):
    """Set cell source from a string (split into per-line list)."""
    # Preserve the ipynb convention of per-line storage
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    cell["source"] = result


def patch_gold_set_rubric(nb: dict) -> int:
    """Add `rubric` key to mini_gold entries that are missing it."""
    rubrics = {
        "Q1": {"must_have_keywords": ["selective patching", "SRS", "representation space", "dynamic reassembly"]},
        "Q2": {"must_have_keywords": ["memory replay", "edge-weight", "traversal path"]},
        "Q3": {"must_have_keywords": ["consensus", "planning problem", "primal", "dual"]},
        "Q4": {"must_have_keywords": ["79.38", "64.95", "Multi-Hop", "Deepseek"]},
        "Q5": {"must_have_keywords": []},
        "Q6": {"must_have_keywords": ["KG construction", "traversal", "enhance", "penalize"]},
    }

    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = get_source(cell)
        # Look for cells defining mini_gold
        if "mini_gold" not in src:
            continue
        if '"rubric"' in src or "'rubric'" in src:
            continue  # already patched

        # For each query, insert rubric after answer_criteria or gold_evidence_ids
        for qid, rubric in rubrics.items():
            rubric_json = json.dumps(rubric)
            # Pattern: insert 'rubric' key after 'answer_criteria' block or 'gold_evidence_ids'
            # We use a simple approach: add rubric line after gold_evidence_ids line
            pattern = rf'("gold_evidence_ids"\s*:\s*\[.*?\])'
            # More robust: find the query_id and add rubric nearby
            if f'"{qid}"' in src and '"rubric"' not in src:
                # Find the gold_evidence_ids for this specific query block
                # Simple approach: add rubric as a new key in the dict
                pass

        # Simpler approach: just add a comment + rubric dict at the top
        if "rubric" not in src and "answer_criteria" in src:
            # Add rubric keys by string replacement
            for qid, rubric in rubrics.items():
                rubric_str = json.dumps(rubric)
                # Find the answer_criteria closing bracket for this query
                # Pattern: after citation_format line, before the closing },
                pattern = f'"citation_format": '
                if pattern in src:
                    # Add rubric before citation_format
                    src = src.replace(
                        f'"citation_format": ',
                        f'"rubric": {rubric_str},\n        "citation_format": ',
                        1  # only first occurrence per pass
                    )
            set_source(cell, src)
            count += 1

    return count


def patch_streamlit_cell(nb: dict) -> int:
    """Fix the Streamlit skeleton cell: missing quote + faithfulness logic."""
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = get_source(cell)
        if "streamlit_code" not in src or "MINI_GOLD" not in src:
            continue

        patched = src
        # Fix missing opening quote on Q1 question
        patched = patched.replace(
            '{"question": What is the Selective',
            '{"question": "What is the Selective',
        )
        # Fix faithfulness_pass always-Yes bug
        patched = patched.replace(
            '"faithfulness_pass": "Yes" if answer != MISSING_EVIDENCE_MSG else "Yes"',
            '"faithfulness_pass": "Yes" if answer != MISSING_EVIDENCE_MSG else "No"',
        )
        # Fix missing_evidence_behavior hardcoded Pass
        patched = patched.replace(
            '"missing_evidence_behavior": "Pass"  # update with your rule if needed',
            '"missing_evidence_behavior": "Pass" if evidence else "Fail"',
        )
        # Add Q6
        if '"Q5"' in patched and '"Q6"' not in patched:
            patched = patched.replace(
                '    "Q5": {"question": "What reinforcement learning reward function does SRSNet use to train the Selective Patching scorer?", "gold_evidence_ids": [\'N/A\']},\n}',
                '    "Q5": {"question": "What reinforcement learning reward function does SRSNet use to train the Selective Patching scorer?", "gold_evidence_ids": [\'N/A\']},\n'
                '    "Q6": {"question": "What are the key stages shown in the ReMindRAG overall workflow diagram?", "gold_evidence_ids": [\'img::figure3.png\']},\n}',
            )

        if patched != src:
            set_source(cell, patched)
            count += 1

    return count


def patch_retry_logic(nb: dict) -> int:
    """Add retry/backoff to generate_llm_answer if it exists."""
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = get_source(cell)
        if "def generate_llm_answer" not in src:
            continue
        if "retry" in src or "backoff" in src:
            continue  # already patched

        # Add a simple retry wrapper
        retry_code = '''
import time as _time

_ORIGINAL_generate_llm_answer = generate_llm_answer

def generate_llm_answer(question, context, max_retries=3):
    """Wrapper with exponential backoff for rate-limit errors."""
    for attempt in range(max_retries):
        result = _ORIGINAL_generate_llm_answer(question, context)
        if "429" not in result and "Rate" not in result:
            return result
        wait = 2 ** attempt * 15  # 15s, 30s, 60s
        print(f"Rate-limited. Retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
        _time.sleep(wait)
    return result
'''
        # Append retry logic to the same cell
        set_source(cell, src + "\n" + retry_code)
        count += 1

    return count


def patch_failure_analysis(nb: dict) -> int:
    """Add failure analysis content to Section 10 markdown cell."""
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = get_source(cell)
        if "Failure analysis" not in src and "failure analysis" not in src:
            continue
        if "## Failure Case 1" in src:
            continue  # already has content

        failure_content = """# 10) Failure Analysis (Required)

## Failure Case 1 — Retrieval Failure

**What happened:**  
Q1 asks about the SRS module in SRSNet (from `doc1_TimerSeries.pdf`). The retrieval returned chunk IDs like `doc1_TimerSeries.pdf::p2` but the gold evidence ID was `doc1_TimerSeries.pdf` (document-level). The evaluation counted 0 gold matches because string comparison was exact — the `::p2` page suffix caused a mismatch.

**Root cause:**  
The evidence canonicalization function `_canon_evidence_id()` only stripped `.txt` extensions but did not handle the `::pN` page suffixes produced by per-page PDF chunking.

**Proposed fix:**  
Update `_canon_evidence_id()` to strip `::pN` suffixes using `re.sub(r'::p\\d+$', '', x)` so that page-level chunk IDs match document-level gold evidence IDs.

---

## Failure Case 2 — Grounding / Hallucination Failure

**What happened:**  
Q5 ("What reinforcement learning reward function does SRSNet use?") is designed as a missing-evidence query — SRSNet does **not** use RL. The Gemini API correctly returned the missing-evidence message. However, when using the local TinyLlama model, it hallucinated: *"SRSNet uses the mean square error (MSE) as the reward function during training."*

**Root cause:**  
TinyLlama (1.1B parameters) lacks sufficient instruction-following capability to refuse answering when the context does not contain relevant information. It generated a plausible-sounding but factually incorrect response.

**Proposed fix:**  
1. Add a confidence threshold: if the maximum retrieval score is below 0.05, bypass the LLM entirely and return the missing-evidence message.
2. If using a local model, add explicit instructions in the prompt: "If the context does not contain information about the question, respond ONLY with: Not enough evidence in the retrieved context."
"""
        set_source(cell, failure_content)
        count += 1

    return count


def patch_project_name(nb: dict) -> int:
    """Replace placeholder project name."""
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = get_source(cell)
        if "YOUR_PROJECT_NAME" in src:
            patched = src.replace("YOUR_PROJECT_NAME", "CS5542_Lab4_RAG")
            set_source(cell, patched)
            count += 1
    return count


def main():
    if not NOTEBOOK.exists():
        print(f"ERROR: Notebook not found at {NOTEBOOK}")
        sys.exit(1)

    print(f"Loading {NOTEBOOK.name} ...")
    nb = load_notebook(NOTEBOOK)

    # Backup
    save_notebook(nb, BACKUP)
    print(f"Backup saved to {BACKUP.name}")

    patches = {
        "Streamlit cell fixes": patch_streamlit_cell(nb),
        "Rubric keys added": patch_gold_set_rubric(nb),
        "Retry logic added": patch_retry_logic(nb),
        "Failure analysis content": patch_failure_analysis(nb),
        "Project name placeholder": patch_project_name(nb),
    }

    total = sum(patches.values())
    print(f"\nApplied patches:")
    for name, n in patches.items():
        status = f"  ✅ {name}: {n} cell(s)" if n > 0 else f"  ⏭️  {name}: no matching cells"
        print(status)

    if total > 0:
        save_notebook(nb, NOTEBOOK)
        print(f"\n✅ Saved patched notebook ({total} cells modified).")
    else:
        print("\n⚠️  No cells were modified. The notebook may already be patched or cell structure differs.")

    print("\nDone. Review the changes in your notebook editor.")


if __name__ == "__main__":
    main()
