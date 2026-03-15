# Lab 8: Domain Adaptation Individual Report

## Student: Daniel Evans

### 1. Description of Individual Contributions
For Component 3, Integration & Demo Dashboard:

* Contributed to `demo_dashboard.py`, a Streamlit application to visually compare baseline and adapted AI responses for logistics rerouting decisions. *(Built collaboratively during team session.)*
* Contributed to `evaluation.py`, the automated evaluation framework executing 15 gold-standard queries (including metamorphic tests) across 5 scoring metrics. *(Built collaboratively during team session.)*

Individual contributions on branch `daniel-lab08-a`:

* Refactored `evaluation.py` to lazy-load `torch`, `transformers`, and `peft` inside `load_real_model()`, enabling `--mode mock` to run without GPU dependencies.
* Added `BitsAndBytesConfig` 4-bit NF4 quantization to the real inference path to match the QLoRA training setup.
* Fixed a `torch` variable scope error in `run_real_inference` that caused failures during the real GPU run.
* Added a `--verbose` flag to `evaluation.py` exposing per-query prompts, raw model output, and scoring detail.
* Ran the adapted model evaluation on a local RTX 3060 (phi-2, 4-bit), producing the adapted accuracy result of **40.0%**.
* Finalized the group report: filled in evaluation results, updated metamorphic test outcomes, and completed the contribution table.

### 2. Contribution Percentage

**33.3%**

### 3. Repository Evidence

* **Commit 1:** `186e3b1` — Lazy-import torch/transformers/peft in evaluation.py
* **Commit 2:** `57b079e` — Use 4-bit quantization in evaluation.py --mode real
* **Commit 3:** `e3479f3` — Fix torch scope error in run_real_inference
* **Commit 4:** `5d5785d` — Add --verbose flag to evaluation.py

### 4. AI Tools Used
The Integration & Demo Dashboard features, including `demo_dashboard.py` and `evaluation.py`, were developed with the assistance of AI agents, **Google Antigravity AI Agent** operating on a local Windows environment and **Claude Code**. This work was done collaboratively during a team consensus call, where the agents were instructed to review the overall system requirements and output the code logic as specified by the group, successfully demonstrating AI-driven problem-solving and task delegation in a real-time setting.