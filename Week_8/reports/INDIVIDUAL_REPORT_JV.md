# Lab 8: Domain Adaptation Individual Report

## Student: Joel Vinas

### 1. Description of Individual Contributions
For Component 2, Advanced Prompt Adaptation:

* Built `prompt_adaptation.py` to implement three specific, advanced prompting strategies to handle user queries regarding rerouting.
* Designed a **Self-Consistent Chain-of-Thought (SC-CoT)** strategy to generate three independent reasoning chains (Disruption Assessment → Route Analysis → Constraint Check → Decision) and aggregate them through majority voting, maximizing accuracy by returning the consensus with the best-justified reasoning trace.
* Implemented a **ReAct (Reasoning + Acting)** paradigm to explicitly show the step-by-step logic, simulating "Thoughts" and "Actions" to assess constraints before determining a final APPROVE/VETO outcome.
* Developed a **Structured System Prompt (Few-Shot)** framework that intelligently injects three relevant examples from the instruction dataset to handle disruptions based on similarities.

### 2. Contribution Percentage
**33.3%**

### 3. Repository Evidence

* **Commit 1:** [[e355f6f](https://github.com/mosomo82/COMP_SCI_5542/commit/e355f6f1bfe22715568ec15870d166b4d6106737)] — Initial adaptation strategy investigation
* **Commit 2:** `d233c01` — Core implementation of `prompt_adaptation.py` strategies
* **Commit 3:** `1a45962` — Update individual report contributions
* **Commit 4:** `a36b14c` — Finalize group report contributions

### 4. AI Tools Used
The Advanced Prompt Adaptation component features and the implementation in `prompt_adaptation.py` were developed with the assistance of AI agents, **Google Antigravity AI Agent** and **Claude Code**. This work was done collaboratively during a team consensus call, where the agent was instructed to review the overall system requirements and output the code logic as specified by the group, successfully demonstrating AI-driven problem-solving and task delegation in a real-time setting.
