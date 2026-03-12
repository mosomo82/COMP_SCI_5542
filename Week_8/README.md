# Week 8 Plan: Domain Adaptation

**Live Demo:** [https://cs5542lab8.streamlit.app/](https://cs5542lab8.streamlit.app/)

This document acts as our roadmap to upgrade HyperLogistics from a standard RAG system to a Domain-Specialized AI Assistant focused on **Explainable Rerouting & Safety Justification**.

## 🚀 Steps to Complete Lab 8

### **Step 1. Dataset Creation (Domain Task)**
A generic LLM doesn't understand "deadhead", "LTL", or DOT clearance limits intuitively. We need to create a dataset to teach it. 
*   **Action:** Generate ~50-100 high-quality query-response pairs. The pairs should simulate Area Managers asking for reroutes due to weather/accidents, and the assistant giving a strictly validated, constraint-aware response.
*   **Implementation:** We can create a Python script inside `src/` to synthesize this data by querying our Snowflake `SILVER.BRIDGE_INVENTORY_GEO` and `SILVER.WEATHER_ALERTS`.

### **Step 2. Model Adaptation Method**
We must choose an adaptation path supported by our architecture (Snowflake Cortex).
*   **Option 1: Prompt Adaptation (Recommended)**
    Implement highly engineered few-shot prompting where expert-level, validated routing examples are injected dynamically into the system prompt based on the disruption type.
*   **Option 2: Instruction Tuning / PEFT**
    If time permits, we can export the synthetic dataset from Step 1, fine-tune a model via LoRA offline, and deploy it to Snowflake, or use Snowflake Cortex Fine-Tuning.

### **Step 3. Execute and Integrate**
*   Update the Streamlit UI to explicitly showcase "Validation Checks" vs generic RAG retrieval.
*   Update the ReMindRAG traversal logic to prioritize DOT Constraints over simple semantic similarity. 

### **Step 4. Evaluation**
Compare the Base Model (our current setup) vs The Adapted Model using domain-specific metrics:
1.  **Safety Compliance Rate:** % of recommendations that do NOT violate DOT bridge limits.
2.  **Domain Jargon Accuracy:** Does it correctly interpret terms like "LTL" or "bobtail"?
3.  **Explainability Score:** Does it clearly cite the weather alert/accident that triggered the reroute?

### **Step 5. Report Writing**
*   **Group:** Summarize architecture changes and evaluation metrics in `GROUP_REPORT.md`.
*   **Individual:** Each team member logs their commits and detailed contributions in `INDIVIDUAL_REPORT_*.md`.
