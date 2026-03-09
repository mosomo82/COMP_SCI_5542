# Lab 8: Domain Adaptation Group Report

## 1. Project Description
HyperLogistics is a Snowflake-native supply chain resilience system that bridges the prediction-action gap for middle-mile logistics. It uses ReMindRAG for knowledge-guided retrieval and SRSNet for adaptive forecasting, ensuring autonomous rerouting strategies grounded in safety and compliance.

## 2. Domain Task
**Domain Task:** Explainable Rerouting & Safety Justification
**Description:** Generating constraint-compliant rerouting justifications for middle-mile logistics dispatchers when real-time disruptions (e.g., weather alerts or accident blackspots) are present.
**Expected Output:** Structured rationale that proposes a reroute, cites the disruption, and confirms compliance with DOT physical constraints (e.g., bridge weight/height limits).

## 3. Dataset Creation Method
To teach the model domain-specific constraints (e.g., DOT bridge weight/height limits) and logistics jargon (e.g., "heavy haul", "LTL", "bobtail"), we generated a synthetic dataset. 

A Python script (`week8/generate_dataset.py`) was created to simulate Area Manager queries regarding real-time disruptions. This script programmatically combined constraints from `SILVER.BRIDGE_INVENTORY_GEO` and simulated events from `SILVER.WEATHER_ALERTS` to generate 100 query-response pairs. These pairs establish the exact reasoning patterns the model must follow to safely veto or approve a route.

The resulting dataset is saved in JSONL format at `week8/synthetic_rerouting_dataset.jsonl`.

## 4. Model Adaptation Method
**Method Chosen:** Prompt Adaptation (Few-Shot Prompting)

Instead of full fine-tuning, we adapted the model by dynamically injecting expert-level, validated routing examples from our synthetic dataset into the system prompt. By using highly engineered few-shot prompting, we condition the Snowflake Cortex inference engine to adopt the rigid, deterministic reasoning required for safety-critical systems while maintaining cost efficiency and agility.

## 5. Evaluation Results
*(To be completed: Present metrics demonstrating the improvement from the base RAG model to the domain-specialized assistant. Include baseline vs. adapted metrics.)*

## 6. System Architecture Updates
*(To be completed: Detail any updates to the diagram or system flow resulting from the domain adaptation.)*

## 7. Contribution Table

| Student | Contribution | Percentage |
| :--- | :--- | :--- |
| **Tony Nguyen** | *(Enter contribution)* | 33.3% |
| **Joel Vinas** | *(Enter contribution)* | 33.3% |
| **Daniel Evans** | *(Enter contribution)* | 33.3% |
