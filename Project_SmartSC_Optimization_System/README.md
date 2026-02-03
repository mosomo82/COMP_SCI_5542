# HyperLogistics: Smart Supply Chain Optimization System

**A Snowflake-Native Supply Chain Resilience System using HyperGraph RAG**

| **Team Members** | **GitHub Role** |
| --- | --- |
| **Daniel Evans** | `@teammate`  Data/Back-End Engineer |
| **Joel Vinas** | `@teammate` Data Engineer/ML Engineer |
| **Tony Nguyen** | `@mosomo82` ML/Full-Stack Engineer |

---

## 🎯 Problem Statement & Objectives

**The Problem:** Logistics networks are plagued by a 'prediction-action' gap, where managers are flooded with disruption alerts but lack the tools to instantly calculate safe alternatives. This inability to unify unstructured real-time news with structured shipment data leads to reactive decision-making, costing billions in avoidable delays and compliance failures.

**The Objective:** To build **HyperLogistics**, a neuro-symbolic engine that doesn't just predict delays but **autonomously generates validated rerouting strategies**.

* **Innovation:** We replace standard Vector RAG with **HyperGraph RAG**. By modeling supply chains as hypergraphs (connecting multiple nodes simultaneously), our system can "reason" about the ripple effects of a disruption.
* **Target Users:** Logistics Network Managers and Compliance Officers.

---

## 🏗️ System Architecture Diagram

Our system is built on the **Snowflake Data Cloud**, utilizing a Client-Server architecture where Google Colab serves as the development environment to orchestrate sophisticated Snowpark pipelines. Data is ingested via Snowpipe into Snowflake, where a NetworkX graph engine (running natively in Snowpark) maps complex supply chain dependencies. The intelligence layer leverages Snowflake Cortex to access Google Gemini, ensuring that generative AI reasoning happens securely next to the data without requiring external API egress.

```mermaid.js
graph TD
    subgraph "Client Layer (Orchestration)"
        Z[Google Colab Notebook] -->|Push Python Code| G
        Z -->|Trigger AI Analysis| L
    end

    subgraph "Ingestion Layer (Snowflake)"
        A[IoT Sensor Logs] -->|Snowpipe| B(RAW_IOT)
        C[Disruption News] -->|Snowpipe| D(RAW_NEWS)
        E[Shipment CSVs] -->|COPY INTO| F(RAW_SHIPMENTS)
    end

    subgraph "Knowledge Layer (Snowpark Python)"
        B & F -->|Snowpark DataFrame| G{Graph Builder \n(NetworkX)}
        G --> H[(Knowledge Graph\nNodes & Edges)]
        D -->|Cortex Embed| I[Vector Store]
    end

    subgraph "Intelligence Layer (Cortex AI)"
        J[User Query: 'Red Sea Blocked'] -->|Search| I
        I -->|Retrieve Context| K[HyperGraph Traversal]
        K -->|Subgraph + Rules| L[Cortex LLM \n(Google Gemini)]
        L --> M[Rerouting Plan JSON]
    end

    subgraph "Application Layer"
        M --> N[Streamlit in Snowflake]
    end
```
## 🛠️ Methods & Technologies
Here is the text formatted as a clean Markdown table for your proposal:

### **Methods & Technologies**

| **Component** | **Tool** | **Description** |
| --- | --- | --- |
| **Development Environment** | **Google Colab** | Used as the client-side notebook to write and execute Snowpark pipelines (instead of local Jupyter). |
| **Graph Processing** | **Snowpark Python** (pushed from Colab) | We write `networkx` code in Colab, but execute it *inside* Snowflake using Snowpark Stored Procedures to keep data secure. |
| **Generative AI** | **Snowflake Cortex** (powered by Google Gemini) | We use `SNOWFLAKE.CORTEX.COMPLETE()` specifying the `'gemini-pro'` model to analyze disruption text without data leaving the platform. |

---

## 📚 Data Sources & References

### **NeurIPS 2025 Reasearch Papers (with Code)**

1. **[Link1](https://example.com)**
* *Application:* describe


2. **[Link2](https://example.com)**
* *Application:* describe


2. **[Link3](https://example.com)**
* *Application:* describe



### **Datasets**

* **[DataLink1](https://example.com):** describe
* **[DataLink2](https://example.com):** describe
* **[DataLink3](https://example.com):** describe

---

## 📂 Repository Structure

```text
/Project_SmartSC_Optimization_System
├── data/                 # Dataset files (CSV, JSON) for ingestion
├── docs/                 # System diagrams, design notes, and meeting logs
├── proposal/             # Your formal PDF proposal and research drafts
├── reproducibility/      # Guides or scripts specifically for reproducing results
├── src/                  # Source code for ingestion, graph building, and the app
└── README.md             # The main project landing page
```

---
