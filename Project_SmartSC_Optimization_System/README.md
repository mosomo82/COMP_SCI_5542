---

# HyperLogistics: Neuro-Symbolic Disruption Recovery Engine 🚢 🧠

**A Snowflake-Native Supply Chain Resilience System using HyperGraph RAG**

| **Team Members** | **GitHub Role** |
| --- | --- |
| **[Your Name]** | `@yourusername` (Lead Architect) |
| **[Teammate Name]** | `@teammate` (Data Engineer) |
| **[Teammate Name]** | `@teammate` (ML Engineer) |

---

## 🚀 Problem Statement & Objectives

**The Problem:** Global supply chains are fragile. When a disruption occurs (e.g., a port strike or canal blockage), standard predictive models only forecast *delays*. They fail to reason about the complex, cascading dependencies between contracts, perishable goods, and alternative routes.

**The Objective:** To build **HyperLogistics**, a neuro-symbolic engine that doesn't just predict delays but **autonomously generates validated rerouting strategies**.

* **Innovation:** We replace standard Vector RAG with **HyperGraph RAG**. By modeling supply chains as hypergraphs (connecting multiple nodes simultaneously), our system can "reason" about the ripple effects of a disruption.
* **Target Users:** Logistics Network Managers and Compliance Officers.

---

## 🏗️ System Architecture

Our system is built entirely on the **Snowflake Data Cloud**, utilizing Snowpark for graph processing and Cortex for Generative AI.

```mermaid
graph TD
    subgraph "Ingestion Layer (Snowflake)"
        A[IoT Sensor Logs] -->|Snowpipe| B(RAW_IOT)
        C[Disruption News] -->|Snowpipe| D(RAW_NEWS)
        E[Shipment CSVs] -->|COPY INTO| F(RAW_SHIPMENTS)
    end

    subgraph "Knowledge Layer (Snowpark)"
        B & F -->|Python UDF| G{Graph Builder}
        G --> H[(Knowledge Graph\nNodes & Edges)]
        D -->|Cortex Embed| I[Vector Store]
    end

    subgraph "Intelligence Layer (Cortex AI)"
        J[User Query: 'Red Sea Blocked'] -->|Search| I
        I -->|Retrieve Context| K[HyperGraph Traversal]
        K -->|Subgraph + Rules| L[Cortex LLM \n(Llama 3)]
        L --> M[Rerouting Plan JSON]
    end

    subgraph "Application Layer"
        M --> N[Streamlit Dashboard]
    end

```

### **Core Components**

1. **Ingestion:** Real-time streams via Snowpipe.
2. **Graph Construction:** Snowpark Python jobs transform tabular shipment data into a network graph.
3. **Reasoning:** Snowflake Cortex (Llama 3) analyzes unstructured news and queries the structured graph to find safe alternative routes.

---

## 📚 Data Sources & References

### **NeurIPS 2025 Papers (Methodology)**

1. **[HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](https://arxiv.org/abs/2408.08174)**
* *Application:* Used to model multi-entity dependencies (Shipment + Port + Union Contract).


2. **[GraphFlow: Retrieving What You Need for Complex Queries](https://neurips.cc/virtual/2025/poster/115922)**
* *Application:* Logic for "walking" the graph to find alternative routes.


3. **[RAG4GFM: Graph Retrieval Augmented Generation](https://neurips.cc/virtual/2025/poster/115562)**
* *Application:* Framework for grounding LLM answers in valid graph data.



### **Datasets**

* **[Global Supply Chain Disruption (Kaggle)](https://www.kaggle.com/datasets/bertnardomariouskono/global-supply-chain-disruption-and-resilience):** Training data for disruption classification.
* **[US Supply Chain Risk Analysis (Kaggle)](https://www.kaggle.com/datasets/yuanchunhong/us-supply-chain-risk-analysis-dataset):** Real-world transaction logs for risk validation.
* **[Financial PhraseBank (Hugging Face)](https://www.google.com/search?q=https://huggingface.co/datasets/financial_phrasebank):** Used to simulate "Live Disruption News" feeds.

---

## 📂 Repository Structure

* `/proposal`: Contains the formal PDF proposal and research notes.
* `/src`: Source code.
* `/ingestion`: SQL scripts for table setup and data loading.
* `/graph_engine`: Snowpark Python scripts for building the Knowledge Graph.
* `/app`: Streamlit dashboard code.


* `/docs`: System diagrams, design documents, and meeting notes.
* `/data`: Sample datasets (subsets) for reproducibility.

---

## 🔁 Reproducibility: How to Run

### **Prerequisites**

* A Snowflake Account (Trial or Standard).
* Python 3.8+ installed locally.
* `pip install snowflake-snowpark-python streamlit pandas`

### **Step 1: Database Setup**

Run the setup script to create the necessary Warehouses, Databases, and Stages.

```bash
snowsql -f src/ingestion/setup_env.sql

```

### **Step 2: Data Ingestion**

Upload the sample data from `/data` to the Snowflake internal stage.

```sql
PUT file://data/shipments.csv @hyperlogistics_stage;
COPY INTO RAW_SHIPMENTS FROM @hyperlogistics_stage/shipments.csv;

```

### **Step 3: Build the Graph**

Execute the Snowpark job to transform rows into graph nodes.

```bash
python src/graph_engine/build_graph.py

```

### **Step 4: Launch the Dashboard**

Start the Streamlit app locally (or deploy to Snowflake).

```bash
streamlit run src/app/dashboard.py

```

---

**Would you like me to write the code for `src/ingestion/setup_env.sql` so you have the actual SQL script ready to commit?**