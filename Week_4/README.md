# CS 5542 — Lab 4: Multimodal RAG Application

A project-aligned **Retrieval-Augmented Generation (RAG)** application that ingests research papers (PDFs) and figures (images), retrieves relevant evidence using multiple retrieval strategies, and generates grounded answers with inline citations.

## 🔗 Deployment

> **Streamlit Cloud:** [Lab4_Streamlit_Cloud](https://compsci5542-qhebkbpsl3muncwyxbsf6n.streamlit.app/)


---

## 📂 Project Dataset Description

The corpus consists of **three research papers** and their associated figures, chosen from course project topics:

| File | Description | Pages |
|------|-------------|-------|
| `doc1_TimeSeries.pdf` | **SRSNet** — Selective Representation Space Network for time series forecasting using learnable patch selection | ~15 |
| `doc2_ReMindRAG.pdf` | **ReMindRAG** — Memory-augmented RAG with knowledge graph traversal and replay mechanisms | ~12 |
| `doc3_CPP.pdf` | **Consensus Planning Problem** — Multi-agent consensus optimization with primal/dual/proximal interfaces | ~10 |
| `data/images/` | **15 figures** extracted from the papers (architecture diagrams, tables, workflow visualizations) | — |

**Ingestion pipeline:**
- PDFs → per-page text chunks via PyMuPDF (`chunk_id: doc1_TimeSeries.pdf::p1`)
- Images → placeholder text entries (`chunk_id: img::figure8.png`)
- Total corpus: ~50 evidence items (text pages + image entries)

---

## 🏗 Project Structure

```
Week_4/
├── app/
│   └── main.py              # Streamlit UI (query + evaluation dashboard)
├── api/
│   └── server.py             # FastAPI REST backend
├── rag/
│   ├── __init__.py
│   └── pipeline.py           # Shared RAG pipeline (827 lines)
├── data/
│   ├── docs/                  # PDF papers (3 files)
│   └── images/                # Extracted figures (15 files)
├── logs/
│   └── query_metrics.csv      # Auto-generated evaluation log
├── src/                 # Lab 4 Jupyter notebook
│   └── CS5542_Lab4_Notebook.ipynb 
├── .streamlit/
│   └── config.toml            # Streamlit config & theme
├── Procfile                   # Deployment entrypoint
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🚀 Run Instructions

### Prerequisites

- Python 3.10+
- (Optional) Gemini API key for LLM-based answer generation

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/mosomo82/COMP_SCI_5542.git
cd COMP_SCI_5542/Week_4

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app/main.py
```

### Optional: FastAPI Backend

```bash
uvicorn api.server:app --reload --port 8000
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Optional | Google Gemini API key (only for LLM answer mode) |

---

## 🔍 Retrieval Modes

The application implements **5 retrieval strategies**:

| Mode | Method | Description |
|------|--------|-------------|
| `tfidf` | TF-IDF + Cosine Similarity | Sparse bag-of-words baseline |
| `sparse_BM25` | BM25 (Okapi) | Probabilistic sparse retrieval |
| `dense` | Sentence-Transformers + FAISS | Dense semantic search (`all-MiniLM-L6-v2`) |
| `hybrid_BM25&TF-IDF` | Reciprocal Rank Fusion | Combines TF-IDF + BM25 scores |
| `hybrid_rerank` | Hybrid + CrossEncoder | Dense+BM25 hybrid, re-scored with `ms-marco-MiniLM-L-6-v2` |

---

## 📊 Results Snapshot

### Evaluation Queries (Q1–Q6)

| ID | Query | Type | Gold Evidence |
|----|-------|------|---------------|
| Q1 | What is the SRS module in SRSNet and how does it differ from conventional adjacent patching? | Text + Image | `doc1_TimeSeries.pdf`, `img::figure8.png`, `img::figure9.png` |
| Q2 | How does ReMindRAG's memory replay mechanism improve retrieval for similar or repeated queries? | Text + Image | `doc2_ReMindRAG.pdf`, `img::figure3.png`, `img::figure7.png` |
| Q3 | What real-world applications of the CPP are described, and what agent interfaces does each use? | Text + Image | `doc3_CPP.pdf`, `img::figure2.png` |
| Q4 | What is the Multi-Hop QA accuracy of ReMindRAG vs HippoRAG2 with Deepseek-V3? | Multimodal/Table | `doc2_ReMindRAG.pdf`, `img::figure6.png` |
| Q5 | What RL reward function does SRSNet use to train the Selective Patching scorer? | Missing Evidence | `N/A` (trick question — SRSNet uses gradient-based, not RL) |
| Q6 | What are the key stages shown in the ReMindRAG overall workflow diagram? | Image-Only | `img::figure3.png` |

### Batch Evaluation Results (6 queries × 5 modes)

<img width="2874" height="470" alt="image" src="https://github.com/user-attachments/assets/107bd25f-04e7-4c2c-a2b8-8ab885776071" />

*Exact values depend on your hardware. Run batch evaluation to get precise numbers.*

---

## 📸 Screenshots

### Query Interface
<img width="2874" height="1208" alt="image" src="https://github.com/user-attachments/assets/276ec1e3-ca17-4999-8271-e6a224cd2fb4" />

*Screenshot: The main query tab showing the question input, answer output with citations, evidence panel, and metrics sidebar.*

### Evaluation Dashboard
<img width="2280" height="924" alt="image" src="https://github.com/user-attachments/assets/df14a6a4-808b-468a-a758-9d22c0574313" />

<img width="2262" height="1238" alt="image" src="https://github.com/user-attachments/assets/e51a16ee-27cc-40aa-b0cb-e0208ff066f9" />

<img width="2280" height="1114" alt="image" src="https://github.com/user-attachments/assets/c508e477-7cc2-46de-8d09-537dbf3825cf" />

<img width="980" height="450" alt="image" src="https://github.com/user-attachments/assets/4bd81e39-be72-4895-8811-ad53c49a9fe0" />

*Screenshot: The evaluation dashboard showing Precision@5 and Recall@10 bar charts across retrieval modes, per-query breakdown table, and latency distribution.*

### Metadata Filtering
<img width="596" height="978" alt="image" src="https://github.com/user-attachments/assets/da018101-1f69-4bbe-b33f-85112b77e991" />

*Screenshot: The sidebar metadata filters showing source document and modality multiselects.*

### Batch Evaluation Results
<img width="2264" height="1238" alt="image" src="https://github.com/user-attachments/assets/f411ba22-cbf9-4a9e-bb2c-855c46adfd47" />

*Screenshot: Batch evaluation results showing the pivot table comparing all queries across all retrieval modes.*

---

## ⚠️ Failure Analysis

### 1. Image Evidence Retrieval (Q6 — Near-Zero Scores)

**Problem:** Q6 asks about the ReMindRAG workflow diagram, but its only gold evidence is `img::figure3.png`. All text-based retrievers (TF-IDF, BM25) score **0.0** for both P@5 and R@10.

**Root Cause:** Images are represented as placeholder text (`[Image: figure3.png]`), which has zero lexical overlap with the query "What are the key stages shown in the ReMindRAG overall workflow diagram?" Text-based retrievers cannot bridge this semantic gap.

**Mitigation:** Dense and hybrid modes perform slightly better because sentence embeddings can capture partial semantic similarity. True multimodal retrieval would require OCR, image captioning, or vision-language models (e.g., CLIP) to generate richer text representations for images.

### 2. Missing Evidence Detection (Q5 — False Positives)

**Problem:** Q5 is a trick question about a non-existent RL component in SRSNet. The extractive answer mode may still return text snippets mentioning "patching" or "scorer" even though the correct behavior is to say "insufficient evidence."

**Root Cause:** The extractive answerer uses keyword overlap, which will match superficially related terms. Only the LLM-based answer modes (Gemini, Flan-T5) can reason about whether the retrieved evidence actually answers the specific question asked.

**Mitigation:** The `missing_evidence_behavior` metric checks whether low-scoring evidence triggers the "insufficient information" fallback. Adjusting the score threshold (default 0.05) can tune sensitivity.

### 3. Table/Figure Data Extraction (Q4)

**Problem:** Q4 asks for specific numeric values from Table 1 in the ReMindRAG paper. Extractive answers may retrieve the correct page but fail to extract the exact numbers.

**Root Cause:** PDF text extraction via PyMuPDF often garbles table layouts, merging columns or losing alignment. The table data exists in the corpus but may not be cleanly parseable.

**Mitigation:** LLM-based answer modes (Gemini) handle this significantly better by reasoning over messy table text. Future improvements could add table-specific extraction (e.g., Camelot, Tabula) as a preprocessing step.

---

## 💡 Reflection

Building this RAG application exposed the gap between simple keyword retrieval and true multimodal understanding. The most critical finding was that retrieval strategy impacts performance far more than answer generation—switching from TF-IDF to a hybrid+rerank approach improved Precision@5 by 30–40%, whereas generation adjustments had negligible impact.

To move beyond qualitative checks, We identified the necessity of a 'Gold Standard' benchmarking mode—using a curated dataset and a predefined query set to rigorously measure system yield and accuracy. This, combined with the evaluation dashboard, allowed for data-driven iteration and highlighted the persistent challenge of 'invisible' visual data. If we were to extend this project, we would prioritize formally implementing this automated benchmarking suite, alongside CLIP-based image embeddings and table-aware parsing, to fully bridge the multimodal gap.

---

## 🛠 Extensions Implemented

1. **Evaluation Dashboard** — Second tab with P@5, R@10 charts, latency distribution, per-query breakdown, failure analysis, and CSV download
2. **Batch Evaluation Pipeline** — One-click evaluation of all 6 queries × 5 retrieval modes (30 runs)
3. **Metadata Filtering** — Sidebar filters for source document and modality (text/image)
4. **Response Caching** — `@st.cache_data` with 1hr TTL for instant repeated-query responses
5. **Multiple Retrieval Modes** — 5 strategies: TF-IDF, BM25, Dense (FAISS), Hybrid, Reranked
6. **LLM Answer Generation** — 3 modes: extractive heuristic, Gemini API, local Flan-T5-Small
7. **CrossEncoder Reranking** — Two-stage retrieval with `ms-marco-MiniLM-L-6-v2`
8. **Query Management** - Edit/Upload/Change Golden Query Set
9. **Comparison Mode** - Option to compare between two retrieval modes on selected generation.

---

## 📄 License

This project is for academic use as part of CS 5542 at the University of Missouri–Kansas City.
