# CS 5542 — Lab 3: Multimodal RAG Systems & Retrieval Evaluation

**Student:** Tony Nguyen  
**ID:** tmnc62@umkc.edu  
**Course:** CS 5542 - Deep Learning  

---

## 1. Executive Summary
This project implements a **Multimodal Retrieval-Augmented Generation (RAG)** system capable of ingesting PDF documents and images (charts/diagrams) to answer complex domain-specific queries. The system utilizes:
* **Hybrid Retrieval:** Combining Sparse (TF-IDF) and Dense (SentenceTransformers/FAISS) vector search.
* **Multimodal Ingestion:** Extracting text from PDFs via `PyMuPDF` and semantic content from images via **OCR (Tesseract)**.
* **Generation:** Comparing a lightweight extractive baseline, a local LLM (TinyLlama), and a cloud API (Gemini).

The experiments demonstrate that **Multimodal RAG** significantly improves answer coverage for queries relying on visual data (flowcharts, maps) compared to text-only baselines.

---

## 2. Methodology

### A. Data Ingestion & OCR
* **Text:** Extracted from PDFs (`doc1.pdf` - `doc4.pdf`) using `PyMuPDF`.
* **Images:** 10 figures (charts, maps) processed using **Tesseract OCR**. Captions were combined with OCR text to create searchable "Image Items."
* **Chunking Strategy:**
    * *Baseline:* Page-based chunking.
    * *Ablation:* Fixed-size sliding window chunking (`CHUNK_SIZE=900`, `OVERLAP=150`).

### B. Retrieval Strategies
We evaluated five retrieval configurations:
1.  **Sparse Only:** TF-IDF (keyword matching).
2.  **Dense Only:** `all-MiniLM-L6-v2` embeddings + FAISS (semantic matching).
3.  **Hybrid:** Weighted fusion of Sparse + Dense scores (`Alpha=0.5`).
4.  **Hybrid + Rerank:** Re-scoring top candidates using a Cross-Encoder (`ms-marco-MiniLM`).
5.  **Multimodal:** Hybrid Text retrieval + Sparse Image retrieval.

### C. Generators
1.  **Extractive:** Returns top-2 retrieved snippets (Baseline).
2.  **Local LLM:** `TinyLlama-1.1B-Chat` (Quantized, runs on T4 GPU).
3.  **Cloud API:** Google `gemini-2.5-flash` (Ground Truth/Gold Standard).

---

## 3. Evaluation Results

### A. Quantitative Retrieval Metrics
*Metric: Precision@5 (P@5) and Recall@10 (R@10) against ground truth keywords.*

| Query | Method | Precision@5 | Recall@10 | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Q1 (Dispute Process)** | Sparse Only | 0.60 | 0.42 | Baseline performed well on specific legal terms. |
| | **Hybrid** | **0.60** | **0.50** | **Best Recall.** Captured diverse evidence. |
| | Dense Only | 0.20 | 0.42 | Struggled with specific procedural steps. |
| **Q2 (Data Min.)** | Sparse Only | 0.60 | 0.44 | |
| | **Hybrid** | 0.40 | **0.56** | **Best Recall.** Found distributed map data. |
| | Dense Only | 0.60 | 0.33 | High precision but low recall. |
| **Q3 (Adolescent)** | Sparse Only | 0.80 | 0.40 | |
| | **Multimodal** | **1.00** | **0.45** | **Perfect Precision.** Successfully retrieved privacy scorecards. |

**Observation:** The **Hybrid** method generally offered the best balance of Recall, while **Multimodal/Dense** methods achieved higher Precision on queries requiring conceptual matching (Q3).

### B. Generator Model Comparison
*Qualitative grading of generated answers.*

| Model | Avg P@5 | Avg Faithfulness | Avg Coverage (1-5) | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Extractive** | 0.67 | **1.0 (High)** | 2.0 (Low) | Accurate but disjointed; missed synthesis. |
| **TinyLlama (Local)** | 0.67 | 0.67 (Med) | 3.3 (Med) | Good structure, but **hallucinated** on Q3. |
| **Gemini (API)** | 0.67 | **1.0 (High)** | **5.0 (High)** | Perfect synthesis; correctly identified missing data. |

### C. Ablation Study: Text-Only vs. Multimodal
*Does adding images help answer the questions?*

| Query | Modality | Images Retrieved | Answer Quality |
| :--- | :--- | :--- | :--- |
| **Q1** | Text-Only | 0 | Generic procedural text. |
| | **Multimodal** | **3** | Referenced the **"FCRA Compliance Flow Chart"** correctly. |
| **Q2** | Text-Only | 0 | Missed the state comparison visual nuances. |
| | **Multimodal** | **3** | Retrieved **"State Privacy Law Map"** and **"Schumer Box"**. |
| **Q3** | Text-Only | 0 | Failed to find age thresholds. |
| | **Multimodal** | **3** | Retrieved **"Privacy Scorecard"** images. |

**Conclusion:** Multimodal RAG is essential for Q1 and Q2, as the text chunks alone described the *rules* but the images contained the *workflow* and *geographic distribution*.

---

## 4. Failure Analysis (Q3 Case Study)

**The Failure:**
For Q3 (*"how do age thresholds vary between Delaware, New Jersey, and Oregon"*), the **TinyLlama** generator failed.
* **Observed Behavior:** It hallucinated specific details or gave a generic answer ("states vary") without citing specific numbers.
* **Gemini Behavior:** Correctly stated "Not enough evidence" for specific age thresholds.

**Root Cause:**
1.  **Retrieval Gap:** The specific age numbers (e.g., "13-16") were likely embedded inside a complex table in `doc4.pdf`.
2.  **Chunking Error:** The fixed-size chunking likely split the **Header (State Name)** from the **Row Data (Age Limit)**, creating disjointed chunks that could not be retrieved together.

**Proposed Fix:**
* Implement **Semantic Chunking** (grouping by table/section) rather than fixed character counts.
* Use **Visual Layout Analysis (VLA)** models (like LayoutLM) to parse tables before chunking.

---

## 5. How to Run This Project

1.  **Install Dependencies:**
    ```bash
    pip install PyMuPDF sentence-transformers faiss-cpu reportlab pytesseract
    sudo apt-get install tesseract-ocr  # If running on Linux/Colab
    ```
2.  **Data Setup:**
    Ensure `project_data_mm/` exists. If not, the notebook will automatically download the dataset or generate sample data.
3.  **Execution:**
    Run all cells in `CS5542_Lab3.ipynb`.
    * **Note:** To run the LLM Generator without an API key, rely on the `TinyLlama` (Local) section.

---
**Generated:** February 2, 2026