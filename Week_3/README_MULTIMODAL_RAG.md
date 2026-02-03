# CS 5542 — Lab 3: Multimodal RAG Systems & Retrieval Evaluation

> **Course:** CS 5542 — Big Data Analytics and Applications  
> **Lab:** Multimodal RAG Systems & Retrieval Evaluation     
> **Student Name:** Tony Nguyen
> **GitHub Username:** mosomo82
> **Date:** 01/28/2026

---

## 1. Executive Summary
This project implements a **Multimodal Retrieval-Augmented Generation (RAG) & Retrieval Evaluation** system capable of ingesting PDF documents and images (charts/diagrams) to answer complex domain-specific queries. The system utilizes:
* **Hybrid Retrieval:** Combining Sparse (TF-IDF) and Dense (SentenceTransformers/FAISS) vector search.
* **Multimodal Ingestion:** Extracting text from PDFs via `PyMuPDF` and semantic content from images via **OCR (Tesseract)**.
* **Generation:** Comparing a lightweight extractive baseline, a local LLM (TinyLlama), and a cloud API (Gemini).

The experiments demonstrate that **Multimodal RAG** significantly improves answer coverage for queries relying on visual data (flowcharts, maps) compared to text-only baselines.

---

## 2. Project Dataset
- **Domain:** Consumer Protection 
- **# Documents:**	4 pdf files and 10 images (scanned pages, flowchart, tables, figures and screenshots etc.)
- **Data Source (URL / Description):** [mosomo82 Github Repository](https://github.com/mosomo82/COMP_SCI_5542/tree/main/Week_3/project_data)

---

## 3. Queries + Rubrics

The following table outlines the three test queries used to evaluate the RAG system, along with the keyword-based ground truth rubric used for automated scoring.

### **Query 1: FCRA Dispute Process**

* **ID:** `Q1`
* **Question:** Explain the step-by-step process for a consumer to dispute inaccurate information on their credit report under the FCRA, and detail the specific obligations a 'furnisher of information' has once they receive a dispute from a Credit Reporting Agency (CRA).

| Category | Keywords (Ground Truth) |
| --- | --- |
| **Must Have** | `30 days`, `investigate`, `modify,delete or block`, `notification`, `irrelavent` |
| **Optional** | `Consumer Finance Protection Bureau (CFPB)`, `15 days`, `indentify theft` |

---

### **Query 2: Data Minimization Evolution**

* **ID:** `Q2`
* **Question:** How is the concept of 'data minimization' evolving? Specifically, contrast the traditional 'procedural' standard found in WPA-style laws with the new 'substantive' standard introduced by Maryland.

| Category | Keywords (Ground Truth) |
| --- | --- |
| **Must Have** | `procedural`, `substantive`, `adaquate, relevant and reasonably necessary`, `secondary use`, `stricly necessary`, `reasonably necssary` |
| **Optional** | `opt-in consent`, `online data privacy act`, `product or service requested` |

---

### **Query 3: Adolescent Data Protections**

* **ID:** `Q3`
* **Question:** Analyze the 'heightened protections' for adolescent data in recent state privacy laws. Which specific activities (e.g., sale, targeted advertising) now typically require opt-in consent for teenagers, and how do the age thresholds vary between states like Delaware, New Jersey, and Oregon?

| Category | Keywords (Ground Truth) |
| --- | --- |
| **Must Have** | `opt-in consent`, `targeted advertising`, `sale of personal data`, `profiling`, `actual knowledge`, `willfully disregards` |
| **Optional** | `under 18`, `duty of care`, `under 17`, `under 16` |

---

## 4. Methodology

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

## 5. Evaluation Results

### A. Ingestion Report following Track B
![ingestion_report](https://github.com/user-attachments/assets/3d571f37-fe5c-4bd0-b513-d7914b187712)

### B. Lightweight Generator
![generator_lightweight](https://github.com/user-attachments/assets/0cfd7d85-96ce-4eb0-bd24-93c1f1dc4ffc)

### C. LLM API Call Generator
![generator_API_Call](https://github.com/user-attachments/assets/61563d18-b1fd-4ad5-a7bc-c1db671df481)

### D. LLM Local Generator
![generator_LLM_Local](https://github.com/user-attachments/assets/34cf34d5-7bd7-4282-97e1-2eddb01ff475)

### E. Quantitative Retrieval Metrics
*=== Final Deliverable Table (Query x Method x Metrics) ===*

|index|Query|Method|Precision@5|Recall@10|Total\_Rel\_In\_Corpus|
|---|---|---|---|---|---|
|0|Q1|Sparse Only|0\.60|0\.42|12|
|1|Q1|Dense Only|0\.20|0\.42|12|
|2|Q1|Hybrid|0\.60|0\.50|12|
|3|Q1|Hybrid + Rerank|0\.60|0\.42|12|
|4|Q1|Multimodal|0\.60|0\.42|12|
|5|Q2|Sparse Only|0\.60|0\.44|9|
|6|Q2|Dense Only|0\.60|0\.33|9|
|7|Q2|Hybrid|0\.40|0\.56|9|
|8|Q2|Hybrid + Rerank|0\.60|0\.44|9|
|9|Q2|Multimodal|0\.60|0\.44|9|
|10|Q3|Sparse Only|0\.80|0\.40|20|
|11|Q3|Dense Only|1\.00|0\.40|20|
|12|Q3|Hybrid|0\.60|0\.35|20|
|13|Q3|Hybrid + Rerank|1\.00|0\.45|20|
|14|Q3|Multimodal|1\.00|0\.45|20|

### F. Manual Evaluation Metrics for Generator Models
![evaluation_metrics](https://github.com/user-attachments/assets/7cb287fd-41e7-444d-9704-ca9945409589)

### G. Ablation Study: Text-Only vs. Multimodal
*Does adding images help answer the questions?*

| Query | Modality | Images Retrieved | Answer Quality | Image_Paths |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | Text-Only | 0 | Generic procedural text. | None |
| | **Multimodal** | **3** | Referenced the **"FCRA Compliance Flow Chart"** correctly. | ['Fair Credit Reporting Act (FCRA) Compliance Flow Chart_1.png', 'accuracy_in_concumer_reporting_process_blog_post-finalfinal.png', 'Dispute-Resolution--Unraveling-the-FCBA-s-Path-to-Fair-Credit-Billing--Step-by-Step-Guide-to-Resolving-Billing-Disputes-under-the-FCBA.png'] |
| **Q2** | Text-Only | 0 | Missed the state comparison visual nuances. | None |
| | **Multimodal** | **3** | Retrieved **"State Privacy Law Map"** and **"Schumer Box"**. | ['State-Data-Privacy-Law-Status_v2.png', 'Schumer_Box_example.png', 'State_Comp_Privacy_Law_Map.png'] |
| **Q3** | Text-Only | 0 | Failed to find age thresholds. | None |
| | **Multimodal** | **3** | Retrieved **"Privacy Scorecard"** images. | ['State-Data-Privacy-Law-Status_v2.png', 'lead-state-privacy-scorecard-graphic-attempt-1.png', 'CCPAvsGDPR-Nov.png'] |

**Conclusion:** Multimodal RAG is essential for Q1 and Q2, as the text chunks alone described the *rules* but the images contained the *workflow* and *geographic distribution*.

---

## 6. Failure Analysis (Q3 Case Study)

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

## 7. How to Run This Project

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

## Reproducibility Checklist
- [x] Project dataset included or linked  
- [x] Queries + rubric filled  
- [x] Results table completed  
- [x] Screenshots included in repo  
- [x] Notebook runs end-to-end  

---

> *CS 5542 — UMKC School of Science & Engineering*
