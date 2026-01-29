# Lab 2 — Advanced RAG Results (CS 5542)

> **Course:** CS 5542 — Big Data Analytics and Apps  
> **Lab:** Advanced RAG Systems Engineering  
> **Student Name:** Tony Nguyen
> **GitHub Username:** mosomo82
> **Date:** 01/28/2026

---

## 1. Project Dataset
- **Domain:** Consumer Protection  
- **# Documents:**	8 text files (covering customer protection, credit card, fraud, etc.)
- **Data Source (URL / Description):**  https://github.com/mosomo82/COMP_SCI_5542/tree/main/Week_2/project_data
- **Why this dataset fits my use case:**  
  This dataset provides realistic, fact-dense legal statutes and consumer rights information that require high precision to answer correctly. It is a perfect use case for Advanced RAG because the documents contain specific deadlines and "Tips" that general LLMs might hallucinate or overlook without grounded retrieval.

---

## 2. Queries + Mini Rubric

### Q1
- **Query:**  
- **Relevant Evidence (keywords / entities / constraints):**  
- **Correct Answer Criteria (1–2 bullets):**  

### Q2
- **Query:**  
- **Relevant Evidence (keywords / entities / constraints):**  
- **Correct Answer Criteria (1–2 bullets):**  

### Q3 (Ambiguous / Edge Case)
- **Query:**  
- **Relevant Evidence (keywords / entities / constraints):**  
- **Correct Answer Criteria (1–2 bullets):**  

---

## 3. System Design

- **Chunking Strategy:** Fixed / Semantic  
- **Chunk Size / Overlap:**  
- **Embedding Model:**  
- **Vector Store / Index:** (FAISS / Chroma / Other)  
- **Keyword Retriever:** (TF-IDF / BM25)  
- **Hybrid α Value(s):**  
- **Re-ranking Method:** (Cross-Encoder / LLM Judge / None)  

### Design Rationale
*(3–5 sentences explaining why you chose these system settings and how they affect performance, latency, or accuracy.)*

---

## 4. Results

| Query | Method (Keyword / Vector / Hybrid) | Precision@5 | Recall@10 |
|-------|-----------------------------------|-------------|-----------|
| Q1    |                                   |             |           |
| Q2    |                                   |             |           |
| Q3    |                                   |             |           |

---

## 5. Failure Case

- **What failed?**  
- **Which layer failed?** (Chunking / Retrieval / Re-ranking / Generation)  
- **Proposed system-level fix:**  

---

## 6. Evidence of Grounding

Provide one example of a **RAG-grounded answer with citations**:

> **Answer:**  
> *(Paste answer here)*  
>  
> **Citations:** [Chunk 1], [Chunk 2]

---

## 7. Reflection (3–5 Sentences)
What did you learn about **system design vs. model performance** when building this RAG pipeline?

---

## Reproducibility Checklist
- [ ] Project dataset included or linked  
- [ ] Queries + rubric filled  
- [ ] Results table completed  
- [ ] Screenshots included in repo  
- [ ] Notebook runs end-to-end  

---

> *CS 5542 — UMKC School of Science & Engineering*