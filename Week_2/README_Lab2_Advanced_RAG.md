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
- **Data Source (URL / Description):** [mosomo82 Github Repository](https://github.com/mosomo82/COMP_SCI_5542/tree/main/Week_2/project_data)
- **Why this dataset fits my use case:**  
  This dataset provides realistic, fact-dense legal statutes and consumer rights information that require high precision to answer correctly. It is a perfect use case for Advanced RAG because the documents contain specific deadlines and "Tips" that general LLMs might hallucinate or overlook without grounded retrieval.

---

## 2. Queries + Mini Rubric

### Q1
- **Query:**  Under the Fair Credit Billing Act, what are the specific deadlines a creditor must follow once a consumer disputes a charge in writing?
- **Relevant Evidence (keywords / entities / constraints):**  Fair Credit Billing Act (FCBA), 60 days (consumer notice), 30 days (acknowledgment), 90 days (resolution).  
- **Correct Answer Criteria (1–2 bullets):**
  - Must specify the 60-day window for consumers and the 30/90-day response sequence for creditors.
  - Must mention the right to withhold payment during the investigation.


### Q2
- **Query:**  If a painting company starts a job but only paints the eaves and never returns, can they recover any money from the homeowners? What is this situation called?
- **Relevant Evidence (keywords / entities / constraints):**  Partial Performance, Ace Painting Company, homeowner rejection, liability for cost difference.
- **Correct Answer Criteria (1–2 bullets):**
  - Correctly identifies the situation as "Partial Performance."
  - Explains that no money is recovered if rejected and the breaching party may owe the homeowner the cost difference.

### Q3 (Ambiguous / Edge Case)
- **Query:**  A 17-year-old living on their own signs a contract for an apartment and a high-end smartphone. If they stop paying for both, can the providers legally enforce these contracts?
- **Relevant Evidence (keywords / entities / constraints):**  Minor/Maturity (Age 18), Necessity exception (Shelter), Cell phone exclusion (Tip).
- **Correct Answer Criteria (1–2 bullets):**
  - Must distinguish between the apartment (enforceable as a necessity) and the phone (unenforceable).
  - Must cite the specific document tip stating phones are not necessities.

---

## 3. System Design

- **Chunking Strategy:** Semantic  (Paragraph-based merging)
- **Chunk Size / Overlap:**  Size 1200 / Overlap 200 (for fixed) or 400-char-buffer (for semnatic)
- **Embedding Model:**  sentence-transformers/all-MiniLM-L6-v2
- **Vector Store / Index:** FAISS (IndexFlatIP with normalized vectors)
- **Keyword Retriever:** BM25  
- **Hybrid α Value(s):** 0.5 (Selected for balanced retrieval)  
- **Re-ranking Method:** (Cross-Encoder/ms-marco-MiniLM-L-6-v2)

### Design Rationale
*Semantic chunking was chosen to keep complete legal thoughts and "Tips" together, preventing critical context from being cut in half. Hybrid retrieval with $\alpha = 0.5$ balances the precision of keyword matching (essential for specific Act names) with the semantic flexibility of vectors (essential for natural language queries). The Cross-Encoder reranker was added to improve Precision@K by performing a deeper analysis of the query-document relationship, which is vital for distinguishing between similar but distinct consumer laws.*

---

## 4. Lab2 - Advanced RAG Results

| Query | Method (Keyword / Vector / Hybrid) | Precision@5 | Recall@10 |
|-------|-----------------------------------|-------------|-----------|
| Q1    | Keyword                           |      0.6    |       1.0 |
| Q2    | Keyword                           |      0.4    |       6.67|
| Q3    | Keyword                           |      0.2    |       6.67|

### Semantic vs. Fixed Chunking Comparison
<img width="386" height="82" alt="image" src="https://github.com/user-attachments/assets/45184f4d-ac6a-4dd8-b253-dfc13049f697" />

### The RAG vs. Prompt-Only Comparison
<img width="582" height="137" alt="image" src="https://github.com/user-attachments/assets/4d926301-af28-40a0-bd6e-0dcc654e756a" />

### Reranking: Before vs. After for Query 1
<img width="322" height="882" alt="image" src="https://github.com/user-attachments/assets/21f2d770-405d-4d12-9257-945e7f5cd6fa" />
---

## 5. Failure Case

- **What failed?**  Retrieval of the specific cell phone "Tip" when using Keyword-only search.
- **Which layer failed?** Retrieval layer BM25
- **Proposed system-level fix:**  Increase the weight of Vector search ($\alpha=0.2$) or use a larger chunk size to ensure the "Tip" is always attached to the broader "Necessity" definition, as keyword search failed when the query used terms like "living on their own" instead of "parent or guardian."

---

## 6. Evidence of Grounding

Provide one example of a **RAG-grounded answer with citations**: 

> **Answer:**  
> *Under the Fair Credit Billing Act, a consumer must notify the creditor within 60 days of the bill being mailed [Chunk 1]. The creditor then has 30 days to respond and must resolve the dispute within 90 days [Chunk 1]. While in dispute, the consumer is not required to pay the disputed amount [Chunk 2].*
>  
> **Citations:** [Chunk 1], [Chunk 2]

---

## 7. Reflection (3–5 Sentences)
Through this lab, I learned that model performance is only as good as the retrieval system feeding it. Even a high-end LLM cannot answer specific legal questions correctly if the chunking strategy splits a rule from its exception. I discovered that Hybrid retrieval offers a "safety net" where keywords catch the technical terms and vectors catch the intent. Finally, I realized that while reranking adds latency, it is essential for high-precision tasks like legal analysis where the top-ranked result must be the most relevant.

---

## Reproducibility Checklist
- [x] Project dataset included or linked  
- [x] Queries + rubric filled  
- [x] Results table completed  
- [x] Screenshots included in repo  
- [x] Notebook runs end-to-end  

---

> *CS 5542 — UMKC School of Science & Engineering*
