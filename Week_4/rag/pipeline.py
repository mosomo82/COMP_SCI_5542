"""
CS5542 Lab 4 — Shared RAG Pipeline
===================================
Extracted from the Lab 4 notebook so that `app/main.py` and `api/server.py`
can import and use the same retrieval + generation + logging logic.
"""

import csv
import glob
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────
MISSING_EVIDENCE_MSG = "Not enough evidence in the retrieved context."

# Default paths (relative to Week_4 root)
DEFAULT_DOCS_DIR = "data/docs"
DEFAULT_IMAGES_DIR = "data/images"
DEFAULT_LOG_FILE = "logs/query_metrics.csv"

LOG_HEADER = [
    "timestamp", "query_id", "retrieval_mode", "top_k", "latency_ms",
    "Precision@5", "Recall@10",
    "evidence_ids_returned", "gold_evidence_ids",
    "faithfulness_pass", "missing_evidence_behavior",
]


# ── Data Ingestion ─────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract per-page text from a PDF using PyMuPDF.
    
    Falls back to raw binary-to-text if PyMuPDF is not installed.
    Returns list of dicts with keys: chunk_id, text, source, page.
    """
    basename = os.path.basename(pdf_path)

    # Try PyMuPDF first (best quality)
    try:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            import pymupdf as fitz  # PyMuPDF >= 1.24

        pages = []
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "chunk_id": f"{basename}::p{page_num + 1}",
                    "text": text,
                    "source": pdf_path,
                    "page": page_num + 1,
                    "modality": "text",
                })
        doc.close()
        return pages

    except ImportError:
        # Fallback: read PDF as raw text (lossy but functional)
        print(f"WARNING: PyMuPDF not installed. Reading {basename} as raw text.")
        try:
            with open(pdf_path, "rb") as f:
                raw = f.read()
            # Extract readable ASCII/UTF-8 fragments from the PDF binary
            text = raw.decode("utf-8", errors="ignore")
            # Strip PDF binary noise — keep lines with mostly printable chars
            clean_lines = []
            for line in text.split("\n"):
                printable = sum(1 for c in line if c.isprintable() or c in " \t")
                if len(line) > 0 and printable / len(line) > 0.8:
                    clean_lines.append(line.strip())
            clean_text = "\n".join(clean_lines).strip()
            if clean_text:
                return [{
                    "chunk_id": f"{basename}::p1",
                    "text": clean_text[:5000],  # cap length
                    "source": pdf_path,
                    "page": 1,
                    "modality": "text",
                }]
        except Exception as e:
            print(f"WARNING: Could not read {basename}: {e}")
        return []


def load_images(images_dir: str) -> List[Dict[str, Any]]:
    """Load image evidence items from a directory.
    
    Each image gets an evidence ID prefixed with ``img::`` per project convention.
    """
    items = []
    if not os.path.isdir(images_dir):
        return items

    for fname in sorted(os.listdir(images_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            fpath = os.path.join(images_dir, fname)
            items.append({
                "chunk_id": f"img::{fname}",
                "text": f"[Image: {fname}]",  # placeholder text for TF-IDF
                "source": fpath,
                "modality": "image",
            })
    return items


def load_corpus(
    docs_dir: str = DEFAULT_DOCS_DIR,
    images_dir: str = DEFAULT_IMAGES_DIR,
) -> List[Dict[str, Any]]:
    """Load the full corpus (PDFs + images) and return a flat evidence list."""
    evidence: List[Dict[str, Any]] = []

    # PDFs
    for pdf_path in sorted(glob.glob(os.path.join(docs_dir, "*.pdf"))):
        evidence.extend(extract_pdf_pages(pdf_path))

    # Plain-text files
    for txt_path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        basename = os.path.basename(txt_path)
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            evidence.append({
                "chunk_id": basename,
                "text": text,
                "source": txt_path,
                "modality": "text",
            })

    # Images
    evidence.extend(load_images(images_dir))
    return evidence


# ── TF-IDF Retriever ──────────────────────────────────────────────────────────

class TfidfRetriever:
    """TF-IDF cosine-similarity retriever."""

    def __init__(self, evidence: List[Dict[str, Any]]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.evidence = evidence
        texts = [item.get("text", "") for item in evidence]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return list of (index, score) tuples, descending by score."""
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        idxs = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idxs]


# ── BM25 (Sparse) Retriever ────────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25 (Okapi) retriever using rank_bm25."""

    def __init__(self, evidence: List[Dict[str, Any]]):
        from rank_bm25 import BM25Okapi
        self.evidence = evidence
        tokenized = [item.get("text", "").lower().split() for item in evidence]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        idxs = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idxs]


# ── Dense (FAISS + Sentence-Transformers) Retriever ───────────────────────────

class DenseRetriever:
    """Dense retriever using sentence-transformers embeddings + FAISS L2 index."""

    def __init__(self, evidence: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.evidence = evidence
        self.model = SentenceTransformer(model_name)

        texts = [item.get("text", "") for item in evidence]
        self.corpus_embeddings = self.model.encode(texts, show_progress_bar=False)

        d = self.corpus_embeddings.shape[1]  # embedding dimension (384 for MiniLM)
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.corpus_embeddings)
        print(f"Dense index built: {self.index.ntotal} vectors, dim={d}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_emb = self.model.encode([query])
        distances, indices = self.index.search(query_emb, top_k)
        # Convert L2 distance to similarity score (lower distance = better)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            # Convert distance to a 0-1 similarity: sim = 1 / (1 + dist)
            sim = 1.0 / (1.0 + float(dist))
            results.append((int(idx), sim))
        return results


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """Combines two retrievers via weighted reciprocal rank fusion."""

    def __init__(self, retriever_a, retriever_b, alpha: float = 0.5):
        self.retriever_a = retriever_a
        self.retriever_b = retriever_b
        self.alpha = alpha  # weight for retriever_a; (1 - alpha) for retriever_b

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        n = len(self.retriever_a.evidence) if hasattr(self.retriever_a, 'evidence') else top_k * 3
        pool_k = min(top_k * 3, n)
        hits_a = self.retriever_a.retrieve(query, top_k=pool_k)
        hits_b = self.retriever_b.retrieve(query, top_k=pool_k)

        rrf_k = 60  # standard RRF constant
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(hits_a):
            scores[idx] = scores.get(idx, 0.0) + self.alpha / (rrf_k + rank + 1)
        for rank, (idx, _) in enumerate(hits_b):
            scores[idx] = scores.get(idx, 0.0) + (1 - self.alpha) / (rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(idx, score) for idx, score in ranked]


# ── Reranked Retriever ────────────────────────────────────────────────────────

class RerankedRetriever:
    """Wraps a base retriever and re-scores candidates with a CrossEncoder.

    Over-fetches ``rerank_depth`` candidates from the base retriever, then
    re-scores each (query, doc_text) pair using a CrossEncoder model.
    """

    def __init__(
        self,
        base_retriever,
        evidence: List[Dict[str, Any]],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_depth: int = 20,
    ):
        from sentence_transformers import CrossEncoder
        self.base = base_retriever
        self.evidence = evidence
        self.reranker = CrossEncoder(model_name)
        self.rerank_depth = rerank_depth
        print(f"Reranker loaded: {model_name}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        # 1. Over-fetch from the base retriever
        candidates = self.base.retrieve(query, top_k=self.rerank_depth)

        if not candidates:
            return []

        # 2. Build (query, doc_text) pairs for the CrossEncoder
        pairs = []
        for idx, _ in candidates:
            text = self.evidence[idx].get("text", "")
            pairs.append([query, text])

        # 3. Score with CrossEncoder
        rerank_scores = self.reranker.predict(pairs)

        # 4. Attach new scores and sort
        reranked = [
            (idx, float(score))
            for (idx, _), score in zip(candidates, rerank_scores)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


# ── Retriever Factory ─────────────────────────────────────────────────────────

def build_retrievers(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build all retriever variants from the evidence list.

    Returns dict mapping mode name → retriever instance.
    Mode names match the Streamlit sidebar labels.
    """
    tfidf = TfidfRetriever(evidence)
    bm25 = BM25Retriever(evidence)
    hybrid_sparse = HybridRetriever(tfidf, bm25, alpha=0.5)

    # Dense retriever (optional — needs faiss-cpu + sentence-transformers)
    try:
        dense = DenseRetriever(evidence)
        hybrid_dense = HybridRetriever(dense, bm25, alpha=0.6)
    except Exception as e:
        print(f"WARNING: Dense retriever unavailable ({e}). Falling back to TF-IDF.")
        dense = tfidf
        hybrid_dense = hybrid_sparse

    # CrossEncoder reranker (optional — needs sentence-transformers)
    try:
        reranked = RerankedRetriever(hybrid_dense, evidence)
    except Exception as e:
        print(f"WARNING: Reranker unavailable ({e}). Falling back to hybrid.")
        reranked = hybrid_dense

    return {
        "tfidf": tfidf,
        "sparse_BM25": bm25,
        "dense": dense,
        "hybrid_BM25&TF-IDF": hybrid_sparse,
        "hybrid_rerank": reranked,
        "multimodal": reranked,  # Reranked Dense+BM25 — best for text + image captions
    }


# ── Context Building ──────────────────────────────────────────────────────────

def build_context(
    evidence: List[Dict[str, Any]],
    hit_indices: List[int],
    max_chars: int = 4000,
) -> str:
    """Join retrieved evidence into a context string with citation tags.
    
    Always includes at least the first entry, truncated if needed.
    """
    parts = []
    total = 0
    for idx in hit_indices:
        item = evidence[idx]
        tag = f"[{item['chunk_id']}]"
        text = item.get("text", "")
        entry = f"{tag} {text}"
        # Always include at least one entry
        if parts and total + len(entry) > max_chars:
            break
        # Truncate individual entries if very long
        if len(entry) > 1000:
            entry = entry[:1000] + " …"
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


# ── Answer Generation ─────────────────────────────────────────────────────────

def extractive_answer(query: str, context: str) -> str:
    """Heuristic extractive answer: return top-3 evidence blocks most
    overlapping with the query terms, preserving citation tags."""
    if not context or not context.strip():
        return MISSING_EVIDENCE_MSG

    # Split context into blocks separated by double-newlines (one per evidence chunk)
    blocks = [b.strip() for b in context.split("\n\n") if b.strip()]

    if not blocks:
        return MISSING_EVIDENCE_MSG

    q_words = set(re.findall(r"[A-Za-z]{2,}", query.lower()))

    scored = []
    for block in blocks:
        words_in_block = set(re.findall(r"[A-Za-z]{2,}", block.lower()))
        score = len(q_words & words_in_block)
        scored.append((score, block))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top-3 blocks. If no overlap found, still return the first block
    # (we already have retrieved evidence, so show something useful)
    best = [b for _, b in scored[:3]]
    if not best:
        best = blocks[:1]

    # Truncate each block to ~500 chars for readability
    truncated = []
    for b in best:
        if len(b) > 500:
            truncated.append(b[:500] + " …")
        else:
            truncated.append(b)

    return "\n\n".join(truncated)


# ── LLM Answer Generation (Gemini) ────────────────────────────────────────────

def generate_llm_answer(
    question: str,
    context: str,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> str:
    """Generate a grounded answer using Google Gemini.

    Falls back to extractive_answer if the API call fails.
    """
    if not context or not context.strip():
        return MISSING_EVIDENCE_MSG

    # Resolve API key
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return f"[LLM unavailable — no GEMINI_API_KEY set]\n\n{extractive_answer(question, context)}"

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(model_name)

        prompt = f"""You are a helpful assistant for a Multimodal RAG system.
Use the following retrieved context (text chunks and image descriptions) to answer the user's question.

RULES:
1. Answer ONLY using the provided context. If the answer is not in the context, say "{MISSING_EVIDENCE_MSG}"
2. Cite your sources! When you use information, append the source ID like [doc1.pdf::p1] or [img::figure1.png].
3. Be concise and direct.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Fallback to extractive answer
        return f"[LLM error: {e}]\n\n{extractive_answer(question, context)}"


# ── LLM Answer Generation (Local HuggingFace) ────────────────────────────────

def generate_local_llm_answer(
    question: str,
    context: str,
    model_name: str = "google/flan-t5-small",
) -> str:
    """Generate an answer using a local HuggingFace model (Flan-T5-Small, ~77 MB).

    Uses text2text-generation (seq2seq) which is better for QA than causal LMs.
    Falls back to extractive_answer if torch/transformers are unavailable.
    """
    if not context or not context.strip():
        return MISSING_EVIDENCE_MSG

    try:
        from transformers import pipeline as hf_pipeline

        llm = hf_pipeline(
            "text2text-generation",
            model=model_name,
        )

        # Truncate context to stay within T5's 512-token limit
        tokenizer = llm.tokenizer
        max_context_tokens = 400  # leave room for question + answer tokens
        tokenized = tokenizer(context, truncation=False, return_tensors="pt")["input_ids"]
        if tokenized.shape[1] > max_context_tokens:
            tokenized = tokenized[:, :max_context_tokens]
            context = tokenizer.decode(tokenized[0], skip_special_tokens=True)

        prompt = (
            f"Answer the question based on the context below. "
            f"If the answer is not in the context, say 'Not enough evidence.'\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}"
        )

        output = llm(prompt, max_new_tokens=200)
        return output[0]["generated_text"].strip()
    except Exception as e:
        return f"[Local LLM error: {e}]\n\n{extractive_answer(question, context)}"


# ── Evidence-ID Canonicalization ──────────────────────────────────────────────

def canon_evidence_id(x: str) -> str:
    """Normalize an evidence ID for comparison.
    
    - Strips whitespace
    - Keeps ``img::`` prefix intact
    - Removes ``.txt`` extension
    - Strips ``::pN`` page suffix so ``doc1_TimerSeries.pdf::p2``
      matches gold ID ``doc1_TimerSeries.pdf``
    """
    x = str(x).strip()
    if x.startswith("img::"):
        return x
    # strip page suffix  e.g.  ::p2
    x = re.sub(r"::p\d+$", "", x)
    if x.endswith(".txt"):
        x = x[:-4]
    return x


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def precision_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int = 5,
) -> Optional[float]:
    """Precision@k.  Returns ``None`` if gold is empty or N/A."""
    if not gold_ids or gold_ids == ["N/A"]:
        return None
    canon_gold = {canon_evidence_id(g) for g in gold_ids}
    canon_ret = [canon_evidence_id(r) for r in retrieved_ids[:k]]
    if k == 0:
        return None
    return len(set(canon_ret) & canon_gold) / float(k)


def recall_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int = 10,
) -> Optional[float]:
    """Recall@k.  Returns ``None`` if gold is empty or N/A."""
    if not gold_ids or gold_ids == ["N/A"]:
        return None
    canon_gold = {canon_evidence_id(g) for g in gold_ids}
    canon_ret = [canon_evidence_id(r) for r in retrieved_ids[:k]]
    denom = float(len(canon_gold))
    return (len(set(canon_ret) & canon_gold) / denom) if denom > 0 else None


def faithfulness_heuristic(answer: str, evidence: List[Dict]) -> bool:
    """True if answer contains at least one citation tag from the evidence,
    OR if answer is the missing-evidence message."""
    if answer.strip() == MISSING_EVIDENCE_MSG:
        return True
    tags = [
        f"[{e.get('chunk_id', e.get('id', ''))}]"
        for e in evidence[:5]
    ]
    return any(tag in answer for tag in tags if tag != "[]")


def missing_evidence_behavior(
    answer: str,
    evidence: List[Dict],
    score_key: str = "score",
    threshold: float = 0.05,
) -> str:
    """Check whether the system correctly handles missing evidence.
    
    - If the best evidence score < threshold → expect MISSING_EVIDENCE_MSG.
    - Otherwise → expect a substantive answer.
    """
    has_ev = bool(evidence) and max(
        e.get(score_key, 0.0) for e in evidence
    ) >= threshold
    if not has_ev:
        return "Pass" if answer.strip() == MISSING_EVIDENCE_MSG else "Fail"
    else:
        return "Pass" if answer.strip() != MISSING_EVIDENCE_MSG else "Fail"


# ── CSV Logging ───────────────────────────────────────────────────────────────

def ensure_logfile(path: str):
    """Create the log CSV with header if it doesn't exist yet."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADER)


def log_query(
    log_path: str,
    query_id: str,
    retrieval_mode: str,
    top_k: int,
    latency_ms: float,
    retrieved_ids: List[str],
    gold_ids: List[str],
    answer: str,
    evidence: List[Dict],
) -> Dict[str, Any]:
    """Compute metrics, append a row to the log CSV, and return metrics dict."""
    ensure_logfile(log_path)

    p5 = precision_at_k(retrieved_ids, gold_ids, k=5)
    r10 = recall_at_k(retrieved_ids, gold_ids, k=10)
    faithful = faithfulness_heuristic(answer, evidence)
    meb = missing_evidence_behavior(answer, evidence)

    row = [
        datetime.now(timezone.utc).isoformat(),
        query_id,
        retrieval_mode,
        top_k,
        round(latency_ms, 2),
        p5 if p5 is not None else "",
        r10 if r10 is not None else "",
        json.dumps(retrieved_ids),
        json.dumps(gold_ids),
        "Yes" if faithful else "No",
        meb,
    ]
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return {
        "Precision@5": p5,
        "Recall@10": r10,
        "faithfulness_pass": faithful,
        "missing_evidence_behavior": meb,
    }


# ── Gold Set ──────────────────────────────────────────────────────────────────

MINI_GOLD = {
    # ── Q1  (typical project query – doc1: SRSNet) ───────────────────────────
    "Q1": {
        "question": (
            "What is the Selective Representation Space (SRS) module in SRSNet "
            "and how does it differ from conventional adjacent patching?"
        ),
        "gold_evidence_ids": [
            "doc1_TimerSeries.pdf",   # main paper text describing SRS
            "figure8.png",            # Figure 2 – overall SRS pipeline
            "figure9.png",            # Figure 3 – detailed architecture
        ],
        "answer_criteria": [
            "Explains that SRS uses Selective Patching (gradient-based, learnable patch selection with stride=1 scanning) "
            "instead of fixed-length adjacent patching",
            "Mentions the Dynamic Reassembly step that re-orders the selected patches based on learned scores",
            "Notes the Adaptive Fusion that integrates embeddings from both conventional and selective patching",
            "Includes at least one citation to a document or figure",
        ],
        "citation_format": "[doc1_TimerSeries.pdf] or [figure8.png] / [figure9.png]",
    },

    # ── Q2  (typical project query – doc2: ReMindRAG) ────────────────────────
    "Q2": {
        "question": (
            "How does ReMindRAG's memory replay mechanism improve retrieval "
            "for similar or repeated queries?"
        ),
        "gold_evidence_ids": [
            "doc2_ReMindRAG.pdf",     # main paper text
            "figure3.png",            # Figure 1 – overall workflow showing memorize/replay
            "figure7.png",            # Figure 5 – memory replay under Same Query setting
        ],
        "answer_criteria": [
            "Describes the enhance/penalize edge-weight update after the first query",
            "Explains that on a subsequent similar/same query the system reuses the memorized traversal path "
            "(skipping full LLM-guided KG traversal)",
            "Includes at least one citation",
        ],
        "citation_format": "[doc2_ReMindRAG.pdf] or [figure3.png] / [figure7.png]",
    },

    # ── Q3  (typical project query – doc3: Consensus Planning Problem) ───────
    "Q3": {
        "question": (
            "What real-world applications of the Consensus Planning Problem (CPP) "
            "are described, and what agent interfaces does each application use?"
        ),
        "gold_evidence_ids": [
            "doc3_CPP.pdf",           # main paper text
            "figure2.png",            # Table 1 – examples of consensus problems
        ],
        "answer_criteria": [
            "Lists at least three applications (Fullness Optimization, Throughput Coordination, "
            "Transportation Optimization, Arrivals & Throughput Coordination)",
            "States the interface type (primal / dual / proximal) used by the agents in each application",
            "Includes a citation to Table 1 or the document",
        ],
        "citation_format": "[doc3_CPP.pdf] or [figure2.png, Table 1]",
    },

    # ── Q4  (multimodal / table-heavy query – doc2 Table 1 + doc1 Table 2) ──
    "Q4": {
        "question": (
            "According to Table 1 in the ReMindRAG paper, what is the Multi-Hop QA "
            "accuracy of ReMindRAG with the Deepseek-V3 backbone, and how does it "
            "compare to HippoRAG2 on the same task and backbone?"
        ),
        "gold_evidence_ids": [
            "doc2_ReMindRAG.pdf",
            "figure6.png",            # Table 1 – Effectiveness Performance
        ],
        "answer_criteria": [
            "Extracts the correct numeric value for ReMindRAG Multi-Hop QA / Deepseek-V3: 79.38%",
            "Extracts HippoRAG2 Multi-Hop QA / Deepseek-V3: 64.95%",
            "Notes that ReMindRAG outperforms HippoRAG2 by ~14.43 percentage points",
            "Includes a citation to Table 1 or figure6",
        ],
        "citation_format": "[doc2_ReMindRAG.pdf, Table 1] or [figure6.png]",
    },

    # ── Q5  (missing-evidence / ambiguous query – must trigger safe behavior)
    "Q5": {
        "question": (
            "What reinforcement learning reward function does SRSNet use to train "
            "the Selective Patching scorer?"
        ),
        "gold_evidence_ids": ["N/A"],  # SRSNet does NOT use RL; the scorer is gradient-based
        "answer_criteria": [
            "Returns a missing-evidence / 'insufficient information' response",
            "Does NOT hallucinate a reinforcement learning component – "
            "SRSNet's scorer is gradient-based (Gumbel-Softmax), not RL-based",
            "Optionally clarifies that the actual mechanism is gradient-based, citing the document",
        ],
        "citation_format": "",
    },

    # ── Q6  (multimodal query – image evidence via caption surrogate) ────────
    "Q6": {
        "question": (
            "What are the key stages shown in the ReMindRAG overall workflow diagram?"
        ),
        "gold_evidence_ids": ["img::figure3.png"],  # Figure 1 – ReMindRAG overall workflow
        "answer_criteria": [
            "Mentions KG construction from documents (Build stage: Document → Chunks → KG)",
            "Mentions LLM-guided KG traversal with seed node selection and path expansion",
            "Mentions the Enhance/Penalize edge-weight update (memory) after the first query",
            "Mentions the fast retrieval shortcut for subsequent similar/same queries",
        ],
        "citation_format": "[img::figure3.png]",
    },
}
