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
from dataclasses import dataclass

import numpy as np
import fitz  # PyMuPDF

try:
    from PIL import Image
    import pytesseract
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False
    print("WARNING: PIL/pytesseract not installed. Image OCR disabled.")


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

@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    page_num: int
    text: str

@dataclass
class ImageItem:
    item_id: str
    path: str
    caption: str  # simple text to make image retrieval runnable

def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_pdf_pages(pdf_path: str) -> List[TextChunk]:
    doc_id = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    out: List[TextChunk] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = clean_text(page.get_text("text"))
        if text:
            out.append(TextChunk(
                chunk_id=f"{doc_id}::p{i+1}",
                doc_id=doc_id,
                page_num=i+1,
                text=text
            ))
    return out


def load_images(images_dir: str) -> List[ImageItem]:
    items: List[ImageItem] = []
    if not os.path.isdir(images_dir):
        return items
    print(f"Scanning images in {images_dir} with OCR...")

    for p in sorted(glob.glob(os.path.join(images_dir, "*.*"))):
        base = os.path.basename(p)
        if not base.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            continue

        # 1. Generate Caption (Filename based)
        simple_caption = os.path.splitext(base)[0].replace("_", " ")

        # 2. Run OCR (Tesseract) to get text inside the image
        ocr_text = ""
        if _HAS_OCR:
            try:
                image = Image.open(p)
                ocr_text = pytesseract.image_to_string(image).strip()
                ocr_text = re.sub(r"\s+", " ", ocr_text)
            except Exception as e:
                print(f"OCR Failed for {base}: {e}")

        final_text = f"Caption: {simple_caption}. Content: {ocr_text}"

        items.append(ImageItem(item_id=base, path=p, caption=final_text))

    return items


def load_corpus(
    docs_dir: str = DEFAULT_DOCS_DIR,
    images_dir: str = DEFAULT_IMAGES_DIR,
) -> List[Dict[str, Any]]:
    """Load the full corpus (PDFs + images) and return a flat evidence list.

    Converts TextChunk / ImageItem dataclasses into the dict format
    expected by all downstream retrievers and the Streamlit UI.
    """
    evidence: List[Dict[str, Any]] = []

    # PDFs → TextChunk → dict
    for pdf_path in sorted(glob.glob(os.path.join(docs_dir, "*.pdf"))):
        for chunk in extract_pdf_pages(pdf_path):
            evidence.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": pdf_path,
                "page": chunk.page_num,
                "modality": "text",
            })

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

    # Images → ImageItem → dict
    for img in load_images(images_dir):
        evidence.append({
            "chunk_id": f"img::{img.item_id}",
            "text": img.caption,
            "source": img.path,
            "modality": "image",
        })

    return evidence


# ── TF-IDF Retriever ──────────────────────────────────────────────────────────

class TfidfRetriever:
    """TF-IDF retriever with L2-normalized cosine similarity.

    Uses normalized sparse matrix multiplication (X @ q.T) which is
    equivalent to cosine similarity but faster for sparse matrices.
    """

    def __init__(self, evidence: List[Dict[str, Any]]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        self.evidence = evidence
        texts = [item.get("text", "") for item in evidence]
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        X = self.vectorizer.fit_transform(texts)
        self.matrix = normalize(X)  # L2-normalize rows

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return list of (index, score) tuples, descending by score."""
        from sklearn.preprocessing import normalize
        q = self.vectorizer.transform([query])
        q = normalize(q)  # L2-normalize query
        scores = (self.matrix @ q.T).toarray().ravel()
        idxs = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idxs]


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
    """Dense retriever with separate FAISS L2 indices for text and image captions.

    Builds two indices:
      - ``index_text``  — embeddings of text evidence (PDF pages, txt files)
      - ``index_caption`` — embeddings of image caption/OCR text

    At query time both indices are searched and results are merged,
    sorted by L2 distance, then converted to a similarity score.
    """

    def __init__(self, evidence: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.evidence = evidence
        self.model = SentenceTransformer(model_name)

        # Partition evidence into text vs image by global index
        text_indices = []
        text_corpus = []
        img_indices = []
        img_corpus = []

        for i, item in enumerate(evidence):
            if item.get("modality") == "image":
                img_indices.append(i)
                img_corpus.append(item.get("text", ""))
            else:
                text_indices.append(i)
                text_corpus.append(item.get("text", ""))

        self._text_idx_map = text_indices   # local → global index
        self._img_idx_map = img_indices

        # Build text FAISS index
        text_emb = self.model.encode(text_corpus, show_progress_bar=False)
        d = text_emb.shape[1]
        self.index_text = faiss.IndexFlatL2(d)
        self.index_text.add(text_emb)
        print(f"✅ Dense text index built: {self.index_text.ntotal} vectors, dim={d}")

        # Build image caption FAISS index
        if img_corpus:
            cap_emb = self.model.encode(img_corpus, show_progress_bar=False)
            d_cap = cap_emb.shape[1]
            self.index_caption = faiss.IndexFlatL2(d_cap)
            self.index_caption.add(cap_emb)
            print(f"✅ Dense caption index built: {self.index_caption.ntotal} images via OCR/caption")
        else:
            self.index_caption = None

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_emb = self.model.encode([query])

        merged: List[Tuple[int, float]] = []

        # Search text index
        k_text = min(top_k, self.index_text.ntotal)
        if k_text > 0:
            dists_t, idxs_t = self.index_text.search(query_emb, k_text)
            for local_idx, dist in zip(idxs_t[0], dists_t[0]):
                if local_idx < 0:
                    continue
                global_idx = self._text_idx_map[int(local_idx)]
                sim = 1.0 / (1.0 + float(dist))
                merged.append((global_idx, sim))

        # Search image caption index
        if self.index_caption is not None:
            k_img = min(top_k, self.index_caption.ntotal)
            if k_img > 0:
                dists_c, idxs_c = self.index_caption.search(query_emb, k_img)
                for local_idx, dist in zip(idxs_c[0], dists_c[0]):
                    if local_idx < 0:
                        continue
                    global_idx = self._img_idx_map[int(local_idx)]
                    sim = 1.0 / (1.0 + float(dist))
                    merged.append((global_idx, sim))

        # Sort by similarity (descending) and return top_k
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:top_k]


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
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Truncate context to stay within T5's 512-token limit
        max_context_tokens = 400  # leave room for question + answer tokens
        tokenized_ctx = tokenizer(context, truncation=False, return_tensors="pt")["input_ids"]
        if tokenized_ctx.shape[1] > max_context_tokens:
            tokenized_ctx = tokenized_ctx[:, :max_context_tokens]
            context = tokenizer.decode(tokenized_ctx[0], skip_special_tokens=True)

        prompt = (
            f"Answer the question based on the context below. "
            f"If the answer is not in the context, say 'Not enough evidence.'\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
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


# ── LLM-as-a-Judge ───────────────────────────────────────────────────────────

def llm_judge(
    question: str,
    answer: str,
    evidence_texts: List[str],
    answer_criteria: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """Use an LLM (Gemini) to grade a RAG answer against gold criteria.

    Returns a dict with keys:
        relevance      (1-5) — does the answer address the question?
        completeness   (1-5) — does it cover all required criteria?
        citation_quality (1-5) — are sources cited and correct?
        faithfulness   (1-5) — is the answer grounded in the evidence?
        overall        (1-5) — mean of the four axes
        feedback       (str) — brief textual explanation from the judge
        error          (str|None) — set only on failure
    """
    # Resolve API key
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {
            "relevance": None, "completeness": None,
            "citation_quality": None, "faithfulness": None,
            "overall": None, "feedback": "No GEMINI_API_KEY available.",
            "error": "missing_api_key",
        }

    criteria_text = "\n".join(f"  - {c}" for c in answer_criteria) if answer_criteria else "  (no specific criteria provided)"
    evidence_text = "\n---\n".join(evidence_texts[:5]) if evidence_texts else "(no evidence provided)"

    prompt = f"""You are an impartial expert evaluator for a Retrieval-Augmented Generation (RAG) system.

TASK: Grade the ANSWER on a 1-5 scale for each axis below.

QUESTION:
{question}

GOLD CRITERIA (what a perfect answer should cover):
{criteria_text}

RETRIEVED EVIDENCE (the context the system had):
{evidence_text}

SYSTEM ANSWER:
{answer}

GRADING AXES (1 = very poor, 5 = excellent):
1. **relevance**: Does the answer address the question asked?
2. **completeness**: Does the answer cover all the gold criteria listed above?
3. **citation_quality**: Does the answer cite sources and are those citations traceable to the evidence?
4. **faithfulness**: Is the answer grounded in the provided evidence (no hallucinations)?

RESPOND IN EXACTLY THIS FORMAT (no extra text):
relevance: <1-5>
completeness: <1-5>
citation_quality: <1-5>
faithfulness: <1-5>
feedback: <one-sentence explanation>
"""

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse structured response
        scores: Dict[str, Any] = {}
        feedback_line = ""
        for line in text.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            if key in ("relevance", "completeness", "citation_quality", "faithfulness"):
                try:
                    scores[key] = int(val[0])  # first digit
                except (ValueError, IndexError):
                    scores[key] = None
            elif key == "feedback":
                feedback_line = val

        # Compute overall
        valid_scores = [v for v in scores.values() if isinstance(v, (int, float))]
        overall = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None

        return {
            "relevance": scores.get("relevance"),
            "completeness": scores.get("completeness"),
            "citation_quality": scores.get("citation_quality"),
            "faithfulness": scores.get("faithfulness"),
            "overall": overall,
            "feedback": feedback_line or text[:200],
            "error": None,
        }
    except Exception as e:
        return {
            "relevance": None, "completeness": None,
            "citation_quality": None, "faithfulness": None,
            "overall": None, "feedback": str(e),
            "error": str(e),
        }


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


# ── Batch Evaluation ─────────────────────────────────────────────────────────

def batch_evaluate(
    evidence: List[Dict[str, Any]],
    retrievers: Dict[str, Any],
    gold: Dict[str, Dict],
    top_k: int = 10,
    log_path: str = "logs/query_metrics.csv",
    answer_fn=None,
) -> List[Dict[str, Any]]:
    """Run every gold query across every retrieval mode and return results.

    Parameters
    ----------
    evidence : list of evidence dicts (from load_corpus)
    retrievers : dict of mode_name -> retriever (from build_retrievers)
    gold : dict mapping query_id -> {question, gold_evidence_ids, ...}
    top_k : int
    log_path : str — CSV file for logging each run
    answer_fn : callable(question, context) -> str, defaults to extractive_answer

    Returns a list of dicts with keys: query_id, mode, P@5, R@10, latency_ms,
    faithfulness_pass, answer (truncated).
    """
    if answer_fn is None:
        answer_fn = extractive_answer

    results = []
    for qid, qobj in gold.items():
        question = qobj["question"]
        gold_ids = qobj.get("gold_evidence_ids", [])

        for mode_name, retriever in retrievers.items():
            t0 = time.time()

            # Retrieve
            hits = retriever.retrieve(question, top_k=top_k)
            hit_indices = [idx for idx, _ in hits]
            retrieved_ids = [evidence[idx]["chunk_id"] for idx, _ in hits]

            # Build context & generate answer
            context = build_context(evidence, hit_indices)
            answer = answer_fn(question, context)

            latency_ms = round((time.time() - t0) * 1000, 2)

            # Log to CSV
            evidence_results = [
                {"chunk_id": evidence[idx]["chunk_id"],
                 "text": evidence[idx].get("text", "")[:200]}
                for idx, _ in hits
            ]
            metrics = log_query(
                log_path=log_path,
                query_id=qid,
                retrieval_mode=mode_name,
                top_k=top_k,
                latency_ms=latency_ms,
                retrieved_ids=retrieved_ids,
                gold_ids=gold_ids,
                answer=answer,
                evidence=evidence_results,
            )

            results.append({
                "query_id": qid,
                "mode": mode_name,
                "Precision@5": metrics["Precision@5"],
                "Recall@10": metrics["Recall@10"],
                "latency_ms": latency_ms,
                "faithfulness": "Yes" if metrics["faithfulness_pass"] else "No",
                "answer": answer[:150] + "…" if len(answer) > 150 else answer,
            })

    return results


# ── Gold Set ──────────────────────────────────────────────────────────────────

MINI_GOLD = {
    # ── Q1  (typical project query – doc1: SRSNet) ───────────────────────────
    "Q1": {
        "question": (
            "What is the Selective Representation Space (SRS) module in SRSNet "
            "and how does it differ from conventional adjacent patching?"
        ),
        "gold_evidence_ids": [
            "doc1_TimeSeries.pdf",    # main paper text describing SRS
            "img::figure8.png",       # Figure 2 – overall SRS pipeline
            "img::figure9.png",       # Figure 3 – detailed architecture
        ],
        "answer_criteria": [
            "Explains that SRS uses Selective Patching (gradient-based, learnable patch selection with stride=1 scanning) "
            "instead of fixed-length adjacent patching",
            "Mentions the Dynamic Reassembly step that re-orders the selected patches based on learned scores",
            "Notes the Adaptive Fusion that integrates embeddings from both conventional and selective patching",
            "Includes at least one citation to a document or figure",
        ],
        "citation_format": "[doc1_TimeSeries.pdf] or [img::figure8.png] / [img::figure9.png]",
    },

    # ── Q2  (typical project query – doc2: ReMindRAG) ────────────────────────
    "Q2": {
        "question": (
            "How does ReMindRAG's memory replay mechanism improve retrieval "
            "for similar or repeated queries?"
        ),
        "gold_evidence_ids": [
            "doc2_ReMindRAG.pdf",     # main paper text
            "img::figure3.png",       # Figure 1 – overall workflow showing memorize/replay
            "img::figure7.png",       # Figure 5 – memory replay under Same Query setting
        ],
        "answer_criteria": [
            "Describes the enhance/penalize edge-weight update after the first query",
            "Explains that on a subsequent similar/same query the system reuses the memorized traversal path "
            "(skipping full LLM-guided KG traversal)",
            "Includes at least one citation",
        ],
        "citation_format": "[doc2_ReMindRAG.pdf] or [img::figure3.png] / [img::figure7.png]",
    },

    # ── Q3  (typical project query – doc3: Consensus Planning Problem) ───────
    "Q3": {
        "question": (
            "What real-world applications of the Consensus Planning Problem (CPP) "
            "are described, and what agent interfaces does each application use?"
        ),
        "gold_evidence_ids": [
            "doc3_CPP.pdf",           # main paper text
            "img::figure2.png",       # Table 1 – examples of consensus problems
        ],
        "answer_criteria": [
            "Lists at least three applications (Fullness Optimization, Throughput Coordination, "
            "Transportation Optimization, Arrivals & Throughput Coordination)",
            "States the interface type (primal / dual / proximal) used by the agents in each application",
            "Includes a citation to Table 1 or the document",
        ],
        "citation_format": "[doc3_CPP.pdf] or [img::figure2.png, Table 1]",
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
            "img::figure6.png",       # Table 1 – Effectiveness Performance
        ],
        "answer_criteria": [
            "Extracts the correct numeric value for ReMindRAG Multi-Hop QA / Deepseek-V3: 79.38%",
            "Extracts HippoRAG2 Multi-Hop QA / Deepseek-V3: 64.95%",
            "Notes that ReMindRAG outperforms HippoRAG2 by ~14.43 percentage points",
            "Includes a citation to Table 1 or figure6",
        ],
        "citation_format": "[doc2_ReMindRAG.pdf, Table 1] or [img::figure6.png]",
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

