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
    """Simple TF-IDF retriever over a list of evidence dicts."""

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


# ── Context Building ──────────────────────────────────────────────────────────

def build_context(
    evidence: List[Dict[str, Any]],
    hit_indices: List[int],
    max_chars: int = 2000,
) -> str:
    """Join retrieved evidence into a context string with citation tags."""
    parts = []
    total = 0
    for idx in hit_indices:
        item = evidence[idx]
        tag = f"[{item['chunk_id']}]"
        entry = f"{tag} {item.get('text', '')}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


# ── Answer Generation ─────────────────────────────────────────────────────────

def extractive_answer(query: str, context: str) -> str:
    """Heuristic extractive answer: return top-3 sentences most
    overlapping with the query terms."""
    if not context or not context.strip():
        return MISSING_EVIDENCE_MSG

    q_words = set(re.findall(r"[A-Za-z]+", query.lower()))
    sents = re.split(r"(?<=[.!?])\s+", context.strip())

    scored = []
    for s in sents:
        w = set(re.findall(r"[A-Za-z]+", s.lower()))
        score = len(q_words & w)
        if score > 0:
            scored.append((score, s.strip()))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for _, s in scored[:3]]
    return " ".join(best) if best else MISSING_EVIDENCE_MSG


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
    "Q1": {
        "question": (
            "What is the Selective Representation Space (SRS) module in SRSNet "
            "and how does it differ from conventional adjacent patching?"
        ),
        "gold_evidence_ids": ["doc1_TimeSeries.pdf", "figure8.png", "figure9.png"],
        "rubric": {
            "must_have_keywords": [
                "selective patching", "SRS", "representation space",
                "dynamic reassembly",
            ],
        },
    },
    "Q2": {
        "question": (
            "How does ReMindRAG's memory replay mechanism improve retrieval "
            "for similar or repeated queries?"
        ),
        "gold_evidence_ids": ["doc2_ReMindRAG.pdf", "figure3.png", "figure7.png"],
        "rubric": {
            "must_have_keywords": [
                "memory replay", "edge-weight", "traversal path",
            ],
        },
    },
    "Q3": {
        "question": (
            "What real-world applications of the Consensus Planning Problem "
            "are described, and what agent interfaces does each application use?"
        ),
        "gold_evidence_ids": ["doc3_CPP.pdf", "figure2.png"],
        "rubric": {
            "must_have_keywords": [
                "consensus", "planning problem", "primal", "dual",
            ],
        },
    },
    "Q4": {
        "question": (
            "According to Table 1 in the ReMindRAG paper, what is the Multi-Hop "
            "QA accuracy of ReMindRAG with the Deepseek-V3 backbone compared to "
            "HippoRAG2?"
        ),
        "gold_evidence_ids": ["doc2_ReMindRAG.pdf", "figure6.png"],
        "rubric": {
            "must_have_keywords": [
                "79.38", "64.95", "Multi-Hop", "Deepseek",
            ],
        },
    },
    "Q5": {
        "question": (
            "What reinforcement learning reward function does SRSNet use to "
            "train the Selective Patching scorer?"
        ),
        "gold_evidence_ids": ["N/A"],
        "rubric": {
            "must_have_keywords": [],  # should trigger missing-evidence
        },
    },
    "Q6": {
        "question": (
            "What are the key stages shown in the ReMindRAG overall workflow diagram?"
        ),
        "gold_evidence_ids": ["img::figure3.png"],
        "rubric": {
            "must_have_keywords": [
                "KG construction", "traversal", "enhance", "penalize",
            ],
        },
    },
}
