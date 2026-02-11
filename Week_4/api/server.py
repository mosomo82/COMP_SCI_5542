"""
CS5542 Lab 4 — FastAPI RAG Backend
====================================
Provides ``POST /query`` endpoint backed by the shared pipeline.
"""

import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import (
    MISSING_EVIDENCE_MSG,
    TfidfRetriever,
    build_context,
    extractive_answer,
    load_corpus,
)

# ── Global State ──────────────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "evidence": [],
    "retriever": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load corpus and build index on startup."""
    docs_dir = str(PROJECT_ROOT / "data" / "docs")
    images_dir = str(PROJECT_ROOT / "data" / "images")
    evidence = load_corpus(docs_dir, images_dir)
    _state["evidence"] = evidence
    if evidence:
        _state["retriever"] = TfidfRetriever(evidence)
        print(f"Server ready: indexed {len(evidence)} evidence items.")
    else:
        print("WARNING: No documents found. Server has empty index.")
    yield
    # Shutdown: nothing to clean up


app = FastAPI(title="CS5542 Lab 4 RAG Backend", lifespan=lifespan)


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryIn(BaseModel):
    question: str
    top_k: int = 10
    retrieval_mode: str = "hybrid"
    use_multimodal: bool = True


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/query")
def query(q: QueryIn) -> Dict[str, Any]:
    retriever = _state["retriever"]
    evidence = _state["evidence"]

    if retriever is None:
        return {
            "answer": MISSING_EVIDENCE_MSG,
            "evidence": [],
            "metrics": {"top_k": q.top_k, "retrieval_mode": q.retrieval_mode},
            "failure_flag": True,
        }

    t0 = time.time()

    # 1. Retrieve
    hits = retriever.retrieve(q.question, top_k=q.top_k)

    # 2. Format evidence
    evidence_list = []
    hit_indices = []
    for idx, score in hits:
        item = evidence[idx]
        evidence_list.append({
            "chunk_id": item["chunk_id"],
            "citation_tag": f"[{item['chunk_id']}]",
            "score": round(score, 4),
            "source": item.get("source", ""),
            "text": item.get("text", "")[:500],
        })
        hit_indices.append(idx)

    # 3. Generate answer
    context = build_context(evidence, hit_indices)
    answer = extractive_answer(q.question, context)
    latency_ms = round((time.time() - t0) * 1000, 2)

    return {
        "answer": answer,
        "evidence": evidence_list,
        "metrics": {
            "top_k": q.top_k,
            "retrieval_mode": q.retrieval_mode,
            "latency_ms": latency_ms,
        },
        "failure_flag": answer == MISSING_EVIDENCE_MSG,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
