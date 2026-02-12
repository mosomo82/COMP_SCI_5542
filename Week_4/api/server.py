"""
CS5542 Lab 4 — FastAPI RAG Backend
====================================
Full-featured REST API backed by the shared pipeline.

Endpoints:
    POST /query          — run a single retrieval + answer query
    POST /batch_evaluate — run all gold queries across all modes
    GET  /modes          — list available retrieval & answer modes
    GET  /health         — health check
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import (
    MINI_GOLD,
    MISSING_EVIDENCE_MSG,
    batch_evaluate,
    build_context,
    build_retrievers,
    extractive_answer,
    generate_llm_answer,
    generate_local_llm_answer,
    load_corpus,
    log_query,
)

# ── Global State ──────────────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "evidence": [],
    "retrievers": {},
}

DEFAULT_LOG_PATH = str(PROJECT_ROOT / "logs" / "query_metrics.csv")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load corpus and build all retriever variants on startup."""
    docs_dir = str(PROJECT_ROOT / "data" / "docs")
    images_dir = str(PROJECT_ROOT / "data" / "images")

    evidence = load_corpus(docs_dir, images_dir)
    _state["evidence"] = evidence

    if evidence:
        _state["retrievers"] = build_retrievers(evidence)
        modes = list(_state["retrievers"].keys())
        print(f"Server ready: {len(evidence)} evidence items, modes={modes}")
    else:
        print("WARNING: No documents found. Server has empty index.")
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="CS5542 Lab 4 RAG Backend",
    description="Project-aligned RAG API with 5 retrieval modes, LLM answer generation, and batch evaluation.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryIn(BaseModel):
    question: str
    top_k: int = 10
    retrieval_mode: str = "tfidf"
    answer_mode: str = "extractive"  # "extractive", "llm_gemini", "llm_local"
    gemini_api_key: Optional[str] = None
    query_id: Optional[str] = None  # for evaluation logging
    sources_filter: Optional[List[str]] = None  # metadata filter
    modalities_filter: Optional[List[str]] = None  # metadata filter


class BatchIn(BaseModel):
    top_k: int = 10
    log_path: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "corpus_size": len(_state["evidence"]),
        "modes": list(_state["retrievers"].keys()),
    }


@app.get("/modes")
def modes():
    """List available retrieval and answer modes."""
    return {
        "retrieval_modes": list(_state["retrievers"].keys()),
        "answer_modes": ["extractive", "llm_gemini", "llm_local"],
    }


@app.post("/query")
def query(q: QueryIn) -> Dict[str, Any]:
    """Run a single retrieval + answer query."""
    retrievers = _state["retrievers"]
    evidence = _state["evidence"]

    if not retrievers:
        return {
            "answer": MISSING_EVIDENCE_MSG,
            "evidence": [],
            "metrics": {"top_k": q.top_k, "retrieval_mode": q.retrieval_mode},
            "failure_flag": True,
        }

    retriever = retrievers.get(q.retrieval_mode, retrievers.get("tfidf"))
    if retriever is None:
        return {
            "error": f"Unknown retrieval mode: {q.retrieval_mode}",
            "available_modes": list(retrievers.keys()),
        }

    t0 = time.time()

    # 1. Retrieve (over-fetch for filtering)
    fetch_k = q.top_k * 3 if (q.sources_filter or q.modalities_filter) else q.top_k
    hits = retriever.retrieve(q.question, top_k=fetch_k)

    # 2. Apply metadata filters
    evidence_list = []
    hit_indices = []
    for idx, score in hits:
        item = evidence[idx]

        # Source filter
        if q.sources_filter:
            src_base = os.path.basename(item.get("source", ""))
            if src_base not in q.sources_filter:
                continue

        # Modality filter
        if q.modalities_filter:
            modality = item.get("modality", "text")
            if modality not in q.modalities_filter:
                continue

        evidence_list.append({
            "chunk_id": item["chunk_id"],
            "citation_tag": f"[{item['chunk_id']}]",
            "score": round(score, 4),
            "source": item.get("source", ""),
            "text": item.get("text", "")[:500],
            "modality": item.get("modality", "text"),
        })
        hit_indices.append(idx)
        if len(evidence_list) >= q.top_k:
            break

    # 3. Build context and generate answer
    context = build_context(evidence, hit_indices)

    if q.answer_mode == "llm_gemini":
        answer = generate_llm_answer(
            q.question, context, api_key=q.gemini_api_key or None,
        )
    elif q.answer_mode == "llm_local":
        answer = generate_local_llm_answer(q.question, context)
    else:
        answer = extractive_answer(q.question, context)

    latency_ms = round((time.time() - t0) * 1000, 2)

    # 4. Log metrics (if query_id provided)
    metrics = {}
    if q.query_id and q.query_id in MINI_GOLD:
        gold = MINI_GOLD[q.query_id]
        retrieved_ids = [item["chunk_id"] for item in evidence_list]
        metrics = log_query(
            log_path=DEFAULT_LOG_PATH,
            query_id=q.query_id,
            retrieval_mode=q.retrieval_mode,
            top_k=q.top_k,
            latency_ms=latency_ms,
            retrieved_ids=retrieved_ids,
            gold_ids=gold["gold_evidence_ids"],
            answer=answer,
            evidence=evidence,
        )

    return {
        "answer": answer,
        "evidence": evidence_list,
        "metrics": {
            "top_k": q.top_k,
            "retrieval_mode": q.retrieval_mode,
            "answer_mode": q.answer_mode,
            "latency_ms": latency_ms,
            **metrics,
        },
        "failure_flag": answer == MISSING_EVIDENCE_MSG,
    }


@app.post("/batch_evaluate")
def batch_eval(b: BatchIn) -> Dict[str, Any]:
    """Run all gold queries across all retrieval modes."""
    evidence = _state["evidence"]
    retrievers = _state["retrievers"]

    if not retrievers:
        return {"error": "No retrievers available. Corpus may be empty."}

    log_path = b.log_path or DEFAULT_LOG_PATH

    results = batch_evaluate(
        evidence=evidence,
        retrievers=retrievers,
        gold=MINI_GOLD,
        top_k=b.top_k,
        log_path=log_path,
    )

    # Compute summary by mode
    mode_summary: Dict[str, Dict[str, Any]] = {}
    for r in results:
        mode = r["mode"]
        if mode not in mode_summary:
            mode_summary[mode] = {"p5_vals": [], "r10_vals": [], "latency_vals": []}
        if r.get("Precision@5") is not None:
            mode_summary[mode]["p5_vals"].append(r["Precision@5"])
        if r.get("Recall@10") is not None:
            mode_summary[mode]["r10_vals"].append(r["Recall@10"])
        mode_summary[mode]["latency_vals"].append(r.get("latency_ms", 0))

    summary = {}
    for mode, vals in mode_summary.items():
        summary[mode] = {
            "avg_precision_5": round(sum(vals["p5_vals"]) / max(len(vals["p5_vals"]), 1), 4),
            "avg_recall_10": round(sum(vals["r10_vals"]) / max(len(vals["r10_vals"]), 1), 4),
            "avg_latency_ms": round(sum(vals["latency_vals"]) / max(len(vals["latency_vals"]), 1), 2),
        }

    return {
        "total_runs": len(results),
        "queries": len(MINI_GOLD),
        "modes": list(retrievers.keys()),
        "summary_by_mode": summary,
        "results": results,
        "log_path": log_path,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
