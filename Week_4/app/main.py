import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd

# ── Ensure project root is on sys.path ────────────────────────────────────────
# Works both locally (Week_4/app/main.py) and on Streamlit Cloud
_THIS_DIR = Path(__file__).resolve().parent          # .../Week_4/app
PROJECT_ROOT = _THIS_DIR.parent                       # .../Week_4
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import (
    MINI_GOLD,
    MISSING_EVIDENCE_MSG,
    build_context,
    build_retrievers,
    extractive_answer,
    generate_llm_answer,
    generate_local_llm_answer,
    load_corpus,
    log_query,
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CS5542 Lab 4 — Project RAG App", layout="wide")
st.title("CS 5542 Lab 4 — Project RAG Application")
st.caption("Project-aligned Streamlit UI + automatic logging + failure monitoring")

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.header("Retrieval Settings")
retrieval_mode = st.sidebar.selectbox(
    "retrieval_mode",
    ["tfidf", "dense", "sparse_BM25", "hybrid_BM25&TF-IDF", "hybrid_rerank", "multimodal"],
)
top_k = st.sidebar.slider("top_k", min_value=1, max_value=30, value=10, step=1)
use_multimodal = st.sidebar.checkbox("use_multimodal", value=True)

st.sidebar.header("Answer Generation")
answer_mode = st.sidebar.selectbox(
    "answer_mode",
    ["extractive", "llm (Gemini)", "llm (Local)"],
    help="Extractive = heuristic overlap. Gemini = API call. Local = TinyLlama on device.",
)

# Gemini API key — from Streamlit secrets or manual input
gemini_key = ""
if answer_mode == "llm (Gemini)":
    # Try Streamlit secrets first
    try:
        gemini_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        gemini_key = st.sidebar.text_input(
            "Gemini API Key", type="password",
            help="Paste your Gemini API key. It won't be stored.",
        )

st.sidebar.header("Logging")
log_path = st.sidebar.text_input("log file", value="logs/query_metrics.csv")

st.sidebar.header("Evaluation")
query_id = st.sidebar.selectbox("query_id (for logging)", list(MINI_GOLD.keys()))
use_gold_question = st.sidebar.checkbox("Use the gold-set question text", value=True)

# ── Load Corpus & Build All Retrievers (cached) ──────────────────────────────

@st.cache_resource(show_spinner="Loading corpus and building retrieval indices …")
def get_retrievers():
    docs_path = PROJECT_ROOT / "data" / "docs"
    images_path = PROJECT_ROOT / "data" / "images"

    if not docs_path.exists():
        st.error(f"❌ Critical Error: The directory `{docs_path}` does not exist.")
        st.info(f"Current working directory: {os.getcwd()}")
        st.info(f"Contents of {PROJECT_ROOT}: {list(PROJECT_ROOT.glob('*'))}")
        st.stop()

    evidence = load_corpus(str(docs_path), str(images_path))
    retrievers = build_retrievers(evidence)
    return evidence, retrievers


evidence_store, all_retrievers = get_retrievers()
st.sidebar.info(f"📚 Corpus: **{len(evidence_store)}** items · Modes: {list(all_retrievers.keys())}")

# ── Main Query ────────────────────────────────────────────────────────────────
default_q = MINI_GOLD[query_id]["question"] if use_gold_question else ""
question = st.text_area("Enter your question", value=default_q, height=120)
run_btn = st.button("Run Query")

colA, colB = st.columns([2, 1])

if run_btn and question.strip():
    t0 = time.time()

    # 1. Retrieve using selected mode
    retriever = all_retrievers.get(retrieval_mode, all_retrievers["tfidf"])
    hits = retriever.retrieve(question, top_k=top_k)

    # 2. Format evidence
    evidence_results = []
    hit_indices = []
    for idx, score in hits:
        item = evidence_store[idx]
        evidence_results.append({
            "chunk_id": item["chunk_id"],
            "citation_tag": f"[{item['chunk_id']}]",
            "score": round(score, 4),
            "source": item.get("source", ""),
            "text": item.get("text", "")[:500],
        })
        hit_indices.append(idx)

    # 3. Generate answer
    context = build_context(evidence_store, hit_indices)

    if answer_mode == "llm (Gemini)":
        with st.spinner("🤖 Calling Gemini API..."):
            answer = generate_llm_answer(question, context, api_key=gemini_key or None)
    elif answer_mode == "llm (Local)":
        with st.spinner("🤖 Running local LLM (Flan-T5-Small)..."):
            answer = generate_local_llm_answer(question, context)
    else:
        answer = extractive_answer(question, context)

    latency_ms = round((time.time() - t0) * 1000, 2)

    # 4. Evaluation
    retrieved_ids = [e["chunk_id"] for e in evidence_results]
    gold_ids = MINI_GOLD[query_id].get("gold_evidence_ids", [])

    # 5. Log
    metrics = log_query(
        log_path=log_path,
        query_id=query_id,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        latency_ms=latency_ms,
        retrieved_ids=retrieved_ids,
        gold_ids=gold_ids,
        answer=answer,
        evidence=evidence_results,
    )

    # ── Display ───────────────────────────────────────────────────────────
    with colA:
        st.subheader("Answer")
        if answer_mode == "llm (Gemini)":
            st.markdown(answer)
        else:
            st.write(answer)

        st.subheader("Evidence (Top-K)")
        for ev in evidence_results:
            with st.expander(f"{ev['citation_tag']}  —  score {ev['score']}"):
                st.write(ev["text"])

    with colB:
        st.subheader("Metrics")
        st.metric("Latency (ms)", latency_ms)
        st.caption(f"Answer mode: **{answer_mode}** · Retrieval: **{retrieval_mode}**")
        col1, col2 = st.columns(2)
        with col1:
            p5_val = metrics["Precision@5"]
            st.metric("Precision@5", f"{p5_val:.2f}" if p5_val is not None else "N/A")
        with col2:
            r10_val = metrics["Recall@10"]
            st.metric("Recall@10", f"{r10_val:.2f}" if r10_val is not None else "N/A")
        st.write({
            "faithfulness_pass": "Yes" if metrics["faithfulness_pass"] else "No",
            "missing_evidence_behavior": metrics["missing_evidence_behavior"],
        })

    st.success(f"Logged {query_id} to {log_path}")
