import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd

# ── Ensure project root is on sys.path ────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent          # .../Week_4/app
PROJECT_ROOT = _THIS_DIR.parent                       # .../Week_4
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

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CS5542 Lab 4 — Project RAG App", layout="wide")
st.title("CS 5542 Lab 4 — Project RAG Application")
st.caption("Project-aligned Streamlit UI + automatic logging + evaluation dashboard")

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.header("Retrieval Settings")
retrieval_mode = st.sidebar.selectbox(
    "retrieval_mode",
    ["tfidf", "dense", "sparse_BM25", "hybrid_BM25&TF-IDF", "hybrid_rerank"],
)
top_k = st.sidebar.slider("top_k", min_value=1, max_value=30, value=10, step=1)
use_multimodal = st.sidebar.checkbox("use_multimodal", value=True)

st.sidebar.header("Answer Generation")
answer_mode = st.sidebar.selectbox(
    "answer_mode",
    ["extractive", "llm (Gemini)", "llm (Local)"],
    help="Extractive = heuristic overlap. Gemini = API call. Local = Flan-T5-Small on device.",
)

# Gemini API key — from Streamlit secrets or manual input
gemini_key = ""
if answer_mode == "llm (Gemini)":
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
log_path_input = st.sidebar.text_input("log file", value="logs/query_metrics.csv")
# Resolve to absolute so both tabs use the same path
_lp = Path(log_path_input)
log_path = str(_lp if _lp.is_absolute() else PROJECT_ROOT / _lp)

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

# ── Metadata Filters ──────────────────────────────────────────────────────────
st.sidebar.header("Metadata Filters")

# Extract unique sources (basename) and modalities from the corpus
_all_sources = sorted({os.path.basename(e.get("source", "unknown")) for e in evidence_store})
_all_modalities = sorted({e.get("modality", "text") for e in evidence_store})

selected_sources = st.sidebar.multiselect(
    "Source documents", _all_sources, default=_all_sources,
    help="Filter evidence to only include items from selected source files.",
)
selected_modalities = st.sidebar.multiselect(
    "Modality", _all_modalities, default=_all_modalities,
    help="Filter by evidence type (text pages, images, tables, etc.).",
)

# ── Response Caching ──────────────────────────────────────────────────────────
st.sidebar.header("Cache")
if st.sidebar.button("🗑️ Clear Response Cache"):
    _cached_query.clear()
    st.sidebar.success("Cache cleared!")


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_query(
    _question: str,
    _retrieval_mode: str,
    _answer_mode: str,
    _top_k: int,
    _gemini_key: str,
    _sources_key: str,
    _modalities_key: str,
):
    """Cache query results keyed on (question, mode, top_k, filters).

    Identical queries return instantly without re-running retrieval + LLM.
    """
    # Parse filter keys back to sets
    allowed_sources = set(_sources_key.split("|")) if _sources_key else set()
    allowed_modalities = set(_modalities_key.split("|")) if _modalities_key else set()

    retriever = all_retrievers.get(_retrieval_mode, all_retrievers["tfidf"])
    # Over-fetch then filter so we still get top_k results after filtering
    hits = retriever.retrieve(_question, top_k=_top_k * 3)

    evidence_results = []
    hit_indices = []
    for idx, score in hits:
        item = evidence_store[idx]
        src_base = os.path.basename(item.get("source", ""))
        modality = item.get("modality", "text")
        # Apply metadata filters
        if src_base not in allowed_sources:
            continue
        if modality not in allowed_modalities:
            continue
        evidence_results.append({
            "chunk_id": item["chunk_id"],
            "citation_tag": f"[{item['chunk_id']}]",
            "score": round(score, 4),
            "source": item.get("source", ""),
            "text": item.get("text", "")[:500],
        })
        hit_indices.append(idx)
        if len(evidence_results) >= _top_k:
            break

    context = build_context(evidence_store, hit_indices)

    if _answer_mode == "llm (Gemini)":
        answer = generate_llm_answer(_question, context, api_key=_gemini_key or None)
    elif _answer_mode == "llm (Local)":
        answer = generate_local_llm_answer(_question, context)
    else:
        answer = extractive_answer(_question, context)

    return {
        "evidence_results": evidence_results,
        "hit_indices": hit_indices,
        "answer": answer,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_query, tab_dashboard = st.tabs(["🔍 Query", "📊 Evaluation Dashboard"])

# ── TAB 1: Query Interface ────────────────────────────────────────────────────
with tab_query:
    default_q = MINI_GOLD[query_id]["question"] if use_gold_question else ""
    question = st.text_area("Enter your question", value=default_q, height=120)
    run_btn = st.button("Run Query")

    colA, colB = st.columns([2, 1])

    if run_btn and question.strip():
        t0 = time.time()

        # Use cached query (dedup identical calls)
        with st.spinner(f"⏳ Querying ({retrieval_mode} + {answer_mode}) …"):
            sources_key = "|".join(sorted(selected_sources))
            modalities_key = "|".join(sorted(selected_modalities))
            result = _cached_query(
                question, retrieval_mode, answer_mode, top_k,
                gemini_key or "", sources_key, modalities_key,
            )

        evidence_results = result["evidence_results"]
        answer = result["answer"]

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

        # ── Display ───────────────────────────────────────────────────────
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


# ── TAB 2: Evaluation Dashboard ──────────────────────────────────────────────
with tab_dashboard:
    st.subheader("📊 Evaluation Dashboard")
    st.caption("Visualize logged query metrics across retrieval modes")

    # ── Batch Evaluation Button ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Batch Evaluation")
    st.caption("Run all 6 gold queries × all retrieval modes in one click")

    batch_col1, batch_col2 = st.columns([1, 2])
    with batch_col1:
        batch_btn = st.button("▶ Run Batch Evaluation", type="primary")

    if batch_btn:
        with st.spinner(f"Running {len(MINI_GOLD)} queries × {len(all_retrievers)} modes …"):
            results = batch_evaluate(
                evidence=evidence_store,
                retrievers=all_retrievers,
                gold=MINI_GOLD,
                top_k=top_k,
                log_path=log_path,
            )
        batch_df = pd.DataFrame(results)
        st.success(f"✅ Batch complete — {len(results)} runs logged to `{log_path}`")

        # Pivot: rows=query, cols=mode, values=P@5
        st.markdown("**Precision@5 — Query × Mode**")
        pivot_p5 = batch_df.pivot_table(
            index="query_id", columns="mode",
            values="Precision@5", aggfunc="mean"
        ).round(3)
        st.dataframe(pivot_p5, use_container_width=True)

        # Pivot: rows=query, cols=mode, values=R@10
        st.markdown("**Recall@10 — Query × Mode**")
        pivot_r10 = batch_df.pivot_table(
            index="query_id", columns="mode",
            values="Recall@10", aggfunc="mean"
        ).round(3)
        st.dataframe(pivot_r10, use_container_width=True)

        # Summary by mode
        st.markdown("**Average Metrics by Mode**")
        mode_summary = batch_df.groupby("mode").agg(
            Avg_P5=("Precision@5", "mean"),
            Avg_R10=("Recall@10", "mean"),
            Avg_Latency_ms=("latency_ms", "mean"),
        ).round(3)
        st.dataframe(mode_summary, use_container_width=True)

        # Full results table
        with st.expander("📋 All Batch Results"):
            st.dataframe(batch_df, use_container_width=True)

    st.markdown("---")

    # ── Load log CSV ──────────────────────────────────────────────────────
    log_file = Path(log_path)

    if not log_file.exists():
        st.info("📭 No log data yet. Click **▶ Run Batch Evaluation** above or run queries in the **Query** tab!")
    else:
        df = pd.read_csv(log_file)

        if df.empty:
            st.info("Log file is empty. Run some queries to populate data.")
        else:
            # Clean up numeric columns
            for col in ["Precision@5", "Recall@10", "latency_ms"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            st.metric("Total Logged Queries", len(df))

            # ── 1. Metrics by Retrieval Mode ──────────────────────────────
            st.markdown("---")
            st.subheader("📈 Metrics by Retrieval Mode")

            if "retrieval_mode" in df.columns:
                mode_stats = df.groupby("retrieval_mode").agg(
                    Queries=("query_id", "count"),
                    Avg_P5=("Precision@5", "mean"),
                    Avg_R10=("Recall@10", "mean"),
                    Avg_Latency_ms=("latency_ms", "mean"),
                    Faithfulness_Rate=("faithfulness_pass", lambda x: (x == "Yes").mean()),
                ).round(3)

                st.dataframe(mode_stats, use_container_width=True)

                col_p5, col_r10 = st.columns(2)
                with col_p5:
                    st.markdown("**Avg Precision@5 by Mode**")
                    st.bar_chart(mode_stats["Avg_P5"])
                with col_r10:
                    st.markdown("**Avg Recall@10 by Mode**")
                    st.bar_chart(mode_stats["Avg_R10"])

                st.markdown("**Avg Latency (ms) by Mode**")
                st.bar_chart(mode_stats["Avg_Latency_ms"])

            # ── 2. Per-Query Breakdown ────────────────────────────────────
            st.markdown("---")
            st.subheader("🔎 Per-Query Breakdown")

            if "query_id" in df.columns:
                query_stats = df.groupby(["query_id", "retrieval_mode"]).agg(
                    Avg_P5=("Precision@5", "mean"),
                    Avg_R10=("Recall@10", "mean"),
                    Avg_Latency_ms=("latency_ms", "mean"),
                ).round(3)

                st.dataframe(query_stats, use_container_width=True)

                if len(df["retrieval_mode"].unique()) > 1:
                    st.markdown("**Precision@5 — Query × Mode**")
                    pivot = df.pivot_table(
                        index="query_id", columns="retrieval_mode",
                        values="Precision@5", aggfunc="mean"
                    ).round(3)
                    st.dataframe(pivot, use_container_width=True)

            # ── 3. Latency Distribution ───────────────────────────────────
            st.markdown("---")
            st.subheader("⏱️ Latency Distribution")

            if "latency_ms" in df.columns:
                col_hist, col_stats = st.columns([2, 1])
                with col_hist:
                    st.line_chart(df.set_index(df.index)["latency_ms"])
                with col_stats:
                    st.metric("Min", f"{df['latency_ms'].min():.1f} ms")
                    st.metric("Median", f"{df['latency_ms'].median():.1f} ms")
                    st.metric("Max", f"{df['latency_ms'].max():.1f} ms")
                    st.metric("Mean", f"{df['latency_ms'].mean():.1f} ms")

            # ── 4. Failure Analysis ───────────────────────────────────────
            st.markdown("---")
            st.subheader("⚠️ Failure Analysis")

            if "faithfulness_pass" in df.columns:
                faith_counts = df["faithfulness_pass"].value_counts()
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.markdown("**Faithfulness Distribution**")
                    st.bar_chart(faith_counts)
                with col_f2:
                    fail_rows = df[df["faithfulness_pass"] == "No"]
                    st.metric("Failed Queries", len(fail_rows))
                    if not fail_rows.empty:
                        st.dataframe(
                            fail_rows[["query_id", "retrieval_mode", "Precision@5"]],
                            use_container_width=True,
                        )

            # ── 5. Raw Log ────────────────────────────────────────────────
            st.markdown("---")
            with st.expander("📋 Raw Log Data"):
                st.dataframe(df, use_container_width=True)
                csv_data = df.to_csv(index=False)
                st.download_button("⬇️ Download CSV", csv_data, "query_metrics.csv", "text/csv")

