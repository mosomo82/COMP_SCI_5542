import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

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
    llm_judge,
    load_corpus,
    log_query,
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CS5542 Lab 4 — Project RAG App", layout="wide")
st.title("CS 5542 Lab 4 — Project RAG Application")
st.caption("Project-aligned Streamlit UI + automatic logging + evaluation dashboard")

# ── Sidebar Controls ──────────────────────────────────────────────────────────
ALL_MODES = ["tfidf", "dense", "sparse_BM25", "hybrid_BM25&TF-IDF", "hybrid_rerank"]

st.sidebar.header("Retrieval Settings")
retrieval_mode = st.sidebar.selectbox(
    "retrieval_mode",
    ALL_MODES,
)
top_k = st.sidebar.slider("top_k", min_value=1, max_value=30, value=10, step=1)
use_multimodal = st.sidebar.checkbox("use_multimodal", value=True)

# ── Advanced Retrieval Controls ────────────────────────────────────────────────
st.sidebar.header("⚙️ Advanced Retrieval")

embedding_model = st.sidebar.selectbox(
    "Embedding Model (dense / hybrid / rerank)",
    [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "paraphrase-MiniLM-L6-v2",
    ],
    help="Sentence-Transformer model used for dense vector search. "
         "Changing this will re-build the FAISS index.",
)

chunk_size = st.sidebar.slider(
    "Chunk size (chars)", 0, 2000, 0, step=100,
    help="0 = full-page chunks (default). >0 splits each page into sub-chunks of this character length.",
)

chunk_overlap = st.sidebar.slider(
    "Chunk overlap (chars)", 0, 500, 0, step=50,
    help="Number of overlapping characters between consecutive sub-chunks.",
)

# Comparison mode
st.sidebar.header("⚔️ Comparison Mode")
comparison_enabled = st.sidebar.checkbox("Enable side-by-side comparison", value=False)
retrieval_mode_b = retrieval_mode  # default same
if comparison_enabled:
    other_modes = [m for m in ALL_MODES if m != retrieval_mode]
    retrieval_mode_b = st.sidebar.selectbox(
        "Compare with",
        other_modes,
        help="Select a second retrieval mode to compare against.",
    )

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
def get_retrievers(_embedding_model: str, _chunk_size: int, _chunk_overlap: int):
    docs_path = PROJECT_ROOT / "data" / "docs"
    images_path = PROJECT_ROOT / "data" / "images"

    if not docs_path.exists():
        st.error(f"❌ Critical Error: The directory `{docs_path}` does not exist.")
        st.info(f"Current working directory: {os.getcwd()}")
        st.info(f"Contents of {PROJECT_ROOT}: {list(PROJECT_ROOT.glob('*'))}")
        st.stop()

    evidence = load_corpus(
        str(docs_path), str(images_path),
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
    )
    retrievers = build_retrievers(evidence, embedding_model=_embedding_model)
    return evidence, retrievers


evidence_store, all_retrievers = get_retrievers(embedding_model, chunk_size, chunk_overlap)
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


# Now that _cached_query is defined, add the clear button
if st.sidebar.button("🗑️ Clear Response Cache"):
    _cached_query.clear()
    st.sidebar.success("Cache cleared!")


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_GOLD_PATH = str(PROJECT_ROOT / "data" / "gold_set.json")

tab_query, tab_dashboard, tab_goldset = st.tabs(
    ["🔍 Query", "📊 Evaluation Dashboard", "🏆 Gold Set"]
)


# ── Helper: run a single retrieval+answer and display in a column ─────────────
def _run_and_display(col, mode_label: str, ret_mode: str):
    """Execute query and render results inside the given column."""
    with col:
        st.markdown(f"### 🏷️ `{ret_mode}`")
        t0 = time.time()
        with st.spinner(f"⏳ {ret_mode} + {answer_mode} …"):
            sources_key = "|".join(sorted(selected_sources))
            modalities_key = "|".join(sorted(selected_modalities))
            result = _cached_query(
                question, ret_mode, answer_mode, top_k,
                gemini_key or "", sources_key, modalities_key,
            )
        latency_ms = round((time.time() - t0) * 1000, 2)

        evidence_results = result["evidence_results"]
        answer = result["answer"]

        # Visual Debug: Show exactly what the #1 retrieved item was
        if evidence_results:
            top_hit = evidence_results[0]
            top_id = top_hit["chunk_id"]
            top_score = top_hit["score"]
            
            # Color-code based on score to see confidence differences
            if top_score > 0.5:
                st.success(f"🥇 **Top Hit:** `{top_id}` (Score: {top_score:.4f})")
            else:
                st.warning(f"🥇 **Top Hit:** `{top_id}` (Low Score: {top_score:.4f})")
        else:
            st.error("❌ No evidence retrieved.")
        # ----------------------

        retrieved_ids = [e["chunk_id"] for e in evidence_results]
        gold_ids = MINI_GOLD[query_id].get("gold_evidence_ids", [])

        metrics = log_query(
            log_path=log_path,
            query_id=query_id,
            retrieval_mode=ret_mode,
            top_k=top_k,
            latency_ms=latency_ms,
            retrieved_ids=retrieved_ids,
            gold_ids=gold_ids,
            answer=answer,
            evidence=evidence_results,
        )

        # Answer
        st.subheader("Answer")
        if answer_mode == "llm (Gemini)":
            st.markdown(answer)
        else:
            st.write(answer)

        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Latency", f"{latency_ms:.0f} ms")
        with m2:
            p5 = metrics["Precision@5"]
            st.metric("P@5", f"{p5:.2f}" if p5 is not None else "N/A")
        with m3:
            r10 = metrics["Recall@10"]
            st.metric("R@10", f"{r10:.2f}" if r10 is not None else "N/A")

        faith = "Yes" if metrics["faithfulness_pass"] else "No"
        st.caption(f"Faithfulness: **{faith}** · Missing-evidence: **{metrics['missing_evidence_behavior']}**")

        # Evidence
        st.subheader(f"Evidence (Top-{top_k})")
        for ev in evidence_results:
            with st.expander(f"{ev['citation_tag']}  —  score {ev['score']}"):
                st.write(ev["text"])

        # LLM Judge grading
        judge_key = f"judge_{ret_mode}_{hash(question)}"
        if st.button(f"🤖 Grade with LLM Judge", key=judge_key):
            gold_criteria = MINI_GOLD.get(query_id, {}).get("answer_criteria", [])
            ev_texts = [ev.get("text", "") for ev in evidence_results]
            with st.spinner("🧑‍⚖️ LLM Judge is grading …"):
                verdict = llm_judge(
                    question=question,
                    answer=answer,
                    evidence_texts=ev_texts,
                    answer_criteria=gold_criteria,
                    api_key=gemini_key or None,
                )
            if verdict.get("error"):
                st.warning(f"Judge error: {verdict['error']}")
            else:
                jc1, jc2, jc3, jc4, jc5 = st.columns(5)
                for jcol, axis in zip(
                    [jc1, jc2, jc3, jc4, jc5],
                    ["relevance", "completeness", "citation_quality", "faithfulness", "overall"],
                ):
                    with jcol:
                        v = verdict.get(axis)
                        label = axis.replace("_", " ").title()
                        st.metric(label, f"{v}/5" if v is not None else "N/A")
                st.info(f"💬 **Feedback**: {verdict.get('feedback', '')}")

    return {
        "latency_ms": latency_ms,
        "Precision@5": metrics["Precision@5"],
        "Recall@10": metrics["Recall@10"],
        "faithfulness": faith,
        "mode": ret_mode,
        "answer": answer,
        "evidence_results": evidence_results,
    }


# ── TAB 1: Query Interface ────────────────────────────────────────────────────
with tab_query:
    default_q = MINI_GOLD[query_id]["question"] if use_gold_question else ""
    question = st.text_area("Enter your question", value=default_q, height=120)

    if comparison_enabled:
        st.info(f"⚔️ **Comparison Mode**: `{retrieval_mode}` vs `{retrieval_mode_b}`")

    run_btn = st.button("Run Query")

    if run_btn and question.strip():
        if comparison_enabled:
            # ── Side-by-side comparison ────────────────────────────────────
            col_left, col_right = st.columns(2)
            stats_a = _run_and_display(col_left, "Method A", retrieval_mode)
            stats_b = _run_and_display(col_right, "Method B", retrieval_mode_b)

            # ── Comparison summary ────────────────────────────────────────
            st.markdown("---")
            st.subheader(f"📊 Comparison: `{retrieval_mode}` vs `{retrieval_mode_b}` (top_k={top_k})")

            cmp1, cmp2 = st.columns(2)
            with cmp1:
                # Delta metrics
                lat_a, lat_b = stats_a["latency_ms"], stats_b["latency_ms"]
                st.metric(
                    f"Latency: {retrieval_mode}",
                    f"{lat_a:.0f} ms",
                    delta=f"{lat_a - lat_b:.0f} ms vs {retrieval_mode_b}",
                    delta_color="inverse",  # lower is better
                )
                for metric_name in ["Precision@5", "Recall@10"]:
                    va = stats_a[metric_name]
                    vb = stats_b[metric_name]
                    if va is not None and vb is not None:
                        st.metric(
                            f"{metric_name}: {retrieval_mode}",
                            f"{va:.3f}",
                            delta=f"{va - vb:+.3f} vs {retrieval_mode_b}",
                        )

            with cmp2:
                # Radar chart
                import plotly.graph_objects as go
                categories = ["Precision@5", "Recall@10", "Speed (1/latency)"]
                def _safe(v, default=0.0):
                    return float(v) if v is not None else default
                vals_a = [
                    _safe(stats_a["Precision@5"]),
                    _safe(stats_a["Recall@10"]),
                    1000.0 / max(stats_a["latency_ms"], 1),
                ]
                vals_b = [
                    _safe(stats_b["Precision@5"]),
                    _safe(stats_b["Recall@10"]),
                    1000.0 / max(stats_b["latency_ms"], 1),
                ]
                # Normalize speed to 0-1 range for radar
                max_speed = max(vals_a[2], vals_b[2], 0.001)
                vals_a[2] /= max_speed
                vals_b[2] /= max_speed

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_a + [vals_a[0]], theta=categories + [categories[0]],
                    fill="toself", name=retrieval_mode, opacity=0.6,
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_b + [vals_b[0]], theta=categories + [categories[0]],
                    fill="toself", name=retrieval_mode_b, opacity=0.6,
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=f"Radar: {retrieval_mode} vs {retrieval_mode_b}",
                    showlegend=True,
                    height=350,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.success(f"Logged both runs ({query_id}) to {log_path}")

        else:
            # ── Single query mode (original) ──────────────────────────────
            colA, colB = st.columns([2, 1])
            stats = _run_and_display(colA, "Result", retrieval_mode)

            with colB:
                st.subheader("Metrics")
                st.metric("Latency (ms)", f"{stats['latency_ms']:.0f}")
                st.caption(f"Answer mode: **{answer_mode}** · Retrieval: **{retrieval_mode}**")
                m1, m2 = st.columns(2)
                with m1:
                    p5 = stats["Precision@5"]
                    st.metric("Precision@5", f"{p5:.2f}" if p5 is not None else "N/A")
                with m2:
                    r10 = stats["Recall@10"]
                    st.metric("Recall@10", f"{r10:.2f}" if r10 is not None else "N/A")
                st.write({
                    "faithfulness_pass": stats["faithfulness"],
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

    # ── LLM-as-a-Judge Batch Grading ──────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 LLM-as-a-Judge — Automated Grading")
    st.caption(
        "Grade all gold-set answers using Gemini as an impartial judge. "
        "Requires a Gemini API key (set in sidebar or environment)."
    )

    judge_api_key = gemini_key or os.environ.get("GEMINI_API_KEY", "")
    if not judge_api_key:
        st.warning("⚠️ No Gemini API key available. Set one in the sidebar under *Answer Generation*.")
    else:
        if st.button("▶ Run LLM Judge on All Gold Queries"):
            judge_results = []
            progress = st.progress(0, text="Grading …")
            gold_items = list(MINI_GOLD.items())

            for i, (qid, qobj) in enumerate(gold_items):
                question = qobj["question"]
                criteria = qobj.get("answer_criteria", [])

                # Use the primary retriever to get an answer to grade
                retriever = all_retrievers.get(retrieval_mode, list(all_retrievers.values())[0])
                hits = retriever.retrieve(question, top_k=top_k)
                hit_indices = [idx for idx, _ in hits]
                context = build_context(evidence_store, hit_indices)

                if answer_mode == "llm (Gemini)":
                    ans = generate_llm_answer(question, context, api_key=judge_api_key)
                elif answer_mode == "llm (Local)":
                    from rag.pipeline import generate_local_llm_answer as _gen_local
                    ans = _gen_local(question, context)
                else:
                    ans = extractive_answer(question, context)

                ev_texts = [evidence_store[idx].get("text", "")[:300] for idx in hit_indices[:5]]

                verdict = llm_judge(
                    question=question,
                    answer=ans,
                    evidence_texts=ev_texts,
                    answer_criteria=criteria,
                    api_key=judge_api_key,
                )
                verdict["query_id"] = qid
                verdict["retrieval_mode"] = retrieval_mode
                verdict["answer_preview"] = ans[:100] + "…" if len(ans) > 100 else ans
                judge_results.append(verdict)
                progress.progress((i + 1) / len(gold_items), text=f"Grading {qid} …")

            progress.empty()
            st.success(f"✅ Graded {len(judge_results)} queries")

            judge_df = pd.DataFrame(judge_results)

            # Results table
            display_cols = ["query_id", "retrieval_mode", "relevance", "completeness",
                            "citation_quality", "faithfulness", "overall", "feedback"]
            avail_cols = [c for c in display_cols if c in judge_df.columns]
            st.dataframe(judge_df[avail_cols], use_container_width=True)

            # Averages
            score_axes = ["relevance", "completeness", "citation_quality", "faithfulness", "overall"]
            avail_axes = [a for a in score_axes if a in judge_df.columns]

            if avail_axes:
                avg_scores = judge_df[avail_axes].mean().round(2)

                # Bar chart
                avg_df = avg_scores.reset_index()
                avg_df.columns = ["Axis", "Avg Score"]
                fig_judge = px.bar(
                    avg_df, x="Axis", y="Avg Score",
                    color="Axis",
                    title=f"LLM Judge — Avg Scores ({retrieval_mode}, top_k={top_k})",
                    text_auto=".2f",
                    range_y=[0, 5],
                )
                fig_judge.update_layout(showlegend=False)
                st.plotly_chart(fig_judge, use_container_width=True)

                # Radar per query
                import plotly.graph_objects as go
                radar_axes = [a for a in ["relevance", "completeness", "citation_quality", "faithfulness"] if a in judge_df.columns]
                if len(radar_axes) >= 3:
                    fig_j_radar = go.Figure()
                    for _, row in judge_df.iterrows():
                        vals = [float(row[a]) if pd.notna(row.get(a)) else 0 for a in radar_axes]
                        # Normalize to 0-1 for radar
                        vals_norm = [v / 5.0 for v in vals]
                        fig_j_radar.add_trace(go.Scatterpolar(
                            r=vals_norm + [vals_norm[0]],
                            theta=radar_axes + [radar_axes[0]],
                            fill="toself",
                            name=str(row.get("query_id", "")),
                            opacity=0.5,
                        ))
                    fig_j_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title="LLM Judge — Per-Query Radar",
                        showlegend=True,
                        height=450,
                    )
                    st.plotly_chart(fig_j_radar, use_container_width=True)

            # Download
            csv_judge = judge_df.to_csv(index=False)
            st.download_button("⬇️ Download Judge Results", csv_judge, "llm_judge_results.csv", "text/csv")

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
                    fig_p5 = px.bar(
                        mode_stats.reset_index(),
                        x="retrieval_mode", y="Avg_P5",
                        color="retrieval_mode",
                        title=f"Avg Precision@5 by Mode (top_k={top_k})",
                        labels={"Avg_P5": "Precision@5", "retrieval_mode": "Mode"},
                        text_auto=".3f",
                    )
                    fig_p5.update_layout(showlegend=False)
                    st.plotly_chart(fig_p5, use_container_width=True)
                with col_r10:
                    fig_r10 = px.bar(
                        mode_stats.reset_index(),
                        x="retrieval_mode", y="Avg_R10",
                        color="retrieval_mode",
                        title=f"Avg Recall@10 by Mode (top_k={top_k})",
                        labels={"Avg_R10": "Recall@10", "retrieval_mode": "Mode"},
                        text_auto=".3f",
                    )
                    fig_r10.update_layout(showlegend=False)
                    st.plotly_chart(fig_r10, use_container_width=True)

                fig_lat = px.bar(
                    mode_stats.reset_index(),
                    x="retrieval_mode", y="Avg_Latency_ms",
                    color="retrieval_mode",
                    title=f"Avg Latency (ms) by Mode (top_k={top_k})",
                    labels={"Avg_Latency_ms": "Latency (ms)", "retrieval_mode": "Mode"},
                    text_auto=".1f",
                )
                fig_lat.update_layout(showlegend=False)
                st.plotly_chart(fig_lat, use_container_width=True)

                # ── Radar: all modes across metrics ───────────────────────
                st.markdown("---")
                st.subheader("🕸️ Radar: All Modes Compared")

                import plotly.graph_objects as go

                radar_categories = ["Precision@5", "Recall@10", "Speed", "Faithfulness"]
                ms = mode_stats.reset_index()

                # Normalize latency → speed (0-1, higher is better)
                max_lat = ms["Avg_Latency_ms"].max()
                if max_lat > 0:
                    ms["Speed"] = 1.0 - (ms["Avg_Latency_ms"] / max_lat)
                else:
                    ms["Speed"] = 1.0

                fig_radar = go.Figure()
                for _, row in ms.iterrows():
                    vals = [
                        float(row["Avg_P5"]) if pd.notna(row["Avg_P5"]) else 0,
                        float(row["Avg_R10"]) if pd.notna(row["Avg_R10"]) else 0,
                        float(row["Speed"]),
                        float(row["Faithfulness_Rate"]) if pd.notna(row.get("Faithfulness_Rate", 0)) else 0,
                    ]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=radar_categories + [radar_categories[0]],
                        fill="toself",
                        name=str(row["retrieval_mode"]),
                        opacity=0.55,
                    ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=f"Method Comparison Radar (top_k={top_k}, {len(df)} logged queries)",
                    showlegend=True,
                    height=450,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

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
                    pivot = df.pivot_table(
                        index="query_id", columns="retrieval_mode",
                        values="Precision@5", aggfunc="mean"
                    ).round(3)
                    st.dataframe(pivot, use_container_width=True)

                    # Grouped bar chart: P@5 per query per mode
                    fig_pq = px.bar(
                        df.dropna(subset=["Precision@5"]),
                        x="query_id", y="Precision@5",
                        color="retrieval_mode", barmode="group",
                        title=f"Precision@5 per Query × Mode (top_k={top_k})",
                        labels={"query_id": "Query", "retrieval_mode": "Mode"},
                    )
                    st.plotly_chart(fig_pq, use_container_width=True)

            # ── 3. Latency Distribution ───────────────────────────────────
            st.markdown("---")
            st.subheader("⏱️ Latency Distribution")

            if "latency_ms" in df.columns:
                col_hist, col_stats = st.columns([2, 1])
                with col_hist:
                    fig_hist = px.histogram(
                        df, x="latency_ms",
                        color="retrieval_mode" if "retrieval_mode" in df.columns else None,
                        nbins=20,
                        title=f"Latency Distribution ({len(df)} queries, top_k={top_k})",
                        labels={"latency_ms": "Latency (ms)", "count": "Frequency"},
                        marginal="rug",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col_stats:
                    st.metric("Min", f"{df['latency_ms'].min():.1f} ms")
                    st.metric("Median", f"{df['latency_ms'].median():.1f} ms")
                    st.metric("Max", f"{df['latency_ms'].max():.1f} ms")
                    st.metric("Mean", f"{df['latency_ms'].mean():.1f} ms")

            # ── 4. Failure Analysis ───────────────────────────────────────
            st.markdown("---")
            st.subheader("⚠️ Failure Analysis")

            if "faithfulness_pass" in df.columns:
                faith_counts = df["faithfulness_pass"].value_counts().reset_index()
                faith_counts.columns = ["Result", "Count"]
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    fig_faith = px.pie(
                        faith_counts, names="Result", values="Count",
                        title=f"Faithfulness Distribution ({len(df)} queries)",
                        color="Result",
                        color_discrete_map={"Yes": "#2ecc71", "No": "#e74c3c"},
                        hole=0.4,
                    )
                    st.plotly_chart(fig_faith, use_container_width=True)
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


# ── TAB 3: Gold Set Management ────────────────────────────────────────────────
with tab_goldset:
    st.subheader("🏆 Gold Set Management")
    st.caption("View, edit, upload, or export the gold evaluation set used for scoring")

    # ── Helper: MINI_GOLD dict ↔ flat DataFrame ───────────────────────────
    def _gold_to_df(gold: dict) -> pd.DataFrame:
        """Convert MINI_GOLD dict to a flat DataFrame for editing."""
        rows = []
        for qid, entry in gold.items():
            rows.append({
                "query_id": qid,
                "question": entry.get("question", ""),
                "gold_evidence_ids": "; ".join(entry.get("gold_evidence_ids", [])),
                "answer_criteria": "; ".join(entry.get("answer_criteria", [])),
                "citation_format": entry.get("citation_format", ""),
            })
        return pd.DataFrame(rows)

    def _df_to_gold(df: pd.DataFrame) -> dict:
        """Convert flat DataFrame back to MINI_GOLD dict format."""
        gold = {}
        for _, row in df.iterrows():
            qid = str(row["query_id"]).strip()
            if not qid:
                continue
            gold[qid] = {
                "question": str(row.get("question", "")),
                "gold_evidence_ids": [
                    s.strip() for s in str(row.get("gold_evidence_ids", "")).split(";")
                    if s.strip()
                ],
                "answer_criteria": [
                    s.strip() for s in str(row.get("answer_criteria", "")).split(";")
                    if s.strip()
                ],
                "citation_format": str(row.get("citation_format", "")),
            }
        return gold

    # ── Current Gold Set (editable) ───────────────────────────────────────
    st.markdown("### ✏️ Edit Current Gold Set")
    st.caption(
        "Edit cells directly. Separate multiple evidence IDs or criteria with `;`. "
        "Add rows with the ➕ button below the table."
    )

    gold_df = _gold_to_df(MINI_GOLD)

    edited_df = st.data_editor(
        gold_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "query_id": st.column_config.TextColumn("Query ID", width="small"),
            "question": st.column_config.TextColumn("Question", width="large"),
            "gold_evidence_ids": st.column_config.TextColumn(
                "Gold Evidence IDs", width="medium",
                help="Separate multiple IDs with semicolons (;)",
            ),
            "answer_criteria": st.column_config.TextColumn(
                "Answer Criteria", width="large",
                help="Separate multiple criteria with semicolons (;)",
            ),
            "citation_format": st.column_config.TextColumn("Citation Format", width="medium"),
        },
        key="gold_editor",
    )

    save_col1, save_col2, save_col3 = st.columns(3)

    with save_col1:
        if st.button("💾 Apply to Session", type="primary",
                     help="Update the in-memory gold set for this session (no file write)"):
            updated_gold = _df_to_gold(edited_df)
            MINI_GOLD.clear()
            MINI_GOLD.update(updated_gold)
            st.success(f"✅ Session gold set updated — {len(MINI_GOLD)} queries")
            st.rerun()

    with save_col2:
        if st.button("💾 Save to Disk",
                     help=f"Write to {DEFAULT_GOLD_PATH}"):
            updated_gold = _df_to_gold(edited_df)
            MINI_GOLD.clear()
            MINI_GOLD.update(updated_gold)
            Path(DEFAULT_GOLD_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_GOLD_PATH, "w", encoding="utf-8") as f:
                json.dump(MINI_GOLD, f, indent=2, ensure_ascii=False)
            st.success(f"✅ Saved {len(MINI_GOLD)} queries to `{DEFAULT_GOLD_PATH}`")

    with save_col3:
        # Export current gold set
        export_json = json.dumps(MINI_GOLD, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇️ Export JSON", export_json, "gold_set.json", "application/json",
        )

    # ── Upload New Gold Set ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📤 Upload New Gold Set")
    st.caption("Upload a CSV or JSON file to replace or merge with the current gold set.")

    upload_mode = st.radio(
        "Upload mode",
        ["Replace entire gold set", "Merge (add/update queries, keep existing)"],
        horizontal=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV or JSON file",
        type=["csv", "json"],
        help=(
            "**CSV format**: columns `query_id`, `question`, `gold_evidence_ids` (;-separated), "
            "`answer_criteria` (;-separated), `citation_format`.\n\n"
            "**JSON format**: same structure as MINI_GOLD dict."
        ),
    )

    if uploaded_file is not None:
        try:
            fname = uploaded_file.name.lower()
            if fname.endswith(".json"):
                import io
                raw = json.load(io.TextIOWrapper(uploaded_file, encoding="utf-8"))
                if isinstance(raw, dict):
                    new_gold = raw
                elif isinstance(raw, list):
                    # List of dicts with query_id key
                    new_gold = {}
                    for item in raw:
                        qid = item.pop("query_id", None) or f"Q{len(new_gold)+1}"
                        new_gold[qid] = item
                else:
                    st.error("JSON must be a dict (keyed by query_id) or a list of objects.")
                    new_gold = None
            else:  # CSV
                csv_df = pd.read_csv(uploaded_file)
                new_gold = _df_to_gold(csv_df)

            if new_gold:
                st.success(f"✅ Parsed {len(new_gold)} queries from `{uploaded_file.name}`")

                # Preview
                with st.expander("👀 Preview uploaded gold set"):
                    st.dataframe(_gold_to_df(new_gold), use_container_width=True)

                if st.button("✅ Apply Upload", type="primary"):
                    if upload_mode == "Replace entire gold set":
                        MINI_GOLD.clear()
                    MINI_GOLD.update(new_gold)

                    # Also persist
                    Path(DEFAULT_GOLD_PATH).parent.mkdir(parents=True, exist_ok=True)
                    with open(DEFAULT_GOLD_PATH, "w", encoding="utf-8") as f:
                        json.dump(MINI_GOLD, f, indent=2, ensure_ascii=False)

                    st.success(
                        f"✅ Gold set {'replaced' if upload_mode.startswith('Replace') else 'merged'} "
                        f"— {len(MINI_GOLD)} total queries. Saved to `{DEFAULT_GOLD_PATH}`"
                    )
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Error parsing upload: {e}")

    # ── Log Editing ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📝 Edit Query Log")
    st.caption("Fix incorrect entries in the evaluation log and save changes back.")

    log_file = Path(log_path)
    if not log_file.exists():
        st.info("No log file found. Run queries first to create one.")
    else:
        log_df = pd.read_csv(log_file)
        if log_df.empty:
            st.info("Log file is empty.")
        else:
            edited_log = st.data_editor(
                log_df,
                num_rows="dynamic",
                use_container_width=True,
                key="log_editor",
            )

            if st.button("💾 Save Log Changes"):
                edited_log.to_csv(log_file, index=False)
                st.success(f"✅ Log saved to `{log_file}` ({len(edited_log)} rows)")
