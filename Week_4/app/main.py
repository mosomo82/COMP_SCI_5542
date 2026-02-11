
import json, time
from pathlib import Path
import streamlit as st
import pandas as pd

# --- Import your team pipeline here ---
# from rag.pipeline import retrieve, generate_answer, run_query_and_log

MISSING_EVIDENCE_MSG = "Not enough evidence in the retrieved context."

st.set_page_config(page_title="CS5542 Lab 4 — Project RAG App", layout="wide")
st.title("CS 5542 Lab 4 — Project RAG Application")
st.caption("Project-aligned Streamlit UI + automatic logging + failure monitoring")

# Sidebar controls
st.sidebar.header("Retrieval Settings")
retrieval_mode = st.sidebar.selectbox("retrieval_mode", ["tfidf", "dense", "sparse", "hybrid", "hybrid_rerank"])
top_k = st.sidebar.slider("top_k", min_value=1, max_value=30, value=10, step=1)
use_multimodal = st.sidebar.checkbox("use_multimodal", value=True)

st.sidebar.header("Logging")
log_path = st.sidebar.text_input("log file", value="logs/query_metrics.csv")

# --- Mini gold set (replace with your team's Q1–Q5) ---
# Tip: keep the same structure as in your Lab 4 notebook so IDs match logs.
MINI_GOLD = {
    "Q1": {"question": What is the Selective Representation Space (SRS) module in SRSNet and how does it differ from conventional adjacent patching?", "gold_evidence_ids": ['doc1_TimerSeries.pdf']
    },
    "Q2": {"question": "How does ReMindRAG\'s memory replay mechanism improve retrieval for similar or repeated queries?", "gold_evidence_ids": ['doc2_ReMindRAG.pdf']},
    "Q3": {"question": "What real-world applications of the Consensus Planning Problem are described, and what agent interfaces does each application use?", "gold_evidence_ids": ['doc3_CPP.pdf']},
    "Q4": {"question": "According to Table 1 in the ReMindRAG paper, what is the Multi-Hop QA accuracy of ReMindRAG with the Deepseek-V3 backbone compared to HippoRAG2?", "gold_evidence_ids": ['img::figure6.png']},
    "Q5": {"question": "What reinforcement learning reward function does SRSNet use to train the Selective Patching scorer?", "gold_evidence_ids": ['N/A']},
}

st.sidebar.header("Evaluation")
query_id = st.sidebar.selectbox("query_id (for logging)", list(MINI_GOLD.keys()))
use_gold_question = st.sidebar.checkbox("Use the gold-set question text", value=True)

# Main query
default_q = MINI_GOLD[query_id]["question"] if use_gold_question else ""
question = st.text_area("Enter your question", value=default_q, height=120)
run_btn = st.button("Run Query")

colA, colB = st.columns([2, 1])

def ensure_logfile(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        df = pd.DataFrame(columns=[
            "timestamp","query_id","retrieval_mode","top_k","latency_ms",
            "Precision@5","Recall@10","evidence_ids_returned","gold_evidence_ids",
            "faithfulness_pass","missing_evidence_behavior"
        ])
        df.to_csv(p, index=False)

def precision_at_k(retrieved_ids, gold_ids, k=5):
    if not gold_ids:
        return None
    topk = retrieved_ids[:k]
    hits = sum(1 for x in topk if x in set(gold_ids))
    return hits / k

def recall_at_k(retrieved_ids, gold_ids, k=10):
    if not gold_ids:
        return None
    topk = retrieved_ids[:k]
    hits = sum(1 for x in topk if x in set(gold_ids))
    return hits / max(1, len(gold_ids))

# ---- Placeholder demo logic (replace with imports from your /rag module) ----
def retrieve_demo(q: str, top_k: int):
    return [{"chunk_id":"demo_doc","citation_tag":"[demo_doc]","score":0.9,"source":"data/docs/demo_doc.txt","text":"demo evidence..."}]

def answer_demo(q: str, evidence: list):
    if not evidence:
        return MISSING_EVIDENCE_MSG
    return f"Grounded answer using {evidence[0]['citation_tag']} {evidence[0]['citation_tag']}"

def log_row(path: str, row: dict):
    ensure_logfile(path)
    df = pd.read_csv(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
# --------------------------------------------------------------------------

if run_btn and question.strip():
    t0 = time.time()
    evidence = retrieve_demo(question, top_k=top_k)
    answer = answer_demo(question, evidence)
    latency_ms = round((time.time() - t0)*1000, 2)

    retrieved_ids = [e["chunk_id"] for e in evidence]
    gold_ids = MINI_GOLD[query_id].get("gold_evidence_ids", [])

    p5 = precision_at_k(retrieved_ids, gold_ids, k=5)
    r10 = recall_at_k(retrieved_ids, gold_ids, k=10)

    with colA:
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Evidence (Top-K)")
        st.json(evidence)

    with colB:
        st.subheader("Metrics")
        st.write({"latency_ms": latency_ms, "Precision@5": p5, "Recall@10": r10})

    # Log the query using the selected Q1–Q5 ID (not ad-hoc)
    row = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "query_id": query_id,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "latency_ms": latency_ms,
        "Precision@5": p5,
        "Recall@10": r10,
        "evidence_ids_returned": json.dumps(retrieved_ids),
        "gold_evidence_ids": json.dumps(gold_ids),
        "faithfulness_pass": "Yes" if answer != MISSING_EVIDENCE_MSG else "Yes",
        "missing_evidence_behavior": "Pass"  # update with your rule if needed
    }
    log_row(log_path, row)
    st.success(f"Logged {query_id} to CSV.")
