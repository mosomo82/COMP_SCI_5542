
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="CS5542 Lab 4 RAG Backend")

MISSING_EVIDENCE_MSG = "Not enough evidence in the retrieved context."

class QueryIn(BaseModel):
    question: str
    top_k: int = 10
    retrieval_mode: str = "hybrid"
    use_multimodal: bool = True

@app.post("/query")
def query(q: QueryIn) -> Dict[str, Any]:
    # TODO: import your real pipeline:
    # evidence = retrieve(q.question, top_k=q.top_k, mode=q.retrieval_mode, use_multimodal=q.use_multimodal)
    # answer = generate_answer(q.question, evidence)
    evidence = [{"chunk_id":"demo_doc","citation_tag":"[demo_doc]","score":0.9,"source":"data/docs/demo_doc.txt","text":"demo evidence..."}]
    answer = f"Grounded answer using {evidence[0]['citation_tag']} {evidence[0]['citation_tag']}"
    return {
        "answer": answer,
        "evidence": evidence,
        "metrics": {"top_k": q.top_k, "retrieval_mode": q.retrieval_mode},
        "failure_flag": False
    }
