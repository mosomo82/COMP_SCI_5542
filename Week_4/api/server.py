from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path
import time
import re

# Ensure project root is on sys.path
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from RAG pipeline
from rag.pipeline import (
    load_corpus,
    build_retrievers,
    build_context,
    extractive_answer,
    generate_llm_answer,
    generate_local_llm_answer,
    MISSING_EVIDENCE_MSG
)

app = FastAPI(title="RAG Backend API")

# Global state
class State:
    evidence: List[Dict[str, Any]] = []
    retrievers: Dict[str, Any] = {}
    config: Dict[str, Any] = {
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 0,
        "chunk_overlap": 0
    }

state = State()

# Pydantic models
class ConfigureRequest(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 0
    chunk_overlap: int = 0

class RetrieveRequest(BaseModel):
    question: str
    retrieval_mode: str
    top_k: int = 10
    sources_key: Optional[str] = None  # pipe-separated allowed sources
    modalities_key: Optional[str] = None # pipe-separated allowed modalities

class GenerateRequest(BaseModel):
    question: str
    context: str
    answer_mode: str
    gemini_key: Optional[str] = None

class RagRequest(BaseModel):
    question: str
    retrieval_mode: str
    answer_mode: str
    top_k: int = 10
    gemini_key: Optional[str] = None
    sources_key: Optional[str] = None
    modalities_key: Optional[str] = None

# Startup
@app.on_event("startup")
def startup_event():
    print("Loading corpus with default config...")
    _reload_corpus()

def _reload_corpus():
    docs_path = PROJECT_ROOT / "data" / "docs"
    images_path = PROJECT_ROOT / "data" / "images"
    
    if not docs_path.exists():
        print(f"ERROR: {docs_path} not found")
        return

    # Call pipeline's load_corpus
    # We must ensure we're calling the updated version that accepts chunk_size/overlap
    try:
        updated_evidence = load_corpus(
            str(docs_path), 
            str(images_path),
            chunk_size=state.config.get("chunk_size", 0),
            chunk_overlap=state.config.get("chunk_overlap", 0)
        )
        state.evidence = updated_evidence
        
        # Build retrievers
        state.retrievers = build_retrievers(
            state.evidence, 
            embedding_model=state.config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        print(f"Corpus loaded: {len(state.evidence)} items. Retrievers ready.")
    except Exception as e:
        print(f"Error reloading corpus: {e}")

# Endpoints

@app.get("/status")
def get_status():
    return {
        "status": "ok",
        "corpus_size": len(state.evidence),
        "available_modes": list(state.retrievers.keys()),
        "config": state.config
    }

@app.post("/configure")
def configure_endpoint(req: ConfigureRequest):
    """Re-load corpus and retrievers if config changed."""
    # Check if config actually changed to avoid redundant reloads
    current = state.config
    if (req.embedding_model == current["embedding_model"] and 
        req.chunk_size == current["chunk_size"] and 
        req.chunk_overlap == current["chunk_overlap"]):
        return {"message": "No configuration changes detected.", "config": current}
    
    # Update config
    state.config["embedding_model"] = req.embedding_model
    state.config["chunk_size"] = req.chunk_size
    state.config["chunk_overlap"] = req.chunk_overlap
    
    print(f"Re-configuring with: {state.config}")
    _reload_corpus()
    
    return {"message": "Configuration updated and corpus reloaded.", "config": state.config}

@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    mode = req.retrieval_mode
    # Fallback to tfidf if mode not found
    retriever = state.retrievers.get(mode, state.retrievers.get("tfidf"))
    
    if not retriever:
        # Should ideally not happen if tfidf is always built
        raise HTTPException(status_code=500, detail="No retrievers available (corpus load failed?)")

    # Over-fetch for filtering
    hits = retriever.retrieve(req.question, top_k=req.top_k * 3)

    # Parse filters
    allowed_sources = set(req.sources_key.split("|")) if req.sources_key else set()
    allowed_modalities = set(req.modalities_key.split("|")) if req.modalities_key else set()

    results = []
    hit_indices = []
    
    count = 0
    for idx, score in hits:
        if idx >= len(state.evidence):
            continue
            
        item = state.evidence[idx]
        src_base = os.path.basename(item.get("source", ""))
        modality = item.get("modality", "text")
        
        # Apply metadata filters
        if allowed_sources and src_base not in allowed_sources:
             continue
        if allowed_modalities and modality not in allowed_modalities:
            continue
        
        results.append({
            "chunk_id": item["chunk_id"],
            "citation_tag": f"[{item['chunk_id']}]",
            "score": float(score), # ensure JSON serializable
            "source": item.get("source", ""),
            "text": item.get("text", "")[:500],
            # Helper for debugging/inspection if needed
            "full_text_len": len(item.get("text", "")),
        })
        hit_indices.append(int(idx))
        count += 1
        if count >= req.top_k:
            break
            
    return {"results": results, "hit_indices": hit_indices}

@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    if req.answer_mode == "llm (Gemini)":
        ans = generate_llm_answer(req.question, req.context, api_key=req.gemini_key)
    elif req.answer_mode == "llm (Local)":
        ans = generate_local_llm_answer(req.question, req.context)
    else:
        ans = extractive_answer(req.question, req.context)
    return {"answer": ans}

@app.post("/rag")
def rag_pipeline_endpoint(req: RagRequest):
    """Full RAG pipeline: Retrieve -> Context -> Generate"""
    
    # 1. Retrieve
    # Re-use logic from retrieve_endpoint but via function call
    # Construct a RetrieveRequest object manually or call retrieval logic directly?
    # Direct usage is cleaner to avoid serialization overhead
    
    mode = req.retrieval_mode
    retriever = state.retrievers.get(mode, state.retrievers.get("tfidf"))
    if not retriever:
         raise HTTPException(status_code=500, detail="Backend not ready: No retrievers loaded.")

    hits = retriever.retrieve(req.question, top_k=req.top_k * 3)
    
    allowed_sources = set(req.sources_key.split("|")) if req.sources_key else set()
    allowed_modalities = set(req.modalities_key.split("|")) if req.modalities_key else set()

    evidence_results = []
    final_indices = []
    
    count = 0
    for idx, score in hits:
        if idx >= len(state.evidence): continue
        
        item = state.evidence[idx]
        src_base = os.path.basename(item.get("source", ""))
        modality = item.get("modality", "text")
        
        if allowed_sources and src_base not in allowed_sources: continue
        if allowed_modalities and modality not in allowed_modalities: continue
        
        evidence_results.append({
            "chunk_id": item["chunk_id"],
            "citation_tag": f"[{item['chunk_id']}]",
            "score": round(float(score), 4),
            "source": item.get("source", ""),
            "text": item.get("text", "")[:500],
        })
        final_indices.append(int(idx))
        count += 1
        if count >= req.top_k:
            break
            
    # 2. Build Context
    context = build_context(state.evidence, final_indices)
    
    # 3. Generate
    if req.answer_mode == "llm (Gemini)":
        ans = generate_llm_answer(req.question, context, api_key=req.gemini_key)
    elif req.answer_mode == "llm (Local)":
        ans = generate_local_llm_answer(req.question, context)
    else:
        ans = extractive_answer(req.question, context)

    return {
        "answer": ans,
        "evidence_results": evidence_results,
        "context_preview": context[:200]
    }
