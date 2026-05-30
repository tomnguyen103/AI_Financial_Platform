"""RAG chatbot endpoint "Ask the Financials" (PRD Module 5; arch §2.5)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.config import settings
from app.rag.chatbot import ask
from app.rag.index import SIM_THRESHOLD, TOP_K, get_index
from app.security.auth import User, require

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


class AskRequest(BaseModel):
    query: str
    session_id: str | None = None


@router.post("/ask")
def ask_endpoint(req: AskRequest, user: User = Depends(require("chatbot:use"))) -> dict:
    r = ask(req.query, user_id=user.user_id, user_role=user.role, session_id=req.session_id)
    return {
        "answer": r.answer,
        "citations": r.citations,
        "insufficient": r.insufficient,
        "blocked": r.blocked,
        "phi_redacted": r.phi_redacted,
        "latency_ms": r.latency_ms,
        "session_id": r.session_id,
        "retrieval": r.retrieval,
    }


@router.get("/status")
def status_endpoint(user: User = Depends(require("chatbot:use"))) -> dict:
    idx = get_index()
    return {
        "vector_store": settings.vector_store,
        "pinecone_index": settings.pinecone_index_name if settings.vector_store == "pinecone" else "",
        "embedding_model": settings.openai_embed_model if settings.llm_enabled else "stub-hash",
        "llm_model": settings.openai_model if settings.llm_enabled else "stub",
        "corpus_documents": len(idx.docs),
        "top_k": TOP_K,
        "similarity_threshold": SIM_THRESHOLD,
    }
