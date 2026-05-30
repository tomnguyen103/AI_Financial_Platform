"""Vector index selection for the RAG chatbot.

Default local mode uses an in-memory vector store. Portfolio deployments can set
VECTOR_STORE=pinecone after running scripts/build_pinecone_index.py.
"""
from __future__ import annotations

from app.config import settings
from app.rag.vector_store import InMemoryVectorStore, VectorStore

TOP_K = 8
SIM_THRESHOLD = 0.75 if settings.llm_enabled else 0.15

_index: VectorStore | None = None


def _build_store() -> VectorStore:
    if settings.vector_store == "pinecone":
        from app.rag.pinecone_store import PineconeVectorStore

        store = PineconeVectorStore()
        store.build()
        return store

    store = InMemoryVectorStore()
    store.build()
    return store


def get_index(rebuild: bool = False) -> VectorStore:
    global _index
    if _index is None or rebuild:
        _index = _build_store()
    return _index
