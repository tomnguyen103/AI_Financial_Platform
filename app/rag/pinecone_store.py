"""Pinecone vector store for portfolio RAG deployments."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.config import settings
from app.llm.client import get_llm
from app.rag.corpus import Document, build_corpus
from app.rag.vector_store import SearchHit


def _doc_to_record(doc: Document, vector: list[float]) -> dict:
    metadata = {
        "source_doc_id": doc.source_doc_id,
        "entity_type": doc.entity_type,
        "entity_id": doc.entity_id,
        "date": doc.date,
        "text": doc.text,
        **doc.metadata,
    }
    return {"id": doc.source_doc_id, "values": vector, "metadata": metadata}


def _match_to_hit(match: Any) -> SearchHit:
    if hasattr(match, "to_dict"):
        match = match.to_dict()
    metadata = dict(match.get("metadata") or {})
    return SearchHit(
        source_doc_id=metadata.get("source_doc_id") or match.get("id", ""),
        entity_type=metadata.get("entity_type", ""),
        entity_id=metadata.get("entity_id", ""),
        date=metadata.get("date", ""),
        text=metadata.get("text", ""),
        score=float(match.get("score") or 0.0),
        metadata={
            k: v
            for k, v in metadata.items()
            if k not in {"source_doc_id", "entity_type", "entity_id", "date", "text"}
        },
    )


class PineconeVectorStore:
    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is required when VECTOR_STORE=pinecone")
        self.docs: list[Document] = []
        self._index = self._build_index_client()

    def _build_index_client(self):
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=settings.pinecone_api_key)
        existing = {idx["name"] for idx in pc.list_indexes()}
        if settings.pinecone_index_name not in existing:
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )
        return pc.Index(settings.pinecone_index_name)

    def build(self) -> int:
        self.docs = build_corpus()
        return len(self.docs)

    def upsert_documents(
        self,
        docs: Iterable[Document],
        batch_size: int = 100,
    ) -> int:
        docs = list(docs)
        total = 0
        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            vectors = get_llm().embed([doc.text for doc in batch])
            records = [_doc_to_record(doc, vector) for doc, vector in zip(batch, vectors, strict=True)]
            self._index.upsert(vectors=records)
            total += len(records)
        return total

    def search(
        self,
        query: str,
        top_k: int = 8,
        entity_id: str | None = None,
    ) -> list[SearchHit]:
        vector = get_llm().embed([query])[0]
        filter_expr = {"entity_id": {"$eq": entity_id}} if entity_id else None
        result = self._index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_expr,
        )
        matches = result.get("matches", []) if isinstance(result, dict) else result.matches
        return [_match_to_hit(match) for match in matches]
