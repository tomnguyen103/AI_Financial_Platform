"""Vector store abstractions for RAG retrieval.

The app keeps an in-memory implementation for local/offline demos and tests.
Production portfolio deployments can switch to Pinecone via VECTOR_STORE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.llm.client import get_llm
from app.rag.corpus import Document


@dataclass(frozen=True)
class SearchHit:
    source_doc_id: str
    entity_type: str
    entity_id: str
    date: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_document(cls, doc: Document, score: float) -> SearchHit:
        return cls(
            source_doc_id=doc.source_doc_id,
            entity_type=doc.entity_type,
            entity_id=doc.entity_id,
            date=doc.date,
            text=doc.text,
            score=score,
            metadata=dict(doc.metadata),
        )


class VectorStore(Protocol):
    docs: list[Document]

    def build(self) -> int:
        """Build or refresh the searchable index from the local corpus."""

    def search(
        self,
        query: str,
        top_k: int = 8,
        entity_id: str | None = None,
    ) -> list[SearchHit]:
        """Return ranked semantic matches."""


class InMemoryVectorStore:
    def __init__(self, docs: list[Document] | None = None) -> None:
        self.docs: list[Document] = docs or []
        self.matrix: np.ndarray | None = None

    def build(self) -> int:
        from app.rag.corpus import build_corpus

        self.docs = build_corpus()
        if not self.docs:
            self.matrix = None
            return 0
        embeds = get_llm().embed([d.text for d in self.docs])
        self.matrix = np.array(embeds, dtype=float)
        return len(self.docs)

    def search(
        self,
        query: str,
        top_k: int = 8,
        entity_id: str | None = None,
    ) -> list[SearchHit]:
        if self.matrix is None or not self.docs:
            return []

        idxs = list(range(len(self.docs)))
        if entity_id:
            filtered = [i for i in idxs if self.docs[i].entity_id == entity_id]
            if filtered:
                idxs = filtered

        q = np.array(get_llm().embed([query])[0], dtype=float)
        sub = self.matrix[idxs]
        denom = (np.linalg.norm(sub, axis=1) * np.linalg.norm(q)) + 1e-9
        sims = sub @ q / denom
        order = np.argsort(-sims)[:top_k]
        return [
            SearchHit.from_document(self.docs[idxs[i]], float(sims[i]))
            for i in order
        ]
