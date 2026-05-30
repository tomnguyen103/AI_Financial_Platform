"""Build the Pinecone RAG index from de-identified financial summaries."""
from __future__ import annotations

import argparse

from app.db import init_db
from app.rag.corpus import build_corpus
from app.rag.pinecone_store import PineconeVectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Build corpus without upserting")
    args = parser.parse_args()

    init_db()
    docs = build_corpus()
    print(f"Built corpus: {len(docs)} documents")

    if args.dry_run:
        for doc in docs[:5]:
            print(f"- {doc.source_doc_id}: {doc.text[:120]}")
        return

    store = PineconeVectorStore()
    count = store.upsert_documents(docs)
    print(f"Upserted to Pinecone: {count} vectors")


if __name__ == "__main__":
    main()
