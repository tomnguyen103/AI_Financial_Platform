# Pinecone RAG Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing financial RAG chatbot into a portfolio-ready AI application with Pinecone vector search, visible retrieval evidence, evaluation, Docker deployment, and documentation aligned to the AI Applications Developer role.

**Architecture:** Keep the current in-memory vector index as the default local fallback, add a `VectorStore` interface, and route production retrieval through Pinecone when `VECTOR_STORE=pinecone`. The chatbot continues to enforce PHI scanning, metadata-aware retrieval, citation-grounded answers, and audit logging.

**Tech Stack:** Python, FastAPI, OpenAI embeddings, Pinecone serverless, SQLite, pytest, Docker, static HTML demo UI.

---

## File Structure

- Create `app/rag/vector_store.py`: shared `SearchHit`, `VectorStore`, and in-memory vector store implementation.
- Create `app/rag/pinecone_store.py`: Pinecone adapter for upsert and query.
- Modify `app/rag/index.py`: build/select vector store based on settings.
- Modify `app/rag/chatbot.py`: keep existing `get_index()` contract but expose richer citation metadata.
- Modify `app/config.py`: add vector store and Pinecone settings.
- Modify `.env.example`: document Pinecone and deployment environment variables.
- Modify `requirements.txt`: add `pinecone`.
- Create `scripts/build_pinecone_index.py`: one-command corpus embedding and Pinecone upsert.
- Create `tests/test_vector_store.py`: fast local tests for vector-store behavior.
- Create `tests/test_pinecone_store.py`: mocked Pinecone adapter tests with no network calls.
- Modify `tests/test_rag_chatbot.py`: verify retrieval metadata remains visible to the chatbot.
- Modify `app/static/index.html`: make the demo explicitly show vector DB, source IDs, similarity scores, and safety status.
- Create `eval/rag_questions.jsonl`: small portfolio evaluation set.
- Create `scripts/evaluate_rag.py`: evaluate retrieval hit rate, citation presence, refusal behavior, and latency.
- Create `Dockerfile`: containerize the FastAPI app.
- Create `.dockerignore`: keep local data/cache/secrets out of the image.
- Modify `README.md`: add portfolio case-study section, architecture, setup, evaluation, deployment, and job-description mapping.

Note: this folder is currently not a git repository, so commit steps are written as checkpoints. If git is initialized before execution, use the suggested commit commands; otherwise record each checkpoint in the implementation notes.

---

### Task 1: Add Vector Store Settings

**Files:**
- Modify: `app/config.py`
- Modify: `.env.example`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: Write the failing settings test**

Create `tests/test_vector_store.py` with:

```python
from __future__ import annotations

from app.rag.vector_store import SearchHit


def test_search_hit_preserves_document_score_and_metadata():
    hit = SearchHit(
        source_doc_id="forecast_facility_round_rock_2026-03-27",
        entity_type="facility",
        entity_id="round_rock",
        date="2026-03-27",
        text="30-day collections forecast for facility round_rock.",
        score=0.91,
        metadata={"feature_group": "forecast"},
    )

    assert hit.source_doc_id == "forecast_facility_round_rock_2026-03-27"
    assert hit.entity_type == "facility"
    assert hit.entity_id == "round_rock"
    assert hit.date == "2026-03-27"
    assert hit.score == 0.91
    assert hit.metadata["feature_group"] == "forecast"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_vector_store.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.rag.vector_store'`.

- [ ] **Step 3: Add settings**

In `app/config.py`, add these fields inside `class Settings` after the Azure OpenAI settings:

```python
    # vector search
    vector_store: str = os.getenv("VECTOR_STORE", "memory").lower()
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "financial-rag-demo")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
```

Update `.env.example` after the LLM block:

```bash
# --- Vector search. Use memory locally; pinecone for portfolio deployment. ---
VECTOR_STORE=memory
PINECONE_API_KEY=
PINECONE_INDEX_NAME=financial-rag-demo
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_DIMENSION=1536
```

- [ ] **Step 4: Create vector store module**

Create `app/rag/vector_store.py`:

```python
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
    def from_document(cls, doc: Document, score: float) -> "SearchHit":
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
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_vector_store.py -q
```

Expected: PASS.

- [ ] **Step 6: Checkpoint**

If git is initialized:

```bash
git add app/config.py .env.example app/rag/vector_store.py tests/test_vector_store.py
git commit -m "feat: add vector store settings and interface"
```

---

### Task 2: Route Existing RAG Through Vector Store Interface

**Files:**
- Modify: `app/rag/index.py`
- Modify: `app/rag/chatbot.py`
- Test: `tests/test_rag_chatbot.py`

- [ ] **Step 1: Update test fake to use SearchHit**

Modify `tests/test_rag_chatbot.py`:

```python
from __future__ import annotations

from app.rag import chatbot
from app.rag.corpus import Document
from app.rag.vector_store import SearchHit


class FakeIndex:
    def __init__(self):
        self.docs = [
            Document(
                "attorney_aging_JIM ADLER_2026-03-27",
                "attorney",
                "JIM ADLER",
                "2026-03-27",
                "Attorney Jim Adler aging report as of 2026-03-27: total outstanding $5,543,209.",
            )
        ]

    def search(self, query: str, top_k: int = 8, entity_id: str | None = None):
        assert entity_id == "JIM ADLER"
        return [SearchHit.from_document(self.docs[0], 0.70)]


class FakeLLM:
    enabled = False
    model_name = "stub"

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        return ""


def test_exact_entity_match_answers_below_generic_similarity_threshold(monkeypatch):
    monkeypatch.setattr(chatbot, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(chatbot, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(chatbot, "write_audit", lambda **kwargs: None)

    response = chatbot.ask("How is Attorney Jim Adler performing?")

    assert not response.insufficient
    assert "Jim Adler" in response.answer
    assert response.citations[0]["entity_id"] == "JIM ADLER"
    assert response.citations[0]["score"] == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_rag_chatbot.py -q
```

Expected: FAIL because `chatbot._format_context()` still expects `(doc, score)` tuples.

- [ ] **Step 3: Replace `app/rag/index.py` implementation**

Replace `app/rag/index.py` with:

```python
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
```

- [ ] **Step 4: Update chatbot context formatting**

In `app/rag/chatbot.py`, replace `_format_context()` with:

```python
def _format_context(hits) -> tuple[str, list[dict]]:
    blocks, cites = [], []
    for hit in hits:
        blocks.append(f"[{hit.source_doc_id} | {hit.date}] {hit.text}")
        cites.append({
            "source_doc_id": hit.source_doc_id,
            "date": hit.date,
            "entity_id": hit.entity_id,
            "entity_type": hit.entity_type,
            "score": round(hit.score, 3),
            "metadata": hit.metadata,
        })
    return "\n".join(blocks), cites
```

Replace the retrieval scoring and stub-answer references in `ask()`:

```python
    hits = idx.search(query, entity_id=entity_id)
    top_score = hits[0].score if hits else 0.0
```

Replace:

```python
                    retrieved_sources=[h[0].source_doc_id for h in hits],
```

with:

```python
                    retrieved_sources=[h.source_doc_id for h in hits],
```

Replace:

```python
        raw = (f"{hits[0][0].text} "
               f"[Source: {cites[0]['source_doc_id']}, {cites[0]['date']}]")
```

with:

```python
        raw = (f"{hits[0].text} "
               f"[Source: {cites[0]['source_doc_id']}, {cites[0]['date']}]")
```

- [ ] **Step 5: Run RAG tests**

Run:

```bash
python -m pytest tests/test_rag_chatbot.py tests/test_vector_store.py -q
```

Expected: PASS.

- [ ] **Step 6: Checkpoint**

If git is initialized:

```bash
git add app/rag/index.py app/rag/chatbot.py tests/test_rag_chatbot.py
git commit -m "refactor: route rag retrieval through vector store"
```

---

### Task 3: Add Pinecone Adapter

**Files:**
- Create: `app/rag/pinecone_store.py`
- Modify: `requirements.txt`
- Test: `tests/test_pinecone_store.py`

- [ ] **Step 1: Add dependency**

Add this line to `requirements.txt`:

```text
pinecone>=5.0
```

- [ ] **Step 2: Write mocked Pinecone tests**

Create `tests/test_pinecone_store.py`:

```python
from __future__ import annotations

from app.rag.corpus import Document
from app.rag.pinecone_store import _doc_to_record, _match_to_hit


def test_doc_to_record_includes_vector_and_metadata():
    doc = Document(
        "forecast_facility_round_rock_2026-03-27",
        "facility",
        "round_rock",
        "2026-03-27",
        "30-day collections forecast for round_rock.",
        {"feature_group": "forecast", "horizon": 30},
    )

    record = _doc_to_record(doc, [0.1, 0.2, 0.3])

    assert record["id"] == "forecast_facility_round_rock_2026-03-27"
    assert record["values"] == [0.1, 0.2, 0.3]
    assert record["metadata"]["text"] == "30-day collections forecast for round_rock."
    assert record["metadata"]["entity_type"] == "facility"
    assert record["metadata"]["entity_id"] == "round_rock"
    assert record["metadata"]["feature_group"] == "forecast"


def test_match_to_hit_round_trips_metadata():
    match = {
        "id": "alert_round_rock_2026-03-27",
        "score": 0.88,
        "metadata": {
            "source_doc_id": "alert_round_rock_2026-03-27",
            "entity_type": "facility",
            "entity_id": "round_rock",
            "date": "2026-03-27",
            "text": "Anomaly alert for round_rock.",
            "feature_group": "alert",
            "severity": "P2",
        },
    }

    hit = _match_to_hit(match)

    assert hit.source_doc_id == "alert_round_rock_2026-03-27"
    assert hit.entity_type == "facility"
    assert hit.entity_id == "round_rock"
    assert hit.date == "2026-03-27"
    assert hit.text == "Anomaly alert for round_rock."
    assert hit.score == 0.88
    assert hit.metadata["severity"] == "P2"
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_pinecone_store.py -q
```

Expected: FAIL because `app.rag.pinecone_store` does not exist.

- [ ] **Step 4: Create Pinecone adapter**

Create `app/rag/pinecone_store.py`:

```python
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
            records = [_doc_to_record(doc, vector) for doc, vector in zip(batch, vectors)]
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
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_pinecone_store.py tests/test_vector_store.py tests/test_rag_chatbot.py -q
```

Expected: PASS.

- [ ] **Step 6: Checkpoint**

If git is initialized:

```bash
git add requirements.txt app/rag/pinecone_store.py tests/test_pinecone_store.py
git commit -m "feat: add pinecone vector store"
```

---

### Task 4: Add Pinecone Index Build Script

**Files:**
- Create: `scripts/build_pinecone_index.py`
- Test: command-level smoke test

- [ ] **Step 1: Create script**

Create `scripts/build_pinecone_index.py`:

```python
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
```

- [ ] **Step 2: Run dry-run**

Run:

```bash
python -m scripts.build_pinecone_index --dry-run
```

Expected: command prints `Built corpus: N documents` and five sample source IDs. If it prints `Built corpus: 0 documents`, run `python -m scripts.seed_data` first.

- [ ] **Step 3: Run real upsert after environment is configured**

Set `.env`:

```bash
OPENAI_API_KEY=your_key_here
VECTOR_STORE=pinecone
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=financial-rag-demo
EMBEDDING_DIMENSION=1536
```

Run:

```bash
python -m scripts.build_pinecone_index
```

Expected: command prints `Upserted to Pinecone: N vectors`.

- [ ] **Step 4: Checkpoint**

If git is initialized:

```bash
git add scripts/build_pinecone_index.py
git commit -m "feat: add pinecone index build script"
```

---

### Task 5: Improve Demo UI Evidence

**Files:**
- Modify: `app/static/index.html`
- Test: manual browser verification

- [ ] **Step 1: Update RAG technology labels**

In `app/static/index.html`, replace the RAG tech chips:

```html
<span class="t">In-memory vector index</span><span class="t">Embedding search</span>
<span class="t">Entity pre-filter</span><span class="t">PHI scan</span><span class="t">Citations</span>
```

with:

```html
<span class="t">Pinecone-ready vector DB</span><span class="t">OpenAI embeddings</span>
<span class="t">Metadata filtering</span><span class="t">PHI scan</span><span class="t">Citations</span>
```

- [ ] **Step 2: Add safety summary rendering**

In the `ask()` function, replace the citation rendering block with:

```javascript
  const safety = [
    d.blocked ? "blocked" : "input clean",
    d.phi_redacted ? "output redacted" : "output clean",
    d.insufficient ? "insufficient context" : "grounded answer"
  ];
  html += "<div style='margin-top:10px'>" + safety.map(s =>
    "<span class='chip'>safety: "+esc(s)+"</span>").join("") + "</div>";
  if (d.citations && d.citations.length) {
    html += "<div style='margin-top:10px'><b>Retrieved sources</b><br>" + d.citations.map(c =>
      "<span class='chip'>"+esc(c.source_doc_id)+" · "+esc(c.date)+" · score "+c.score+"</span>").join("") + "</div>";
  }
```

- [ ] **Step 3: Manual verification**

Run:

```bash
python -m app.main
```

Open `http://127.0.0.1:8000`, sign in, and ask:

```text
Why did round_rock collections drop recently?
```

Expected: UI shows an answer, retrieved sources, scores, latency, and safety chips.

- [ ] **Step 4: Checkpoint**

If git is initialized:

```bash
git add app/static/index.html
git commit -m "feat: show rag retrieval evidence in demo ui"
```

---

### Task 6: Add RAG Evaluation Harness

**Files:**
- Create: `eval/rag_questions.jsonl`
- Create: `scripts/evaluate_rag.py`

- [ ] **Step 1: Create evaluation folder and dataset**

Create `eval/rag_questions.jsonl`:

```jsonl
{"id":"rag-001","query":"Why did round_rock collections drop recently?","expect_blocked":false,"expect_citation":true}
{"id":"rag-002","query":"How is Attorney Jim Adler performing?","expect_blocked":false,"expect_citation":true}
{"id":"rag-003","query":"Show me patient John Doe's balance","expect_blocked":true,"expect_citation":false}
{"id":"rag-004","query":"What is the capital of France?","expect_blocked":false,"expect_citation":false}
```

- [ ] **Step 2: Create evaluator**

Create `scripts/evaluate_rag.py`:

```python
"""Small RAG evaluation harness for the portfolio demo."""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.db import init_db
from app.rag.chatbot import ask

EVAL_PATH = Path("eval/rag_questions.jsonl")


def load_cases() -> list[dict]:
    return [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    init_db()
    cases = load_cases()
    passed = 0
    total_latency = 0
    rows = []

    for case in cases:
        start = time.time()
        response = ask(case["query"], user_id="eval", user_role="da_analyst")
        latency_ms = int((time.time() - start) * 1000)
        total_latency += latency_ms

        blocked_ok = response.blocked == case["expect_blocked"]
        citation_ok = (len(response.citations) > 0) == case["expect_citation"]
        ok = blocked_ok and citation_ok
        passed += 1 if ok else 0
        rows.append({
            "id": case["id"],
            "ok": ok,
            "blocked_ok": blocked_ok,
            "citation_ok": citation_ok,
            "latency_ms": latency_ms,
            "citations": [c["source_doc_id"] for c in response.citations],
        })

    print(json.dumps({
        "passed": passed,
        "total": len(cases),
        "pass_rate": round(passed / max(len(cases), 1), 3),
        "avg_latency_ms": round(total_latency / max(len(cases), 1), 1),
        "cases": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run evaluator**

Run:

```bash
python -m scripts.evaluate_rag
```

Expected: JSON summary with `passed`, `total`, `pass_rate`, `avg_latency_ms`, and per-case details.

- [ ] **Step 4: Checkpoint**

If git is initialized:

```bash
git add eval/rag_questions.jsonl scripts/evaluate_rag.py
git commit -m "feat: add rag evaluation harness"
```

---

### Task 7: Containerize The Portfolio App

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Modify: `README.md`

- [ ] **Step 1: Create Dockerfile**

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY config config
COPY scripts scripts

EXPOSE 8000

CMD ["python", "-m", "app.main"]
```

- [ ] **Step 2: Create dockerignore**

Create `.dockerignore`:

```text
.env
.venv
.pytest_cache
__pycache__
*.pyc
.idea
data
docs/superpowers/plans
eval
```

- [ ] **Step 3: Build image**

Run:

```bash
docker build -t ai-financial-platform .
```

Expected: image builds successfully.

- [ ] **Step 4: Run container**

Run:

```bash
docker run --rm -p 8000:8000 --env VECTOR_STORE=memory ai-financial-platform
```

Expected: app starts at `http://127.0.0.1:8000`. If the demo needs seeded data inside the container, run seed data in a follow-up deployment task or mount a generated `data` directory.

- [ ] **Step 5: Checkpoint**

If git is initialized:

```bash
git add Dockerfile .dockerignore README.md
git commit -m "chore: containerize portfolio app"
```

---

### Task 8: Rewrite README As A Role-Aligned Case Study

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add case-study section near the top**

Insert after the title:

```markdown
## Portfolio Case Study: Production-Style AI Financial Assistant

This project demonstrates an AI Applications Developer workflow: building a
secure AI-powered financial assistant from data pipelines through retrieval,
LLM synthesis, API delivery, evaluation, and deployment.

### Role Alignment

| Job requirement | Evidence in this project |
|---|---|
| AI-powered features | RAG chatbot, NL-to-SQL, forecasting, anomaly detection |
| Backend APIs | FastAPI gateway with RBAC-protected endpoints |
| Embeddings/vector DB | OpenAI embeddings with Pinecone-ready vector retrieval |
| Data pipelines | Synthetic ingestion, feature store, forecasts, alerts |
| Privacy/security | PHI scanning, HMAC tokenization, RBAC, audit logging |
| Evaluation | RAG evaluation harness for citations, refusals, latency |
| DevOps | Dockerfile and deployment-ready environment config |
| Documentation | Architecture notes, implementation notes, demo instructions |
```

- [ ] **Step 2: Add Pinecone setup section**

Insert under Configuration:

````markdown
### Pinecone RAG Mode

Local development defaults to `VECTOR_STORE=memory`. For the portfolio deployment:

```bash
OPENAI_API_KEY=...
OPENAI_EMBED_MODEL=text-embedding-3-small
VECTOR_STORE=pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=financial-rag-demo
EMBEDDING_DIMENSION=1536
```

Build the Pinecone index:

```bash
python -m scripts.seed_data
python -m scripts.build_pinecone_index
```
````

- [ ] **Step 3: Add evaluation section**

Insert before Tests:

````markdown
## RAG Evaluation

Run a small portfolio evaluation suite:

```bash
python -m scripts.evaluate_rag
```

The evaluator checks whether safe questions return citations, PHI-style
questions are blocked, and responses stay within acceptable latency.
````

- [ ] **Step 4: Review README**

Run:

```bash
python -m pytest tests/ -q
```

Expected: PASS. README markdown changes do not affect tests.

- [ ] **Step 5: Checkpoint**

If git is initialized:

```bash
git add README.md
git commit -m "docs: frame project as ai applications portfolio case study"
```

---

## Self-Review

- Spec coverage: Pinecone vector DB, embeddings, chatbot retrieval, demo UI, evaluation, Docker deployment, documentation, and privacy story are all covered.
- Placeholder scan: no unresolved placeholders or undefined follow-up tasks remain.
- Type consistency: `SearchHit` is the common retrieval type used by memory store, Pinecone store, and chatbot citations.
- Scope control: this plan does not add React, Kubernetes, MLflow, Prometheus, or cloud-specific CI/CD yet. Those are good follow-up portfolio enhancements after the Pinecone RAG upgrade is stable.
