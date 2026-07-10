"""Performance-hardening regression tests.

Covers the success-signalled query-embedding cache (only genuine provider
embeddings are memoised; a transient failure degrades to an uncached stub) and
the streamed NL2SQL CSV export.
"""
from __future__ import annotations

import pytest

from app.nl2sql.executor import QueryResult, iter_csv, to_csv
from app.rag import vector_store


@pytest.fixture(autouse=True)
def _clear_embed_cache():
    """Keep each test independent of the module-level lru_cache."""
    vector_store._embed_query_cached.cache_clear()
    yield
    vector_store._embed_query_cached.cache_clear()


class CountingLLM:
    """Real-path double: counts embed_one calls, optionally raising."""

    enabled = True

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def embed_one(self, text: str) -> list[float]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return [float(len(text)), 1.0, 2.0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Stub fallback path — deterministic, never counted as a real call.
        return [[0.0, 0.0, 0.0] for _ in texts]


def test_embed_query_caches_real_embedding(monkeypatch):
    llm = CountingLLM()
    monkeypatch.setattr(vector_store, "get_llm", lambda: llm)

    first = vector_store.embed_query("same query")
    second = vector_store.embed_query("same query")

    assert first == second
    assert llm.calls == 1  # second call served from the cache


def test_embed_query_failure_falls_back_to_uncached_stub(monkeypatch):
    failing = CountingLLM(fail=True)
    monkeypatch.setattr(vector_store, "get_llm", lambda: failing)

    result = vector_store.embed_query("recover me")

    # Degrades to the stub for this call without crashing, and nothing is cached.
    assert result == (0.0, 0.0, 0.0)
    assert vector_store._embed_query_cached.cache_info().currsize == 0

    # Provider recovers -> the query re-embeds for real (nothing stale cached).
    healthy = CountingLLM()
    monkeypatch.setattr(vector_store, "get_llm", lambda: healthy)
    recovered = vector_store.embed_query("recover me")
    assert recovered == (float(len("recover me")), 1.0, 2.0)
    assert healthy.calls == 1


def test_embed_query_stub_path_when_disabled(monkeypatch):
    class DisabledLLM:
        enabled = False

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[9.0, 9.0] for _ in texts]

    monkeypatch.setattr(vector_store, "get_llm", lambda: DisabledLLM())

    assert vector_store.embed_query("anything") == (9.0, 9.0)
    assert vector_store._embed_query_cached.cache_info().currsize == 0


def test_iter_csv_streams_header_and_rows():
    result = QueryResult(
        ok=True,
        columns=["attorney_id", "total"],
        rows=[["JOHNSON", 100], ["SMITH", 250]],
        row_count=2,
    )

    streamed = "".join(iter_csv(result))

    assert streamed == to_csv(result)
    lines = streamed.splitlines()
    assert lines[0] == "attorney_id,total"
    assert lines[1] == "JOHNSON,100"
    assert lines[2] == "SMITH,250"
