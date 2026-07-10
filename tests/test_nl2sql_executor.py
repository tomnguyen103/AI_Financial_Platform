"""NL-to-SQL executor tests (arch §2.6): row cap, preview cap, timeout and
execution-error paths, and CSV rendering."""
from __future__ import annotations

from app.nl2sql import executor
from app.nl2sql.executor import QueryResult


def test_mutation_request_rejected_before_generation(seeded_db):
    r = executor.run_query("delete from collections where 1=1", user_role="da_analyst")
    assert not r.ok
    assert "mutation" in r.error.lower()
    assert r.attempts == 0


def test_run_query_success_and_row_cap(seeded_db, monkeypatch):
    """A hard row cap truncates results and flags truncated=True."""
    monkeypatch.setattr(executor, "generate_sql", lambda q: "SELECT collection_id FROM collections")
    monkeypatch.setattr(executor, "ROW_CAP", 5)
    r = executor.run_query("all collection ids")
    assert r.ok
    assert r.row_count == 5
    assert r.truncated is True
    assert r.columns == ["collection_id"]


def test_run_query_preview_cap(seeded_db, monkeypatch):
    monkeypatch.setattr(executor, "generate_sql", lambda q: "SELECT collection_id FROM collections")
    monkeypatch.setattr(executor, "PREVIEW_ROW_CAP", 3)
    r = executor.run_query("all collection ids", preview=True)
    assert r.ok
    assert r.row_count == 3
    assert r.truncated is True


def test_run_query_execution_error_path(seeded_db, monkeypatch):
    # Valid SELECT (table whitelisted) that references a non-existent column ->
    # passes validation but fails at execution -> exec_error branch.
    monkeypatch.setattr(executor, "generate_sql",
                        lambda q: "SELECT no_such_column FROM collections")
    r = executor.run_query("boom")
    assert not r.ok
    assert "Execution error" in r.error


def test_run_query_timeout_path(seeded_db, monkeypatch):
    monkeypatch.setattr(executor, "generate_sql", lambda q: "SELECT collection_id FROM collections")

    def _timeout(_sql):
        raise TimeoutError("Query exceeded 30s and was cancelled.")

    monkeypatch.setattr(executor, "_execute", _timeout)
    r = executor.run_query("slow query")
    assert not r.ok
    assert "cancelled" in r.error


def test_to_csv_and_iter_csv_agree():
    result = QueryResult(ok=True, columns=["a", "b"], rows=[[1, 2], [3, 4]], row_count=2)
    csv_text = executor.to_csv(result)
    streamed = "".join(executor.iter_csv(result))
    assert csv_text == streamed
    lines = csv_text.strip().splitlines()
    assert lines[0] == "a,b"
    assert lines[1] == "1,2"
    assert lines[2] == "3,4"
