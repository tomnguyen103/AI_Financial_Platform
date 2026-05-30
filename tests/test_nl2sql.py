"""NL-to-SQL safety + execution tests."""
from __future__ import annotations

from app.nl2sql.validator import validate_sql
from app.nl2sql.executor import run_query


def test_select_allowed():
    ok, _ = validate_sql("SELECT * FROM collections LIMIT 5")
    assert ok


def test_non_select_rejected():
    for sql in ("DROP TABLE collections", "UPDATE collections SET amount_collected=0",
                "DELETE FROM visits"):
        ok, _ = validate_sql(sql)
        assert not ok


def test_table_whitelist_enforced():
    ok, why = validate_sql("SELECT * FROM audit_log")
    assert not ok and "not permitted" in why


def test_multi_statement_rejected():
    ok, _ = validate_sql("SELECT visit_id FROM visits; DELETE FROM visits")
    assert not ok


def test_prose_is_rejected_without_exception():
    ok, why = validate_sql("I cannot help with that request.")
    assert not ok
    assert "parse error" in why.lower() or "tokenizing" in why.lower()


def test_column_named_update_not_flagged():
    ok, _ = validate_sql("SELECT report_date FROM attorney_aging")
    assert ok


def test_explicit_mutation_request_rejected_before_generation(monkeypatch):
    def fail_generate_sql(question: str) -> str:
        raise AssertionError("unsafe mutation prompts should not reach the LLM")

    monkeypatch.setattr("app.nl2sql.executor.generate_sql", fail_generate_sql)
    monkeypatch.setattr("app.nl2sql.executor.write_audit", lambda **kwargs: None)

    result = run_query("DROP TABLE collections", user_id="test", user_role="da_analyst")

    assert not result.ok
    assert "unsafe database mutation" in result.error.lower()
