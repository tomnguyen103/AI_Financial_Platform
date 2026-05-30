"""LLM SQL generation (arch §2.6; data design §7.1).

Builds the schema-grounded prompt and asks the LLM for a single SELECT. Offline,
the deterministic stub in llm.client._stub_sql produces a plausible query (the
system prompt below contains the "Return ONLY a SQL" trigger the stub keys on).

A second `fix_sql` entry point implements the §7.3 retry: hand the LLM the bad
SQL + validator error and ask for a corrected query.
"""
from __future__ import annotations

import re

from app.llm.client import get_llm
from app.nl2sql.glossary import schema_prompt

SYSTEM_PROMPT = """You are a SQL generator for a medical billing analytics database (SQLite).
Return ONLY a SQL SELECT statement — no prose, no markdown fences, no explanation.

Rules:
1. SELECT statements only. Never INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.
2. Only use the tables and columns listed below. Never invent columns.
3. Use the business glossary to map domain terms to SQL.
4. Add a sensible LIMIT (<= 1000) unless the question asks for an aggregate.
5. Dates are stored as ISO TEXT (YYYY-MM-DD); compare them as strings.

{schema}
"""


def _strip_fences(sql: str) -> str:
    """Remove ```sql ... ``` fences and leading labels the LLM may add."""
    sql = sql.strip()
    fence = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", sql, re.DOTALL | re.IGNORECASE)
    if fence:
        sql = fence.group(1).strip()
    if sql.lower().startswith("sql:"):
        sql = sql[4:].strip()
    return sql.strip().rstrip(";")


def generate_sql(question: str) -> str:
    llm = get_llm()
    system = SYSTEM_PROMPT.format(schema=schema_prompt())
    user = f"Question: {question}\n\nReturn ONLY the SQL SELECT statement."
    raw = llm.complete(system, user)
    return _strip_fences(raw)


def fix_sql(question: str, bad_sql: str, error: str) -> str:
    """§7.3 attempt 2: ask the LLM to repair SQL that failed validation."""
    llm = get_llm()
    system = SYSTEM_PROMPT.format(schema=schema_prompt())
    user = (
        f"Question: {question}\n\n"
        f"This SQL was rejected:\n{bad_sql}\n\n"
        f"Reason: {error}\n\n"
        "Return ONLY a corrected SQL SELECT statement that fixes the problem."
    )
    raw = llm.complete(system, user)
    return _strip_fences(raw)
