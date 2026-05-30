"""SQL safety validation (data design §7.2).

Defense in depth before any generated SQL reaches the database:
  1. Parse with sqlglot (reject unparseable).
  2. Statement must be a single SELECT.
  3. No forbidden DML/DDL keywords anywhere in the text.
  4. Every referenced table must be in the whitelist.

Execution is *also* run on a read-only connection (db.get_readonly_conn), so this
is the first of two independent guards.
"""
from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp

from app.nl2sql.glossary import ALLOWED_TABLES

FORBIDDEN = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
             "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "ATTACH", "PRAGMA",
             "VACUUM"]


def validate_sql(sql: str) -> tuple[bool, str]:
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return False, "Empty SQL."

    try:
        # Reject multiple statements early (parse_one would only see the first).
        statements = [s for s in sqlglot.parse(sql) if s is not None]
        if len(statements) > 1:
            return False, "Only a single statement is permitted."
        parsed = sqlglot.parse_one(sql)
    except Exception as e:  # noqa: BLE001 - surface parse error to caller
        return False, f"SQL parse error: {e}"

    if not isinstance(parsed, exp.Select):
        return False, "Only SELECT statements are permitted."

    sql_upper = sql.upper()
    for keyword in FORBIDDEN:
        # word-boundary-ish check to avoid matching column names like "updated_at"
        if f" {keyword} " in f" {sql_upper} " or sql_upper.startswith(keyword + " "):
            return False, f"Statement contains forbidden keyword: {keyword}"

    referenced = {t.name for t in parsed.find_all(exp.Table)}
    for table in referenced:
        if table not in ALLOWED_TABLES:
            return False, f"Table not permitted: {table}"
    if not referenced:
        return False, "No permitted table referenced."

    return True, "OK"
