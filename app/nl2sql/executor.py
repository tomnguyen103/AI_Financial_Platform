"""NL-to-SQL orchestration + safe execution (arch §2.6; data design §7.2/§7.3).

Pipeline: generate SQL -> validate -> (retry once on failure) -> execute on a
read-only connection with a hard row cap and a wall-clock timeout -> optional CSV
export -> audit. Two independent safety layers protect the database:
  * validator.validate_sql (SELECT-only, table whitelist, keyword denylist)
  * db.get_readonly_conn (URI mode=ro + PRAGMA query_only)

Timeout: SQLite has no per-statement timeout, so a watchdog thread calls
conn.interrupt() after TIMEOUT_S, raising OperationalError in the executing
thread (documented in implementation-notes.md).
"""
from __future__ import annotations

import csv
import io
import re
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.db import get_readonly_conn
from app.nl2sql.generator import fix_sql, generate_sql
from app.nl2sql.validator import validate_sql
from app.security.audit import write_audit

ROW_CAP = 10_000
PREVIEW_ROW_CAP = 200  # interactive /nl2sql/query response cap; CSV export keeps the full ROW_CAP
TIMEOUT_S = 30
MUTATION_PATTERNS = [
    r"\bdrop\s+table\b",
    r"\bdelete\s+from\b",
    r"\bupdate\s+\w+\s+set\b",
    r"\binsert\s+into\b",
    r"\balter\s+table\b",
    r"\bcreate\s+table\b",
    r"\btruncate\s+table\b",
]


@dataclass
class QueryResult:
    ok: bool
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str = ""
    latency_ms: int = 0
    attempts: int = 0


def _execute(sql: str) -> tuple[list[str], list[list], bool]:
    """Run SELECT on a read-only conn with a row cap and interrupt watchdog."""
    conn = get_readonly_conn()
    timed_out = {"hit": False}

    def _watchdog() -> None:
        timed_out["hit"] = True
        conn.interrupt()

    timer = threading.Timer(TIMEOUT_S, _watchdog)
    timer.start()
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        fetched = cur.fetchmany(ROW_CAP + 1)
        truncated = len(fetched) > ROW_CAP
        rows = [list(r) for r in fetched[:ROW_CAP]]
        return columns, rows, truncated
    except Exception as e:  # noqa: BLE001
        if timed_out["hit"]:
            raise TimeoutError(f"Query exceeded {TIMEOUT_S}s and was cancelled.") from e
        raise
    finally:
        timer.cancel()
        conn.close()


def to_csv(result: QueryResult) -> str:
    """Render a successful result as CSV text (PRD AC-6.6 export)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    if result.columns:
        writer.writerow(result.columns)
    writer.writerows(result.rows)
    return buf.getvalue()


def iter_csv(result: QueryResult) -> Iterator[str]:
    """Yield a successful result as CSV row-by-row (header first, then rows).

    Streaming the export keeps a large (up to ROW_CAP) download from being fully
    materialised in memory; each yielded chunk is a single serialised CSV row.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)

    def _flush() -> str:
        chunk = buf.getvalue()
        buf.seek(0)
        buf.truncate(0)
        return chunk

    if result.columns:
        writer.writerow(result.columns)
        yield _flush()
    for row in result.rows:
        writer.writerow(row)
        yield _flush()


def _explicit_mutation_request(question: str) -> bool:
    q = question.lower()
    return any(re.search(pattern, q) for pattern in MUTATION_PATTERNS)


def run_query(question: str, *, user_id: str = "anon", user_role: str = "da_analyst",
              session_id: str | None = None, preview: bool = False) -> QueryResult:
    """Run the NL-to-SQL pipeline.

    ``preview=True`` (used by the interactive /nl2sql/query endpoint) caps the
    rows returned in the response to PREVIEW_ROW_CAP; the CSV export path
    (preview=False, the default) keeps up to the full ROW_CAP so downloads
    aren't truncated more aggressively than the documented hard cap.
    """
    start = time.time()
    session_id = session_id or str(uuid.uuid4())

    if _explicit_mutation_request(question):
        latency = int((time.time() - start) * 1000)
        reason = "Unsafe database mutation request rejected before SQL generation."
        write_audit(user_id=user_id, user_role=user_role, service="nl2sql",
                    action="rejected", query_text=question, generated_sql="",
                    response_latency_ms=latency, session_id=session_id,
                    detail={"reason": reason, "attempts": 0})
        return QueryResult(ok=False, error=reason, latency_ms=latency, attempts=0)

    # 1. Generate (attempt 1).
    sql = generate_sql(question)
    ok, reason = validate_sql(sql)
    attempts = 1

    # 2. Retry once: hand the validator error back to the LLM (§7.3 step 2).
    if not ok:
        fixed = fix_sql(question, sql, reason)
        ok2, reason2 = validate_sql(fixed)
        attempts = 2
        if ok2:
            sql, ok, reason = fixed, True, "OK"
        else:
            sql, reason = fixed, reason2  # surface the latest failure

    if not ok:
        latency = int((time.time() - start) * 1000)
        write_audit(user_id=user_id, user_role=user_role, service="nl2sql",
                    action="rejected", query_text=question, generated_sql=sql,
                    response_latency_ms=latency, session_id=session_id,
                    detail={"reason": reason, "attempts": attempts})
        return QueryResult(ok=False, sql=sql, error=f"Could not produce safe SQL: {reason}",
                           latency_ms=latency, attempts=attempts)

    # 3. Execute.
    try:
        columns, rows, truncated = _execute(sql)
    except TimeoutError as e:
        latency = int((time.time() - start) * 1000)
        write_audit(user_id=user_id, user_role=user_role, service="nl2sql",
                    action="timeout", query_text=question, generated_sql=sql,
                    response_latency_ms=latency, session_id=session_id,
                    detail={"attempts": attempts})
        return QueryResult(ok=False, sql=sql, error=str(e), latency_ms=latency, attempts=attempts)
    except Exception as e:  # noqa: BLE001
        latency = int((time.time() - start) * 1000)
        write_audit(user_id=user_id, user_role=user_role, service="nl2sql",
                    action="exec_error", query_text=question, generated_sql=sql,
                    response_latency_ms=latency, session_id=session_id,
                    detail={"error": str(e), "attempts": attempts})
        return QueryResult(ok=False, sql=sql, error=f"Execution error: {e}",
                           latency_ms=latency, attempts=attempts)

    latency = int((time.time() - start) * 1000)
    write_audit(user_id=user_id, user_role=user_role, service="nl2sql",
                action="answer", query_text=question, generated_sql=sql,
                response_latency_ms=latency, session_id=session_id,
                detail={"row_count": len(rows), "truncated": truncated, "attempts": attempts})
    if preview and len(rows) > PREVIEW_ROW_CAP:
        rows = rows[:PREVIEW_ROW_CAP]
        truncated = True
    return QueryResult(ok=True, sql=sql, columns=columns, rows=rows, row_count=len(rows),
                       truncated=truncated, latency_ms=latency, attempts=attempts)


if __name__ == "__main__":
    from app.db import init_db
    init_db()
    r = run_query("Show total collected by facility")
    print("SQL:", r.sql)
    print("ok:", r.ok, "rows:", r.row_count, "truncated:", r.truncated)
    if r.ok:
        print(r.columns)
        for row in r.rows[:5]:
            print(row)
    else:
        print("error:", r.error)
