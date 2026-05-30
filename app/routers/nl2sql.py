"""NL-to-SQL endpoints (PRD Module 6; arch §2.6).

`/nl2sql/query` returns the generated SQL + result preview. `/nl2sql/export`
returns the same query rendered as a CSV download.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.nl2sql.executor import run_query, to_csv
from app.security.auth import User, require

router = APIRouter(prefix="/nl2sql", tags=["nl2sql"])


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


@router.post("/query")
def query(req: QueryRequest, user: User = Depends(require("nl2sql:use"))) -> dict:
    r = run_query(req.question, user_id=user.user_id, user_role=user.role,
                  session_id=req.session_id)
    return {
        "ok": r.ok,
        "sql": r.sql,
        "columns": r.columns,
        "rows": r.rows,
        "row_count": r.row_count,
        "truncated": r.truncated,
        "error": r.error,
        "latency_ms": r.latency_ms,
        "attempts": r.attempts,
    }


@router.post("/export", response_class=PlainTextResponse)
def export(req: QueryRequest, user: User = Depends(require("nl2sql:use"))) -> PlainTextResponse:
    r = run_query(req.question, user_id=user.user_id, user_role=user.role,
                  session_id=req.session_id)
    if not r.ok:
        return PlainTextResponse(f"# error: {r.error}\n", status_code=400)
    return PlainTextResponse(
        to_csv(r), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=query_result.csv"},
    )
