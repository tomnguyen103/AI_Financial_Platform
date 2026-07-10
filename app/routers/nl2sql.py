"""NL-to-SQL endpoints (PRD Module 6; arch §2.6).

`/nl2sql/query` returns the generated SQL + a capped result preview (see
PREVIEW_ROW_CAP in app.nl2sql.executor). `/nl2sql/export` returns the same
query rendered as a CSV download with the full row cap.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.nl2sql.executor import iter_csv, run_query
from app.security.auth import User, require

router = APIRouter(prefix="/nl2sql", tags=["nl2sql"])


class QueryRequest(BaseModel):
    # Bound the free-text field so an oversized body can't drive expensive SQL
    # generation / validation work before the rate limiter trips.
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class Nl2SqlResponse(BaseModel):
    ok: bool
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    error: str
    latency_ms: int
    attempts: int


@router.post("/query", response_model=Nl2SqlResponse)
def query(req: QueryRequest, user: User = Depends(require("nl2sql:use"))) -> Nl2SqlResponse:
    r = run_query(req.question, user_id=user.user_id, user_role=user.role,
                  session_id=req.session_id, preview=True)
    return Nl2SqlResponse(
        ok=r.ok,
        sql=r.sql,
        columns=r.columns,
        rows=r.rows,
        row_count=r.row_count,
        truncated=r.truncated,
        error=r.error,
        latency_ms=r.latency_ms,
        attempts=r.attempts,
    )


@router.post("/export")
def export(req: QueryRequest, user: User = Depends(require("nl2sql:use"))):
    r = run_query(req.question, user_id=user.user_id, user_role=user.role,
                  session_id=req.session_id)
    if not r.ok:
        return PlainTextResponse(f"# error: {r.error}\n", status_code=400)
    return StreamingResponse(
        iter_csv(r), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=query_result.csv"},
    )
