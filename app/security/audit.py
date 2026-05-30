"""Append-only audit logging (04 §9.4; PRD §9.1).

Append-only is enforced by convention: only `write_audit` inserts; no update/
delete paths exist. In prod this maps to an immutable store; here it's a plain
table with no exposed mutation API.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid

from app.db import tx


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def query_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_audit(
    *,
    user_id: str,
    user_role: str,
    service: str,
    action: str,
    query_text: str | None = None,
    input_phi_scan: str = "n/a",
    output_phi_scan: str = "n/a",
    llm_model: str | None = None,
    retrieved_sources: list[str] | None = None,
    generated_sql: str | None = None,
    response_latency_ms: int | None = None,
    session_id: str | None = None,
    detail: dict | None = None,
) -> str:
    event_id = str(uuid.uuid4())
    with tx() as conn:
        conn.execute(
            """INSERT INTO audit_log (event_id, ts, user_id, user_role, service, action,
                 query_hash, input_phi_scan, output_phi_scan, llm_model, retrieved_sources,
                 generated_sql, response_latency_ms, session_id, detail_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id, _utcnow(), user_id, user_role, service, action,
                query_hash(query_text) if query_text else None,
                input_phi_scan, output_phi_scan, llm_model,
                json.dumps(retrieved_sources or []), generated_sql,
                response_latency_ms, session_id, json.dumps(detail or {}),
            ),
        )
    return event_id


def log_phi_detection(service: str, user_id: str, user_role: str, matches: list[str], where: str) -> None:
    """Compliance alert path for any PHI scanner trigger (04 §9.3)."""
    write_audit(
        user_id=user_id, user_role=user_role, service=service,
        action=f"PHI_DETECTED_{where}", input_phi_scan="blocked",
        detail={"matches": matches, "compliance_alert": True},
    )
