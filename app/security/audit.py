"""Append-only audit logging (04 §9.4; PRD §9.1).

Append-only is enforced by convention: only `write_audit` inserts; no update/
delete paths exist. In prod this maps to an immutable store; here it's a plain
table with no exposed mutation API.

The log is now *cryptographically chained*: each row stores ``prev_hash`` (the
``row_hash`` of the row before it) and ``row_hash = sha256(prev_hash + canonical
JSON of this row's fields)``. Any in-place edit or deletion of a historical row
breaks the chain and is caught by ``verify_audit_chain()`` — the append-only
guarantee is therefore tamper-*evident*, not merely convention.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid

from app.db import tx

# Ordered field names that feed the canonical serialization / row hash. Keep in
# lockstep with the INSERT column list below and with verify_audit_chain().
_CHAINED_FIELDS = (
    "event_id", "ts", "user_id", "user_role", "service", "action", "query_hash",
    "input_phi_scan", "output_phi_scan", "llm_model", "retrieved_sources",
    "generated_sql", "response_latency_ms", "session_id", "detail_json",
)


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def query_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _row_hash(prev_hash: str, values: dict) -> str:
    """Deterministic SHA-256 over prev_hash + canonical JSON of the row fields."""
    canonical = json.dumps(
        {k: values[k] for k in _CHAINED_FIELDS},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    )
    return hashlib.sha256((prev_hash + canonical).encode()).hexdigest()


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
    values = {
        "event_id": event_id,
        "ts": _utcnow(),
        "user_id": user_id,
        "user_role": user_role,
        "service": service,
        "action": action,
        "query_hash": query_hash(query_text) if query_text else None,
        "input_phi_scan": input_phi_scan,
        "output_phi_scan": output_phi_scan,
        "llm_model": llm_model,
        "retrieved_sources": json.dumps(retrieved_sources or []),
        "generated_sql": generated_sql,
        "response_latency_ms": response_latency_ms,
        "session_id": session_id,
        "detail_json": json.dumps(detail or {}),
    }
    # Read the tail hash and insert within ONE transaction so the chain cannot
    # interleave between concurrent writers.
    with tx() as conn:
        row = conn.execute(
            "SELECT row_hash FROM audit_log ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        prev_hash = (row["row_hash"] if row and row["row_hash"] is not None else "")
        row_hash = _row_hash(prev_hash, values)
        conn.execute(
            """INSERT INTO audit_log (event_id, ts, user_id, user_role, service, action,
                 query_hash, input_phi_scan, output_phi_scan, llm_model, retrieved_sources,
                 generated_sql, response_latency_ms, session_id, detail_json,
                 prev_hash, row_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (*[values[k] for k in _CHAINED_FIELDS], prev_hash, row_hash),
        )
    return event_id


def verify_audit_chain() -> tuple[bool, int]:
    """Walk audit_log by rowid and recompute the hash chain.

    Returns ``(is_intact, first_broken_rowid)``. ``first_broken_rowid`` is -1 when
    the chain is intact. A break means a historical row's chained fields, its
    stored ``prev_hash``, or its ``row_hash`` no longer agree with the recomputed
    value — i.e. an in-place edit or deletion has occurred.
    """
    with tx() as conn:
        rows = conn.execute(
            f"SELECT rowid, {', '.join(_CHAINED_FIELDS)}, prev_hash, row_hash "
            "FROM audit_log ORDER BY rowid ASC"
        ).fetchall()

    prev_hash = ""
    for row in rows:
        values = {k: row[k] for k in _CHAINED_FIELDS}
        if row["prev_hash"] != prev_hash:
            return False, row["rowid"]
        expected = _row_hash(prev_hash, values)
        if row["row_hash"] != expected:
            return False, row["rowid"]
        prev_hash = row["row_hash"]
    return True, -1


def log_phi_detection(service: str, user_id: str, user_role: str, matches: list[str], where: str) -> None:
    """Compliance alert path for any PHI scanner trigger (04 §9.3)."""
    write_audit(
        user_id=user_id, user_role=user_role, service=service,
        action=f"PHI_DETECTED_{where}", input_phi_scan="blocked",
        detail={"matches": matches, "compliance_alert": True},
    )
