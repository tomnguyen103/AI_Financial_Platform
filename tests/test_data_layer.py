"""Data-layer tests: required indexes exist and the read-only connection
genuinely rejects writes (least-privilege design, see app/db.py)."""
from __future__ import annotations

import sqlite3

import pytest

from app.db import get_readonly_conn, tx

EXPECTED_INDEXES = {
    "idx_visits_billing_status",
    "idx_collections_attorney",
    "idx_collections_case_type",
    "idx_alerts_status_created",
    "idx_alerts_severity_created",
}


def test_expected_indexes_exist():
    with tx() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    names = {r[0] for r in rows}
    missing = EXPECTED_INDEXES - names
    assert not missing, f"missing indexes: {missing}"


def test_readonly_conn_rejects_write():
    conn = get_readonly_conn()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO alerts (alert_id, created_at, severity, entity_type, entity_id, "
                "metric, expected, actual, deviation_pct, detector, driver_narrative, status, "
                "acknowledged_by, acknowledged_at, payload_json) "
                "VALUES ('x','x','P3','facility','x','x',0,0,0,'x','x','open',NULL,NULL,'{}')"
            )
    finally:
        conn.close()


def test_readonly_conn_allows_select():
    conn = get_readonly_conn()
    try:
        conn.execute("SELECT 1").fetchone()
    finally:
        conn.close()
