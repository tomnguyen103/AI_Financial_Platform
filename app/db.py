"""Thin SQLite layer + schema bootstrap.

Two logical roles are modeled to honor the spec's least-privilege design:
  - get_conn(): full read/write connection (system / pipeline use)
  - get_readonly_conn(): SELECT-only connection used by the NL-to-SQL executor
    (enforced via SQLite query_only PRAGMA + URI mode=ro).
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from app.config import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_readonly_conn() -> sqlite3.Connection:
    """Read-only connection for untrusted NL-to-SQL execution."""
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA = """
-- ============ Curated business tables (also queried by NL-to-SQL) ============
CREATE TABLE IF NOT EXISTS visits (
    visit_id TEXT PRIMARY KEY, facility_id TEXT, case_type TEXT, visit_date TEXT,
    billing_status TEXT, billed_amount REAL, paid_amount REAL, provider_id TEXT
);
CREATE TABLE IF NOT EXISTS collections (
    collection_id TEXT PRIMARY KEY, facility_id TEXT, attorney_id TEXT, case_type TEXT,
    collection_date TEXT, amount_collected REAL, days_outstanding INTEGER
);
CREATE TABLE IF NOT EXISTS attorney_aging (
    attorney_id TEXT, facility_id TEXT, bucket_0_30 REAL, bucket_31_60 REAL,
    bucket_61_90 REAL, bucket_91_180 REAL, bucket_180_plus REAL, report_date TEXT,
    PRIMARY KEY (attorney_id, facility_id, report_date)
);
CREATE TABLE IF NOT EXISTS settlements (
    settlement_id TEXT PRIMARY KEY, attorney_id TEXT, case_type TEXT, open_date TEXT,
    close_date TEXT, settlement_amount REAL, settlement_status TEXT
);
CREATE TABLE IF NOT EXISTS lop (
    lop_id TEXT PRIMARY KEY, facility_id TEXT, case_type TEXT, issued_date TEXT,
    returned_date TEXT, status TEXT, rejection_reason TEXT
);

-- ============ Feature store ============
CREATE TABLE IF NOT EXISTS feature_values (
    feature_group TEXT, entity_key TEXT, event_date TEXT,
    features_json TEXT, computed_at TEXT,
    PRIMARY KEY (feature_group, entity_key, event_date)
);
CREATE TABLE IF NOT EXISTS feature_freshness (
    feature_group TEXT PRIMARY KEY, last_updated TEXT, row_count INTEGER
);

-- ============ Forecasting + registry ============
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, entity_type TEXT, entity_id TEXT,
    horizon_days INTEGER, forecast_date TEXT, target_date TEXT,
    predicted REAL, ci_lower REAL, ci_upper REAL, p50 REAL, p80 REAL, p95 REAL,
    model_name TEXT, model_version INTEGER, feature_snapshot_ts TEXT, generated_at TEXT
);
CREATE TABLE IF NOT EXISTS model_registry (
    model_name TEXT, version INTEGER, stage TEXT, mape REAL, rmse REAL, coverage REAL,
    bias REAL, params_json TEXT, artifact_path TEXT, created_at TEXT, git_commit TEXT,
    PRIMARY KEY (model_name, version)
);

-- ============ Anomaly + alerting ============
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY, created_at TEXT, severity TEXT, entity_type TEXT,
    entity_id TEXT, metric TEXT, expected REAL, actual REAL, deviation_pct REAL,
    detector TEXT, driver_narrative TEXT, status TEXT, acknowledged_by TEXT,
    acknowledged_at TEXT, payload_json TEXT
);

-- ============ Audit + sessions ============
CREATE TABLE IF NOT EXISTS audit_log (
    event_id TEXT PRIMARY KEY, ts TEXT, user_id TEXT, user_role TEXT, service TEXT,
    action TEXT, query_hash TEXT, input_phi_scan TEXT, output_phi_scan TEXT,
    llm_model TEXT, retrieved_sources TEXT, generated_sql TEXT,
    response_latency_ms INTEGER, session_id TEXT, detail_json TEXT
);
CREATE TABLE IF NOT EXISTS ingest_audit (
    run_id TEXT PRIMARY KEY, ts TEXT, source_record_count INTEGER,
    passed_validation INTEGER, failed_validation INTEGER, phi_masked INTEGER,
    schema_ok INTEGER, quality_ok INTEGER, operator TEXT, detail_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_collections_facility_date ON collections(facility_id, collection_date);
CREATE INDEX IF NOT EXISTS idx_visits_facility_date ON visits(facility_id, visit_date);
CREATE INDEX IF NOT EXISTS idx_forecasts_entity ON forecasts(entity_type, entity_id, horizon_days, forecast_date);
"""


def init_db() -> None:
    with tx() as conn:
        conn.executescript(SCHEMA)


if __name__ == "__main__":
    init_db()
    print(f"Initialized schema at {DB_PATH}")
