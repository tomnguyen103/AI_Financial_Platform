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
    # foreign_keys is a per-connection PRAGMA and must be set every time.
    # journal_mode=WAL is a persistent DB-level setting applied once in init_db().
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
-- ============ Money-as-REAL: deliberate decision (do NOT convert to integer cents) ============
-- Money columns (billed_amount, paid_amount, amount_collected, settlement_amount,
-- aging buckets, forecast/alert expected/actual, etc.) are REAL on purpose.
-- The NL-to-SQL feature surfaces RAW column values directly to end users, so integer
-- cents would render as unformatted integers (e.g. 12345 instead of 123.45) — a UX
-- regression. Tradeoff accepted: retain REAL for raw-display fidelity; callers doing
-- SUM/AVG aggregations should ROUND(...) at read time to avoid float drift.
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
    model_name TEXT, version INTEGER,
    stage TEXT CHECK (stage IN ('Staging','Production','Archived')),
    mape REAL, rmse REAL, coverage REAL,
    bias REAL, params_json TEXT, artifact_path TEXT, created_at TEXT, git_commit TEXT,
    PRIMARY KEY (model_name, version)
);

-- ============ Anomaly + alerting ============
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY, created_at TEXT,
    severity TEXT CHECK (severity IN ('P1','P2','P3')),
    entity_type TEXT,
    entity_id TEXT, metric TEXT, expected REAL, actual REAL, deviation_pct REAL,
    detector TEXT, driver_narrative TEXT,
    status TEXT CHECK (status IN ('open','acknowledged')),
    acknowledged_by TEXT,
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
CREATE INDEX IF NOT EXISTS idx_visits_billing_status ON visits(billing_status);
CREATE INDEX IF NOT EXISTS idx_collections_attorney ON collections(attorney_id, collection_date);
CREATE INDEX IF NOT EXISTS idx_collections_case_type ON collections(case_type, collection_date);
CREATE INDEX IF NOT EXISTS idx_alerts_status_created ON alerts(status, created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_severity_created ON alerts(severity, created_at);
"""


# ---------------------------------------------------------------------------
# Versioned migrations (PRAGMA user_version)
# ---------------------------------------------------------------------------
# The base SCHEMA above uses CREATE TABLE IF NOT EXISTS, so it is the v0/bootstrap
# and is a silent no-op on an existing DB. That means a future column/constraint
# change would never apply. This runner is the forward-looking mechanism: it reads
# PRAGMA user_version, applies each migration whose target is greater (each in its
# own transaction), and advances user_version. It is idempotent and safe to run on
# every startup.
#
# NOTE: enum CHECK constraints live directly in the base CREATE TABLE statements
# (SQLite cannot ALTER TABLE to add a CHECK), so they only bind on a FRESH DB. An
# existing DB created before this change keeps its old, unconstrained tables until
# rebuilt — acceptable for this app's regenerable synthetic data.
MIGRATIONS: list[tuple[int, str]] = [
    # v1: baseline marker. The CHECK-constrained schema is expressed in the base
    # CREATE TABLE statements (can't be ALTERed in). This documented no-op records
    # that a v1-aware runner initialized the DB and is the template for real future
    # migrations (each entry: (target_version, sql_script)).
    (1, "-- v1 baseline: schema with enum CHECK constraints; no forward DDL needed.\n"),
]


def _run_migrations(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for target, sql in MIGRATIONS:
        if target <= current:
            continue
        # executescript() implicitly commits any pending work, runs the migration,
        # then we stamp user_version and commit — each migration in its own tx.
        conn.executescript(sql)
        conn.execute(f"PRAGMA user_version = {int(target)}")
        conn.commit()


def init_db() -> None:
    conn = get_conn()
    try:
        # WAL is a persistent DB-level setting; apply once here (not per-connection).
        # Must run outside an open transaction, so do it before any DML/DDL.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.commit()
        _run_migrations(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized schema at {DB_PATH}")
