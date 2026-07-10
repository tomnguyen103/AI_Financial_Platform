"""Data-layer v2 tests: versioned migrations, enum CHECK constraints, WAL-once,
and the batched (no-N+1) anomaly driver-narrative path.

These use a fresh temporary SQLite file (not the shared session DB) so the CHECK
constraints — which only bind on a freshly created schema — are exercised, and so
migration/user_version assertions don't depend on session state.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

import app.db as db


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Point app.db at a brand-new SQLite file and initialize it."""
    dbfile = tmp_path / "v2.sqlite"
    monkeypatch.setattr(db, "DB_PATH", dbfile)
    db.init_db()
    return dbfile


def test_migration_sets_and_reads_user_version(fresh_db):
    with db.tx() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    expected = max(t for t, _ in db.MIGRATIONS)
    assert version == expected


def test_migrations_idempotent(fresh_db):
    """Running init_db again keeps user_version stable (no re-apply)."""
    with db.tx() as conn:
        first = conn.execute("PRAGMA user_version").fetchone()[0]
    db.init_db()
    with db.tx() as conn:
        second = conn.execute("PRAGMA user_version").fetchone()[0]
    assert first == second == max(t for t, _ in db.MIGRATIONS)


def test_wal_enabled_after_init(fresh_db):
    with db.tx() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_check_rejects_bad_severity(fresh_db):
    with pytest.raises(sqlite3.IntegrityError):
        with db.tx() as conn:
            conn.execute(
                "INSERT INTO alerts (alert_id, severity, status) VALUES (?,?,?)",
                ("a1", "P9", "open"),
            )


def test_check_rejects_bad_status(fresh_db):
    with pytest.raises(sqlite3.IntegrityError):
        with db.tx() as conn:
            conn.execute(
                "INSERT INTO alerts (alert_id, severity, status) VALUES (?,?,?)",
                ("a2", "P1", "closed"),
            )


def test_check_rejects_bad_stage(fresh_db):
    with pytest.raises(sqlite3.IntegrityError):
        with db.tx() as conn:
            conn.execute(
                "INSERT INTO model_registry (model_name, version, stage) VALUES (?,?,?)",
                ("m1", 1, "Live"),
            )


def test_check_allows_valid_enum_values(fresh_db):
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO alerts (alert_id, severity, status) VALUES (?,?,?)",
            ("ok1", "P2", "acknowledged"),
        )
        conn.execute(
            "INSERT INTO model_registry (model_name, version, stage) VALUES (?,?,?)",
            ("m_ok", 1, "Production"),
        )
    with db.tx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 1


def test_readonly_conn_still_works_under_wal(fresh_db):
    """WAL is persistent; the read-only path must still open and SELECT."""
    conn = db.get_readonly_conn()
    try:
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    finally:
        conn.close()


def _seed_settlement(conn, entity_key, vel, event_date):
    conn.execute(
        "INSERT INTO feature_values (feature_group, entity_key, event_date, features_json, computed_at) "
        "VALUES ('settlement_pipeline', ?, ?, ?, ?)",
        (entity_key, event_date, json.dumps({"settlement_velocity_change_30d": vel}), event_date),
    )


def test_batched_narrative_single_connection(fresh_db, monkeypatch):
    """The batched settlement fetch must use exactly ONE connection (no N+1)."""
    # Import after DB_PATH is patched so the module reads the temp DB.
    from app.anomaly import alerting

    with db.tx() as conn:
        _seed_settlement(conn, "alice|mva", -0.30, "2026-01-01")
        _seed_settlement(conn, "alice|mva", -0.40, "2026-02-01")  # newer wins
        _seed_settlement(conn, "bob|slip", -0.05, "2026-02-01")   # below -0.15 threshold

    # Count how many connections the batched path opens.
    calls = {"n": 0}
    real_tx = db.tx

    import contextlib

    @contextlib.contextmanager
    def counting_tx():
        calls["n"] += 1
        with real_tx() as conn:
            yield conn

    monkeypatch.setattr(alerting, "tx", counting_tx)

    result = alerting._settlement_online_features()
    assert calls["n"] == 1  # single query, no per-entity fan-out
    # Newest event_date wins for alice.
    assert result["alice|mva"]["settlement_velocity_change_30d"] == -0.40


def test_batched_narrative_text(fresh_db):
    """Narrative reports the largest (most negative) settlement-velocity driver."""
    from app.anomaly import alerting

    with db.tx() as conn:
        _seed_settlement(conn, "alice|mva", -0.40, "2026-02-01")
        _seed_settlement(conn, "bob|slip", -0.20, "2026-02-01")

    narrative = alerting._driver_narrative("facility-1")
    assert narrative == "Likely driver: settlement velocity down 40% (Attorney Alice)."
