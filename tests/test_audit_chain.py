"""Tamper-evident audit hash-chain + migration tests."""
from __future__ import annotations

from app.db import tx
from app.security.audit import verify_audit_chain, write_audit


def _clear_audit() -> None:
    with tx() as conn:
        conn.execute("DELETE FROM audit_log")


def test_migration_bumped_version_and_columns_exist(_hermetic_db):
    """After init_db(), user_version is >= 2 and the chain columns exist."""
    with tx() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(audit_log)")}
    assert version >= 2
    assert {"prev_hash", "row_hash"} <= cols


def test_two_writes_are_hash_linked(_hermetic_db):
    _clear_audit()
    write_audit(user_id="u1", user_role="admin", service="test", action="A1")
    write_audit(user_id="u2", user_role="admin", service="test", action="A2")

    with tx() as conn:
        rows = conn.execute(
            "SELECT prev_hash, row_hash FROM audit_log ORDER BY rowid ASC"
        ).fetchall()

    assert len(rows) == 2
    # First row's prev_hash is the empty-string genesis anchor.
    assert rows[0]["prev_hash"] == ""
    assert rows[0]["row_hash"]
    # Second row is chained to the first.
    assert rows[1]["prev_hash"] == rows[0]["row_hash"]
    assert rows[1]["row_hash"] != rows[0]["row_hash"]


def test_verify_intact_on_clean_log(_hermetic_db):
    _clear_audit()
    for i in range(3):
        write_audit(user_id=f"u{i}", user_role="admin", service="test", action=f"A{i}")

    intact, broken = verify_audit_chain()
    assert intact is True
    assert broken == -1


def test_tampering_a_historical_row_is_detected(_hermetic_db):
    _clear_audit()
    write_audit(user_id="u1", user_role="admin", service="test", action="A1")
    write_audit(user_id="u2", user_role="admin", service="test", action="A2")
    write_audit(user_id="u3", user_role="admin", service="test", action="A3")

    # Directly mutate the SECOND row's user_id, bypassing write_audit — this is
    # exactly the tamper the chain must expose.
    with tx() as conn:
        target = conn.execute(
            "SELECT rowid FROM audit_log ORDER BY rowid ASC LIMIT 1 OFFSET 1"
        ).fetchone()["rowid"]
        conn.execute(
            "UPDATE audit_log SET user_id='HACKED' WHERE rowid=?", (target,)
        )

    intact, broken = verify_audit_chain()
    assert intact is False
    assert broken == target
