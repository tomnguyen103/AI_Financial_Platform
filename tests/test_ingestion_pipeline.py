"""Ingestion pipeline tests (PRD Module 1).

Covers the end-to-end run (source -> validate -> mask -> zones -> quality ->
SQLite -> ingest_audit) plus focused tests for the quality gate and the
Pydantic schema-validation failure path.
"""
from __future__ import annotations

import pandas as pd

from app.db import tx
from app.ingestion import pipeline, quality
from app.ingestion.schemas import ENTITY_MODELS
from app.security.phi import tokenize


def test_run_ingest_end_to_end(seeded_db):
    """One synthetic ingest exercises the whole pipeline and writes an audit row."""
    result = seeded_db["ingest"]
    assert result["source_records"] > 0
    assert result["passed"] == result["source_records"]  # synthetic feed is all-valid
    assert result["failed"] == 0
    assert result["schema_ok"] and result["quality_ok"]
    assert result["aborted"] is False

    # Curated data landed in the business tables.
    with tx() as conn:
        collections = conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
        visits = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        # The audit row was written with matching counts.
        audit = conn.execute(
            "SELECT source_record_count, passed_validation, failed_validation, phi_masked, "
            "schema_ok, quality_ok FROM ingest_audit WHERE run_id=?",
            (result["run_id"],),
        ).fetchone()
    assert collections > 0 and visits > 0
    assert audit["source_record_count"] == result["source_records"]
    assert audit["passed_validation"] == result["passed"]
    assert audit["phi_masked"] == 1
    assert audit["schema_ok"] == 1 and audit["quality_ok"] == 1


def test_curated_visits_have_no_raw_phi_columns(seeded_db):
    """PHI columns are dropped from the curated business table (AC-1.4)."""
    with tx() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(visits)").fetchall()]
    for phi_col in ("patient_name", "patient_dob", "patient_ssn_last4", "patient_address"):
        assert phi_col not in cols


def test_phi_masking_is_deterministic_hmac():
    """Same PHI value -> same token (referential integrity), tokenized not raw."""
    masked = pipeline.mask_record("visits", {"visit_id": "v1", "patient_name": "Alex Doe"})
    assert masked["patient_name"] == tokenize("Alex Doe")
    assert masked["patient_name"].startswith("tok_")
    assert masked["visit_id"] == "v1"  # non-PHI untouched


def test_validate_rows_counts_and_samples_failures():
    """A row missing a required field is dropped, counted, and sampled (not raw)."""
    good = {"collection_id": "c1", "facility_id": "round_rock", "attorney_id": "smith",
            "case_type": "PI", "collection_date": "2026-01-01",
            "amount_collected": 100.0, "days_outstanding": 10}
    bad = dict(good)
    del bad["amount_collected"]  # required float missing -> ValidationError
    valid, failed, samples = pipeline._validate_rows("collections", [good, bad])
    assert len(valid) == 1
    assert failed == 1
    assert samples  # a field-level summary was recorded
    assert any("amount_collected" in s for s in samples)


def test_check_schema_flags_missing_required_and_extra_columns():
    model, _ = ENTITY_MODELS["collections"]
    # Missing a required column ('amount_collected') and carrying an unexpected one.
    df = pd.DataFrame([{"collection_id": "c1", "facility_id": "f", "attorney_id": "a",
                        "case_type": "PI", "collection_date": "2026-01-01",
                        "days_outstanding": 5, "surprise_col": 1}])
    res = quality.check_schema("collections", df)
    assert not res.ok
    joined = " ".join(res.failures)
    assert "missing required columns" in joined
    assert "unexpected new columns" in joined


def test_check_quality_flags_negative_and_null_rate():
    df = pd.DataFrame({
        "facility_id": ["f", "f", "f", "f", "f"],
        "collection_date": ["2026-01-01"] * 5,
        "amount_collected": [-1.0, 2.0, 3.0, 4.0, 5.0],  # a negative value
    })
    res = quality.check_quality("collections", df)
    assert not res.ok
    assert any("negative values" in f for f in res.failures)

    # High null rate on a non_null column trips the null-rate rule.
    nulls = pd.DataFrame({
        "facility_id": [None, None, "f"],
        "collection_date": ["2026-01-01"] * 3,
        "amount_collected": [1.0, 2.0, 3.0],
    })
    res2 = quality.check_quality("collections", nulls)
    assert not res2.ok
    assert any("null rate" in f for f in res2.failures)


def test_check_record_count_blocks_or_warns_on_large_drop():
    # Blocking mode: a >20% drop is a failure.
    blocked = quality.check_record_count("collections", current=50, prior=100)
    assert not blocked.ok
    assert blocked.failures

    # Permissive mode: same drop is a warning, not a failure.
    warned = quality.check_record_count("collections", current=50, prior=100, blocking=False)
    assert warned.ok
    assert warned.warnings

    # No prior data -> nothing to compare, passes clean.
    first_run = quality.check_record_count("collections", current=50, prior=0)
    assert first_run.ok and not first_run.failures


def test_run_ingest_aborts_and_preserves_curated_on_schema_failure(monkeypatch):
    """A schema-contract failure aborts the load without touching business tables.

    An entity that yields zero rows from the source fails the schema contract in
    (non-permissive) default mode, which must abort the whole load so the prior
    day's curated data is preserved.
    """
    def _empty_source() -> dict[str, list[dict]]:
        return {"collections": []}

    monkeypatch.setattr(pipeline, "_load_source", _empty_source)
    # Count rows before; the aborted run must not change them.
    with tx() as conn:
        before = conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0]

    result = pipeline.run_ingest()
    assert result["schema_ok"] is False
    assert result["aborted"] is True

    with tx() as conn:
        after = conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
    assert after == before  # curated business table untouched on abort
