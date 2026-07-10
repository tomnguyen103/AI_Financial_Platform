"""Ingestion pipeline: OData -> validate -> mask PHI -> raw/curated zones -> SQLite.

Spec refs: arch §2.1 data flow; PRD Module 1 (US-1.1..1.5, AC-1.1..1.7).

Flow per run:
  1. Pull from synthetic OData source.
  2. Pydantic-validate rows (drop+count failures).
  3. Schema-contract check (AC-1.3): on failure, abort and DO NOT overwrite curated.
  4. Write RAW zone parquet (retains PHI; "elevated access" simulated by zone dir).
  5. PHI-mask -> write CURATED zone parquet (no raw PHI; AC-1.4).
  6. Data-quality + record-count checks (AC-1.6).
  7. Load curated into SQLite business tables (consumed downstream + NL-to-SQL).
  8. Write ingest_audit row (AC-1.7).
"""
from __future__ import annotations

import datetime as dt
import json
import uuid

import pandas as pd

from app.config import CURATED_ZONE, RAW_ZONE, settings
from app.db import tx
from app.ingestion import quality
from app.ingestion.schemas import ENTITY_MODELS
from app.security.phi import PHI_FIELDS, mask_record


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _load_source() -> dict[str, list[dict]]:
    """Load rows from the synthetic OData source generator."""
    from app.ingestion.odata_source import generate as synth_generate
    return synth_generate()


def _validate_rows(entity: str, rows: list[dict]) -> tuple[list[dict], int]:
    model, _ = ENTITY_MODELS[entity]
    valid, failed = [], 0
    for r in rows:
        try:
            valid.append(model(**r).model_dump())
        except Exception:
            failed += 1
    return valid, failed


def _prior_counts() -> dict[str, int]:
    with tx() as conn:
        out = {}
        for entity in ENTITY_MODELS:
            try:
                out[entity] = conn.execute(f"SELECT COUNT(*) FROM {entity}").fetchone()[0]
            except Exception:
                out[entity] = 0
    return out


def run_ingest(*, fail_on_quality: bool = True) -> dict:
    run_id = str(uuid.uuid4())
    raw = _load_source()
    prior = _prior_counts()

    schema_ok = True
    quality_ok = True
    total_source = 0
    total_passed = 0
    total_failed = 0
    failures: list[str] = []
    warnings: list[str] = []
    curated_frames: dict[str, pd.DataFrame] = {}

    permissive = getattr(settings, "ingest_permissive", False)

    for entity, rows in raw.items():
        total_source += len(rows)
        valid, failed = _validate_rows(entity, rows)
        total_passed += len(valid)
        total_failed += failed

        df_raw = pd.DataFrame(valid)

        # An entity can legitimately be empty on a small sample DB (e.g. derived
        # attorney_aging when no consult has an aged balance). An empty frame has
        # no columns, which would trip the schema-contract check -> skip it
        # gracefully in permissive mode instead of aborting the whole load.
        if df_raw.empty:
            msg = f"[{entity}] no rows from source; skipped"
            if permissive:
                warnings.append(msg)
            else:
                failures.append(msg)
                schema_ok = False
            continue

        # 3. schema contract
        sc = quality.check_schema(entity, df_raw)
        if not sc.ok:
            schema_ok = False
            failures.extend(sc.failures)
            continue  # do not propagate this entity

        # 4. RAW zone (PHI retained)
        df_raw.to_parquet(RAW_ZONE / f"{entity}.parquet", index=False)

        # 5. PHI masking -> curated
        if entity in PHI_FIELDS:
            masked = [mask_record(entity, r) for r in valid]
        else:
            masked = valid
        df_cur = pd.DataFrame(masked)
        # drop PHI columns entirely from curated business columns
        _, curated_cols = ENTITY_MODELS[entity]
        df_cur = df_cur[[c for c in curated_cols if c in df_cur.columns]]

        # 6. quality + record count. In permissive mode a large count drop is a
        # warning (deliberate sample), not a blocking failure.
        q = quality.check_quality(entity, df_cur)
        rc = quality.check_record_count(
            entity, len(df_cur), prior.get(entity), blocking=not permissive
        )
        if not q.ok:
            quality_ok = False
            failures.extend(q.failures)
        if not rc.ok:
            quality_ok = False
            failures.extend(rc.failures)
        warnings.extend(q.warnings)
        warnings.extend(rc.warnings)

        curated_frames[entity] = df_cur

    abort = (not schema_ok) or (fail_on_quality and not quality_ok)

    if not abort:
        for entity, df_cur in curated_frames.items():
            df_cur.to_parquet(CURATED_ZONE / f"{entity}.parquet", index=False)
            _load_to_sqlite(entity, df_cur)

    # 8. audit
    with tx() as conn:
        conn.execute(
            """INSERT INTO ingest_audit (run_id, ts, source_record_count, passed_validation,
                 failed_validation, phi_masked, schema_ok, quality_ok, operator, detail_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (run_id, _utcnow(), total_source, total_passed, total_failed, 1,
             int(schema_ok), int(quality_ok), "system",
             json.dumps({"failures": failures, "warnings": warnings, "aborted": abort})),
        )

    return {
        "run_id": run_id, "source_records": total_source, "passed": total_passed,
        "failed": total_failed, "schema_ok": schema_ok, "quality_ok": quality_ok,
        "aborted": abort, "failures": failures, "warnings": warnings,
        "permissive": permissive,
    }


def _load_to_sqlite(entity: str, df: pd.DataFrame) -> None:
    cols = list(df.columns)
    placeholders = ",".join("?" * len(cols))
    collist = ",".join(cols)
    rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    with tx() as conn:
        conn.execute(f"DELETE FROM {entity}")
        conn.executemany(
            f"INSERT OR REPLACE INTO {entity} ({collist}) VALUES ({placeholders})", rows
        )


if __name__ == "__main__":
    from app.db import init_db
    init_db()
    print(json.dumps(run_ingest(), indent=2))
