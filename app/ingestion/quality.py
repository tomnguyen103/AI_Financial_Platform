"""Lightweight data-quality + schema-contract checks (arch §2.1; PRD AC-1.3/1.6).

Stands in for Great Expectations. Each check returns a structured result; any
failure blocks the curated-zone write (the pipeline keeps the prior day's data).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.ingestion.schemas import ENTITY_MODELS


@dataclass
class CheckResult:
    ok: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.failures.append(msg)


# Expected column sets derived from the Pydantic models (incl. PHI fields).
def _expected_columns(entity: str) -> set[str]:
    model, _ = ENTITY_MODELS[entity]
    return set(model.model_fields.keys())


def check_schema(entity: str, df: pd.DataFrame) -> CheckResult:
    """Detect added/renamed/dropped fields vs. the contract (AC-1.3)."""
    res = CheckResult()
    expected = _expected_columns(entity)
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected
    # Optional PHI/nullable fields may legitimately be absent; only required ones block.
    required = {
        f for f, info in ENTITY_MODELS[entity][0].model_fields.items()
        if info.is_required()
    }
    blocking_missing = (missing & required)
    if blocking_missing:
        res.fail(f"[{entity}] missing required columns: {sorted(blocking_missing)}")
    if extra:
        res.fail(f"[{entity}] unexpected new columns (possible upstream schema change): {sorted(extra)}")
    return res


# Value-range / null-rate rules per entity (AC-1.6).
_RULES = {
    "collections": {"non_null": ["facility_id", "collection_date", "amount_collected"],
                    "non_negative": ["amount_collected"]},
    "visits": {"non_null": ["facility_id", "visit_date", "billing_status"],
               "non_negative": ["billed_amount", "paid_amount"]},
    "attorney_aging": {"non_null": ["attorney_id", "report_date"],
                       "non_negative": ["bucket_0_30", "bucket_180_plus"]},
    "settlements": {"non_null": ["settlement_id", "open_date"],
                    "non_negative": ["settlement_amount"]},
    "lop": {"non_null": ["lop_id", "issued_date", "status"], "non_negative": []},
}


def check_quality(entity: str, df: pd.DataFrame, max_null_rate: float = 0.02) -> CheckResult:
    res = CheckResult()
    rules = _RULES.get(entity, {})
    for col in rules.get("non_null", []):
        if col not in df.columns:
            continue
        null_rate = float(df[col].isna().mean())
        if null_rate > max_null_rate:
            res.fail(f"[{entity}.{col}] null rate {null_rate:.1%} exceeds {max_null_rate:.0%}")
    for col in rules.get("non_negative", []):
        if col in df.columns and (df[col] < 0).any():
            res.fail(f"[{entity}.{col}] contains negative values")
    return res


def check_record_count(entity: str, current: int, prior: int | None,
                       max_drop: float = 0.20, *, blocking: bool = True) -> CheckResult:
    """Alert if record count drops > 20% vs. prior run (US-1.4 / AC-1.6).

    A large drop is usually a real upstream problem, so by default it blocks the
    curated write. But it also fires on a *deliberate* dataset change (e.g.
    switching PG_DATABASE from the full DB to a 100-row sample). Pass
    blocking=False (the pipeline does this when settings.ingest_permissive is set)
    to record it as a warning instead of aborting.
    """
    res = CheckResult()
    if prior and prior > 0:
        drop = (prior - current) / prior
        if drop > max_drop:
            msg = f"[{entity}] record count dropped {drop:.1%} (prior={prior}, now={current})"
            if blocking:
                res.fail(msg)
            else:
                res.warnings.append(msg)
    return res
