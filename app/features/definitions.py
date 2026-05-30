"""Feature group definitions (data design §3; PRD Module 2 / AC-2.5).

Version-controlled in Git per AC-2.5. Each group declares its entity key(s),
event-timestamp column, and the feature names it produces. The compute layer
reads these; the store layer serves them point-in-time.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    entity_keys: tuple[str, ...]
    event_ts: str
    features: tuple[str, ...]


FEATURE_GROUPS: dict[str, FeatureGroup] = {
    "facility_collections": FeatureGroup(
        "facility_collections", ("facility_id",), "date",
        ("collections_1d", "collections_7d_rolling", "collections_30d_rolling",
         "collections_mom_growth", "collections_yoy_growth", "day_of_week",
         "month", "is_month_end"),
    ),
    "attorney_aging": FeatureGroup(
        "attorney_aging", ("attorney_id",), "report_date",
        ("bucket_0_30_balance", "bucket_31_60_balance", "bucket_61_90_balance",
         "bucket_91_180_balance", "bucket_180_plus_balance", "total_outstanding",
         "pct_180_plus", "avg_days_outstanding", "aging_migration_rate_30d"),
    ),
    "visit_velocity": FeatureGroup(
        "visit_velocity", ("facility_id", "case_type"), "date",
        ("new_visits_7d", "new_visits_30d", "visit_cancellation_rate_7d",
         "new_case_open_rate_7d", "visit_billing_conversion_rate"),
    ),
    "settlement_pipeline": FeatureGroup(
        "settlement_pipeline", ("attorney_id", "case_type"), "date",
        ("open_settlements_count", "settlements_closed_30d",
         "avg_days_to_settlement_90d", "settlement_velocity_change_30d",
         "high_value_open_count"),
    ),
    "lop_metrics": FeatureGroup(
        "lop_metrics", ("facility_id",), "date",
        ("lop_issued_7d", "lop_returned_7d", "lop_rejection_rate_30d",
         "lop_turnaround_days_p50", "lop_turnaround_days_p90", "lop_backlog_count"),
    ),
}

HIGH_VALUE_SETTLEMENT_THRESHOLD = 50000.0  # data design §3.4 "$X TBD with DA"
