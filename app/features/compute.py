"""Nightly feature computation from the curated SQLite zone (PRD AC-2.1).

Writes one row per (feature_group, entity_key, event_date) into feature_values
and updates feature_freshness (AC-2.3/2.4).

Some spec features lack a direct source field in the synthetic feed; these are
approximated and noted here and in implementation-notes.md:
  - visit_cancellation_rate_7d: no 'cancelled' status exists -> proxied by the
    share of 'pending' visits (unbilled). Swap to a real cancellation field when
    the OData feed provides one.
  - aging_migration_rate_30d: approximated as the 30-day change in pct_180_plus.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from app.db import tx
from app.features.definitions import FEATURE_GROUPS, HIGH_VALUE_SETTLEMENT_THRESHOLD


def _read(table: str) -> pd.DataFrame:
    with tx() as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_group(group: str, records: list[tuple[str, str, dict]]) -> int:
    now = _utcnow()
    with tx() as conn:
        conn.execute("DELETE FROM feature_values WHERE feature_group=?", (group,))
        conn.executemany(
            """INSERT OR REPLACE INTO feature_values
               (feature_group, entity_key, event_date, features_json, computed_at)
               VALUES (?,?,?,?,?)""",
            [(group, ek, ed, json.dumps(fv), now) for ek, ed, fv in records],
        )
        conn.execute(
            "INSERT OR REPLACE INTO feature_freshness (feature_group, last_updated, row_count) VALUES (?,?,?)",
            (group, now, len(records)),
        )
    return len(records)


def compute_facility_collections() -> int:
    df = _read("collections")
    if df.empty:  # empty on a small sample -> no features, skip gracefully
        return _write_group("facility_collections", [])
    df["collection_date"] = pd.to_datetime(df["collection_date"])
    daily = (df.groupby(["facility_id", "collection_date"])["amount_collected"]
               .sum().reset_index().rename(columns={"amount_collected": "collections_1d"}))
    out: list[tuple[str, str, dict]] = []
    for fac, g in daily.groupby("facility_id"):
        g = (g[["collection_date", "collections_1d"]].sort_values("collection_date")
             .set_index("collection_date").asfreq("D", fill_value=0.0))
        g["collections_7d_rolling"] = g["collections_1d"].rolling(7, min_periods=1).sum()
        g["collections_30d_rolling"] = g["collections_1d"].rolling(30, min_periods=1).sum()
        monthly = g["collections_1d"].resample("MS").sum()
        mom = monthly.pct_change()
        for date, row in g.iterrows():
            m_key = pd.Timestamp(date.year, date.month, 1)
            yoy_prev = g["collections_30d_rolling"].get(date - pd.Timedelta(days=365))
            yoy = ((row["collections_30d_rolling"] - yoy_prev) / yoy_prev
                   if yoy_prev and yoy_prev > 0 else None)
            out.append((fac, date.date().isoformat(), {
                "collections_1d": float(row["collections_1d"]),
                "collections_7d_rolling": float(row["collections_7d_rolling"]),
                "collections_30d_rolling": float(row["collections_30d_rolling"]),
                "collections_mom_growth": (float(mom.get(m_key)) if pd.notna(mom.get(m_key)) else None),
                "collections_yoy_growth": (float(yoy) if yoy is not None else None),
                "day_of_week": int(date.weekday()),
                "month": int(date.month),
                "is_month_end": bool(date.day >= 28),
            }))
    return _write_group("facility_collections", out)


def compute_attorney_aging() -> int:
    df = _read("attorney_aging")
    if df.empty:  # empty on a small sample -> no features, skip gracefully
        return _write_group("attorney_aging", [])
    df["report_date"] = pd.to_datetime(df["report_date"])
    bcols = ["bucket_0_30", "bucket_31_60", "bucket_61_90", "bucket_91_180", "bucket_180_plus"]
    mids = np.array([15, 45, 75, 135, 270])  # bucket midpoints for weighted days
    agg = df.groupby(["attorney_id", "report_date"])[bcols].sum().reset_index()
    out: list[tuple[str, str, dict]] = []
    for att, g in agg.groupby("attorney_id"):
        g = g.sort_values("report_date")
        g["total"] = g[bcols].sum(axis=1)
        g["pct_180"] = np.where(g["total"] > 0, g["bucket_180_plus"] / g["total"], 0.0)
        g["pct_180_30d_ago"] = g["pct_180"].shift(4)  # weekly snapshots -> ~30d
        for _, r in g.iterrows():
            total = float(r["total"])
            avg_days = float((r[bcols].values * mids).sum() / total) if total > 0 else 0.0
            migration = (float(r["pct_180"] - r["pct_180_30d_ago"])
                         if pd.notna(r["pct_180_30d_ago"]) else None)
            out.append((att, r["report_date"].date().isoformat(), {
                "bucket_0_30_balance": float(r["bucket_0_30"]),
                "bucket_31_60_balance": float(r["bucket_31_60"]),
                "bucket_61_90_balance": float(r["bucket_61_90"]),
                "bucket_91_180_balance": float(r["bucket_91_180"]),
                "bucket_180_plus_balance": float(r["bucket_180_plus"]),
                "total_outstanding": total,
                "pct_180_plus": float(r["pct_180"]),
                "avg_days_outstanding": avg_days,
                "aging_migration_rate_30d": migration,
            }))
    return _write_group("attorney_aging", out)


def compute_visit_velocity() -> int:
    df = _read("visits")
    if df.empty:  # empty on a small sample -> no features, skip gracefully
        return _write_group("visit_velocity", [])
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df["is_billed"] = (df["billed_amount"] > 0).astype(int)
    df["is_pending"] = (df["billing_status"] == "pending").astype(int)
    grp = (df.groupby(["facility_id", "case_type", "visit_date"])
             .agg(n=("visit_id", "count"), billed=("is_billed", "sum"),
                  pending=("is_pending", "sum")).reset_index())
    out: list[tuple[str, str, dict]] = []
    for (fac, ct), g in grp.groupby(["facility_id", "case_type"]):
        g = (g[["visit_date", "n", "billed", "pending"]].sort_values("visit_date")
             .set_index("visit_date").asfreq("D", fill_value=0))
        nv7 = g["n"].rolling(7, min_periods=1).sum()
        nv30 = g["n"].rolling(30, min_periods=1).sum()
        billed7 = g["billed"].rolling(7, min_periods=1).sum()
        pending7 = g["pending"].rolling(7, min_periods=1).sum()
        for date, r in g.iterrows():
            n7 = float(nv7[date])
            out.append((f"{fac}|{ct}", date.date().isoformat(), {
                "new_visits_7d": int(nv7[date]),
                "new_visits_30d": int(nv30[date]),
                "visit_cancellation_rate_7d": float(pending7[date] / n7) if n7 > 0 else 0.0,
                "new_case_open_rate_7d": float(n7 / 7.0),
                "visit_billing_conversion_rate": float(billed7[date] / n7) if n7 > 0 else 0.0,
            }))
    return _write_group("visit_velocity", out)


def compute_settlement_pipeline() -> int:
    df = _read("settlements")
    if df.empty:  # empty on a small sample -> no features, skip gracefully
        return _write_group("settlement_pipeline", [])
    df["open_date"] = pd.to_datetime(df["open_date"])
    df["close_date"] = pd.to_datetime(df["close_date"])
    dates = pd.date_range(df["open_date"].min(), pd.Timestamp("2026-05-27"), freq="7D")
    out: list[tuple[str, str, dict]] = []
    for (att, ct), g in df.groupby(["attorney_id", "case_type"]):
        for d in dates:
            opened = g[g["open_date"] <= d]
            open_now = opened[(opened["close_date"].isna()) | (opened["close_date"] > d)]
            closed_30 = g[(g["close_date"] > d - pd.Timedelta(days=30)) & (g["close_date"] <= d)]
            closed_prev30 = g[(g["close_date"] > d - pd.Timedelta(days=60)) & (g["close_date"] <= d - pd.Timedelta(days=30))]
            closed_90 = g[(g["close_date"] > d - pd.Timedelta(days=90)) & (g["close_date"] <= d)].copy()
            if len(closed_90):
                days = (closed_90["close_date"] - closed_90["open_date"]).dt.days
                avg_days = float(days.mean())
            else:
                avg_days = None
            prev = len(closed_prev30)
            vel = float((len(closed_30) - prev) / prev) if prev > 0 else None
            out.append((f"{att}|{ct}", d.date().isoformat(), {
                "open_settlements_count": int(len(open_now)),
                "settlements_closed_30d": int(len(closed_30)),
                "avg_days_to_settlement_90d": avg_days,
                "settlement_velocity_change_30d": vel,
                "high_value_open_count": int((open_now["settlement_amount"] > HIGH_VALUE_SETTLEMENT_THRESHOLD).sum()),
            }))
    return _write_group("settlement_pipeline", out)


def compute_lop_metrics() -> int:
    df = _read("lop")
    if df.empty:  # empty on a small sample -> no features, skip gracefully
        return _write_group("lop_metrics", [])
    df["issued_date"] = pd.to_datetime(df["issued_date"])
    df["returned_date"] = pd.to_datetime(df["returned_date"])
    dates = pd.date_range(df["issued_date"].min(), pd.Timestamp("2026-05-27"), freq="7D")
    out: list[tuple[str, str, dict]] = []
    for fac, g in df.groupby("facility_id"):
        for d in dates:
            issued7 = g[(g["issued_date"] > d - pd.Timedelta(days=7)) & (g["issued_date"] <= d)]
            returned7 = g[(g["returned_date"] > d - pd.Timedelta(days=7)) & (g["returned_date"] <= d)]
            window30 = g[(g["returned_date"] > d - pd.Timedelta(days=30)) & (g["returned_date"] <= d)].copy()
            rej_rate = float((window30["status"] == "rejected").mean()) if len(window30) else 0.0
            if len(window30):
                tat = (window30["returned_date"] - window30["issued_date"]).dt.days
                p50, p90 = float(tat.quantile(0.5)), float(tat.quantile(0.9))
            else:
                p50 = p90 = None
            backlog = g[(g["issued_date"] <= d - pd.Timedelta(days=60)) &
                        ((g["returned_date"].isna()) | (g["returned_date"] > d))]
            out.append((fac, d.date().isoformat(), {
                "lop_issued_7d": int(len(issued7)),
                "lop_returned_7d": int(len(returned7)),
                "lop_rejection_rate_30d": rej_rate,
                "lop_turnaround_days_p50": p50,
                "lop_turnaround_days_p90": p90,
                "lop_backlog_count": int(len(backlog)),
            }))
    return _write_group("lop_metrics", out)


def compute_all() -> dict[str, int]:
    return {
        "facility_collections": compute_facility_collections(),
        "attorney_aging": compute_attorney_aging(),
        "visit_velocity": compute_visit_velocity(),
        "settlement_pipeline": compute_settlement_pipeline(),
        "lop_metrics": compute_lop_metrics(),
    }


if __name__ == "__main__":
    print(json.dumps(compute_all(), indent=2))
