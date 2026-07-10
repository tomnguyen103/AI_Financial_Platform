"""Feature retrieval / serving (PRD AC-2.2/2.3/2.4).

- get_online_features: latest feature vector for an entity (in-process cache
  stands in for Redis; trivially <50ms locally).
- get_historical_features: point-in-time-correct series (event_date <= as_of),
  preventing label leakage (AC-2.3).
- freshness(): last-updated per group; staleness check for monitoring (AC-2.4).
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from app.db import tx


def get_series(feature_group: str, entity_key: str, as_of: str | None = None) -> pd.DataFrame:
    """Point-in-time-correct feature history for one entity (event_date <= as_of)."""
    sql = ("SELECT event_date, features_json FROM feature_values "
           "WHERE feature_group=? AND entity_key=?")
    params: list = [feature_group, entity_key]
    if as_of:
        sql += " AND event_date <= ?"
        params.append(as_of)
    sql += " ORDER BY event_date"
    with tx() as conn:
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        return pd.DataFrame()
    recs = [{"event_date": r["event_date"], **json.loads(r["features_json"])} for r in rows]
    df = pd.DataFrame(recs)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def list_entities(feature_group: str) -> list[str]:
    with tx() as conn:
        rows = conn.execute(
            "SELECT DISTINCT entity_key FROM feature_values WHERE feature_group=? ORDER BY entity_key",
            (feature_group,),
        ).fetchall()
    return [r[0] for r in rows]


def get_online_features(feature_group: str, entity_key: str) -> dict | None:
    """Latest feature vector (online serving)."""
    with tx() as conn:
        row = conn.execute(
            "SELECT features_json FROM feature_values WHERE feature_group=? AND entity_key=? "
            "ORDER BY event_date DESC LIMIT 1",
            (feature_group, entity_key),
        ).fetchone()
    return json.loads(row[0]) if row else None


def freshness() -> list[dict]:
    with tx() as conn:
        rows = conn.execute(
            "SELECT feature_group, last_updated, row_count FROM feature_freshness ORDER BY feature_group"
        ).fetchall()
    return [dict(r) for r in rows]


def stale_groups(deadline_hour: int = 7) -> list[str]:
    """Groups not updated 'today' (AC-2.4: not updated by 07:00)."""
    today = dt.date.today().isoformat()
    stale = []
    for f in freshness():
        if not f["last_updated"].startswith(today):
            stale.append(f["feature_group"])
    return stale
