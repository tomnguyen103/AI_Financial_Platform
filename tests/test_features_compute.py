"""Feature computation + serving tests (PRD Module 2).

`compute_all` over the seeded curated data is the high-yield path; focused
assertions check a known group's shape/values and the store's freshness /
staleness boundary.
"""
from __future__ import annotations

import datetime as dt
import json

from app.db import tx
from app.features import store
from app.features.definitions import FEATURE_GROUPS


def test_compute_all_populates_every_group(seeded_db):
    counts = seeded_db["features"]
    # Every declared feature group produced rows.
    assert set(counts) == set(FEATURE_GROUPS)
    for group, n in counts.items():
        assert n > 0, f"{group} produced no feature rows"
    # feature_freshness has one row per group with a matching row_count.
    fresh = {f["feature_group"]: f["row_count"] for f in store.freshness()}
    for group, n in counts.items():
        assert fresh[group] == n


def test_facility_collections_feature_shape_and_values(seeded_db):
    """A known group exposes exactly its declared feature names with sane values."""
    group = FEATURE_GROUPS["facility_collections"]
    latest = store.get_online_features("facility_collections", "round_rock")
    assert latest is not None
    # Exactly the declared feature names are present.
    assert set(latest) == set(group.features)
    # Structural invariants from the definitions.
    assert latest["collections_1d"] >= 0.0
    assert latest["collections_7d_rolling"] >= latest["collections_1d"] - 1e-6
    assert 0 <= latest["day_of_week"] <= 6
    assert 1 <= latest["month"] <= 12
    assert isinstance(latest["is_month_end"], bool)


def test_get_series_is_point_in_time_correct(seeded_db):
    """as_of caps the returned history to event_date <= as_of (no leakage)."""
    full = store.get_series("facility_collections", "round_rock")
    assert not full.empty
    cutoff = full["event_date"].iloc[len(full) // 2]
    as_of = cutoff.date().isoformat()
    capped = store.get_series("facility_collections", "round_rock", as_of=as_of)
    assert capped["event_date"].max() <= cutoff
    assert len(capped) < len(full)


def test_list_entities_returns_seeded_facilities(seeded_db):
    entities = store.list_entities("facility_collections")
    assert "round_rock" in entities
    assert entities == sorted(entities)


def test_stale_groups_boundary(seeded_db, monkeypatch):
    """A group whose last_updated is 'today' is fresh; backdating it makes it stale."""
    # Seeded groups were just written with a UTC 'now' timestamp -> not stale today.
    assert store.stale_groups() == []

    # Backdate one group's freshness to yesterday -> it becomes stale.
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat() + "T00:00:00+00:00"
    with tx() as conn:
        conn.execute(
            "UPDATE feature_freshness SET last_updated=? WHERE feature_group=?",
            (yesterday, "lop_metrics"),
        )
    assert "lop_metrics" in store.stale_groups()

    # Restore so later tests relying on freshness see a clean slate.
    now = dt.datetime.now(dt.UTC).isoformat()
    with tx() as conn:
        conn.execute(
            "UPDATE feature_freshness SET last_updated=? WHERE feature_group=?",
            (now, "lop_metrics"),
        )


def test_feature_values_json_roundtrips(seeded_db):
    """Stored features_json is valid JSON matching the online vector."""
    with tx() as conn:
        row = conn.execute(
            "SELECT features_json FROM feature_values WHERE feature_group=? AND entity_key=? "
            "ORDER BY event_date DESC LIMIT 1",
            ("facility_collections", "round_rock"),
        ).fetchone()
    parsed = json.loads(row[0])
    assert parsed == store.get_online_features("facility_collections", "round_rock")
