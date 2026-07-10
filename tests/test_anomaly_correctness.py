"""Anomaly + alerting correctness/reliability tests.

Covers the audit fixes:
  - CUSUM baseline is estimated from a REFERENCE window preceding the detection
    window, so a sustained end-of-series drop fires instead of contaminating its
    own baseline (data design §5.1).
  - Slack delivery failures are logged, not silently swallowed, and never crash
    the run (arch §2.4; AC-4.x).
  - /alerts pagination threads limit/offset through to list_alerts.
"""
from __future__ import annotations

import logging
import uuid

import httpx
import pandas as pd
from fastapi.testclient import TestClient

from app.anomaly import alerting, detectors
from app.db import tx
from app.main import app

_CUSUM_TH = {"cusum": {"k": 0.5, "p2_sigma": 4.0, "p1_sigma": 6.0, "window": 20}}


def _pin_cusum(monkeypatch, series: pd.Series) -> None:
    monkeypatch.setattr(detectors, "daily_collections", lambda et, eid: series)
    monkeypatch.setattr(detectors, "thresholds_for", lambda eid: _CUSUM_TH)


def test_cusum_fires_on_sustained_drop_with_stable_baseline(monkeypatch):
    # 40 stable points (~100 with small noise -> nonzero sigma) then a sustained
    # drop to 50 over the most-recent window. The reference window precedes the
    # drop, so the baseline stays stable and the shift is detectable.
    baseline = [101.0 if i % 2 else 99.0 for i in range(40)]
    drop = [50.0] * 20
    series = pd.Series(baseline + drop)
    _pin_cusum(monkeypatch, series)

    findings = detectors.cusum("facility", "round_rock")

    assert len(findings) == 1
    assert findings[0].severity == "P1"
    assert findings[0].detector == "cusum"


def test_cusum_stays_quiet_on_flat_series(monkeypatch):
    # A perfectly flat series has zero variance in the reference window -> the
    # detector degrades cleanly to no finding rather than dividing by zero.
    series = pd.Series([100.0] * 60)
    _pin_cusum(monkeypatch, series)

    assert detectors.cusum("facility", "round_rock") == []


def test_slack_failure_is_logged_not_silent_and_does_not_raise(monkeypatch, caplog):
    monkeypatch.setattr(alerting.settings, "slack_webhook_url", "https://hooks.example/x")

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(alerting.httpx, "post", _boom)
    payload = {"severity": "P1", "entity_id": "round_rock", "metric": "collections_7d",
               "detector": "cusum", "expected": None, "actual": None,
               "deviation_pct": None, "driver": "test", "dashboard_link": "/d",
               "alert_id": "abc"}

    with caplog.at_level(logging.WARNING):
        alerting._deliver(payload)  # must NOT raise

    assert any("slack delivery failed" in r.message for r in caplog.records)


def test_slack_non_2xx_status_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(alerting.settings, "slack_webhook_url", "https://hooks.example/x")

    def _bad_status(*args, **kwargs):
        return httpx.Response(500, request=httpx.Request("POST", kwargs.get("url", "http://x")))

    monkeypatch.setattr(alerting.httpx, "post", _bad_status)
    payload = {"severity": "P2", "entity_id": "e", "metric": "m", "detector": "cusum",
               "expected": None, "actual": None, "deviation_pct": None,
               "driver": "d", "dashboard_link": "/d", "alert_id": "z"}

    with caplog.at_level(logging.WARNING):
        alerting._deliver(payload)

    assert any("slack delivery failed" in r.message for r in caplog.records)


def _seed_alerts(n: int) -> None:
    # Unique ids so the test is idempotent against the persistent session DB
    # (other tests and prior runs may already hold rows).
    with tx() as conn:
        for i in range(n):
            conn.execute(
                """INSERT INTO alerts (alert_id, created_at, severity, entity_type,
                     entity_id, metric, detector, driver_narrative, status)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), f"2026-01-{i + 1:02d}T00:00:00+00:00", "P2",
                 "facility", f"e{i}", "collections_7d", "cusum", "n", "open"),
            )


def test_alerts_route_respects_limit_and_offset():
    client = TestClient(app)
    token = client.post("/auth/token", json={"user_id": "t", "role": "da_analyst"}).json()
    headers = {"Authorization": f"Bearer {token['access_token']}"}
    _seed_alerts(5)

    body = client.get("/alerts?limit=2", headers=headers).json()
    assert body["count"] == 2
    assert len(body["alerts"]) == 2

    # limit is clamped to <= 500 by the route validator
    assert client.get("/alerts?limit=999", headers=headers).status_code == 422
    # offset skips rows; combined with a limit it bounds the page size
    paged = client.get("/alerts?limit=3&offset=1", headers=headers).json()
    assert paged["count"] == 3
