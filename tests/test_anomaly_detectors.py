"""Anomaly detector + severity tests (PRD Module 4).

Each detector must fire on a synthetic spike/drop and stay quiet on a flat
series; severity band boundaries are checked at the exact threshold value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.anomaly import config, detectors, severity
from app.anomaly.detectors import Finding

# Deterministic thresholds so tests don't depend on config/thresholds.yaml drift.
_TH = {
    "forecast_deviation": {"p2_pct": -0.15, "p1_pct": -0.25},
    "cusum": {"k": 0.5, "p2_sigma": 4.0, "p1_sigma": 6.0, "window": 20},
    "isolation_forest": {"contamination": 0.05, "p2_score": -0.05, "p1_score": -0.10},
}


def _pin(monkeypatch, series=None):
    monkeypatch.setattr(detectors, "thresholds_for", lambda eid: _TH)
    if series is not None:
        monkeypatch.setattr(detectors, "daily_collections", lambda et, eid: series)


# --- forecast_deviation ---------------------------------------------------

def test_forecast_deviation_fires_on_collections_drop(monkeypatch):
    # Forecast implies 700/7d; actual is far below -> deep negative deviation -> P1.
    monkeypatch.setattr(detectors, "get_forecast", lambda et, eid, h: {"predicted": 3000.0})
    series = pd.Series([50.0] * 30)  # 7d actual = 350 vs expected 700 -> -50%
    _pin(monkeypatch, series)
    findings = detectors.forecast_deviation("facility", "round_rock")
    assert len(findings) == 1 and findings[0].severity == "P1"
    assert findings[0].detector == "forecast_deviation"
    assert findings[0].deviation_pct < 0


def test_forecast_deviation_quiet_when_actual_matches_forecast(monkeypatch):
    monkeypatch.setattr(detectors, "get_forecast", lambda et, eid, h: {"predicted": 3000.0})
    # expected_7d = 3000*7/30 = 700; make actual ~ 700 so deviation ~ 0.
    series = pd.Series([100.0] * 30)  # 7d actual = 700
    _pin(monkeypatch, series)
    assert detectors.forecast_deviation("facility", "round_rock") == []


def test_forecast_deviation_severity_boundary_exact_threshold(monkeypatch):
    """A deviation exactly at p2_pct (-0.15) resolves to P2 (boundary is inclusive)."""
    monkeypatch.setattr(detectors, "get_forecast", lambda et, eid, h: {"predicted": 3000.0})
    # expected_7d = 700; want dev == -0.15 -> actual_7d = 700 * 0.85 = 595.
    series = pd.Series([595.0 / 7.0] * 30)
    _pin(monkeypatch, series)
    findings = detectors.forecast_deviation("facility", "round_rock")
    assert len(findings) == 1 and findings[0].severity == "P2"


def test_forecast_deviation_no_forecast_returns_empty(monkeypatch):
    monkeypatch.setattr(detectors, "get_forecast", lambda et, eid, h: None)
    _pin(monkeypatch, pd.Series([100.0] * 30))
    assert detectors.forecast_deviation("facility", "round_rock") == []


# --- CUSUM ----------------------------------------------------------------

def test_cusum_fires_on_sustained_drop(monkeypatch):
    baseline = [101.0 if i % 2 else 99.0 for i in range(40)]
    drop = [50.0] * 20
    _pin(monkeypatch, pd.Series(baseline + drop))
    findings = detectors.cusum("facility", "round_rock")
    assert len(findings) == 1 and findings[0].severity == "P1"


def test_cusum_quiet_on_flat_series(monkeypatch):
    _pin(monkeypatch, pd.Series([100.0] * 60))  # zero variance -> clean no-finding
    assert detectors.cusum("facility", "round_rock") == []


def test_cusum_returns_empty_on_short_history(monkeypatch):
    _pin(monkeypatch, pd.Series([100.0] * 5))  # < window + 10
    assert detectors.cusum("facility", "round_rock") == []


# --- Isolation Forest -----------------------------------------------------

def _if_frame(last_row_anomalous: bool) -> pd.DataFrame:
    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    base = rng.normal(1000, 20, n)
    turnaround = rng.normal(10, 1, n)
    backlog = rng.normal(5, 1, n)
    if last_row_anomalous:
        base[-1] = 50.0            # collections collapse
        turnaround[-1] = 200.0     # turnaround explodes
        backlog[-1] = 400.0
    return pd.DataFrame({
        "event_date": dates,
        "collections_1d": base,
        "lop_turnaround_days_p50": turnaround,
        "lop_backlog_count": backlog,
    })


def test_isolation_forest_fires_on_multivariate_anomaly(monkeypatch):
    monkeypatch.setattr(detectors, "thresholds_for", lambda eid: _TH)
    monkeypatch.setattr(detectors, "_facility_vectors", lambda fid: _if_frame(True))
    findings = detectors.isolation_forest("round_rock")
    assert len(findings) == 1
    assert findings[0].detector == "isolation_forest"
    assert findings[0].severity in {"P1", "P2"}


def test_isolation_forest_quiet_on_normal_series(monkeypatch):
    monkeypatch.setattr(detectors, "thresholds_for", lambda eid: _TH)
    monkeypatch.setattr(detectors, "_facility_vectors", lambda fid: _if_frame(False))
    assert detectors.isolation_forest("round_rock") == []


def test_isolation_forest_empty_on_short_history(monkeypatch):
    monkeypatch.setattr(detectors, "thresholds_for", lambda eid: _TH)
    short = _if_frame(False).iloc[:30]
    monkeypatch.setattr(detectors, "_facility_vectors", lambda fid: short)
    assert detectors.isolation_forest("round_rock") == []


# --- severity.combine -----------------------------------------------------

def _finding(sev: str, detector: str) -> Finding:
    return Finding("facility", "round_rock", "m", detector, None, None, None, sev, {})


def test_combine_returns_none_for_no_findings():
    assert severity.combine([]) is None


def test_combine_takes_highest_single_severity():
    best = severity.combine([_finding("P2", "cusum"), _finding("P3", "forecast_deviation")])
    assert best.severity == "P2"
    assert "escalated" not in best.detail  # single detector -> no escalation


def test_combine_escalates_to_p1_on_two_agreeing_detectors():
    best = severity.combine([_finding("P2", "cusum"), _finding("P2", "isolation_forest")])
    assert best.severity == "P1"
    assert best.detail["escalated"] == "multi_detector_agreement"
    assert set(best.detail["detectors"]) == {"cusum", "isolation_forest"}


# --- config.thresholds_for ------------------------------------------------

def test_thresholds_for_merges_entity_overrides():
    config._load.cache_clear()
    # A facility with an override gets wider forecast_deviation bands than default.
    override = config.thresholds_for("facility_new_braunfels")
    default = config.thresholds_for("round_rock")
    assert override["forecast_deviation"]["p1_pct"] != default["forecast_deviation"]["p1_pct"]
    # Non-overridden sections fall through to defaults unchanged.
    assert override["cusum"] == default["cusum"]
