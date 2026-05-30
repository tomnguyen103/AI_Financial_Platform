"""Anomaly detectors (data design §5): CUSUM, Isolation Forest, forecast deviation.

Each detector returns a list of Finding objects with a severity and the numbers
needed to build an alert payload.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.anomaly.config import thresholds_for
from app.features import store
from app.forecasting.service import daily_collections, get_forecast


@dataclass
class Finding:
    entity_type: str
    entity_id: str
    metric: str
    detector: str
    expected: float | None
    actual: float | None
    deviation_pct: float | None
    severity: str  # P1 / P2 / P3
    detail: dict


def _severity_from_deviation(dev: float, th: dict) -> str | None:
    fd = th["forecast_deviation"]
    if dev <= fd["p1_pct"]:
        return "P1"
    if dev <= fd["p2_pct"]:
        return "P2"
    return None


def forecast_deviation(entity_type: str, entity_id: str) -> list[Finding]:
    """Compare recent 7-day actual collections vs the (scaled) forecast."""
    fc = get_forecast(entity_type, entity_id, 30)
    if not fc:
        return []
    series = daily_collections(entity_type, entity_id)
    if len(series) < 7:
        return []
    actual_7d = float(series.iloc[-7:].sum())
    expected_7d = fc["predicted"] * (7.0 / 30.0)  # scale 30d forecast to 7d
    if expected_7d <= 0:
        return []
    dev = (actual_7d - expected_7d) / expected_7d
    th = thresholds_for(entity_id)
    sev = _severity_from_deviation(dev, th)
    if not sev:
        return []
    return [Finding(entity_type, entity_id, "collections_7d", "forecast_deviation",
                    round(expected_7d, 2), round(actual_7d, 2), round(dev, 4), sev,
                    {"horizon_basis": "30d_scaled_to_7d"})]


def cusum(entity_type: str, entity_id: str) -> list[Finding]:
    """CUSUM on z-scored daily collections (data design §5.1)."""
    series = daily_collections(entity_type, entity_id)
    th = thresholds_for(entity_id)["cusum"]
    window = int(th["window"])
    if len(series) < window + 10:
        return []
    recent = series.iloc[-window:]
    mu, sigma = float(recent.mean()), float(recent.std())
    if sigma == 0:
        return []
    z = (series.values - mu) / sigma
    k = float(th["k"])
    s_neg = 0.0
    min_s = 0.0
    for val in z[-window:]:
        s_neg = min(0.0, s_neg + val + k)  # accumulate downward shifts
        min_s = min(min_s, s_neg)
    magnitude = abs(min_s)
    if magnitude >= th["p1_sigma"]:
        sev = "P1"
    elif magnitude >= th["p2_sigma"]:
        sev = "P2"
    else:
        return []
    return [Finding(entity_type, entity_id, "collections_1d_cusum", "cusum",
                    round(mu, 2), round(float(series.iloc[-1]), 2), None, sev,
                    {"cusum_magnitude_sigma": round(magnitude, 2)})]


# Feature vector for multivariate detection (data design §5.2).
def _facility_vectors(facility_id: str) -> pd.DataFrame:
    fc = store.get_series("facility_collections", facility_id)
    lop = store.get_series("lop_metrics", facility_id)
    if fc.empty:
        return pd.DataFrame()
    df = fc[["event_date", "collections_1d"]].copy()
    if not lop.empty:
        df = df.merge(lop[["event_date", "lop_turnaround_days_p50", "lop_backlog_count"]],
                      on="event_date", how="left")
    return df.fillna(0.0)


def isolation_forest(facility_id: str) -> list[Finding]:
    df = _facility_vectors(facility_id)
    th = thresholds_for(facility_id)["isolation_forest"]
    if len(df) < 60:
        return []
    feats = df.drop(columns=["event_date"]).values
    model = IsolationForest(contamination=float(th["contamination"]), random_state=42)
    model.fit(feats)
    scores = model.decision_function(feats)  # higher = more normal
    last_score = float(scores[-1])
    if last_score <= th["p1_score"]:
        sev = "P1"
    elif last_score <= th["p2_score"]:
        sev = "P2"
    else:
        return []
    return [Finding("facility", facility_id, "multivariate", "isolation_forest",
                    None, None, None, sev, {"if_score": round(last_score, 4)})]
