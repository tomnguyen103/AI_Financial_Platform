"""Forecasting model + service tests (PRD Module 3).

Covers SeasonalTrendForecaster fit/predict (shape, determinism, band ordering),
walk-forward evaluate() metrics including the SIGN of the bias term, and
generate_forecasts() persisting 30/60/90 rows per entity atomically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.db import tx
from app.forecasting import service
from app.forecasting.models import SeasonalTrendForecaster


def _rising_series(n: int = 200, slope: float = 10.0, base: float = 1000.0) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    values = base + slope * np.arange(n) + rng.normal(0, 5, n)
    return pd.Series(values, index=idx)


def test_predict_horizon_shape_and_band_ordering():
    model = SeasonalTrendForecaster().fit(_rising_series())
    pred = model.predict_horizon(30)
    assert set(pred) == {"predicted", "p50", "ci_lower", "ci_upper", "p80", "p95"}
    # Bands are ordered and centred on the point forecast.
    assert pred["ci_lower"] <= pred["predicted"] <= pred["ci_upper"]
    assert pred["p50"] == pred["predicted"]
    assert pred["ci_upper"] <= pred["p95"]  # 80% upper band inside 95% upper band
    assert pred["predicted"] > 0


def test_fit_predict_is_deterministic():
    s = _rising_series()
    a = SeasonalTrendForecaster().fit(s).predict_horizon(60)
    b = SeasonalTrendForecaster().fit(s).predict_horizon(60)
    assert a == b


def test_longer_horizon_widens_the_band():
    model = SeasonalTrendForecaster().fit(_rising_series())
    band_30 = model.predict_horizon(30)
    band_90 = model.predict_horizon(90)
    width_30 = band_30["ci_upper"] - band_30["ci_lower"]
    width_90 = band_90["ci_upper"] - band_90["ci_lower"]
    assert width_90 > width_30  # sqrt(H) scaling of the residual band


def test_state_roundtrip_reproduces_predictions():
    model = SeasonalTrendForecaster().fit(_rising_series())
    restored = SeasonalTrendForecaster.from_state(model.state())
    assert restored.predict_horizon(45) == model.predict_horizon(45)


def test_evaluate_bias_sign_is_negative_when_model_underpredicts():
    """A series that accelerates at the end makes a linear-trend model under-forecast,
    so signed bias (pred-actual)/actual must be negative."""
    idx = pd.date_range("2025-01-01", periods=180, freq="D")
    # Quadratic ramp: recent actuals exceed the linear extrapolation from history.
    values = 1000.0 + 0.5 * (np.arange(180) ** 1.6)
    series = pd.Series(values, index=idx)
    metrics = service.evaluate(series, horizon=30, folds=3)
    assert metrics["folds"] >= 1
    assert metrics["mape"] is not None and metrics["mape"] >= 0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["bias"] is not None
    assert metrics["bias"] < 0  # underprediction -> signed bias negative


def test_evaluate_returns_empty_metrics_on_insufficient_history():
    short = pd.Series(np.ones(20), index=pd.date_range("2025-01-01", periods=20, freq="D"))
    metrics = service.evaluate(short, horizon=30)
    assert metrics == {"mape": None, "rmse": None, "coverage": None, "bias": None, "folds": 0}


def test_generate_forecasts_writes_all_horizons_atomically(seeded_db):
    """After the seeded run, each facility has exactly one row per horizon for the
    latest forecast_date (the delete+insert share one transaction)."""
    with tx() as conn:
        rows = conn.execute(
            "SELECT entity_id, horizon_days, COUNT(*) AS c FROM forecasts "
            "WHERE entity_type='facility' GROUP BY entity_id, horizon_days"
        ).fetchall()
    assert rows, "no facility forecasts were written"
    horizons_by_entity: dict[str, set[int]] = {}
    for r in rows:
        assert r["c"] == 1  # no duplicate (entity, horizon) for the run date
        horizons_by_entity.setdefault(r["entity_id"], set()).add(r["horizon_days"])
    for entity, horizons in horizons_by_entity.items():
        assert horizons == {30, 60, 90}, f"{entity} missing horizons: {horizons}"


def test_generate_forecasts_is_idempotent_per_run_date(seeded_db):
    """Re-running does not accumulate rows for the same forecast_date."""
    with tx() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM forecasts WHERE entity_type='facility'"
        ).fetchone()[0]
    service.generate_forecasts("facility")
    with tx() as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM forecasts WHERE entity_type='facility'"
        ).fetchone()[0]
    assert after == before  # prior batch deleted before re-insert


def test_get_forecast_returns_latest_row(seeded_db):
    fc = service.get_forecast("facility", "round_rock", 30)
    assert fc is not None
    assert fc["entity_id"] == "round_rock"
    assert fc["horizon_days"] == 30
    assert fc["ci_lower"] <= fc["predicted"] <= fc["ci_upper"]
