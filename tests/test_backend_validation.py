"""Backend-correctness tests: entity_type validation, forecast bias sign,
and the NL-to-SQL interactive preview cap.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.forecasting.service as svc
import app.nl2sql.executor as executor
from app.main import app

client = TestClient(app)


def _auth(role: str = "da_analyst") -> dict:
    r = client.post("/auth/token", json={"user_id": "tester", "role": role})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ---------------------------------------------------------------------------
# entity_type must 422, not 500, on an invalid value.
# ---------------------------------------------------------------------------

def test_entities_bad_entity_type_returns_422():
    r = client.get("/forecasts/entities?entity_type=not_a_real_type", headers=_auth())
    assert r.status_code == 422


def test_forecast_bad_entity_type_returns_422():
    r = client.get("/forecasts/not_a_real_type/some-id?horizon=30", headers=_auth())
    assert r.status_code == 422


def test_entities_valid_entity_type_ok():
    r = client.get("/forecasts/entities?entity_type=facility", headers=_auth())
    assert r.status_code == 200
    assert r.json()["entity_type"] == "facility"


# ---------------------------------------------------------------------------
# bias is a signed error (over-forecasting -> positive bias), not abs(MAPE).
# ---------------------------------------------------------------------------

class _FixedForecaster:
    """Deterministic stand-in: always predicts a fixed 30-day total regardless
    of the training data, so the sign of (predicted - actual) is controlled by
    the test's fixture."""

    name = "fixed"
    predicted = 200.0

    def fit(self, series: pd.Series) -> _FixedForecaster:
        return self

    def predict_horizon(self, horizon: int) -> dict:
        return {"predicted": self.predicted, "ci_lower": 0.0, "ci_upper": 400.0,
                "p50": self.predicted, "p80": 250.0, "p95": 300.0}


def test_bias_positive_for_systematic_over_forecast(monkeypatch):
    monkeypatch.setattr(svc, "SeasonalTrendForecaster", _FixedForecaster)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    # Actual daily amount sums to ~100 per 30-day window; fixed model predicts 200
    # -> systematic over-forecast -> bias should be positive.
    series = pd.Series([100.0 / 30] * 200, index=idx)

    metrics = svc.evaluate(series, horizon=30, folds=3)

    assert metrics["bias"] is not None
    assert metrics["bias"] > 0
    assert metrics["mape"] == pytest.approx(metrics["bias"])  # both magnitude 1.0 here


def test_bias_negative_for_systematic_under_forecast(monkeypatch):
    monkeypatch.setattr(svc, "SeasonalTrendForecaster", _FixedForecaster)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    # Actual daily amount sums to ~400 per 30-day window; fixed model predicts 200
    # -> systematic under-forecast -> bias should be negative even though MAPE (abs) is positive.
    series = pd.Series([400.0 / 30] * 200, index=idx)

    metrics = svc.evaluate(series, horizon=30, folds=3)

    assert metrics["bias"] is not None
    assert metrics["bias"] < 0
    assert metrics["mape"] > 0


# ---------------------------------------------------------------------------
# Interactive /nl2sql/query preview cap.
# ---------------------------------------------------------------------------

def _stub_pipeline(monkeypatch, n_rows: int):
    monkeypatch.setattr(executor, "generate_sql", lambda question: "SELECT 1")
    monkeypatch.setattr(executor, "validate_sql", lambda sql: (True, "OK"))
    monkeypatch.setattr(executor, "write_audit", lambda **kwargs: None)
    columns = ["n"]
    rows = [[i] for i in range(n_rows)]
    monkeypatch.setattr(executor, "_execute", lambda sql: (columns, rows, False))


def test_preview_cap_truncates_interactive_query(monkeypatch):
    _stub_pipeline(monkeypatch, executor.PREVIEW_ROW_CAP + 50)

    result = executor.run_query("anything", preview=True)

    assert result.ok
    assert len(result.rows) == executor.PREVIEW_ROW_CAP
    assert result.row_count == executor.PREVIEW_ROW_CAP
    assert result.truncated is True


def test_no_preview_cap_for_export_path(monkeypatch):
    n_rows = executor.PREVIEW_ROW_CAP + 50
    _stub_pipeline(monkeypatch, n_rows)

    result = executor.run_query("anything", preview=False)

    assert result.ok
    assert len(result.rows) == n_rows
    assert result.row_count == n_rows
    assert result.truncated is False
