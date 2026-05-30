"""Forecasting models.

The spec names Prophet / XGBoost / TFT. Those are fragile to build on Python
3.14/Windows, so the MVP ships a dependency-light `SeasonalTrendForecaster`
(numpy): linear trend + weekday + month seasonal indices + residual-based
prediction intervals. It satisfies the spec contract (multi-horizon totals,
confidence bands, MAPE-evaluable, weekly retrainable).

`Forecaster` is a Protocol, so a Prophet/XGBoost implementation can drop in with
no change to the service/registry/API layers.
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

_Z80 = 1.2816
_Z95 = 1.9600


@runtime_checkable
class Forecaster(Protocol):
    name: str
    def fit(self, series: pd.Series) -> "Forecaster": ...
    def predict_horizon(self, horizon: int) -> dict: ...


class SeasonalTrendForecaster:
    """Daily series -> H-day-ahead total collections with 80/95% bands."""

    name = "seasonal_trend"

    def __init__(self) -> None:
        self._slope = 0.0
        self._intercept = 0.0
        self._weekday = np.zeros(7)
        self._month = np.zeros(13)
        self._resid_std = 0.0
        self._n = 0
        self._last_date: pd.Timestamp | None = None

    def fit(self, series: pd.Series) -> "SeasonalTrendForecaster":
        series = series.sort_index().asfreq("D", fill_value=0.0)
        y = series.values.astype(float)
        n = len(y)
        t = np.arange(n)
        self._slope, self._intercept = np.polyfit(t, y, 1)
        trend = self._slope * t + self._intercept
        detrended = y - trend

        weekdays = series.index.weekday.values
        months = series.index.month.values
        self._weekday = np.array([detrended[weekdays == d].mean() if (weekdays == d).any() else 0.0
                                  for d in range(7)])
        deseason_wd = detrended - self._weekday[weekdays]
        self._month = np.zeros(13)
        for m in range(1, 13):
            mask = months == m
            self._month[m] = deseason_wd[mask].mean() if mask.any() else 0.0
        resid = deseason_wd - self._month[months]
        self._resid_std = float(np.std(resid))
        self._n = n
        self._last_date = series.index[-1]
        return self

    def _daily_forecast(self, horizon: int) -> np.ndarray:
        assert self._last_date is not None
        future_t = np.arange(self._n, self._n + horizon)
        future_dates = pd.date_range(self._last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        trend = self._slope * future_t + self._intercept
        wd = self._weekday[future_dates.weekday.values]
        mo = self._month[future_dates.month.values]
        return np.clip(trend + wd + mo, 0.0, None)

    def predict_horizon(self, horizon: int) -> dict:
        daily = self._daily_forecast(horizon)
        total = float(daily.sum())
        # Std of an H-day sum of iid daily residuals ~ resid_std * sqrt(H).
        band = self._resid_std * math.sqrt(horizon)
        return {
            "predicted": round(total, 2),
            "p50": round(total, 2),
            "ci_lower": round(max(0.0, total - _Z80 * band), 2),
            "ci_upper": round(total + _Z80 * band, 2),
            "p80": round(total + _Z80 * band, 2),   # upper 80th band
            "p95": round(total + _Z95 * band, 2),   # upper 95th band
        }

    # --- pickling support for the registry artifact ---
    def state(self) -> dict:
        return {
            "slope": self._slope, "intercept": self._intercept,
            "weekday": self._weekday.tolist(), "month": self._month.tolist(),
            "resid_std": self._resid_std, "n": self._n,
            "last_date": self._last_date.isoformat() if self._last_date is not None else None,
        }

    @classmethod
    def from_state(cls, s: dict) -> "SeasonalTrendForecaster":
        m = cls()
        m._slope, m._intercept = s["slope"], s["intercept"]
        m._weekday = np.array(s["weekday"]); m._month = np.array(s["month"])
        m._resid_std = s["resid_std"]; m._n = s["n"]
        m._last_date = pd.Timestamp(s["last_date"]) if s["last_date"] else None
        return m
