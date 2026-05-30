"""Forecasting service: train, evaluate (walk-forward MAPE), persist, serve.

Spec refs: arch §2.3; data design §4.5, §10.1; PRD Module 3.

Entities forecast: facility, attorney, case_type (AC-3.1). Entities with < 30
days of history are skipped ("insufficient history", data design §4.1).
Horizons: 30/60/90 days (AC). Each forecast carries 80% CI + upper 80/95 bands
(AC-3.2) and is persisted to the `forecasts` table (AC-3.7).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from app.db import tx
from app.forecasting import registry
from app.forecasting.models import SeasonalTrendForecaster

HORIZONS = (30, 60, 90)
MIN_HISTORY_DAYS = 30
ENTITY_COLUMN = {"facility": "facility_id", "attorney": "attorney_id", "case_type": "case_type"}


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def daily_collections(entity_type: str, entity_id: str) -> pd.Series:
    col = ENTITY_COLUMN[entity_type]
    with tx() as conn:
        df = pd.read_sql_query(
            f"SELECT collection_date, amount_collected FROM collections WHERE {col}=?",
            conn, params=(entity_id,),
        )
    if df.empty:
        return pd.Series(dtype=float)
    df["collection_date"] = pd.to_datetime(df["collection_date"])
    s = df.groupby("collection_date")["amount_collected"].sum().sort_index()
    return s.asfreq("D", fill_value=0.0)


def list_entities(entity_type: str) -> list[str]:
    col = ENTITY_COLUMN[entity_type]
    with tx() as conn:
        rows = conn.execute(f"SELECT DISTINCT {col} FROM collections ORDER BY {col}").fetchall()
    return [r[0] for r in rows]


def evaluate(series: pd.Series, horizon: int, folds: int = 4) -> dict:
    """Walk-forward backtest of the H-day-ahead total (data design §10.1)."""
    errors, actuals, in_ci = [], [], []
    n = len(series)
    if n < horizon * 2 + MIN_HISTORY_DAYS:
        folds = max(1, (n - MIN_HISTORY_DAYS) // horizon - 1)
    for i in range(folds):
        cutoff = n - (i + 1) * horizon
        if cutoff < MIN_HISTORY_DAYS:
            break
        train = series.iloc[:cutoff]
        actual = float(series.iloc[cutoff:cutoff + horizon].sum())
        if actual <= 0:
            continue
        m = SeasonalTrendForecaster().fit(train)
        pred = m.predict_horizon(horizon)
        errors.append(abs(pred["predicted"] - actual) / actual)
        actuals.append(actual)
        in_ci.append(pred["ci_lower"] <= actual <= pred["ci_upper"])
        # bias accumulation
    if not errors:
        return {"mape": None, "rmse": None, "coverage": None, "bias": None, "folds": 0}
    mape = float(np.mean(errors))
    coverage = float(np.mean(in_ci))
    return {"mape": mape, "rmse": None, "coverage": coverage,
            "bias": float(np.mean(errors)), "folds": len(errors)}


def train_and_register(entity_type: str = "facility") -> list[dict]:
    """Train one model per entity, evaluate, register champion/challenger."""
    results = []
    for entity_id in list_entities(entity_type):
        series = daily_collections(entity_type, entity_id)
        if len(series) < MIN_HISTORY_DAYS:
            results.append({"entity": entity_id, "skipped": "insufficient_history"})
            continue
        metrics = evaluate(series, horizon=30)
        model = SeasonalTrendForecaster().fit(series)
        model_name = f"collections_{entity_type}_{entity_id}"
        reg = registry.register(
            model_name, state=model.state(),
            metrics=metrics, params={"model": model.name, "history_days": len(series)},
        )
        reg["entity"] = entity_id
        reg["mape"] = metrics["mape"]
        reg["coverage"] = metrics["coverage"]
        results.append(reg)
    return results


def generate_forecasts(entity_type: str = "facility") -> int:
    """Produce + persist 30/60/90 forecasts for each entity's champion model."""
    today = dt.date(2026, 5, 28)
    written = 0
    with tx() as conn:
        conn.execute("DELETE FROM forecasts WHERE entity_type=? AND forecast_date=?",
                     (entity_type, today.isoformat()))
    for entity_id in list_entities(entity_type):
        model_name = f"collections_{entity_type}_{entity_id}"
        champ = registry.champion(model_name)
        if not champ:
            continue
        art = registry.load_artifact(champ["artifact_path"])
        model = SeasonalTrendForecaster.from_state(art["state"])
        for h in HORIZONS:
            pred = model.predict_horizon(h)
            target = (today + dt.timedelta(days=h)).isoformat()
            with tx() as conn:
                conn.execute(
                    """INSERT INTO forecasts (entity_type, entity_id, horizon_days, forecast_date,
                         target_date, predicted, ci_lower, ci_upper, p50, p80, p95,
                         model_name, model_version, feature_snapshot_ts, generated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (entity_type, entity_id, h, today.isoformat(), target,
                     pred["predicted"], pred["ci_lower"], pred["ci_upper"],
                     pred["p50"], pred["p80"], pred["p95"], model_name,
                     champ["version"], _utcnow(), _utcnow()),
                )
            written += 1
    return written


def get_forecast(entity_type: str, entity_id: str, horizon: int) -> dict | None:
    with tx() as conn:
        row = conn.execute(
            "SELECT * FROM forecasts WHERE entity_type=? AND entity_id=? AND horizon_days=? "
            "ORDER BY forecast_date DESC LIMIT 1", (entity_type, entity_id, horizon)
        ).fetchone()
    return dict(row) if row else None


def run_all() -> dict:
    out = {}
    for et in ("facility", "attorney", "case_type"):
        train_and_register(et)
        out[et] = generate_forecasts(et)
    return out


if __name__ == "__main__":
    import json
    from app.db import init_db
    init_db()
    print(json.dumps(run_all(), indent=2))
