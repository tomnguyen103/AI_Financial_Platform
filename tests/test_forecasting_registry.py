from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from app.forecasting import registry


def test_register_handles_null_champion_mape(tmp_path, monkeypatch):
    db_path = tmp_path / "registry.db"
    artifact_dir = tmp_path / "models"
    artifact_dir.mkdir()

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE model_registry (
            model_name TEXT, version INTEGER, stage TEXT, mape REAL, rmse REAL, coverage REAL,
            bias REAL, params_json TEXT, artifact_path TEXT, created_at TEXT, git_commit TEXT,
            PRIMARY KEY (model_name, version)
        );
        INSERT INTO model_registry
            (model_name, version, stage, mape, rmse, coverage, bias, params_json, artifact_path, created_at, git_commit)
        VALUES
            ('collections_attorney_sparse', 1, 'Production', NULL, NULL, NULL, NULL, '{}', 'old.json', 'now', 'local');
        """
    )
    conn.close()

    @contextmanager
    def temp_tx():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(registry, "tx", temp_tx)
    monkeypatch.setattr(registry, "MODEL_DIR", artifact_dir)

    result = registry.register(
        "collections_attorney_sparse",
        state={"n": 30},
        metrics={"mape": None, "rmse": None, "coverage": None, "bias": None},
        params={"model": "seasonal_trend"},
    )

    assert result["stage"] == "Staging"
    assert result["mape"] is None
