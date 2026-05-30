"""Model registry — SQLite + filesystem stand-in for MLflow Model Registry.

Spec refs: arch §2.3, data design §8.2; PRD AC-3.5/3.6, AC-3.4.
Implements: versioned registration, champion (Production)/challenger (Staging)
stages, automated promotion gate (challenger MAPE must be <= champion + 1pp),
and single-call rollback.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re

from app.config import MODEL_DIR
from app.db import tx


def _safe_stem(model_name: str, version: int) -> str:
    """Filesystem-safe artifact filename. Real entity ids (attorney/facility names)
    can contain '/', spaces, etc.; a short hash keeps distinct names from colliding."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")[:80]
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}_v{version}"

PROMOTION_TOLERANCE_PP = 1.0  # challenger MAPE may be at most 1pp worse (data design §4.5)
REGRESSION_BLOCK_PP = 3.0     # block + alert if worse by >3pp (AC-3.5)


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def next_version(model_name: str) -> int:
    with tx() as conn:
        row = conn.execute(
            "SELECT MAX(version) FROM model_registry WHERE model_name=?", (model_name,)
        ).fetchone()
    return (row[0] or 0) + 1


def champion(model_name: str) -> dict | None:
    with tx() as conn:
        row = conn.execute(
            "SELECT * FROM model_registry WHERE model_name=? AND stage='Production' "
            "ORDER BY version DESC LIMIT 1", (model_name,)
        ).fetchone()
    return dict(row) if row else None


def register(model_name: str, *, state: dict, metrics: dict, params: dict,
             git_commit: str = "local") -> dict:
    """Register a new version in Staging; auto-evaluate against champion gate."""
    version = next_version(model_name)
    artifact = MODEL_DIR / f"{_safe_stem(model_name, version)}.json"
    artifact.write_text(json.dumps({"state": state, "params": params}), encoding="utf-8")

    champ = champion(model_name)
    stage = "Staging"
    promoted = False
    blocked = False
    new_mape = metrics.get("mape", float("inf"))

    if champ is None:
        stage, promoted = "Production", True
    else:
        champ_mape = champ["mape"]
        if new_mape is None or champ_mape is None:
            if champ_mape is None and new_mape is not None:
                stage, promoted = "Production", True
        else:
            delta_pp = (new_mape - champ_mape) * 100.0  # mape stored as fraction
            if delta_pp > REGRESSION_BLOCK_PP:
                blocked = True  # promotion blocked + alert (AC-3.5)
            elif delta_pp <= PROMOTION_TOLERANCE_PP:
                stage, promoted = "Production", True

    with tx() as conn:
        if promoted and champ is not None:
            conn.execute(
                "UPDATE model_registry SET stage='Archived' WHERE model_name=? AND stage='Production'",
                (model_name,),
            )
        conn.execute(
            """INSERT INTO model_registry (model_name, version, stage, mape, rmse, coverage,
                 bias, params_json, artifact_path, created_at, git_commit)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (model_name, version, stage, new_mape, metrics.get("rmse"),
             metrics.get("coverage"), metrics.get("bias"), json.dumps(params),
             str(artifact), _utcnow(), git_commit),
        )
    return {"model_name": model_name, "version": version, "stage": stage,
            "promoted": promoted, "blocked": blocked, "mape": new_mape,
            "champion_mape": champ["mape"] if champ else None}


def set_production(model_name: str, version: int) -> dict:
    """Single-call rollback / manual promotion (AC-3.6)."""
    with tx() as conn:
        exists = conn.execute(
            "SELECT 1 FROM model_registry WHERE model_name=? AND version=?",
            (model_name, version),
        ).fetchone()
        if not exists:
            raise ValueError(f"{model_name} v{version} not found")
        conn.execute(
            "UPDATE model_registry SET stage='Archived' WHERE model_name=? AND stage='Production'",
            (model_name,),
        )
        conn.execute(
            "UPDATE model_registry SET stage='Production' WHERE model_name=? AND version=?",
            (model_name, version),
        )
    return {"model_name": model_name, "version": version, "stage": "Production"}


def list_versions(model_name: str) -> list[dict]:
    with tx() as conn:
        rows = conn.execute(
            "SELECT model_name, version, stage, mape, coverage, created_at "
            "FROM model_registry WHERE model_name=? ORDER BY version DESC", (model_name,)
        ).fetchall()
    return [dict(r) for r in rows]


def load_artifact(artifact_path: str) -> dict:
    return json.loads(open(artifact_path, encoding="utf-8").read())
