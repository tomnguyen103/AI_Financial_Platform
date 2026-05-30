"""Admin/MLOps endpoints: model registry, rollback, feature freshness (arch §2.8)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.features.store import freshness, stale_groups
from app.forecasting.registry import champion, list_versions, set_production
from app.security.auth import User, require

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/models/{model_name}")
def models(model_name: str, user: User = Depends(require("admin"))) -> dict:
    return {
        "model_name": model_name,
        "champion": champion(model_name),
        "versions": list_versions(model_name),
    }


@router.post("/models/{model_name}/rollback/{version}")
def rollback(model_name: str, version: int, user: User = Depends(require("admin"))) -> dict:
    promoted = set_production(model_name, version)
    return {"model_name": model_name, "production": promoted}


@router.get("/freshness")
def feature_freshness(user: User = Depends(require("admin"))) -> dict:
    return {"freshness": freshness(), "stale_groups": stale_groups()}
