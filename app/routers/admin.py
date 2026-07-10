"""Admin/MLOps endpoints: model registry, rollback, feature freshness (arch §2.8)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.features.store import freshness, stale_groups
from app.forecasting.registry import champion, list_versions, set_production
from app.security.auth import User, require

router = APIRouter(prefix="/admin", tags=["admin"])


class ModelVersionSummary(BaseModel):
    model_name: str
    version: int
    stage: str
    mape: float | None
    coverage: float | None
    created_at: str


class ChampionModel(BaseModel):
    model_name: str
    version: int
    stage: str
    mape: float | None
    rmse: float | None
    coverage: float | None
    bias: float | None
    params_json: str
    artifact_path: str
    created_at: str
    git_commit: str


class ModelRegistryResponse(BaseModel):
    model_name: str
    champion: ChampionModel | None
    versions: list[ModelVersionSummary]


class ProductionPromotion(BaseModel):
    model_name: str
    version: int
    stage: str


class RollbackResponse(BaseModel):
    model_name: str
    production: ProductionPromotion


class FeatureFreshnessItem(BaseModel):
    feature_group: str
    last_updated: str
    row_count: int


class FreshnessResponse(BaseModel):
    freshness: list[FeatureFreshnessItem]
    stale_groups: list[str]


@router.get("/models/{model_name}", response_model=ModelRegistryResponse)
def models(model_name: str, user: User = Depends(require("admin"))) -> ModelRegistryResponse:
    return ModelRegistryResponse(
        model_name=model_name,
        champion=champion(model_name),
        versions=list_versions(model_name),
    )


@router.post("/models/{model_name}/rollback/{version}", response_model=RollbackResponse)
def rollback(model_name: str, version: int,
             user: User = Depends(require("admin"))) -> RollbackResponse:
    promoted = set_production(model_name, version)
    return RollbackResponse(model_name=model_name, production=promoted)


@router.get("/freshness", response_model=FreshnessResponse)
def feature_freshness(user: User = Depends(require("admin"))) -> FreshnessResponse:
    return FreshnessResponse(freshness=freshness(), stale_groups=stale_groups())
