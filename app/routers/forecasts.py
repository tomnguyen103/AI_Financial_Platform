"""Forecast read endpoints (PRD Module 3; arch §2.3)."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.forecasting.service import get_forecast, list_entities
from app.security.auth import User, require

router = APIRouter(prefix="/forecasts", tags=["forecasts"])

EntityType = Literal["facility", "attorney", "case_type"]


class EntityListResponse(BaseModel):
    entity_type: EntityType
    entities: list[str]


class ForecastResponse(BaseModel):
    id: int
    entity_type: str
    entity_id: str
    horizon_days: int
    forecast_date: str
    target_date: str
    predicted: float
    ci_lower: float
    ci_upper: float
    p50: float
    p80: float
    p95: float
    model_name: str
    model_version: int
    feature_snapshot_ts: str
    generated_at: str


@router.get("/entities", response_model=EntityListResponse)
def entities(entity_type: EntityType = Query("facility"),
             user: User = Depends(require("forecasts:read"))) -> EntityListResponse:
    return EntityListResponse(entity_type=entity_type, entities=list_entities(entity_type))


@router.get("/{entity_type}/{entity_id}", response_model=ForecastResponse)
def forecast(entity_type: EntityType, entity_id: str,
             horizon: int = Query(30, ge=1, le=365),
             user: User = Depends(require("forecasts:read"))) -> ForecastResponse:
    fc = get_forecast(entity_type, entity_id, horizon)
    if not fc:
        raise HTTPException(404, f"no forecast for {entity_type}/{entity_id} @ {horizon}d")
    return ForecastResponse(**fc)
