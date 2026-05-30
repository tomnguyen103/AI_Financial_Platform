"""Forecast read endpoints (PRD Module 3; arch §2.3)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.forecasting.service import get_forecast, list_entities
from app.security.auth import User, require

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@router.get("/entities")
def entities(entity_type: str = Query("facility"),
             user: User = Depends(require("forecasts:read"))) -> dict:
    return {"entity_type": entity_type, "entities": list_entities(entity_type)}


@router.get("/{entity_type}/{entity_id}")
def forecast(entity_type: str, entity_id: str,
             horizon: int = Query(30, ge=1, le=365),
             user: User = Depends(require("forecasts:read"))) -> dict:
    fc = get_forecast(entity_type, entity_id, horizon)
    if not fc:
        raise HTTPException(404, f"no forecast for {entity_type}/{entity_id} @ {horizon}d")
    return fc
