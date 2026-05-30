"""Anomaly alert endpoints (PRD Module 4; arch §2.4)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.anomaly.alerting import acknowledge, list_alerts
from app.security.auth import User, require

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def get_alerts(severity: str | None = Query(None),
               status: str | None = Query(None),
               user: User = Depends(require("alerts:read"))) -> dict:
    rows = list_alerts(severity=severity, status=status)
    return {"count": len(rows), "alerts": rows}


@router.post("/{alert_id}/acknowledge")
def ack(alert_id: str, user: User = Depends(require("alerts:write"))) -> dict:
    if not acknowledge(alert_id, user.user_id):
        raise HTTPException(404, f"unknown alert '{alert_id}'")
    return {"alert_id": alert_id, "status": "acknowledged", "by": user.user_id}
