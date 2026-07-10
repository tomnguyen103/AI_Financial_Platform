"""Local dev token issuance (stands in for Azure AD / Entra ID, arch §2.7).

In production, tokens come from the identity provider; here `/auth/token` mints an
HS256 JWT for a given user_id + role so the platform is exercisable end-to-end.

This endpoint is intentionally unauthenticated (it stands in for the IdP login
flow) and will mint a token for ANY role, including admin. That is fine for this
public demo, but a real deployment MUST set ENABLE_DEV_TOKEN=0 and validate
tokens from its real identity provider instead.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.security.auth import ROLES, create_token, permissions_for

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    user_id: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    permissions: list[str]


@router.post("/token", response_model=TokenResponse)
def issue_token(req: TokenRequest) -> TokenResponse:
    if not settings.enable_dev_token:
        raise HTTPException(403, "dev token issuance disabled")
    if req.role not in ROLES:
        raise HTTPException(400, f"unknown role '{req.role}'. valid: {sorted(ROLES)}")
    return TokenResponse(
        access_token=create_token(req.user_id, req.role),
        role=req.role,
        permissions=permissions_for(req.role),
    )
