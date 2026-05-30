"""Local dev token issuance (stands in for Azure AD / Entra ID, arch §2.7).

In production, tokens come from the identity provider; here `/auth/token` mints an
HS256 JWT for a given user_id + role so the platform is exercisable end-to-end.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.security.auth import ROLES, create_token

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    user_id: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@router.post("/token", response_model=TokenResponse)
def issue_token(req: TokenRequest) -> TokenResponse:
    if req.role not in ROLES:
        raise HTTPException(400, f"unknown role '{req.role}'. valid: {sorted(ROLES)}")
    return TokenResponse(access_token=create_token(req.user_id, req.role), role=req.role)
