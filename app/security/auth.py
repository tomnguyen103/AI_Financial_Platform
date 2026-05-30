"""JWT issuance/verification + RBAC.

Spec refs: arch §2.7 (JWT/RBAC, roles + permission matrix), PRD §9.1.
Local HS256 issuer stands in for Azure AD / Entra ID — same bearer-token
contract, so the issuer can be swapped without touching route guards.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

# Roles and capability matrix (arch §2.7).
ROLES = {"collections", "finance", "da_analyst", "admin"}

# capability -> roles allowed
_CAPS: dict[str, set[str]] = {
    "forecasts:read": {"collections", "finance", "da_analyst", "admin"},
    "forecasts:write": {"da_analyst", "admin"},
    "alerts:read": {"collections", "finance", "da_analyst", "admin"},
    "alerts:write": {"da_analyst", "admin"},
    "chatbot:use": {"collections", "finance", "da_analyst", "admin"},
    "nl2sql:use": {"collections", "finance", "da_analyst", "admin"},
    "admin": {"admin", "da_analyst"},
}

_bearer = HTTPBearer(auto_error=True)


@dataclass
class User:
    user_id: str
    role: str


def create_token(user_id: str, role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + dt.timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_alg)


def decode_token(token: str) -> User:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
    except jwt.PyJWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}")
    return User(user_id=payload["sub"], role=payload.get("role", ""))


def current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> User:
    return decode_token(creds.credentials)


def require(capability: str):
    """Dependency factory enforcing a capability from the matrix."""
    allowed = _CAPS.get(capability, set())

    def _guard(user: User = Depends(current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"role '{user.role}' lacks capability '{capability}'",
            )
        return user

    return _guard
