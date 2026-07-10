"""Tests for the A+ audit security hardening fixes.

Covers: gated dev-token issuer, fail-closed default secrets in production,
security response headers, and proxy-aware rate-limiter client IP.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.routers.auth as auth_router
from app.config import Settings
from app.main import app

client = TestClient(app)


def test_dev_token_disabled_returns_403(monkeypatch):
    monkeypatch.setattr(auth_router.settings, "enable_dev_token", False)
    r = client.post("/auth/token", json={"user_id": "u1", "role": "admin"})
    assert r.status_code == 403


def test_dev_token_enabled_by_default_still_works():
    r = client.post("/auth/token", json={"user_id": "u1", "role": "da_analyst"})
    assert r.status_code == 200


def test_production_with_default_secrets_fails_closed():
    s = Settings(app_env="production", jwt_secret="dev-insecure-change-me-0123456789-please")
    with pytest.raises(RuntimeError):
        s.fail_if_insecure_in_production()


def test_production_with_custom_secrets_is_fine():
    s = Settings(
        app_env="production",
        jwt_secret="a-real-production-secret",
        phi_hmac_key="a-real-production-phi-key",
        enable_dev_token=False,
    )
    s.fail_if_insecure_in_production()  # must not raise


def test_production_with_dev_token_enabled_fails_closed():
    s = Settings(
        app_env="production",
        jwt_secret="a-real-production-secret",
        phi_hmac_key="a-real-production-phi-key",
        enable_dev_token=True,
    )
    with pytest.raises(RuntimeError):
        s.fail_if_insecure_in_production()


def test_development_with_default_secrets_does_not_raise():
    s = Settings(app_env="development")
    s.fail_if_insecure_in_production()  # must not raise outside production


def test_security_headers_present_on_response():
    r = client.get("/health")
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert "Content-Security-Policy" in r.headers
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert "Strict-Transport-Security" in r.headers


def test_rate_limiter_honors_x_forwarded_for_when_trust_proxy(monkeypatch):
    monkeypatch.setattr(main_module.settings, "trust_proxy", True)
    r = client.get("/health", headers={"X-Forwarded-For": "203.0.113.5, 10.0.0.1"})
    assert r.status_code == 200
    assert "203.0.113.5" in main_module._hits


def test_rate_limiter_ignores_x_forwarded_for_when_not_trusted(monkeypatch):
    monkeypatch.setattr(main_module.settings, "trust_proxy", False)
    r = client.get("/health", headers={"X-Forwarded-For": "198.51.100.9"})
    assert r.status_code == 200
    assert "198.51.100.9" not in main_module._hits
