"""Gateway middleware tests (arch §2.7): rate limiting, security headers, and
the uniform 500 error envelope."""
from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main


# A throwaway route that always raises, to drive the middleware's 500 path.
@main.app.get("/_test_boom", include_in_schema=False)
def _boom() -> dict:
    raise RuntimeError("kaboom")


client = TestClient(main.app, raise_server_exceptions=False)


def _reset_rate_limiter(monkeypatch, cap: int) -> None:
    monkeypatch.setattr(main, "_RATE_MAX", cap)
    main._hits.clear()


def test_security_headers_present_on_normal_response():
    resp = client.get("/health")
    assert resp.status_code == 200
    for header in ("X-Content-Type-Options", "X-Frame-Options",
                   "Referrer-Policy", "Strict-Transport-Security",
                   "Content-Security-Policy"):
        assert header in resp.headers
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert "X-Request-ID" in resp.headers  # per-request id echoed back


def test_rate_limiter_returns_429_after_cap(monkeypatch):
    _reset_rate_limiter(monkeypatch, cap=5)
    # First `cap` requests pass; the next is rejected.
    for _ in range(5):
        assert client.get("/health").status_code == 200
    blocked = client.get("/health")
    assert blocked.status_code == 429
    assert blocked.json() == {"error": "rate limit exceeded"}
    # The 429 response still carries the security headers.
    assert blocked.headers["X-Content-Type-Options"] == "nosniff"
    main._hits.clear()


def test_500_envelope_shape(monkeypatch):
    # Give the limiter plenty of headroom so the boom route isn't rate-limited.
    _reset_rate_limiter(monkeypatch, cap=1000)
    resp = client.get("/_test_boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "internal error"
    assert "request_id" in body and body["request_id"]
    # Security headers are attached to the error envelope too.
    assert resp.headers["X-Frame-Options"] == "DENY"
    main._hits.clear()
