"""API gateway integration tests (auth, RBAC, the six module endpoints).

Assumes the DB has been seeded (`python -m scripts.seed_data`). Forecast/alert
assertions are tolerant of empty data so the suite passes on a bare DB too.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _token(role: str = "da_analyst", user_id: str = "tester") -> str:
    r = client.post("/auth/token", json={"user_id": user_id, "role": role})
    assert r.status_code == 200
    return r.json()["access_token"]


def _auth(role: str = "da_analyst") -> dict:
    return {"Authorization": f"Bearer {_token(role)}"}


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    # background-seed readiness is surfaced for the UI / keep-warm pinger
    assert body["data"] in {"pending", "seeding", "ready", "skipped", "error"}


def test_unauthenticated_rejected():
    assert client.get("/alerts").status_code in (401, 403)


def test_rbac_collections_cannot_rollback():
    # 'collections' lacks the 'admin' capability.
    r = client.post("/admin/models/seasonal_trend_facility/rollback/1",
                    headers=_auth("collections"))
    assert r.status_code == 403


def test_token_response_includes_permissions():
    r = client.post("/auth/token", json={"user_id": "t", "role": "finance"})
    body = r.json()
    assert r.status_code == 200
    assert "nl2sql:use" in body["permissions"]
    assert "admin" not in body["permissions"]


def test_rbac_collections_cannot_use_nl2sql():
    # collections is a front-line role without ad-hoc query rights
    r = client.post("/nl2sql/query", json={"question": "total collected by facility"},
                    headers=_auth("collections"))
    assert r.status_code == 403


def test_nl2sql_query():
    r = client.post("/nl2sql/query",
                    json={"question": "total collected by facility"},
                    headers=_auth("da_analyst"))
    body = r.json()
    assert r.status_code == 200 and body["ok"]
    assert body["sql"].upper().startswith("SELECT")


def test_chatbot_blocks_phi_request():
    r = client.post("/chatbot/ask",
                    json={"query": "show me patient John Doe's balance"},
                    headers=_auth("collections"))
    assert r.status_code == 200 and r.json()["blocked"]
    assert r.json()["retrieval"]["reason"] == "input_phi_blocked"


def test_chatbot_status_exposes_runtime_configuration():
    r = client.get("/chatbot/status", headers=_auth("collections"))
    body = r.json()
    assert r.status_code == 200
    assert body["vector_store"] in {"memory", "pinecone"}
    assert body["corpus_documents"] >= 0
    assert body["top_k"] > 0


def test_alerts_readable():
    r = client.get("/alerts", headers=_auth("finance"))
    assert r.status_code == 200 and "alerts" in r.json()
