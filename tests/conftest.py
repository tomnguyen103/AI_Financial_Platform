"""Shared test fixtures and environment pinning.

The application reads configuration from the process environment (and a local
.env) at import time via app.config. That makes test outcomes depend on whatever
happens to be in a developer's .env — e.g. a real OPENAI_API_KEY forces the live
OpenAI client path (and an import of the `openai` package) instead of the
deterministic offline stub, which can break the suite on a machine that has a key
set.

This module pins a hermetic, offline configuration BEFORE app.config is imported,
so the suite behaves identically everywhere: no network, no real keys, in-memory
vector store, deterministic stub LLM.
"""
from __future__ import annotations

import os

import pytest

# --- Pin the environment before any `app.*` import happens ----------------
# pytest imports conftest.py before collecting/importing test modules, so
# setting these here guarantees app.config sees the offline values.
_HERMETIC_ENV = {
    "OPENAI_API_KEY": "",            # -> LLMClient uses the deterministic stub
    "AZURE_OPENAI_ENDPOINT": "",     # -> never selects the Azure client
    "VECTOR_STORE": "memory",        # -> InMemoryVectorStore, no Pinecone
    "PINECONE_API_KEY": "",
    "INGEST_SOURCE": "synthetic",    # -> never touches a real Postgres
    "PG_DATABASE": "",
    "SLACK_WEBHOOK_URL": "",         # -> alerts stay DB-only, no webhook calls
    "JWT_SECRET": "test-only-secret-not-used-in-prod",
    "PHI_HMAC_KEY": "test-only-phi-key",
}

for _key, _val in _HERMETIC_ENV.items():
    os.environ[_key] = _val


@pytest.fixture(scope="session", autouse=True)
def _init_database():
    """Initialize the SQLite schema once for the whole test session.

    Imported lazily (after the env is pinned above) so the offline config is in
    effect. Tests that need data should seed it themselves or monkeypatch the
    relevant accessors; this only guarantees the tables exist.
    """
    from app.db import init_db

    init_db()
    yield
