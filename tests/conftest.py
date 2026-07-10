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

It ALSO repoints every on-disk artifact path (the SQLite DB, the raw/curated
parquet zones, and the model-registry artifact dir) at a throwaway per-session
temp directory. The suite therefore never reads or writes the developer's real
`data/platform.db` — tests are hermetic and can't corrupt local data, and a fresh
run starts from a clean, deterministic schema every time.
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
    "SEED_ON_STARTUP": "0",          # -> app.main lifespan never spawns the seed thread
}

for _key, _val in _HERMETIC_ENV.items():
    os.environ[_key] = _val


@pytest.fixture(scope="session", autouse=True)
def _hermetic_db(tmp_path_factory):
    """Repoint all on-disk artifact paths at a temp dir and create the schema.

    Imported lazily (after the env is pinned above) so the offline config is in
    effect. The DB path constant is captured *by value* into several modules at
    import time (``from app.config import DB_PATH`` etc.), so we patch it at every
    use site. This guarantees the whole suite is hermetic and order-independent:
    no test can see or mutate the developer's real ``data/platform.db``.

    Only the schema is created here (fast). Tests that need populated business
    data request the ``seeded_db`` fixture below, which seeds once per session.
    """
    import app.config as config
    import app.db as db
    import app.forecasting.registry as registry
    import app.ingestion.pipeline as pipeline

    root = tmp_path_factory.mktemp("platform_data")
    db_path = root / "platform.db"
    raw_zone = root / "raw"
    curated_zone = root / "curated"
    model_dir = root / "models"
    for d in (raw_zone, curated_zone, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    mp = pytest.MonkeyPatch()
    # DB path: patched on app.config (source of truth) and app.db (imported by value).
    mp.setattr(config, "DB_PATH", db_path, raising=False)
    mp.setattr(db, "DB_PATH", db_path, raising=False)
    # Data-lake zones: app.config + the pipeline that imported them by value.
    mp.setattr(config, "RAW_ZONE", raw_zone, raising=False)
    mp.setattr(config, "CURATED_ZONE", curated_zone, raising=False)
    mp.setattr(pipeline, "RAW_ZONE", raw_zone, raising=False)
    mp.setattr(pipeline, "CURATED_ZONE", curated_zone, raising=False)
    # Model-registry artifacts: app.config + the registry that imported it.
    mp.setattr(config, "MODEL_DIR", model_dir, raising=False)
    mp.setattr(registry, "MODEL_DIR", model_dir, raising=False)

    db.init_db()
    yield root
    mp.undo()


@pytest.fixture(scope="session")
def seeded_db(_hermetic_db):
    """Populate the hermetic DB once per session via the real ingest->features->
    forecast->detect pipeline (same path as ``scripts.seed_data``).

    Deterministic given SYNTH_SEED, so downstream assertions are stable. Returns a
    small summary dict integration tests can assert against.
    """
    from app.anomaly.alerting import run_detection
    from app.features.compute import compute_all
    from app.forecasting.service import run_all as forecasting_run_all
    from app.ingestion.pipeline import run_ingest

    ingest = run_ingest()
    features = compute_all()
    forecasts = forecasting_run_all()
    detection = run_detection()
    return {
        "ingest": ingest,
        "features": features,
        "forecasts": forecasts,
        "detection": detection,
    }
