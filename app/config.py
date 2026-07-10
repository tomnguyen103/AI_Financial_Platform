"""Central configuration (pydantic-settings).

Reads a local .env automatically (via SettingsConfigDict) plus the process
environment. Every field is typed and validated at startup. Exports `settings`,
`get_settings()`, the derived path constants, and the `llm_enabled` property.
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_ZONE = DATA_DIR / "raw"
CURATED_ZONE = DATA_DIR / "curated"
MODEL_DIR = DATA_DIR / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
DB_PATH = DATA_DIR / "platform.db"

_INSECURE_DEFAULTS = {
    "jwt_secret": "dev-insecure-change-me-0123456789-please",
    "phi_hmac_key": "dev-phi-hmac-key-change-me",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # environment (set APP_ENV=production in real deployments — see .env.example)
    app_env: str = "development"

    # security
    jwt_secret: str = "dev-insecure-change-me-0123456789-please"
    jwt_alg: str = "HS256"
    jwt_expire_minutes: int = 480
    phi_hmac_key: str = "dev-phi-hmac-key-change-me"
    # unauthenticated /auth/token dev issuer (arch §2.7) — mints any role incl.
    # admin. Fine for this public demo; a real deployment MUST set
    # ENABLE_DEV_TOKEN=0 and front the API with its real IdP instead.
    enable_dev_token: bool = True
    # set true when running behind a reverse proxy/load balancer so the rate
    # limiter keys on the real client IP (X-Forwarded-For) instead of the
    # proxy's own address.
    trust_proxy: bool = False

    # llm
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embed_model: str = "text-embedding-3-small"
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-08-01-preview"

    # vector search
    vector_store: str = "memory"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "financial-rag-demo"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_dimension: int = 1536

    # alerting
    slack_webhook_url: str = ""

    # data generation
    synth_months: int = 20
    synth_seed: int = 42

    # ingestion source (synthetic generator only in this public build)
    ingest_source: str = "synthetic"
    ingest_permissive: bool = False

    @model_validator(mode="after")
    def _normalize(self) -> Settings:
        object.__setattr__(self, "vector_store", self.vector_store.lower())
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)

    def warn_if_insecure(self) -> list[str]:
        problems = [
            name for name, insecure in _INSECURE_DEFAULTS.items()
            if getattr(self, name) == insecure
        ]
        if problems:
            warnings.warn(
                "Insecure default secrets still in use: "
                f"{', '.join(problems)}. Set them via .env / Key Vault before prod.",
                stacklevel=2,
            )
        return problems

    def fail_if_insecure_in_production(self) -> None:
        """Fail closed at startup if a production deploy still ships dev secrets."""
        if self.app_env.lower() != "production":
            return
        problems = [
            name for name, insecure in _INSECURE_DEFAULTS.items()
            if getattr(self, name) == insecure
        ]
        if problems:
            raise RuntimeError(
                "APP_ENV=production but insecure default secrets are still in "
                f"use: {', '.join(problems)}. Set them via .env / Key Vault."
            )
        if self.enable_dev_token:
            raise RuntimeError(
                "APP_ENV=production but ENABLE_DEV_TOKEN is still enabled; the "
                "unauthenticated /auth/token issuer mints admin tokens. Set "
                "ENABLE_DEV_TOKEN=0 in production."
            )


@lru_cache
def get_settings() -> Settings:
    for d in (DATA_DIR, RAW_ZONE, CURATED_ZONE, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    s = Settings()
    s.fail_if_insecure_in_production()
    s.warn_if_insecure()
    return s


settings = get_settings()
