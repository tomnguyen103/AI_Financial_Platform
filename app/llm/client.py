"""LLM abstraction over OpenAI / Azure OpenAI with a deterministic local fallback.

User chose OpenAI/Azure OpenAI. When OPENAI_API_KEY is set, real calls are made
(Azure if AZURE_OPENAI_ENDPOINT is also set). With no key, a deterministic stub
is used so the whole platform runs offline — the stub is clearly labelled in
responses and the audit log records model="stub".

Two capabilities are exposed:
  - complete(system, user): chat completion -> str
  - embed(texts): embeddings -> list[list[float]] (stub uses a hashing embedder)
"""
from __future__ import annotations

import hashlib

from app.config import settings
from app.logging_config import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(self) -> None:
        self.enabled = settings.llm_enabled
        self.model = settings.openai_model
        self.embed_model = settings.openai_embed_model
        self._client = None
        if self.enabled:
            self._client = self._build_client()

    def _build_client(self):
        from openai import AzureOpenAI, OpenAI
        if settings.azure_openai_endpoint:
            return AzureOpenAI(
                api_key=settings.openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                timeout=15.0,
            )
        return OpenAI(api_key=settings.openai_api_key, timeout=15.0)

    @property
    def model_name(self) -> str:
        return self.model if self.enabled else "stub"

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        if not self.enabled:
            return self._stub_complete(system, user)
        try:
            resp = self._client.chat.completions.create(
                model=self.model, temperature=temperature,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            # Timeout / API / network failure -> degrade to the deterministic
            # stub so the request still returns a grounded (if generic) answer.
            log.warning("llm complete failed; using stub", extra={"error": str(e)})
            return self._stub_complete(system, user)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            return [self._stub_embed(t) for t in texts]
        try:
            resp = self._client.embeddings.create(model=self.embed_model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:  # noqa: BLE001
            log.warning("llm embed failed; using stub", extra={"error": str(e)})
            return [self._stub_embed(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string, raising on provider failure (no stub fallback).

        Unlike embed(), this never swallows a failure into a stub vector. It is
        the "real-or-raise" path used by the query-embedding cache so that only
        genuine provider results are ever cached — a transient failure must not
        be memoised as if it were a real embedding.
        """
        resp = self._client.embeddings.create(model=self.embed_model, input=[text])
        return resp.data[0].embedding

    # --- deterministic fallbacks -------------------------------------------
    def _stub_complete(self, system: str, user: str) -> str:
        # For NL-to-SQL the caller wraps prompts so the stub can detect intent.
        if "Return ONLY a SQL" in system or "generate a SQL" in system.lower():
            return self._stub_sql(user)
        return ("[stub LLM — set OPENAI_API_KEY for real synthesis] "
                "Based on the provided context, here is a grounded summary. "
                "See the cited sources below for the underlying figures.")

    def _stub_sql(self, user: str) -> str:
        u = user.lower()
        if "180" in u and ("aging" in u or "bucket" in u):
            return ("SELECT attorney_id, bucket_180_plus FROM attorney_aging "
                    "ORDER BY bucket_180_plus DESC LIMIT 10")
        if "unpaid" in u and "visit" in u:
            return ("SELECT visit_id, facility_id, billed_amount FROM visits "
                    "WHERE billing_status = 'unpaid' LIMIT 100")
        if ("attorney" in u and
                any(term in u for term in ("pending", "outstanding", "aging", "balance", "payment"))):
            return (
                "SELECT attorney_id, "
                "SUM(bucket_0_30 + bucket_31_60 + bucket_61_90 + bucket_91_180 + bucket_180_plus) "
                "AS total_pending_payment "
                "FROM attorney_aging "
                "GROUP BY attorney_id "
                "ORDER BY total_pending_payment DESC"
            )
        if "collect" in u and "attorney" in u:
            return ("SELECT attorney_id, SUM(amount_collected) AS total FROM collections "
                    "GROUP BY attorney_id ORDER BY total DESC")
        return "SELECT facility_id, SUM(amount_collected) AS total FROM collections GROUP BY facility_id"

    def _stub_embed(self, text: str, dim: int = 256) -> list[float]:
        """Hash-based bag-of-words embedding (deterministic, offline)."""
        vec = [0.0] * dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
