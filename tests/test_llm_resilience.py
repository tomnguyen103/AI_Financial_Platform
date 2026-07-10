"""LLM client resilience tests.

When the real OpenAI/Azure path is enabled, a timeout / API / network failure
must degrade gracefully to the deterministic offline stub (and be logged) rather
than propagating and taking down the request. The stub logic itself is unchanged.
"""
from __future__ import annotations

import logging
import types

from app.llm import client as client_mod


class _RaisingEndpoint:
    def create(self, **kwargs):
        raise RuntimeError("simulated API failure / timeout")


def _fake_openai() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_RaisingEndpoint()),
        embeddings=_RaisingEndpoint(),
    )


def _enabled_client() -> client_mod.LLMClient:
    # In the hermetic test env there is no key, so the constructed client is
    # disabled (never builds a real client). Flip it into the enabled path with
    # a fake client whose API calls raise, to exercise the fallback branch.
    c = client_mod.LLMClient()
    c.enabled = True
    c.model = "gpt-test"
    c.embed_model = "embed-test"
    c._client = _fake_openai()
    return c


def test_complete_falls_back_to_stub_on_api_exception(caplog):
    c = _enabled_client()
    with caplog.at_level(logging.WARNING):
        out = c.complete("You are a helpful analyst.", "Summarize the figures.")

    assert out.startswith("[stub LLM")  # deterministic offline fallback
    assert any("llm complete failed" in r.message for r in caplog.records)


def test_embed_falls_back_to_stub_on_api_exception(caplog):
    c = _enabled_client()
    with caplog.at_level(logging.WARNING):
        vecs = c.embed(["hello world"])

    assert len(vecs) == 1
    assert len(vecs[0]) == 256  # stub hashing embedder dimension
    assert any("llm embed failed" in r.message for r in caplog.records)


def test_disabled_client_uses_stub_without_touching_network():
    c = client_mod.LLMClient()  # hermetic env -> disabled
    assert not c.enabled
    assert c.model_name == "stub"
    assert c.complete("sys", "user").startswith("[stub LLM")
