"""Round-2 A+ hardening: request-body length bounds + query-embedding cache."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.routers.chatbot import AskRequest
from app.routers.nl2sql import QueryRequest


def test_chatbot_query_rejects_oversized_body():
    with pytest.raises(ValidationError):
        AskRequest(query="x" * 2001)
    with pytest.raises(ValidationError):
        AskRequest(query="")  # min_length=1


def test_nl2sql_question_rejects_oversized_body():
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 2001)


def test_reasonable_bodies_accepted():
    assert AskRequest(query="How is Round Rock performing?").query
    assert QueryRequest(question="top 10 collections by attorney").question


def test_query_embedding_is_cached():
    from app.rag.vector_store import _embed_query

    _embed_query.cache_clear()
    a = _embed_query("How is Round Rock performing?")
    b = _embed_query("How is Round Rock performing?")
    assert a == b
    info = _embed_query.cache_info()
    assert info.hits >= 1          # second call served from cache
    assert isinstance(a, tuple)    # hashable/cacheable embedding
