"""Round-2 A+ hardening: request-body length bounds on free-text fields."""
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
