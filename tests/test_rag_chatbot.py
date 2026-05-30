from __future__ import annotations

from app.rag import chatbot
from app.rag.corpus import Document
from app.rag.vector_store import SearchHit


class FakeIndex:
    def __init__(self):
        self.docs = [
            Document(
                "attorney_aging_JOHNSON_2026-03-27",
                "attorney",
                "JOHNSON",
                "2026-03-27",
                "Attorney Johnson aging report as of 2026-03-27: total outstanding $5,543,209.",
            )
        ]

    def search(self, query: str, top_k: int = 8, entity_id: str | None = None):
        assert entity_id == "JOHNSON"
        return [SearchHit.from_document(self.docs[0], 0.70)]


class FakeLLM:
    enabled = False
    model_name = "stub"

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        return ""


def test_exact_entity_match_answers_below_generic_similarity_threshold(monkeypatch):
    monkeypatch.setattr(chatbot, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(chatbot, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(chatbot, "write_audit", lambda **kwargs: None)

    response = chatbot.ask("How is Attorney Johnson performing?")

    assert not response.insufficient
    assert "Johnson" in response.answer
    assert response.citations[0]["entity_id"] == "JOHNSON"
    assert response.citations[0]["score"] == 0.7
    assert response.retrieval["status"] == "grounded"
    assert response.retrieval["entity_filter"] == "JOHNSON"


def test_unrelated_question_returns_insufficient_without_retrieval(monkeypatch):
    class NoSearchIndex(FakeIndex):
        def search(self, query: str, top_k: int = 8, entity_id: str | None = None):
            raise AssertionError("unrelated questions should not hit vector search")

    monkeypatch.setattr(chatbot, "get_index", lambda: NoSearchIndex())
    monkeypatch.setattr(chatbot, "write_audit", lambda **kwargs: None)

    response = chatbot.ask("What is the capital of France?")

    assert response.insufficient
    assert response.citations == []
    assert response.retrieval["reason"] == "outside_financial_domain"


def test_unknown_facility_question_below_threshold_is_insufficient(monkeypatch):
    class LowConfidenceIndex(FakeIndex):
        def search(self, query: str, top_k: int = 8, entity_id: str | None = None):
            assert entity_id is None
            return [SearchHit.from_document(self.docs[0], 0.40)]

    monkeypatch.setattr(chatbot, "get_index", lambda: LowConfidenceIndex())
    monkeypatch.setattr(chatbot, "SIM_THRESHOLD", 0.75)
    monkeypatch.setattr(chatbot, "write_audit", lambda **kwargs: None)

    response = chatbot.ask("Why did round_rock collections drop recently?")

    assert response.insufficient
    assert response.citations == []
    assert response.retrieval["reason"] == "below_similarity_threshold"
