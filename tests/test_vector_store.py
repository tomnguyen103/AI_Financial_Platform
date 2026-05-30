from __future__ import annotations

from app.rag.vector_store import SearchHit


def test_search_hit_preserves_document_score_and_metadata():
    hit = SearchHit(
        source_doc_id="forecast_facility_round_rock_2026-03-27",
        entity_type="facility",
        entity_id="round_rock",
        date="2026-03-27",
        text="30-day collections forecast for facility round_rock.",
        score=0.91,
        metadata={"feature_group": "forecast"},
    )

    assert hit.source_doc_id == "forecast_facility_round_rock_2026-03-27"
    assert hit.entity_type == "facility"
    assert hit.entity_id == "round_rock"
    assert hit.date == "2026-03-27"
    assert hit.score == 0.91
    assert hit.metadata["feature_group"] == "forecast"
