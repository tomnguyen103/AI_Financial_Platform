from __future__ import annotations

from app.rag.corpus import Document
from app.rag.pinecone_store import _doc_to_record, _match_to_hit


def test_doc_to_record_includes_vector_and_metadata():
    doc = Document(
        "forecast_facility_round_rock_2026-03-27",
        "facility",
        "round_rock",
        "2026-03-27",
        "30-day collections forecast for round_rock.",
        {"feature_group": "forecast", "horizon": 30},
    )

    record = _doc_to_record(doc, [0.1, 0.2, 0.3])

    assert record["id"] == "forecast_facility_round_rock_2026-03-27"
    assert record["values"] == [0.1, 0.2, 0.3]
    assert record["metadata"]["text"] == "30-day collections forecast for round_rock."
    assert record["metadata"]["entity_type"] == "facility"
    assert record["metadata"]["entity_id"] == "round_rock"
    assert record["metadata"]["feature_group"] == "forecast"


def test_match_to_hit_round_trips_metadata():
    match = {
        "id": "alert_round_rock_2026-03-27",
        "score": 0.88,
        "metadata": {
            "source_doc_id": "alert_round_rock_2026-03-27",
            "entity_type": "facility",
            "entity_id": "round_rock",
            "date": "2026-03-27",
            "text": "Anomaly alert for round_rock.",
            "feature_group": "alert",
            "severity": "P2",
        },
    }

    hit = _match_to_hit(match)

    assert hit.source_doc_id == "alert_round_rock_2026-03-27"
    assert hit.entity_type == "facility"
    assert hit.entity_id == "round_rock"
    assert hit.date == "2026-03-27"
    assert hit.text == "Anomaly alert for round_rock."
    assert hit.score == 0.88
    assert hit.metadata["severity"] == "P2"
