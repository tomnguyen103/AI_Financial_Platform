"""Pinecone store mapping-helper tests (pure functions, no network).

`_doc_to_record` and `_match_to_hit` translate between our Document/SearchHit
types and Pinecone's record/match shapes. They are exercised directly so the
mapping is covered without a live Pinecone index.
"""
from __future__ import annotations

from app.rag.corpus import Document
from app.rag.pinecone_store import _doc_to_record, _match_to_hit


def test_doc_to_record_flattens_metadata_and_carries_vector():
    doc = Document(
        source_doc_id="facility_collections_round_rock_2026-05-27",
        entity_type="facility", entity_id="round_rock", date="2026-05-27",
        text="Round Rock collections summary.", metadata={"kind": "summary"},
    )
    record = _doc_to_record(doc, [0.1, 0.2, 0.3])
    assert record["id"] == doc.source_doc_id
    assert record["values"] == [0.1, 0.2, 0.3]
    md = record["metadata"]
    assert md["entity_id"] == "round_rock"
    assert md["text"] == doc.text
    assert md["kind"] == "summary"  # extra metadata merged in flat


def test_match_to_hit_from_dict():
    match = {
        "id": "doc1", "score": 0.87,
        "metadata": {
            "source_doc_id": "doc1", "entity_type": "facility", "entity_id": "round_rock",
            "date": "2026-05-27", "text": "summary text", "extra_key": "extra_val",
        },
    }
    hit = _match_to_hit(match)
    assert hit.source_doc_id == "doc1"
    assert hit.entity_id == "round_rock"
    assert hit.text == "summary text"
    assert hit.score == 0.87
    # Reserved keys are stripped from the residual metadata; extras are kept.
    assert hit.metadata == {"extra_key": "extra_val"}


def test_match_to_hit_uses_to_dict_and_defaults_missing_fields():
    class _Match:
        def to_dict(self):
            return {"id": "fallback_id", "score": None, "metadata": None}

    hit = _match_to_hit(_Match())
    # Missing source_doc_id falls back to the match id; score coerces to 0.0.
    assert hit.source_doc_id == "fallback_id"
    assert hit.score == 0.0
    assert hit.entity_id == ""
    assert hit.metadata == {}
