"""Security-layer tests: PHI tokenization, scanners, JWT/RBAC."""
from __future__ import annotations

import pytest

from app.security.auth import create_token, decode_token, permissions_for
from app.security.phi import mask_record, scan_input, scan_output, tokenize


def test_tokenize_deterministic_and_nonreversible():
    a = tokenize("123-45-6789")
    b = tokenize("123-45-6789")
    assert a == b                       # referential integrity
    assert a.startswith("tok_")
    assert "123-45-6789" not in a       # not reversible by inspection


def test_mask_record_masks_only_phi_fields():
    rec = {"patient_name": "John Doe", "facility_id": "round_rock", "billed_amount": 100}
    out = mask_record("visits", rec)
    assert out["patient_name"].startswith("tok_")
    assert out["facility_id"] == "round_rock"   # business field untouched
    assert out["billed_amount"] == 100


def test_scan_input_blocks_ssn_and_patient_request():
    assert scan_input("what is patient John Doe's balance").blocked
    assert scan_input("show 123-45-6789").blocked
    assert scan_input("patient name lookup").blocked


def test_scan_input_allows_business_entities():
    assert not scan_input("how is Attorney Johnson performing").blocked
    assert not scan_input("Round Rock collections trend").blocked
    assert not scan_input("total collected by facility").blocked


def test_scan_output_redacts_patient_name_not_attorney():
    r = scan_output("Patient Jane Roe owes money")
    assert r.phi_detected and "[REDACTED]" in r.redacted_text
    clean = scan_output("Attorney Garcia leads collections")
    assert not clean.phi_detected


def test_jwt_roundtrip():
    tok = create_token("u1", "da_analyst")
    user = decode_token(tok)
    assert user.user_id == "u1" and user.role == "da_analyst"


def test_permissions_differ_by_role():
    # account types must not be interchangeable
    collections = permissions_for("collections")
    finance = permissions_for("finance")
    analyst = permissions_for("da_analyst")

    assert "nl2sql:use" not in collections      # front-line: no ad-hoc querying
    assert "forecasts:write" not in collections  # read-only operational view
    assert "nl2sql:use" in finance               # finance adds reporting queries
    assert "forecasts:write" not in finance
    assert {"forecasts:write", "alerts:write", "admin"} <= set(analyst)  # power user
    assert collections != finance != analyst
