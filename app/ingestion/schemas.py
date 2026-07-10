"""Pydantic v2 models for OData entities (arch §2.1 validation; data design §2.1).

These define the *expected* contract. The schema-validation step compares the
incoming columns against these models; a missing/renamed/extra field surfaces as
a schema-contract failure (PRD AC-1.3).
"""
from __future__ import annotations

from pydantic import BaseModel


class Visit(BaseModel):
    visit_id: str
    facility_id: str
    case_type: str
    visit_date: str
    billing_status: str
    billed_amount: float
    paid_amount: float
    provider_id: str
    # PHI (masked at ingest)
    patient_name: str | None = None
    patient_dob: str | None = None
    patient_ssn_last4: str | None = None
    patient_address: str | None = None


class Collection(BaseModel):
    collection_id: str
    facility_id: str
    attorney_id: str
    case_type: str
    collection_date: str
    amount_collected: float
    days_outstanding: int


class AttorneyAging(BaseModel):
    attorney_id: str
    facility_id: str
    bucket_0_30: float
    bucket_31_60: float
    bucket_61_90: float
    bucket_91_180: float
    bucket_180_plus: float
    report_date: str


class Settlement(BaseModel):
    settlement_id: str
    attorney_id: str
    case_type: str
    open_date: str
    close_date: str | None = None
    settlement_amount: float
    settlement_status: str
    # PHI
    plaintiff_name: str | None = None
    plaintiff_dob: str | None = None


class LOP(BaseModel):
    lop_id: str
    facility_id: str
    case_type: str
    issued_date: str
    returned_date: str | None = None
    status: str
    rejection_reason: str | None = None
    # PHI
    patient_name: str | None = None
    patient_dob: str | None = None


# entity name -> (model, curated columns persisted to SQLite)
ENTITY_MODELS = {
    "visits": (Visit, ["visit_id", "facility_id", "case_type", "visit_date",
                       "billing_status", "billed_amount", "paid_amount", "provider_id"]),
    "collections": (Collection, ["collection_id", "facility_id", "attorney_id", "case_type",
                                 "collection_date", "amount_collected", "days_outstanding"]),
    "attorney_aging": (AttorneyAging, ["attorney_id", "facility_id", "bucket_0_30", "bucket_31_60",
                                       "bucket_61_90", "bucket_91_180", "bucket_180_plus", "report_date"]),
    "settlements": (Settlement, ["settlement_id", "attorney_id", "case_type", "open_date",
                                 "close_date", "settlement_amount", "settlement_status"]),
    "lop": (LOP, ["lop_id", "facility_id", "case_type", "issued_date", "returned_date",
                  "status", "rejection_reason"]),
}

FACILITIES = ["round_rock", "cedar_park", "new_braunfels", "san_antonio"]
ATTORNEYS = ["johnson", "smith", "garcia", "lee", "patel"]
CASE_TYPES = ["PI", "Commercial", "Athena"]
PROVIDERS = ["prov_a", "prov_b", "prov_c", "prov_d"]
