"""PHI masking + input/output scanners.

Spec refs: 04_data_and_ml_design.md §2.2 (HMAC tokenization), §9.2/§9.3 (scanners).

Tokenization: deterministic HMAC-SHA256 keyed with a secret (Azure Key Vault in
prod, env var locally). Same input -> same token (referential integrity), not
reversible without the key.

NER note: the spec uses a NER model (spaCy) for PERSON detection. To keep the
MVP dependency-light and offline, we use a curated regex/heuristic detector that
covers SSN, DOB, and capitalized proper-name patterns. The detector lives behind
`scan_input`/`scan_output` so a spaCy/transformer NER can be swapped in later
without touching callers. Documented as a tradeoff in implementation-notes.md.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field

from app.config import settings

# PHI field names per entity (04 §2.2).
PHI_FIELDS: dict[str, list[str]] = {
    "visits": ["patient_name", "patient_dob", "patient_ssn_last4", "patient_address"],
    "settlements": ["plaintiff_name", "plaintiff_dob"],
    "lop": ["patient_name", "patient_dob"],
}


def tokenize(value: str) -> str:
    """Deterministic, non-reversible token for a single PHI value."""
    digest = hmac.new(
        settings.phi_hmac_key.encode(), str(value).encode(), hashlib.sha256
    ).hexdigest()
    return f"tok_{digest[:16]}"


def mask_record(entity: str, record: dict) -> dict:
    """Return a copy of `record` with that entity's PHI fields tokenized."""
    fields = PHI_FIELDS.get(entity, [])
    out = dict(record)
    for f in fields:
        if f in out and out[f] not in (None, ""):
            out[f] = tokenize(out[f])
    return out


# --- Scanners -------------------------------------------------------------

_PHI_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (re.compile(r"\b(DOB|date of birth|born on)\b", re.IGNORECASE), "DOB_REF"),
    (re.compile(r"\bpatient name\b", re.IGNORECASE), "PATIENT_NAME_REQUEST"),
    (re.compile(r"\bpatient(?:'s)?\s+(name|dob|ssn|address|identity)\b", re.IGNORECASE), "PATIENT_FIELD_REQUEST"),
]
# Two-word Capitalized proper-name heuristic (NER stand-in).
_PROPER_NAME = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")
# IMPORTANT: only PATIENT / PLAINTIFF names are PHI. Attorney, provider, and
# facility names are legitimate business entities and must NOT be flagged
# (the spec itself references "Attorney Johnson", "Round Rock", etc.). We exempt
# any proper-name match that contains a known role word or business entity token.
# This is the documented limitation of the regex NER stand-in; a real NER with an
# org-entity allowlist would replace it.
_ALLOWED_TOKENS = {
    # role / title prefixes
    "attorney", "attorneys", "provider", "providers", "facility", "facilities",
    "dr", "doctor", "firm", "office", "report", "summary", "collections",
    # facilities
    "round", "rock", "cedar", "park", "new", "braunfels", "san", "antonio",
    # attorney surnames in the dataset
    "johnson", "smith", "garcia", "lee", "patel",
    # domain phrases
    "personal", "injury", "letter", "protection", "commercial", "athena",
}


def _is_allowed_name(match: str) -> bool:
    return any(tok.lower() in _ALLOWED_TOKENS for tok in match.split())


@dataclass
class ScanResult:
    blocked: bool = False
    reason: str = ""
    phi_detected: bool = False
    redacted_text: str | None = None
    matches: list[str] = field(default_factory=list)


def scan_input(query: str) -> ScanResult:
    """Block queries that appear to request individual patient data."""
    for pattern, label in _PHI_PATTERNS:
        if pattern.search(query):
            return ScanResult(blocked=True, reason=label, phi_detected=True, matches=[label])
    for m in _PROPER_NAME.finditer(query):
        if not _is_allowed_name(m.group(0)):
            return ScanResult(
                blocked=True, reason="PERSON_ENTITY_DETECTED",
                phi_detected=True, matches=[m.group(0)],
            )
    return ScanResult(blocked=False)


def scan_output(response: str) -> ScanResult:
    """Redact (not block) PHI accidentally present in an LLM response."""
    matches: list[str] = []
    redacted = response
    for pattern, label in _PHI_PATTERNS:
        if pattern.search(redacted):
            redacted = pattern.sub("[REDACTED]", redacted)
            matches.append(label)
    for m in list(_PROPER_NAME.finditer(redacted)):
        if not _is_allowed_name(m.group(0)):
            redacted = redacted.replace(m.group(0), "[REDACTED]")
            matches.append(m.group(0))
    if matches:
        return ScanResult(blocked=False, phi_detected=True, redacted_text=redacted, matches=matches)
    return ScanResult(blocked=False, phi_detected=False, redacted_text=response)
