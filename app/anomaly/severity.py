"""Combine detector findings into one alert per entity at the highest severity.

data design §5.4: cross-validation across detectors raises confidence; a single
detector firing alone is at most its own severity. We escalate to P1 when two or
more detectors agree at >= P2.
"""
from __future__ import annotations

from app.anomaly.detectors import Finding

_RANK = {"P1": 3, "P2": 2, "P3": 1}


def combine(findings: list[Finding]) -> Finding | None:
    if not findings:
        return None
    best = max(findings, key=lambda f: _RANK[f.severity])
    # cross-validation escalation
    p2_plus = [f for f in findings if _RANK[f.severity] >= 2]
    if len(p2_plus) >= 2 and best.severity != "P1":
        best.severity = "P1"
        best.detail = {**best.detail, "escalated": "multi_detector_agreement",
                       "detectors": [f.detector for f in p2_plus]}
    return best
