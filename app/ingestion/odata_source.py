"""Synthetic OData feed (local stand-in for the real OData endpoint).

Generates internally-consistent financial data with trend + weekly/yearly
seasonality + noise so that forecasting and anomaly detection have something
meaningful to learn. Deterministic given SYNTH_SEED.

A few *intentional anomalies* are injected (a collections stall at one facility
in the most recent weeks) so the anomaly module has a true positive to find.

PHI fields are populated with obviously-fake values to exercise masking.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

from app.config import settings
from app.ingestion.schemas import ATTORNEYS, CASE_TYPES, FACILITIES, PROVIDERS

_FAKE_FIRST = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Jamie"]
_FAKE_LAST = ["Doe", "Roe", "Public", "Smithson", "Tester", "Sample"]

# Per-facility baseline daily collections ($) and trend slope ($/day).
_FACILITY_BASE = {
    "round_rock": (48000, 12.0),
    "cedar_park": (32000, 6.0),
    "new_braunfels": (21000, 18.0),
    "san_antonio": (61000, -4.0),
}


def _rng() -> np.random.Generator:
    return np.random.default_rng(settings.synth_seed)


def _date_range(months: int) -> list[dt.date]:
    end = dt.date(2026, 5, 27)  # day before "today" in spec
    start = end - dt.timedelta(days=int(months * 30.44))
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def _fake_name(rng) -> str:
    return f"{rng.choice(_FAKE_FIRST)} {rng.choice(_FAKE_LAST)}"


def _fake_dob(rng) -> str:
    y = int(rng.integers(1950, 2005))
    m = int(rng.integers(1, 13))
    d = int(rng.integers(1, 28))
    return f"{y:04d}-{m:02d}-{d:02d}"


def _daily_collection_signal(base: float, slope: float, day_idx: int, date: dt.date, rng) -> float:
    trend = base + slope * day_idx
    weekly = 1.0 + 0.18 * math.sin(2 * math.pi * date.weekday() / 7.0)
    yearly = 1.0 + 0.10 * math.sin(2 * math.pi * date.timetuple().tm_yday / 365.0)
    month_end = 1.25 if date.day >= 28 else 1.0
    noise = rng.normal(1.0, 0.06)
    return max(0.0, trend * weekly * yearly * month_end * noise)


def generate() -> dict[str, list[dict]]:
    """Return {entity_name: [row dicts]} mimicking an OData pull."""
    rng = _rng()
    dates = _date_range(settings.synth_months)
    n_days = len(dates)
    anomaly_start = dates[-1] - dt.timedelta(days=18)  # injected stall window

    visits: list[dict] = []
    collections: list[dict] = []
    aging: list[dict] = []
    settlements: list[dict] = []
    lops: list[dict] = []

    vid = cid = sid = lid = 0

    for facility in FACILITIES:
        base, slope = _FACILITY_BASE[facility]
        for day_idx, date in enumerate(dates):
            daily_total = _daily_collection_signal(base, slope, day_idx, date, rng)
            # Inject a collections stall (~ -35%) at one facility in recent weeks.
            if facility == "round_rock" and date >= anomaly_start:
                daily_total *= 0.65
            # Split daily total into a handful of collection events.
            n_events = max(1, int(rng.poisson(4)))
            shares = rng.dirichlet(np.ones(n_events))
            for k in range(n_events):
                cid += 1
                collections.append({
                    "collection_id": f"c{cid}",
                    "facility_id": facility,
                    "attorney_id": str(rng.choice(ATTORNEYS)),
                    "case_type": str(rng.choice(CASE_TYPES, p=[0.6, 0.25, 0.15])),
                    "collection_date": date.isoformat(),
                    "amount_collected": round(float(daily_total * shares[k]), 2),
                    "days_outstanding": int(rng.integers(5, 300)),
                })
            # Visits: a few per facility per day.
            for _ in range(max(1, int(rng.poisson(5)))):
                vid += 1
                billed = round(float(rng.uniform(200, 3000)), 2)
                status = str(rng.choice(["paid", "unpaid", "pending"], p=[0.55, 0.30, 0.15]))
                paid = round(billed * (1.0 if status == "paid" else 0.0), 2)
                visits.append({
                    "visit_id": f"v{vid}", "facility_id": facility,
                    "case_type": str(rng.choice(CASE_TYPES)),
                    "visit_date": date.isoformat(), "billing_status": status,
                    "billed_amount": billed, "paid_amount": paid,
                    "provider_id": str(rng.choice(PROVIDERS)),
                    "patient_name": _fake_name(rng), "patient_dob": _fake_dob(rng),
                    "patient_ssn_last4": f"{int(rng.integers(0,9999)):04d}",
                    "patient_address": f"{int(rng.integers(100,9999))} Main St",
                })

    # Attorney aging: weekly snapshots per attorney+facility.
    for date in dates[::7]:
        for attorney in ATTORNEYS:
            for facility in FACILITIES:
                b = rng.uniform(5000, 40000, size=5) * np.array([1.0, 0.8, 0.6, 0.5, 0.7])
                aging.append({
                    "attorney_id": attorney, "facility_id": facility,
                    "bucket_0_30": round(float(b[0]), 2), "bucket_31_60": round(float(b[1]), 2),
                    "bucket_61_90": round(float(b[2]), 2), "bucket_91_180": round(float(b[3]), 2),
                    "bucket_180_plus": round(float(b[4]), 2), "report_date": date.isoformat(),
                })

    # Settlements: opened across the window, ~70% closed.
    for _ in range(900):
        sid += 1
        open_idx = int(rng.integers(0, n_days))
        open_d = dates[open_idx]
        closed = rng.random() < 0.7
        close_d = None
        status = "open"
        if closed:
            dur = int(rng.integers(20, 200))
            ci = min(open_idx + dur, n_days - 1)
            close_d = dates[ci].isoformat()
            status = "closed"
        settlements.append({
            "settlement_id": f"s{sid}", "attorney_id": str(rng.choice(ATTORNEYS)),
            "case_type": str(rng.choice(CASE_TYPES)), "open_date": open_d.isoformat(),
            "close_date": close_d, "settlement_amount": round(float(rng.uniform(2000, 90000)), 2),
            "settlement_status": status,
            "plaintiff_name": _fake_name(rng), "plaintiff_dob": _fake_dob(rng),
        })

    # LOPs.
    for _ in range(1200):
        lid += 1
        issue_idx = int(rng.integers(0, n_days))
        issued = dates[issue_idx]
        resolved = rng.random() < 0.75
        ret_d = None
        status = "open"
        reason = None
        if resolved:
            dur = int(rng.integers(5, 120))
            ri = min(issue_idx + dur, n_days - 1)
            ret_d = dates[ri].isoformat()
            if rng.random() < 0.12:
                status, reason = "rejected", str(rng.choice(["incomplete", "out_of_network", "duplicate"]))
            else:
                status = "returned"
        lops.append({
            "lop_id": f"l{lid}", "facility_id": str(rng.choice(FACILITIES)),
            "case_type": str(rng.choice(CASE_TYPES)), "issued_date": issued.isoformat(),
            "returned_date": ret_d, "status": status, "rejection_reason": reason,
            "patient_name": _fake_name(rng), "patient_dob": _fake_dob(rng),
        })

    return {
        "visits": visits, "collections": collections, "attorney_aging": aging,
        "settlements": settlements, "lop": lops,
    }
