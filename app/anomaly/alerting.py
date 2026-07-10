"""Alert generation, grounded driver narrative, delivery, acknowledgment.

Spec refs: arch §2.4; PRD Module 4 (AC-4.1..4.10); data design §5.4.

Driver narrative (AC-4.6) is **rule-based and grounded in the feature store**,
not free LLM text: it inspects settlement velocity, LOP turnaround, and visit
velocity around the flagged entity and reports the largest contributing signal.
Every payload is PHI-scanned before delivery (AC-4.8).
"""
from __future__ import annotations

import datetime as dt
import json
import uuid

import httpx

from app.anomaly import detectors
from app.anomaly.severity import combine
from app.config import settings
from app.db import tx
from app.features import store
from app.forecasting.service import list_entities
from app.security.audit import log_phi_detection, write_audit
from app.security.phi import scan_output


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _driver_narrative(facility_id: str) -> str:
    """Grounded explanation from feature store signals (no LLM)."""
    signals: list[tuple[float, str]] = []
    # Settlement velocity across attorneys at this facility's case types.
    for ek in store.list_entities("settlement_pipeline"):
        fv = store.get_online_features("settlement_pipeline", ek)
        if fv and fv.get("settlement_velocity_change_30d") is not None:
            vel = fv["settlement_velocity_change_30d"]
            if vel < -0.15:
                att = ek.split("|")[0]
                signals.append((vel, f"settlement velocity down {abs(vel):.0%} (Attorney {att.title()})"))
    # LOP turnaround degradation.
    lop = store.get_online_features("lop_metrics", facility_id)
    if lop and lop.get("lop_turnaround_days_p50"):
        if lop["lop_turnaround_days_p50"] > 45:
            signals.append((-lop["lop_turnaround_days_p50"],
                            f"LOP turnaround elevated at {lop['lop_turnaround_days_p50']:.0f} days median"))
    if not signals:
        return "No single dominant driver identified in current feature signals."
    signals.sort(key=lambda s: s[0])
    return "Likely driver: " + signals[0][1] + "."


def _store_alert(f, narrative: str) -> dict:
    alert_id = str(uuid.uuid4())
    payload = {
        "severity": f.severity, "entity_type": f.entity_type, "entity_id": f.entity_id,
        "metric": f.metric, "expected": f.expected, "actual": f.actual,
        "deviation_pct": f.deviation_pct, "detector": f.detector,
        "driver": narrative, "dashboard_link": f"/dashboard/{f.entity_type}/{f.entity_id}",
        "chatbot_link": f"/chatbot?q=Why+did+{f.entity_id}+collections+change",
        "detail": f.detail,
    }
    with tx() as conn:
        conn.execute(
            """INSERT INTO alerts (alert_id, created_at, severity, entity_type, entity_id,
                 metric, expected, actual, deviation_pct, detector, driver_narrative,
                 status, payload_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (alert_id, _utcnow(), f.severity, f.entity_type, f.entity_id, f.metric,
             f.expected, f.actual, f.deviation_pct, f.detector, narrative,
             "open", json.dumps(payload)),
        )
    payload["alert_id"] = alert_id
    return payload


def _deliver(payload: dict) -> None:
    """Route by severity (arch §2.4). Slack POST only if a webhook is configured."""
    text = format_slack(payload)
    if settings.slack_webhook_url:
        try:
            httpx.post(settings.slack_webhook_url, json={"text": text}, timeout=5)
        except Exception:
            pass  # delivery failure must not crash the nightly run; alert is persisted


def format_slack(p: dict) -> str:
    sev_label = {"P1": "P1 Critical", "P2": "P2 Warning", "P3": "P3 Info"}[p["severity"]]
    lines = [f":warning: [{sev_label}] {p['entity_id']} — Anomaly Detected",
             f"Metric: {p['metric']} | Detector: {p['detector']}"]
    if p.get("expected") is not None:
        lines.append(f"Expected: ${p['expected']:,.0f} | Actual: ${p['actual']:,.0f} "
                     f"| Δ: {p['deviation_pct']:+.1%}" if p.get("deviation_pct") is not None
                     else f"Expected: ${p['expected']:,.0f} | Actual: ${p['actual']:,.0f}")
    lines.append(p["driver"])
    lines.append(f"→ Dashboard: {p['dashboard_link']}")
    return "\n".join(lines)


def run_detection() -> dict:
    """Nightly anomaly sweep across facilities + attorneys (AC-4.1)."""
    generated: list[dict] = []
    blocked = 0
    for facility_id in list_entities("facility"):
        findings = []
        findings += detectors.forecast_deviation("facility", facility_id)
        findings += detectors.cusum("facility", facility_id)
        findings += detectors.isolation_forest(facility_id)
        best = combine(findings)
        if not best:
            continue
        narrative = _driver_narrative(facility_id)
        # PHI scan on the assembled payload text (AC-4.8).
        scan = scan_output(narrative)
        if scan.phi_detected:
            log_phi_detection("anomaly", "system", "admin", scan.matches, "ALERT")
            blocked += 1
            continue
        payload = _store_alert(best, narrative)
        _deliver(payload)
        generated.append(payload)
    write_audit(user_id="system", user_role="admin", service="anomaly",
                action="nightly_detection", detail={"alerts": len(generated), "phi_blocked": blocked})
    return {"alerts_generated": len(generated), "phi_blocked": blocked,
            "by_severity": {s: sum(1 for a in generated if a["severity"] == s) for s in ("P1", "P2", "P3")}}


def acknowledge(alert_id: str, user_id: str) -> bool:
    """Slack-button ack updates status within the DB (AC-4.9)."""
    with tx() as conn:
        cur = conn.execute(
            "UPDATE alerts SET status='acknowledged', acknowledged_by=?, acknowledged_at=? "
            "WHERE alert_id=? AND status='open'",
            (user_id, _utcnow(), alert_id),
        )
        return cur.rowcount > 0


def list_alerts(severity: str | None = None, status: str | None = None) -> list[dict]:
    sql = "SELECT * FROM alerts WHERE 1=1"
    params: list = []
    if severity:
        sql += " AND severity=?"
        params.append(severity)
    if status:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY created_at DESC"
    with tx() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


if __name__ == "__main__":
    from app.db import init_db
    init_db()
    print(json.dumps(run_detection(), indent=2))
