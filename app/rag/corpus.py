"""Build the RAG document corpus from de-identified, aggregated sources only.

Spec refs: arch §2.5 corpus; data design §6.1. NOTHING patient-level is indexed
— only feature-store aggregates, forecasts, and alert history.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.anomaly.alerting import list_alerts
from app.features import store
from app.forecasting.service import get_forecast, list_entities


@dataclass
class Document:
    source_doc_id: str
    entity_type: str
    entity_id: str
    date: str
    text: str
    metadata: dict = field(default_factory=dict)


def build_corpus() -> list[Document]:
    docs: list[Document] = []

    # Facility collections summaries (latest feature snapshot).
    for fac in store.list_entities("facility_collections"):
        fv = store.get_online_features("facility_collections", fac)
        series = store.get_series("facility_collections", fac)
        if not fv or series.empty:
            continue
        last_date = series["event_date"].max().date().isoformat()
        docs.append(Document(
            f"facility_collections_{fac}_{last_date}", "facility", fac, last_date,
            f"Facility {fac.replace('_',' ').title()} collections summary as of {last_date}: "
            f"latest day ${fv['collections_1d']:,.0f}, 7-day rolling ${fv['collections_7d_rolling']:,.0f}, "
            f"30-day rolling ${fv['collections_30d_rolling']:,.0f}. "
            f"Month-over-month growth "
            f"{(fv.get('collections_mom_growth') or 0):.1%}.",
            {"feature_group": "facility_collections"},
        ))

    # Attorney aging summaries.
    for att in store.list_entities("attorney_aging"):
        fv = store.get_online_features("attorney_aging", att)
        series = store.get_series("attorney_aging", att)
        if not fv or series.empty:
            continue
        last_date = series["event_date"].max().date().isoformat()
        docs.append(Document(
            f"attorney_aging_{att}_{last_date}", "attorney", att, last_date,
            f"Attorney {att.title()} aging report as of {last_date}: "
            f"total outstanding ${fv['total_outstanding']:,.0f}; "
            f"180+ day bucket ${fv['bucket_180_plus_balance']:,.0f} "
            f"({fv['pct_180_plus']:.1%} of total); "
            f"average days outstanding {fv['avg_days_outstanding']:.0f}.",
            {"feature_group": "attorney_aging"},
        ))

    # Forecast summaries.
    for et in ("facility", "attorney", "case_type"):
        for eid in list_entities(et):
            fc = get_forecast(et, eid, 30)
            if not fc:
                continue
            docs.append(Document(
                f"forecast_{et}_{eid}_{fc['forecast_date']}", et, eid, fc["forecast_date"],
                f"30-day collections forecast for {et} {eid} generated {fc['forecast_date']}: "
                f"predicted ${fc['predicted']:,.0f} (80% CI ${fc['ci_lower']:,.0f}–${fc['ci_upper']:,.0f}).",
                {"feature_group": "forecast", "horizon": 30},
            ))

    # Anomaly alert history.
    for a in list_alerts():
        docs.append(Document(
            f"alert_{a['entity_id']}_{a['created_at'][:10]}", a["entity_type"], a["entity_id"],
            a["created_at"][:10],
            f"Anomaly alert [{a['severity']}] for {a['entity_id']} on {a['created_at'][:10]}: "
            f"metric {a['metric']}, expected ${a['expected'] or 0:,.0f}, actual ${a['actual'] or 0:,.0f}. "
            f"{a['driver_narrative']}",
            {"feature_group": "alert", "severity": a["severity"]},
        ))

    return docs
