# Project Overview & Scope
## AI-Powered Financial Reporting & Revenue Intelligence Platform

**Document Version:** 1.0  
**Date:** 2026-05-28  
**Status:** Draft

---

## 1. Executive Summary

The AI-Powered Financial Reporting & Revenue Intelligence Platform replaces today's manual, retrospective financial reporting workflow with a predictive, conversational intelligence system. Where the Data Analytics (DA) team currently pulls OData feeds day-by-day into Azure and surfaces results through Tableau—a process that is slow, reactive, and opaque to non-technical staff—this platform introduces forecasting, anomaly detection, natural-language querying, and AI-synthesized explanations on top of the same underlying data.

The end state: a Collections team member who wakes up to a Slack alert that says "Round Rock collections are tracking 18% below forecast — driven by a 3-week stall in Attorney Johnson's PI case settlements," supported by cited source rows, without anyone having run a single manual query.

---

## 2. Problem Statement

### Current State Pain Points

| Pain Point | Impact |
|---|---|
| Manual OData → Azure → Tableau pipeline requires daily DA analyst time | Slow time-to-insight; analyst bandwidth bottleneck |
| Reporting is purely retrospective | Issues are discovered after they compound, not as they emerge |
| No early-warning system for collection shortfalls | Finance and ops leadership cannot course-correct in time |
| Non-technical staff cannot self-serve data queries | Dependency on DA team for every ad-hoc question |
| No explanation layer — dashboards show *what*, not *why* | Root-cause analysis is manual and time-consuming |
| Forecasting is done via spreadsheets or intuition | No quantifiable confidence intervals or scenario modeling |

### Opportunity

The organization already has structured, high-quality financial and clinical-billing data (visits, settlements, LOPs, attorney aging buckets, facility-level collections) flowing through Azure. The gap is intelligence layered on top of that data — not more data collection.

---

## 3. Goals & Objectives

### Primary Goals

1. **Predictive revenue visibility** — Provide 30/60/90-day collection forecasts at the facility, attorney-provider, and case-type level (Personal Injury vs. Commercial/Athena).
2. **Proactive anomaly alerting** — Detect and surface unusual drops in billable visits, settlement velocity, or LOP turnaround time before they become material shortfalls.
3. **Conversational financial querying** — Allow any staff member to ask natural-language questions about financial data and receive cited, AI-synthesized answers.
4. **Self-service SQL access** — Enable non-technical users to query the Azure database with plain English, eliminating the DA team bottleneck for routine lookups.

### Secondary Goals

- Reduce time-to-insight from "next business day" to near-real-time.
- Establish MLOps foundations (model versioning, drift detection, A/B testing) to support future AI initiatives.
- Build a reusable feature store that can underpin future ML use cases (e.g., case outcome prediction, LOP risk scoring).

---

## 4. Stakeholders

| Role | Name / Team | Involvement |
|---|---|---|
| **Product Owner** | DA Team Lead | Requirements, acceptance testing, rollout decisions |
| **Primary Users** | Collections Team | Daily consumers of alerts, forecasts, and chatbot |
| **Secondary Users** | Finance / Ops Leadership | Executive dashboard, forecast review |
| **Data Provider** | DA Team (OData / Azure) | Data pipeline ownership, schema expertise |
| **Technical Lead** | Platform Engineer (TBD) | Architecture, backend, MLOps |
| **Compliance** | Legal / Privacy Officer | PHI guardrails, audit logging sign-off |
| **Integration Owner** | Mendix / Tableau Admin | Embedding and surface integration |

---

## 5. Success Metrics

### Business KPIs

| Metric | Baseline | Target (6 months post-launch) |
|---|---|---|
| Forecast accuracy (MAPE) for 30-day collections | N/A (no forecasts today) | ≤ 12% MAPE per facility |
| Time from anomaly onset to alert delivery | ~1–2 business days (manual) | < 4 hours |
| % of ad-hoc data queries self-served without DA involvement | ~0% | ≥ 60% |
| Chatbot answer accuracy (human-evaluated sample) | N/A | ≥ 85% factually correct with citations |
| Collections team alert-to-action rate | N/A | ≥ 50% of alerts trigger a documented follow-up |

### Technical KPIs

| Metric | Target |
|---|---|
| OData ingest pipeline SLA | Nightly completion by 06:00 local time |
| Forecast API p95 latency | < 2 seconds |
| Chatbot query response time | < 8 seconds end-to-end |
| NL-to-SQL query accuracy (test set) | ≥ 90% syntactically valid, ≥ 80% semantically correct |
| Model drift detection lag | ≤ 7 days from distribution shift to alert |
| System uptime | 99.5% monthly |

---

## 6. Scope

### In Scope

- **Data Ingestion Layer:** Nightly OData → Azure → Feature Store pipeline (FastAPI + Feast/Postgres or Azure ML Feature Store).
- **Forecasting Service:** 30/60/90-day collection forecasts per facility, attorney-provider, and case type using Prophet, XGBoost, or Temporal Fusion Transformer.
- **Collections Forecast Dashboard:** Read-only Tableau or embedded frontend surface for forecast visualization.
- **Anomaly Detection Layer:** Isolation Forest + statistical control charts; Slack/Teams alerting integration.
- **RAG Chatbot ("Ask the Financials"):** LangChain/LlamaIndex pipeline over Tableau extracts and visit-level data, powered by GPT-4 or Claude, with source-row citations.
- **NL-to-SQL Interface:** Natural-language query layer over the Azure database for non-technical users.
- **MLOps Infrastructure:** Model versioning, A/B testing framework, forecast rollback, drift monitoring.
- **PHI/Privacy Guardrails:** Role-based access controls, audit logging, data masking for PII/PHI in chatbot outputs.

### Out of Scope (Phase 1)

- Case outcome prediction or LOP risk scoring (future phase).
- Integration with Mendix application logic (Mendix embedding is optional and deferred).
- Real-time streaming ingestion (nightly batch is sufficient for Phase 1).
- Mobile application.
- Modifications to existing Tableau dashboards (new surfaces are additive).
- EHR / clinical system integrations beyond the existing OData feed.

---

## 7. Assumptions & Dependencies

### Assumptions

- The existing OData feed provides reliable, schema-stable financial data including visits, collections, settlements, LOPs, and attorney aging buckets.
- Azure infrastructure (existing) is available for feature store hosting without significant new procurement.
- A GPT-4 or Claude API key is available and approved for use with de-identified/masked financial data.
- The DA team will provide subject-matter expertise on data semantics during the design and test phases.
- PHI fields are identifiable in the schema and can be masked before leaving the perimeter.

### Dependencies

| Dependency | Owner | Risk if Delayed |
|---|---|---|
| OData schema documentation | DA Team | Delays feature store design |
| Azure environment access | Infra/Cloud Team | Blocks all backend work |
| LLM API procurement / approval | Legal + IT | Blocks chatbot and NL-to-SQL modules |
| Slack/Teams webhook credentials | IT | Blocks alerting module |
| Tableau extract access for RAG indexing | DA Team | Limits chatbot context quality |

---

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OData schema changes break ingestion | Medium | High | Schema versioning + contract tests in pipeline |
| LLM hallucinations in chatbot | Medium | High | Grounding via RAG + citation enforcement + human eval baseline |
| PHI/PII leakage through chatbot | Low | Critical | Field-level masking, output scanning, access controls, audit log |
| Forecast accuracy insufficient for stakeholder trust | Medium | Medium | Start with transparent confidence intervals; iterate on models |
| DA team bandwidth for SME support | Medium | Medium | Time-box design sessions; document data semantics early |
| Azure cost overrun from feature store storage | Low | Medium | Set budget alerts; archive raw features after 12 months |

---

## 9. High-Level Timeline

| Phase | Duration | Key Deliverables |
|---|---|---|
| **Phase 0 — Discovery & Design** | Weeks 1–3 | Data audit, schema documentation, architecture sign-off, PHI review |
| **Phase 1 — Data Foundation** | Weeks 4–8 | OData ingestion pipeline, feature store, nightly job, data quality tests |
| **Phase 2 — Forecasting & Anomaly** | Weeks 9–14 | Forecasting service, anomaly detection, Slack alerting, forecast dashboard |
| **Phase 3 — Conversational AI** | Weeks 15–20 | RAG chatbot, NL-to-SQL interface, user testing, citation accuracy eval |
| **Phase 4 — MLOps & Hardening** | Weeks 21–24 | Model versioning, A/B framework, drift monitoring, audit logging, load testing |
| **Phase 5 — Rollout** | Weeks 25–26 | Staged rollout (1 facility), feedback loop, full deployment |

---

## 10. Definition of Done

A feature is considered production-ready when:
1. Unit and integration tests pass (≥ 80% coverage on business logic).
2. Acceptance criteria in the PRD are verified by the DA Team Lead.
3. PHI/privacy review signed off by Legal/Compliance.
4. Monitoring dashboards and alerting are live for the feature.
5. Runbook is written and reviewed.
6. Performance targets (latency, accuracy) are met on staging data.
