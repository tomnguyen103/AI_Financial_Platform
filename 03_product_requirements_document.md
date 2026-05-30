# Product Requirements Document (PRD)
## AI-Powered Financial Reporting & Revenue Intelligence Platform

**Document Version:** 1.0  
**Date:** 2026-05-28  
**Status:** Draft

---

## 1. Document Purpose

This PRD defines the functional requirements, user stories, and acceptance criteria for each module of the AI Financial Reporting platform. It is the authoritative reference for what must be built, and is the basis for QA sign-off and stakeholder acceptance.

---

## 2. User Personas

### Persona 1: Collections Specialist (Casey)
- Manages day-to-day collections for 3–5 facilities.
- Not technically proficient — does not write SQL or use Tableau directly.
- Needs to know when something is wrong *before* the end-of-month review.
- Primary tools today: email, spreadsheets, phone calls to attorneys.
- **Key need:** Alerts that tell her *what* is wrong and *why*, so she can take action immediately.

### Persona 2: Finance Lead / Operations Director (Morgan)
- Reviews monthly financial performance across all facilities.
- Comfortable with Tableau; not a data engineer.
- Needs forward-looking visibility to plan staffing and attorney outreach.
- **Key need:** 90-day forecast with confidence intervals to support budget planning.

### Persona 3: DA Analyst (Alex)
- Builds and maintains the OData pipeline and Tableau dashboards.
- Technically proficient; writes Python and SQL.
- Needs to trust the data quality of the AI platform's outputs.
- **Key need:** Observability into pipeline health, model drift, and feature freshness.

### Persona 4: Non-Technical Staff (Jordan)
- Works in billing or case coordination; needs quick data lookups.
- Cannot write SQL or navigate Tableau.
- **Key need:** Ask a plain-English question and get a reliable, specific answer.

---

## 3. Module 1: Data Ingestion Pipeline

### 3.1 Overview
Nightly extraction of the OData financial feed into a structured, validated, PHI-masked data store.

### 3.2 User Stories

**US-1.1** — As a DA Analyst, I want the ingestion job to run automatically every night so that the platform always reflects the latest data without manual intervention.

**US-1.2** — As a DA Analyst, I want schema validation to run after every ingest so that upstream OData changes are caught immediately and don't silently corrupt downstream features.

**US-1.3** — As a Compliance Officer, I want PHI fields (patient name, DOB, SSN) to be masked before any data is written to the feature store or accessible by the AI layer, so that patient privacy is maintained at all times.

**US-1.4** — As a DA Analyst, I want to receive an alert if the nightly job fails or produces anomalous output (e.g., record count drops > 20%) so I can investigate and rerun before the business day begins.

**US-1.5** — As a DA Analyst, I want to view the ingestion audit log (timestamps, record counts, validation results, PHI masking confirmations) so I can troubleshoot issues and demonstrate compliance.

### 3.3 Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-1.1 | Ingest job is triggered automatically at 02:00 local time, 7 days a week. |
| AC-1.2 | Job completes and feature store is updated by 06:00 local time (4-hour SLA). |
| AC-1.3 | If OData schema changes (new field, renamed field, dropped field), the job fails gracefully, writes an alert, and does not overwrite the prior day's curated data. |
| AC-1.4 | All fields defined as PHI in the data dictionary are tokenized before the curated zone is written. Raw zone retains original data with elevated access controls. |
| AC-1.5 | Ingest failure triggers a Slack/Teams alert to the DA Analyst within 15 minutes of failure. |
| AC-1.6 | Data quality rules (non-null rates, value ranges, referential integrity) are enforced; violations produce a detailed report viewable in the monitoring dashboard. |
| AC-1.7 | Audit log captures: ingest timestamp, source record count, records passed validation, records failed validation, PHI masking confirmation (boolean), operator (system). Retained for 7 years. |

---

## 4. Module 2: Feature Store

### 4.1 Overview
Centralized, versioned store of pre-computed financial features used by forecasting and anomaly detection models.

### 4.2 User Stories

**US-2.1** — As an ML Engineer, I want to retrieve point-in-time-correct features for model training so that there is no label leakage and my models are trained on data that was actually available at the prediction time.

**US-2.2** — As an ML Engineer, I want to serve features to the online forecasting API with < 50ms latency so that API response times remain within SLA.

**US-2.3** — As a DA Analyst, I want to see the last-updated timestamp for every feature group so I can confirm data freshness before relying on forecasts.

**US-2.4** — As an ML Engineer, I want feature definitions version-controlled in Git so that any change to a feature is auditable and reversible.

### 4.3 Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-2.1 | All five feature groups (facility_collections, attorney_aging, visit_velocity, settlement_pipeline, lop_metrics) are populated nightly from the curated data zone. |
| AC-2.2 | Online feature serving (Redis) returns feature vectors in < 50ms at p95. |
| AC-2.3 | Historical feature retrieval uses point-in-time-correct joins (no future data leakage). |
| AC-2.4 | Feature freshness monitoring alerts DA Analyst if any feature group is not updated by 07:00. |
| AC-2.5 | Feature definitions are stored as versioned YAML/Python files in the project Git repo. Changes require PR review. |

---

## 5. Module 3: Forecasting Service

### 5.1 Overview
Produces 30/60/90-day collection forecasts per facility, attorney-provider, and case type, surfaced via API and dashboard.

### 5.2 User Stories

**US-3.1** — As a Finance Lead, I want to see a 90-day collections forecast with confidence intervals for each facility so I can plan budget allocations and flag potential shortfalls in advance.

**US-3.2** — As a Finance Lead, I want to drill into the forecast by attorney provider and case type (PI vs. Commercial) so I can understand which practice area is driving variance.

**US-3.3** — As a Collections Specialist, I want a simple "forecast vs. actual" view in the dashboard so I can see at a glance whether collections are tracking above or below expectation.

**US-3.4** — As a DA Analyst, I want the model to retrain weekly on the latest data so that forecasts reflect recent trends, not stale patterns.

**US-3.5** — As a DA Analyst, I want a champion/challenger model registry so I can test new model versions against the current production model before promoting them.

**US-3.6** — As a Finance Lead, I want to be notified when a forecast is significantly revised (> 10% change in the 30-day outlook) so I am not blindsided by model updates.

### 5.3 Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-3.1 | Forecasts are available for all active facilities, all attorney providers with ≥ 30 days of history, and both case types (PI, Commercial/Athena). |
| AC-3.2 | Forecasts include 50th, 80th, and 95th percentile confidence bands. |
| AC-3.3 | Forecast API responds in < 2 seconds at p95 for single-entity requests. |
| AC-3.4 | Model is retrained weekly; MAPE on the held-out validation window is logged to MLflow on every retrain. |
| AC-3.5 | If the new model's validation MAPE is worse than the current champion by > 3pp, promotion is blocked and an alert is sent to the DA Analyst. |
| AC-3.6 | Production model rollback can be executed via a single CLI command or API call. |
| AC-3.7 | Forecast outputs are persisted to the `forecasts` table with model version, generation timestamp, and entity identifiers, enabling historical comparison. |
| AC-3.8 | Collections Forecast Dashboard is available in Tableau (or embedded alternative) showing: 30/60/90-day forecast bands, actual-vs-forecast overlay, and per-facility/attorney/case-type drill-down. |

---

## 6. Module 4: Anomaly Detection & Alerting

### 6.1 Overview
Monitors financial metrics nightly and delivers actionable alerts to Collections staff when anomalies are detected.

### 6.2 User Stories

**US-4.1** — As a Collections Specialist, I want to receive a Slack alert when a facility's collections are tracking significantly below forecast so I can investigate and take corrective action the same day.

**US-4.2** — As a Collections Specialist, I want the alert to include a plain-English description of the likely driver (e.g., "settlement velocity down 22% for Attorney X") so I don't have to dig through dashboards to understand what happened.

**US-4.3** — As a Finance Lead, I want to receive a daily digest of all P2 and P3 anomalies (lower priority than real-time P1 alerts) so I have a full picture each morning without being paged for minor variations.

**US-4.4** — As a DA Analyst, I want to configure alert thresholds per facility (e.g., a new facility with volatile early data has wider thresholds) so we minimize false positives.

**US-4.5** — As a Collections Specialist, I want to acknowledge or dismiss an alert from within Slack so the system knows I've seen it and can track response rates.

**US-4.6** — As a Compliance Officer, I want all alert content to be free of individual patient data so that alerts sent to Slack do not create PHI exposure.

### 6.3 Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-4.1 | Anomaly detection runs nightly after feature store refresh; first alerts delivered by 07:30 local time. |
| AC-4.2 | P1 alerts (critical, > 20% below forecast or sudden visit volume collapse) are delivered to Collections Lead via Slack DM within 15 minutes of detection. |
| AC-4.3 | P2 alerts are posted to a dedicated Slack channel (#collections-alerts). |
| AC-4.4 | P3 alerts are included in the daily digest email only. |
| AC-4.5 | Alert payload includes: entity (facility/attorney), metric, expected value, actual value, % deviation, likely driver narrative, and a deep link to the relevant dashboard view. |
| AC-4.6 | Likely driver narrative is generated using the forecasting service context and feature importance, not raw LLM hallucination. |
| AC-4.7 | Thresholds are configurable per entity in a YAML config file without code changes. |
| AC-4.8 | Alert payloads pass through the PHI scanner before delivery; any PHI detection blocks the alert and triggers a compliance log entry. |
| AC-4.9 | Alert acknowledgment (Slack button click) updates the alert status in the database within 5 seconds. |
| AC-4.10 | False positive rate is measured monthly; target < 15% of P1 alerts are dismissed as false positives by users. |

---

## 7. Module 5: RAG Chatbot ("Ask the Financials")

### 7.1 Overview
A conversational interface allowing any staff member to ask natural-language questions about financial performance and receive AI-synthesized answers with source citations.

### 7.2 User Stories

**US-5.1** — As a Collections Specialist, I want to type "Why did Round Rock collections drop last week?" and receive a specific, cited explanation so I can understand the root cause without needing a DA analyst.

**US-5.2** — As a Finance Lead, I want to ask "Which attorney has the highest balance in the 180+ day aging bucket?" and get an immediate, accurate answer so I can prioritize follow-up calls.

**US-5.3** — As any user, I want the chatbot to cite the source data (e.g., "Source: Attorney Aging Report, 2026-05-21, Row 47") for every factual claim so I can verify the answer if needed.

**US-5.4** — As any user, I want the chatbot to tell me when it doesn't have enough information to answer confidently, rather than making something up, so I can trust its outputs.

**US-5.5** — As a Collections Specialist, I want the chatbot to be accessible from the existing dashboard or a simple web UI so I don't need to learn a new tool.

**US-5.6** — As a Compliance Officer, I want the chatbot to refuse any request that would surface individual patient data (names, DOBs, claim IDs) and log the attempt, so that patient privacy is always protected.

### 7.3 Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-5.1 | Chatbot responds to financial queries within 8 seconds end-to-end (p95). |
| AC-5.2 | Every factual claim in the chatbot's response is accompanied by at least one citation (source document ID, date, and relevant row or section). |
| AC-5.3 | If the RAG retrieval step finds no relevant documents with similarity score > 0.75, the chatbot responds with "I don't have enough information to answer this reliably" rather than fabricating an answer. |
| AC-5.4 | Weekly human evaluation of a random 20-question sample scores ≥ 85% factually correct (grounded in source data). |
| AC-5.5 | Any query that appears to request patient-identifying information (detected via regex + NER classifier) is rejected with a compliance message and logged. |
| AC-5.6 | Chatbot output is scanned for PHI before delivery; detection triggers redaction + compliance log. |
| AC-5.7 | The corpus (Tableau extracts, attorney aging summaries) is re-indexed nightly after the feature store refresh. |
| AC-5.8 | Chatbot is accessible via a web UI embeddable in Tableau or Mendix (iframe-compatible, JWT-authenticated). |
| AC-5.9 | All chatbot sessions are logged (user, timestamp, query, response, retrieved sources, latency) for audit and quality improvement. |

---

## 8. Module 6: NL-to-SQL Interface

### 8.1 Overview
A natural-language query interface over the Azure financial database that allows non-technical staff to retrieve specific data without writing SQL.

### 8.2 User Stories

**US-6.1** — As a Billing Coordinator, I want to type "show me all unpaid visits over 180 days for Attorney Johnson" and see a results table so I can compile a follow-up list without calling the DA team.

**US-6.2** — As a Collections Specialist, I want to export NL-to-SQL query results to CSV so I can use them in my workflow tools.

**US-6.3** — As a DA Analyst, I want to review a log of all NL-to-SQL queries executed (including the generated SQL) so I can audit data access and identify commonly-asked questions that should become named reports.

**US-6.4** — As a DA Analyst, I want to maintain a business glossary (mapping plain-English terms to database columns) that improves SQL generation accuracy so the system improves over time as new terms are added.

**US-6.5** — As a Compliance Officer, I want the NL-to-SQL layer to execute only SELECT statements on a read-only database account so there is zero risk of data modification via this interface.

### 8.3 Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-6.1 | 90% of NL-to-SQL queries on the defined test set produce syntactically valid SQL. |
| AC-6.2 | 80% of NL-to-SQL queries on the defined test set produce semantically correct results (evaluated by DA Analyst on a 50-query benchmark). |
| AC-6.3 | Generated SQL is parsed and validated before execution; any DDL or DML statement (INSERT, UPDATE, DELETE, DROP, etc.) is blocked and logged. |
| AC-6.4 | Query execution uses a read-only database role with SELECT grants only. |
| AC-6.5 | Query execution timeout is 30 seconds; results are capped at 10,000 rows. |
| AC-6.6 | Results are displayed in a formatted table in the UI with an option to export as CSV. |
| AC-6.7 | All executed queries (original NL input, generated SQL, result row count, execution time, user, timestamp) are stored in the audit log. |
| AC-6.8 | The business glossary is editable by DA Analysts via a simple YAML file (no UI required in Phase 1). |
| AC-6.9 | When SQL generation fails or produces invalid SQL after retry, the user receives a helpful error message (e.g., "I couldn't translate that query — try rephrasing or contact the DA team"). |

---

## 9. Cross-Cutting Requirements

### 9.1 Security & Compliance

| Requirement | Detail |
|---|---|
| Authentication | All API endpoints require a valid JWT issued by Azure AD / Entra ID. |
| Authorization | RBAC enforced at the API gateway; roles: Collections, Finance, DA Analyst, Admin. |
| PHI protection | PHI masked at ingest; no patient-identifiable data in chatbot corpus, alerts, or NL-to-SQL results. |
| Audit logging | All user actions (queries, logins, data exports) logged with user identity, timestamp, and action detail. Retained 7 years. |
| Data encryption | All data at rest encrypted (AES-256); all data in transit via TLS 1.3. |

### 9.2 Performance

| Requirement | Target |
|---|---|
| Forecast API latency | < 2 seconds p95 |
| Chatbot end-to-end latency | < 8 seconds p95 |
| NL-to-SQL response time | < 15 seconds p95 (including query execution) |
| Anomaly detection + alerting | Alerts delivered by 07:30 after nightly ingest |
| System uptime | 99.5% monthly |

### 9.3 Usability

- All user-facing interfaces (dashboard, chatbot, NL-to-SQL) require no training beyond a 30-minute onboarding session.
- Error messages are written in plain English, not technical jargon.
- The chatbot and NL-to-SQL tool include 5 example prompts to guide first-time users.

### 9.4 Observability & Monitoring

- All services emit structured JSON logs to Azure Monitor.
- A monitoring dashboard (Grafana or Azure Monitor Workbook) shows: pipeline health, feature freshness, API latency, model MAPE, alert volumes, and chatbot quality scores.
- Alerting on: pipeline failure, API error rate > 1%, model MAPE regression, PHI scanner trigger.

---

## 10. Out-of-Scope Requirements

The following are explicitly not required in Phase 1:

- Write operations to the Azure financial database via any AI interface.
- Individual patient-level data exposed through any user-facing interface.
- Streaming/real-time data ingestion (nightly batch only).
- Mobile application or native desktop client.
- Integration with EHR or clinical systems beyond existing OData.
- Automated actions triggered by anomaly alerts (e.g., auto-drafting attorney emails) — notification only.
