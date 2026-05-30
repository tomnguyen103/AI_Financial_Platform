# Data & ML Design Document
## AI-Powered Financial Reporting & Revenue Intelligence Platform

**Document Version:** 1.0  
**Date:** 2026-05-28  
**Status:** Draft

---

## 1. Purpose

This document defines the data modeling decisions, feature engineering strategy, machine learning model selections, evaluation methodology, MLOps infrastructure, and PHI/privacy safeguards for the platform. It is the technical reference for ML engineers and data engineers building and maintaining the AI components.

---

## 2. Data Inventory & Sources

### 2.1 Source: OData Financial Feed

The primary data source. Fields expected from the OData feed (subject to DA team confirmation during Phase 0):

| Entity | Key Fields | Granularity |
|---|---|---|
| **Visits** | visit_id, facility_id, case_type, visit_date, billing_status, billed_amount, paid_amount, provider_id | Row per visit |
| **Collections** | collection_id, facility_id, attorney_id, case_type, collection_date, amount_collected, days_outstanding | Row per collection event |
| **Attorney Aging** | attorney_id, facility_id, bucket_0_30, bucket_31_60, bucket_61_90, bucket_91_180, bucket_180_plus, report_date | Row per attorney per report date |
| **Settlements** | settlement_id, attorney_id, case_type, open_date, close_date, settlement_amount, settlement_status | Row per settlement case |
| **LOPs (Letters of Protection)** | lop_id, facility_id, case_type, issued_date, returned_date, status, rejection_reason | Row per LOP |

### 2.2 PHI Fields (Mask at Ingest)

The following fields are classified as PHI and must be tokenized before the curated data zone:

| Entity | PHI Fields |
|---|---|
| Visits | patient_name, patient_dob, patient_ssn_last4, patient_address |
| Settlements | plaintiff_name, plaintiff_dob |
| LOPs | patient_name, patient_dob |

**Tokenization approach:** Replace PHI values with a deterministic HMAC-SHA256 token keyed with a secret stored in Azure Key Vault. The same patient across multiple records gets the same token (referential integrity preserved), but the token is not reversible without the key.

---

## 3. Feature Store Design

### 3.1 Feature Group: `facility_collections`

**Entity key:** `facility_id`  
**Event timestamp:** `date`  
**Computation frequency:** Nightly

| Feature Name | Type | Description | Computation |
|---|---|---|---|
| `collections_1d` | float | Total collections for the day | SUM(amount_collected) WHERE collection_date = T |
| `collections_7d_rolling` | float | 7-day rolling sum | SUM over T-6 to T |
| `collections_30d_rolling` | float | 30-day rolling sum | SUM over T-29 to T |
| `collections_mom_growth` | float | Month-over-month growth rate | (current_month - prior_month) / prior_month |
| `collections_yoy_growth` | float | Year-over-year growth rate | (current_30d - same_30d_last_year) / same_30d_last_year |
| `collections_vs_forecast_delta` | float | Actual vs. forecast deviation (%) | (actual - forecast) / forecast |
| `day_of_week` | int | 0=Monday, 6=Sunday | Derived from date |
| `month` | int | 1–12 | Derived from date |
| `is_month_end` | bool | True if last 3 days of month | Business calendar rule |

---

### 3.2 Feature Group: `attorney_aging`

**Entity key:** `attorney_id`  
**Event timestamp:** `report_date`  
**Computation frequency:** Nightly

| Feature Name | Type | Description |
|---|---|---|
| `bucket_0_30_balance` | float | Outstanding balance 0–30 days |
| `bucket_31_60_balance` | float | Outstanding balance 31–60 days |
| `bucket_61_90_balance` | float | Outstanding balance 61–90 days |
| `bucket_91_180_balance` | float | Outstanding balance 91–180 days |
| `bucket_180_plus_balance` | float | Outstanding balance 180+ days |
| `total_outstanding` | float | Sum of all aging buckets |
| `pct_180_plus` | float | % of total outstanding in 180+ bucket |
| `avg_days_outstanding` | float | Weighted average days outstanding |
| `aging_migration_rate_30d` | float | % of balance that moved into a higher bucket over 30 days |

---

### 3.3 Feature Group: `visit_velocity`

**Entity key:** `facility_id`, `case_type`  
**Event timestamp:** `date`  
**Computation frequency:** Nightly

| Feature Name | Type | Description |
|---|---|---|
| `new_visits_7d` | int | New billable visits in last 7 days |
| `new_visits_30d` | int | New billable visits in last 30 days |
| `visit_cancellation_rate_7d` | float | % visits cancelled in last 7 days |
| `new_case_open_rate_7d` | float | New cases opened per day (7-day avg) |
| `visit_billing_conversion_rate` | float | % visits that result in a billed amount |

---

### 3.4 Feature Group: `settlement_pipeline`

**Entity key:** `attorney_id`, `case_type`  
**Event timestamp:** `date`  
**Computation frequency:** Nightly

| Feature Name | Type | Description |
|---|---|---|
| `open_settlements_count` | int | Number of open settlement cases |
| `settlements_closed_30d` | int | Settlements closed in last 30 days |
| `avg_days_to_settlement_90d` | float | 90-day rolling average days from open to close |
| `settlement_velocity_change_30d` | float | % change in settlements_closed vs. prior 30 days |
| `high_value_open_count` | int | Open settlements with estimated value > $X (threshold TBD with DA) |

---

### 3.5 Feature Group: `lop_metrics`

**Entity key:** `facility_id`  
**Event timestamp:** `date`  
**Computation frequency:** Nightly

| Feature Name | Type | Description |
|---|---|---|
| `lop_issued_7d` | int | LOPs issued in last 7 days |
| `lop_returned_7d` | int | LOPs returned/resolved in last 7 days |
| `lop_rejection_rate_30d` | float | % LOPs rejected in last 30 days |
| `lop_turnaround_days_p50` | float | Median days from issued to returned (30d window) |
| `lop_turnaround_days_p90` | float | 90th percentile turnaround days (30d window) |
| `lop_backlog_count` | int | LOPs open > 60 days |

---

## 4. Forecasting Models

### 4.1 Problem Framing

**Task:** Multi-horizon time-series regression. Predict total collections at horizon H ∈ {30, 60, 90} days for entity E (facility, attorney, or case type).

**Label:** `collections_rolling_H` at a future date, computed from actuals.

**Training window:** Minimum 18 months of history. Entities with < 30 days of history are excluded from forecasting (flagged as "insufficient history").

---

### 4.2 Model Candidate: Prophet

**When to use:** Facility-level monthly collections where clear weekly/monthly seasonality exists and interpretability is valued.

**Configuration:**
```python
from prophet import Prophet

model = Prophet(
    changepoint_prior_scale=0.05,    # conservative — financial data doesn't shift abruptly
    seasonality_prior_scale=10,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,          # nightly data; daily noise unhelpful
    interval_width=0.80               # 80% confidence intervals
)
# Add US holiday effects
model.add_country_holidays(country_name='US')
# Custom regressors (from feature store)
model.add_regressor('visit_velocity_30d')
model.add_regressor('settlement_velocity_change_30d')
```

**Strengths:** Fast, interpretable, handles missing data well, confidence intervals built-in.  
**Weaknesses:** Assumes additive/multiplicative seasonality; struggles with complex multi-entity interactions.

---

### 4.3 Model Candidate: XGBoost (Time-Series Features)

**When to use:** Attorney/case-type granularity where seasonality is weaker and tabular feature interactions matter more.

**Feature set for XGBoost:**
- All features from the relevant feature groups.
- Lag features: `collections_rolling_7d_lag1`, `collections_rolling_7d_lag2`, `collections_rolling_7d_lag4`.
- Target-encoded categorical: `attorney_id`, `case_type`, `facility_id`.
- Calendar features: `month`, `day_of_week`, `is_month_end`, `quarter`.

**Training strategy:**
- Walk-forward cross-validation (expanding window). Minimum 3 folds.
- Hyperparameter tuning via Optuna (100 trials, early stopping on validation MAPE).
- Quantile regression for confidence intervals: train separate models for 10th, 50th, and 90th percentiles.

---

### 4.4 Model Candidate: Temporal Fusion Transformer (TFT)

**When to use:** When multi-entity, multi-horizon accuracy is the priority and infra cost is justified. Phase 1 optional — implement only if XGBoost/Prophet MAPE targets are not met.

**Framework:** PyTorch Forecasting library.

**Key advantages over simpler models:**
- Jointly trains across all entities, learning shared patterns.
- Attention mechanism provides interpretable feature importance per prediction.
- Naturally produces calibrated quantile forecasts.

**Infra cost consideration:** Requires GPU for training (Azure NC-series VM or Azure ML compute cluster). Inference can run on CPU.

---

### 4.5 Model Selection Protocol

```
1. Train all candidate models on the same training split.
2. Evaluate on held-out validation window (last 90 days of available data).
3. Compute MAPE, RMSE, and coverage (% of actuals within 80% CI) per entity.
4. Select champion = model with lowest mean MAPE across all entities,
   subject to: coverage ≥ 75% (CI is not too narrow/wide).
5. Register champion in MLflow Model Registry.
6. Weekly retrain: if new champion's MAPE < current champion MAPE by > 1pp → auto-promote.
   Otherwise → human review required.
```

**Target MAPE:** ≤ 12% on 30-day horizon per facility.

---

## 5. Anomaly Detection Design

### 5.1 Detection Layer 1: Statistical Control Charts (CUSUM)

**Purpose:** Detect sustained directional shifts in individual financial metrics.

**Algorithm:** CUSUM (Cumulative Sum Control Chart) applied to standardized daily observations.

**Application:**
- `collections_1d` per facility (z-scored using rolling 90-day mean/std).
- `new_visits_7d` per facility.
- `settlements_closed_7d` per attorney.

**Threshold:** Alert triggered when cumulative sum exceeds ±4σ. Thresholds configurable per entity in YAML.

**Why CUSUM over simple threshold:** CUSUM detects small, sustained deviations that a single-day threshold would miss. Ideal for gradual collection slowdowns.

---

### 5.2 Detection Layer 2: Isolation Forest

**Purpose:** Detect multivariate anomalies where no single metric is extreme but the combination is unusual.

**Feature vector (per facility, per day):**
```
[collections_1d, new_visits_7d, settlement_velocity_change_30d,
 lop_turnaround_days_p50, pct_180_plus, collections_vs_forecast_delta]
```

**Training:** Train on 12 months of historical data per entity. Retrain monthly.

**Contamination parameter:** 0.03 (expect ~3% anomaly rate). Tunable per entity.

**Score threshold:** Isolation Forest anomaly score < -0.1 → flag for alert generation.

---

### 5.3 Detection Layer 3: Forecast Deviation

**Purpose:** Use the forecasting service's prediction as a baseline and alert when actuals deviate materially.

**Logic:**
```python
deviation_pct = (actual_collections_7d - forecast_collections_7d) / forecast_collections_7d
if deviation_pct < -0.15:   # 15% below forecast
    severity = "P2"
if deviation_pct < -0.25:   # 25% below forecast
    severity = "P1"
```

Thresholds configurable per entity.

---

### 5.4 Alert Severity Classification

| Severity | Trigger Conditions | Delivery |
|---|---|---|
| **P1 – Critical** | Forecast deviation > 25% OR Isolation Forest score < -0.3 OR CUSUM > 6σ | Slack DM to Collections Lead, Teams channel |
| **P2 – Warning** | Forecast deviation 15–25% OR Isolation Forest score -0.1 to -0.3 OR CUSUM 4–6σ | Slack #collections-alerts channel |
| **P3 – Informational** | Any single-metric threshold breach without cross-validation | Daily digest email |

---

## 6. RAG Pipeline Design

### 6.1 Document Corpus

| Document Type | Source | Update Frequency | Chunking Strategy |
|---|---|---|---|
| Tableau extract summaries | DA team nightly export | Nightly | By facility + date range |
| Attorney aging reports | Feature store aggregate | Nightly | By attorney + date |
| Visit volume summaries | Feature store aggregate | Nightly | By facility + case type |
| Anomaly alert history | Alert database | Real-time | By entity + date |
| Forecast summaries | Forecast store | Nightly | By entity + horizon |

**What is NOT indexed:** Raw visit-level rows, any record containing PHI, individual patient data.

### 6.2 Chunking & Embedding

**Chunk size:** 512 tokens with 64-token overlap.

**Embedding model:** `text-embedding-3-small` (Azure OpenAI) — 1536 dimensions. Cost-efficient and high quality for domain text.

**Metadata per chunk:**
```json
{
  "source_doc_id": "attorney_aging_2026-05-27_johnson",
  "entity_type": "attorney",
  "entity_id": "johnson_firm",
  "date": "2026-05-27",
  "facility_id": "round_rock",
  "chunk_index": 2
}
```

### 6.3 Retrieval Strategy

1. **Semantic search:** Cosine similarity over Azure AI Search index, top-k=8 chunks.
2. **Metadata filter pre-retrieval:** If query contains a facility name, attorney name, or date reference (extracted via NER), filter the vector index to matching entities before semantic search.
3. **Re-ranking:** Cross-encoder re-ranking on the top-8 results to select top-4 for context assembly.
4. **Similarity threshold:** If max similarity score < 0.75, do not call the LLM — return "insufficient information" response.

### 6.4 System Prompt Design

```
You are a financial analyst assistant for a medical billing and collections organization.
You answer questions about facility collections, attorney performance, visit billing, and 
settlement pipelines using ONLY the provided context documents.

Rules:
1. ONLY use facts present in the provided context. Do not infer or extrapolate.
2. ALWAYS cite your sources using the format: [Source: {source_doc_id}, {date}].
3. If the context does not contain enough information to answer, say: 
   "I don't have enough data to answer this reliably. Please check with the DA team."
4. NEVER include patient names, dates of birth, SSNs, or any patient-identifying information
   in your response. If you see such information in the context, skip it.
5. When presenting numbers, always specify the time period they cover.
6. Be concise and direct — the user is a collections professional, not a data scientist.
```

---

## 7. NL-to-SQL Design

### 7.1 Schema Representation

The LLM receives a compressed schema prompt at query time:

```
Tables available:
- collections(collection_id, facility_id, attorney_id, case_type, collection_date, 
              amount_collected, days_outstanding)
- visits(visit_id, facility_id, case_type, visit_date, billing_status, 
         billed_amount, paid_amount, provider_id)
- attorney_aging(attorney_id, facility_id, bucket_0_30, bucket_31_60, bucket_61_90, 
                  bucket_91_180, bucket_180_plus, report_date)
- settlements(settlement_id, attorney_id, case_type, open_date, close_date, 
              settlement_amount, settlement_status)

Business glossary:
- "unpaid visits" = visits WHERE billing_status = 'unpaid'
- "PI" = Personal Injury (case_type = 'PI')
- "Commercial" or "Athena" = case_type IN ('Commercial', 'Athena')
- "aging bucket" = attorney_aging table
- "LOP" = lop table (lop_id, facility_id, issued_date, returned_date, status)
- "overdue" = days_outstanding > 90
```

### 7.2 SQL Safety Validation

All generated SQL passes through a validation pipeline before execution:

```python
import sqlglot

def validate_sql(sql: str) -> tuple[bool, str]:
    # 1. Parse SQL
    try:
        parsed = sqlglot.parse_one(sql)
    except Exception as e:
        return False, f"SQL parse error: {e}"
    
    # 2. Check statement type (SELECT only)
    if not isinstance(parsed, sqlglot.expressions.Select):
        return False, "Only SELECT statements are permitted."
    
    # 3. Check for forbidden clauses (no subquery DML, no CTEs that modify)
    forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
    sql_upper = sql.upper()
    for keyword in forbidden:
        if keyword in sql_upper:
            return False, f"Statement contains forbidden keyword: {keyword}"
    
    # 4. Check all referenced tables are in the whitelist
    referenced_tables = [t.name for t in parsed.find_all(sqlglot.expressions.Table)]
    allowed_tables = {'collections', 'visits', 'attorney_aging', 'settlements', 'lop'}
    for table in referenced_tables:
        if table not in allowed_tables:
            return False, f"Table not permitted: {table}"
    
    return True, "OK"
```

### 7.3 Retry & Fallback Logic

```
1. Generate SQL from NL query (attempt 1).
2. Validate SQL.
   - If valid → execute.
   - If invalid → send error + original SQL back to LLM with "fix this" prompt (attempt 2).
3. Validate fixed SQL.
   - If valid → execute.
   - If invalid again → return user-facing error message.
4. If execution times out (>30s) → return timeout message.
5. If result set > 10,000 rows → return top 10,000 + "Results truncated" notice.
```

---

## 8. MLOps Infrastructure

### 8.1 Experiment Tracking (MLflow)

All model training runs log to MLflow:

| Logged Item | Description |
|---|---|
| Parameters | Model hyperparameters, training window, feature set version |
| Metrics | MAPE, RMSE, MAE, coverage (per entity and aggregate) |
| Artifacts | Trained model file, feature importance plot, validation MAPE by entity |
| Tags | `entity_type`, `model_type`, `data_version`, `git_commit_hash` |

### 8.2 Model Registry & Promotion

```
Training Run
    ↓
Register model in MLflow Model Registry (stage: "Staging")
    ↓
Automated gate: validation MAPE ≤ current champion MAPE + 1pp?
    ├── YES → Promote to "Production" (auto)
    └── NO  → Alert DA Analyst; human reviews before promotion
```

All model versions are retained indefinitely. Rolling back to a prior version is a single MLflow registry API call, wrapped in a CLI command:

```bash
mlflow models set-production --model-name collections_forecast --version 12
```

### 8.3 A/B Testing Framework

For major model changes (e.g., switching from Prophet to XGBoost at the facility level), a traffic-split A/B test is supported:

- **Traffic split:** Configurable % of forecast API calls routed to challenger vs. champion.
- **Evaluation window:** Minimum 14 days before declaring a winner.
- **Metric:** MAPE on live actuals vs. live predictions (not held-out historical data).
- **Winner declaration:** Champion replaced only if challenger MAPE is lower and the difference is statistically significant (t-test, p < 0.05).

### 8.4 Feature & Data Drift Monitoring

| Drift Type | Detection Method | Alert Threshold |
|---|---|---|
| **Feature distribution drift** | Population Stability Index (PSI) on each feature vs. 30-day baseline | PSI > 0.2 on any feature |
| **Prediction drift** | KL divergence on forecast output distribution | KL > 0.1 |
| **Label drift (concept drift)** | Rolling MAPE trend (7-day vs. 30-day baseline) | MAPE increases > 5pp over 7 days |
| **Data freshness** | Feature group last-updated timestamp vs. expected schedule | Feature group not updated by 07:00 |

All drift metrics are logged to Azure Monitor and visible in the monitoring dashboard.

---

## 9. PHI & Privacy Guardrails

### 9.1 Defense-in-Depth Architecture

```
[OData Source — contains PHI]
        ↓
[Ingest Job — PHI Masking Layer]
  • Tokenize PHI fields using HMAC-SHA256 + Azure Key Vault secret
  • Raw zone: encrypted, access limited to DA Analyst + system account
        ↓
[Curated Zone — no raw PHI]
  • All downstream systems (Feature Store, Vector Index, NL-to-SQL DB) 
    use curated data only
        ↓
[AI Layer — PHI Prevention]
  • RAG corpus: only aggregated, de-identified documents indexed
  • Chatbot system prompt: explicit PHI prohibition
  • Input scanner: regex + NER classifier checks every user query
  • Output scanner: NER classifier checks every LLM response
        ↓
[Delivery Layer — Audit]
  • Any PHI scanner trigger → response blocked + compliance log entry
  • All queries, responses, and SQL executions logged with user identity
```

### 9.2 PHI Input Scanner (Chatbot & NL-to-SQL)

The input scanner runs on every user query before it reaches the LLM:

```python
# Patterns that suggest PHI request
PHI_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',           # SSN
    r'\b(DOB|date of birth|born on)\b',  # DOB reference
    r'\bpatient name\b',                  # Explicit patient name request
    r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',     # Possible proper name (combined with context)
]

def scan_input(query: str) -> ScanResult:
    for pattern in PHI_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return ScanResult(blocked=True, reason="PHI_REQUEST_DETECTED")
    
    # NER check for person entities
    entities = ner_model(query)
    if any(e['label'] == 'PERSON' for e in entities):
        return ScanResult(blocked=True, reason="PERSON_ENTITY_DETECTED")
    
    return ScanResult(blocked=False)
```

### 9.3 PHI Output Scanner

Every LLM response is passed through an output scanner before delivery:

```python
def scan_output(response: str) -> ScanResult:
    # Run same NER classifier over output
    entities = ner_model(response)
    phi_entities = [e for e in entities if e['label'] in ('PERSON', 'DATE', 'ID')]
    
    if phi_entities:
        # Redact and log
        redacted = redact_entities(response, phi_entities)
        log_phi_detection(response, phi_entities, user_id=current_user.id)
        return ScanResult(blocked=False, response=redacted, phi_detected=True)
    
    return ScanResult(blocked=False, response=response, phi_detected=False)
```

### 9.4 Audit Log Schema

Every AI interaction produces an audit log entry:

```json
{
  "event_id": "uuid",
  "timestamp": "2026-05-28T08:23:11Z",
  "user_id": "jsmith@org.com",
  "user_role": "collections",
  "service": "chatbot | nl_to_sql | forecast_api",
  "query_hash": "sha256_of_input",
  "input_phi_scan_result": "clean | blocked",
  "output_phi_scan_result": "clean | redacted | blocked",
  "llm_model_used": "gpt-4o-2024-08-06",
  "retrieved_sources": ["doc_id_1", "doc_id_2"],
  "generated_sql": "SELECT ...",
  "response_latency_ms": 4230,
  "session_id": "uuid"
}
```

Audit logs are append-only (no delete), encrypted at rest, retained 7 years, and accessible only to the Compliance Officer and DA Analyst Admin role.

---

## 10. Model Evaluation Framework

### 10.1 Forecasting Evaluation

| Metric | Description | Target |
|---|---|---|
| **MAPE** | Mean Absolute Percentage Error | ≤ 12% on 30-day horizon |
| **RMSE** | Root Mean Squared Error | Track; no hard target (scale-dependent) |
| **Coverage** | % of actuals within the 80% CI | ≥ 75% |
| **Bias** | Mean signed error (positive = over-forecast) | |bias| < 5% |
| **Per-entity MAPE** | Facility-level disaggregation | Flag any facility with MAPE > 20% |

Evaluation is performed on a **walk-forward validation** scheme: train on months 1–18, validate on months 19–21; train on months 1–21, validate on months 22–24; etc.

### 10.2 Anomaly Detection Evaluation

| Metric | Target |
|---|---|
| **Precision** (P1 alerts that were true anomalies) | ≥ 80% |
| **Recall** (true anomalies that were detected) | ≥ 70% |
| **False positive rate** (P1 alerts dismissed as false) | < 20% |

Ground truth labels are assembled retrospectively: DA Analyst reviews a sample of historical dates where collections were genuinely anomalous (identified from business records).

### 10.3 Chatbot Evaluation

| Metric | Method | Target |
|---|---|---|
| **Factual accuracy** | Human eval on 20-question weekly sample | ≥ 85% correct |
| **Citation rate** | % of factual claims with a citation | ≥ 95% |
| **Hallucination rate** | % of responses containing ungrounded facts | < 5% |
| **Refusal rate** | % of unanswerable queries where chatbot correctly declines | ≥ 90% |
| **PHI leak rate** | PHI scanner detections / total responses | 0% target |

### 10.4 NL-to-SQL Evaluation

Maintain a benchmark of 50 human-written (NL query, expected SQL) pairs covering:
- Simple lookups (single table, simple WHERE clause)
- Aggregations (GROUP BY, SUM, COUNT)
- Joins (collections + attorney_aging)
- Business glossary terms ("unpaid visits over 180 days")
- Edge cases (ambiguous date ranges, multi-attorney queries)

| Metric | Target |
|---|---|
| Syntactic validity | ≥ 90% |
| Semantic correctness | ≥ 80% |
| Exact SQL match | Track only (not a hard target) |

Run benchmark after every glossary update and LLM version change.
