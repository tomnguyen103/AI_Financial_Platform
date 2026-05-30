# Technical Architecture Document
## AI-Powered Financial Reporting & Revenue Intelligence Platform

**Document Version:** 1.0  
**Date:** 2026-05-28  
**Status:** Draft

---

## 1. Architecture Overview

The platform is composed of five loosely coupled layers that sit on top of the organization's existing Azure infrastructure and OData feed. Each layer is independently deployable and communicates via well-defined APIs and message queues.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                           │
│   Tableau Dashboard  │  Chatbot UI  │  NL-to-SQL UI  │  Slack/Teams │
└────────────┬─────────┴──────┬───────┴───────┬────────┴──────┬───────┘
             │                │               │               │
┌────────────▼────────────────▼───────────────▼───────────────▼───────┐
│                         API GATEWAY LAYER                           │
│              FastAPI  ·  JWT Auth  ·  Rate Limiting  ·  RBAC        │
└────────────┬────────────────┬───────────────┬───────────────────────┘
             │                │               │
    ┌────────▼──────┐  ┌──────▼──────┐  ┌────▼──────────┐
    │  Forecasting  │  │   RAG /     │  │  Anomaly      │
    │  Service      │  │  Chatbot    │  │  Detection    │
    │               │  │  Service    │  │  Service      │
    └────────┬──────┘  └──────┬──────┘  └────┬──────────┘
             │                │               │
┌────────────▼────────────────▼───────────────▼───────────────────────┐
│                         FEATURE STORE LAYER                         │
│         Feast on Postgres  ·  Azure ML Feature Store (alt)          │
│         Point-in-time correct feature retrieval  ·  Versioned       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                        INGESTION LAYER                              │
│    OData Feed  →  FastAPI Ingest Job  →  Azure Data Lake (raw)      │
│    Nightly batch  ·  Schema validation  ·  PHI masking              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Descriptions

### 2.1 Ingestion Layer

**Purpose:** Pull financial data from the OData feed nightly, validate it, apply PHI masking, and land it in the raw data store.

**Components:**

- **OData Ingest Job** — A Python FastAPI-based CLI/service that calls the OData endpoints, paginates through results, performs schema validation (Pydantic models), and writes Parquet files to Azure Data Lake Storage Gen2 (ADLS2).
- **PHI Masking Module** — Applied at ingest time. Tokenizes or pseudonymizes patient-identifiable fields (patient name, DOB, SSN fragments) before the data is written to any downstream store. Mapping is stored in an encrypted key vault.
- **Schema Contract Tests** — Great Expectations or Soda Core rules run after each ingest to detect upstream schema changes, null rate spikes, or value range violations. Failures block the downstream pipeline and trigger an alert.
- **Orchestration** — Azure Data Factory (ADF) or Apache Airflow DAG schedules and monitors the nightly job. Retry logic: 3 attempts with exponential back-off.

**Data Flow:**
```
OData Endpoint → Pydantic Validation → PHI Masking → Parquet → ADLS2 (raw zone)
                                    ↓
                          Schema Contract Tests
                                    ↓ (on pass)
                          ADLS2 (curated zone) → Feature Store
```

**Technology:**

| Component | Technology |
|---|---|
| Ingest runtime | Python 3.11, httpx (async OData calls) |
| Validation | Pydantic v2 |
| Storage | Azure Data Lake Storage Gen2 (Parquet) |
| Orchestration | Apache Airflow (or ADF if Azure-native preferred) |
| Data quality | Great Expectations |
| PHI masking | Azure Key Vault + custom tokenization module |

---

### 2.2 Feature Store Layer

**Purpose:** Provide a single, versioned, point-in-time-correct source of truth for all ML features consumed by the forecasting and anomaly detection services.

**Feature Groups:**

| Feature Group | Key Entities | Example Features |
|---|---|---|
| `facility_collections` | facility_id, date | rolling_7d_collections, rolling_30d_collections, mom_growth_rate |
| `attorney_aging` | attorney_id, date | bucket_0_30_balance, bucket_31_60_balance, bucket_180_plus_balance, avg_days_outstanding |
| `visit_velocity` | facility_id, case_type, date | billable_visits_7d, visit_cancellation_rate, new_case_open_rate |
| `settlement_pipeline` | attorney_id, case_type, date | open_settlements_count, avg_days_to_settlement, settlements_closed_30d |
| `lop_metrics` | facility_id, date | lop_turnaround_days_p50, lop_turnaround_days_p90, lop_rejection_rate |

**Technology:**

| Component | Technology |
|---|---|
| Feature store | Feast 0.36+ on PostgreSQL (primary) OR Azure ML Feature Store (if org is Azure ML licensed) |
| Online store | Redis (for low-latency feature serving to the API) |
| Offline store | ADLS2 / PostgreSQL |
| Feature computation | PySpark or Pandas (depending on data volume) |

**Design Decisions:**
- Features are computed at nightly cadence alongside ingest. No streaming features in Phase 1.
- Feature definitions are version-controlled in Git. Drift in feature distributions triggers a monitoring alert.
- Point-in-time correctness is enforced via Feast's `get_historical_features` API to prevent label leakage during model training.

---

### 2.3 Forecasting Service

**Purpose:** Produce 30/60/90-day collection forecasts per facility, attorney-provider, and case type.

**Architecture:**

```
Feature Store (offline)
        ↓
  Training Pipeline (weekly retrain)
        ↓
  Model Registry (MLflow)
        ↓ (champion model)
  Forecasting Service (FastAPI)
        ↓
  Forecast Store (PostgreSQL: forecasts table)
        ↓
  Tableau / Dashboard
```

**Models (evaluated in order of complexity):**

| Model | Use Case | Notes |
|---|---|---|
| **Prophet** | Facility-level monthly collections | Strong seasonality handling, interpretable, fast to retrain |
| **XGBoost (time-series features)** | Attorney/case-type granularity | Higher accuracy where seasonality is weak; requires feature engineering |
| **Temporal Fusion Transformer (TFT)** | Multi-horizon, multi-entity | Best accuracy potential; higher infra cost; Phase 1 optional |

Model selection is governed by MAPE on a held-out validation set (last 90 days). The champion model is promoted via MLflow Model Registry. A/B testing framework allows traffic splitting between challenger and champion.

**API Endpoints (FastAPI):**

```
GET /forecasts/facility/{facility_id}?horizon=30|60|90
GET /forecasts/attorney/{attorney_id}?horizon=30|60|90
GET /forecasts/case_type/{case_type}?horizon=30|60|90
GET /forecasts/summary                  # aggregate view
POST /forecasts/refresh                 # admin: trigger manual reforecast
```

**Response Shape:**
```json
{
  "entity_id": "facility_round_rock",
  "horizon_days": 30,
  "forecast_date": "2026-05-28",
  "predicted_collections": 142500.00,
  "confidence_interval_lower": 128000.00,
  "confidence_interval_upper": 157000.00,
  "model_version": "prophet_v3.2",
  "feature_snapshot_ts": "2026-05-27T06:00:00Z"
}
```

---

### 2.4 Anomaly Detection Service

**Purpose:** Monitor financial metrics in real time (nightly) and fire alerts when statistically significant deviations are detected.

**Detection Methods:**

| Method | Target Signals | Rationale |
|---|---|---|
| **Isolation Forest** | Multi-dimensional feature vectors (visits + collections + LOP) | Catches complex multi-variate anomalies |
| **Statistical Control Charts (CUSUM / Shewhart)** | Individual time series (e.g., daily collections per facility) | Interpretable, low false-positive rate for monotonic shifts |
| **Forecast Deviation** | Actual vs. forecast delta > threshold | Leverage forecasting service as a baseline |

**Alert Routing:**

```
Anomaly Detected
      ↓
Severity Classifier (rule-based + ML)
      ↓
  [P1 - Critical]  →  Slack DM to Collections Lead + Teams channel
  [P2 - Warning]   →  Slack channel message + email digest
  [P3 - Info]      →  Daily digest email only
```

**Alert Payload (Slack):**
```
⚠️ [P2 Warning] Round Rock — Collections Anomaly Detected
Date: 2026-05-27
Metric: Rolling 7-day collections
Expected: $48,200 | Actual: $39,500 | Δ: -18.1%
Likely driver: Settlement velocity down 22% (Attorney Johnson PI pipeline)
→ View in dashboard: [link]
→ Ask the Financials: "Why did Round Rock collections drop?" [link]
```

**Technology:**

| Component | Technology |
|---|---|
| Anomaly models | scikit-learn (Isolation Forest), statsmodels (CUSUM) |
| Alert delivery | Slack Incoming Webhooks, Microsoft Teams Webhooks |
| Severity rules | YAML-configured thresholds, overrideable per facility |

---

### 2.5 RAG Chatbot ("Ask the Financials")

**Purpose:** Allow users to ask natural-language questions about financial data and receive AI-synthesized answers grounded in real source data with citations.

**Architecture:**

```
User Query (natural language)
        ↓
  Query Router
  ├── [Structured data query] → NL-to-SQL Engine → Azure DB → Response Formatter
  └── [Analytical / "Why" question] → RAG Pipeline
              ↓
        Query Embedding (text-embedding-3-small or equivalent)
              ↓
        Vector Retrieval (Azure AI Search / Chroma / Weaviate)
        [Indexed: Tableau extracts, visit-level summaries, attorney aging reports]
              ↓
        Context Assembly + Source Tracking
              ↓
        LLM Synthesis (GPT-4o or Claude 3.5 Sonnet)
        [System prompt: grounding instructions, citation format, PHI rules]
              ↓
        Response with Citations → User
```

**Document Corpus (RAG Index):**
- Tableau extract snapshots (nightly, converted to structured text chunks)
- Attorney aging bucket summaries per facility
- Visit-level aggregated reports (not row-level patient data — masked)
- Anomaly alert history
- Forecast summaries

**PHI Guardrails in Chatbot:**
- Input scanning: regex + NER to detect if user is requesting individual patient data; block and redirect.
- Output scanning: LLM output is passed through a PHI classifier before delivery; any detected PHI triggers redaction + audit log entry.
- Context assembly: only de-identified, aggregated documents enter the RAG context window.

**Technology:**

| Component | Technology |
|---|---|
| RAG framework | LangChain or LlamaIndex |
| Vector store | Azure AI Search (native to Azure) or Chroma (self-hosted) |
| Embeddings | OpenAI text-embedding-3-small or Azure OpenAI |
| LLM | GPT-4o (Azure OpenAI) or Claude 3.5 Sonnet (Anthropic API) |
| Citation format | Source document ID + row range + retrieval score |

---

### 2.6 NL-to-SQL Engine

**Purpose:** Translate plain-English questions into validated SQL queries against the Azure database and return formatted results.

**Architecture:**

```
User Query
    ↓
Schema-aware Prompt Construction
(table names, column names, sample values, business glossary injected)
    ↓
LLM SQL Generation (GPT-4o / Claude)
    ↓
SQL Validation Layer
├── Syntax check (sqlglot parser)
├── Safety check (whitelist SELECT-only; block DDL/DML)
└── Schema binding check (all referenced tables/columns exist)
    ↓
Query Execution (read-only Azure DB connection)
    ↓
Result Formatting (tabular → natural language summary)
    ↓
User Response + Raw Table
```

**Business Glossary Injection:** A maintained YAML glossary maps domain terms ("unpaid visits", "LOP", "PI case") to actual table/column names, reducing LLM hallucination of nonexistent columns.

**Safety Controls:**
- Read-only database user (SELECT grants only, no INSERT/UPDATE/DELETE/DROP).
- Query timeout: 30 seconds.
- Row limit cap: 10,000 rows per query.
- All generated SQL is logged for audit.

---

### 2.7 API Gateway

**Purpose:** Single entry point for all client-facing services. Handles authentication, authorization, rate limiting, and request routing.

**Technology:** FastAPI with the following middleware stack:

| Middleware | Purpose |
|---|---|
| JWT Auth (Azure AD / Entra ID) | Identity verification |
| RBAC middleware | Role-based access (Collections, Finance, Admin, DA) |
| Rate limiter (slowapi) | Prevent abuse; chatbot capped at 60 req/min/user |
| Request logging | Structured JSON logs → Azure Monitor |
| CORS | Whitelist approved frontend origins |

**RBAC Roles:**

| Role | Forecasts | Anomaly Alerts | Chatbot | NL-to-SQL | Admin APIs |
|---|---|---|---|---|---|
| Collections | ✅ Read | ✅ Read | ✅ | ✅ | ❌ |
| Finance Lead | ✅ Read | ✅ Read | ✅ | ✅ | ❌ |
| DA Analyst | ✅ Read/Write | ✅ Read/Write | ✅ | ✅ | ✅ |
| Admin | ✅ Full | ✅ Full | ✅ | ✅ | ✅ |

---

## 3. Infrastructure & Deployment

### 3.1 Environment Strategy

| Environment | Purpose | Data |
|---|---|---|
| **Dev** | Development and unit testing | Synthetic / anonymized data only |
| **Staging** | Integration testing, UAT, model evaluation | Masked production data snapshot |
| **Production** | Live system | Masked production data |

### 3.2 Infrastructure (Azure-native)

```
Azure Subscription
├── Resource Group: ai-financial-platform
│   ├── Azure Data Lake Storage Gen2 (raw + curated zones)
│   ├── Azure Database for PostgreSQL (Feature Store + Forecasts)
│   ├── Azure Container Apps (FastAPI services — auto-scaling)
│   ├── Azure AI Search (RAG vector index)
│   ├── Azure OpenAI Service (GPT-4o + embeddings)
│   ├── Azure Key Vault (secrets, PHI token mapping)
│   ├── Azure Monitor + Log Analytics (observability)
│   ├── Azure Data Factory or Apache Airflow (orchestration)
│   └── MLflow on Azure Container Apps (model registry)
```

### 3.3 CI/CD Pipeline

```
GitHub (or Azure DevOps)
    ↓
PR Checks: lint (ruff), type check (mypy), unit tests (pytest), security scan (bandit)
    ↓
Merge to main → Docker image build → Push to Azure Container Registry
    ↓
Staging deploy (auto)
    ↓
Integration tests + model accuracy gate
    ↓
Production deploy (manual approval for ML model changes)
```

---

## 4. Data Flow: End-to-End

```
[OData API]
    │  nightly 02:00
    ▼
[Ingest Job] — validates schema, masks PHI
    │
    ▼
[ADLS2 Raw Zone] (Parquet)
    │  Great Expectations pass
    ▼
[ADLS2 Curated Zone]
    │
    ├──► [Feature Store (Feast/Postgres)] ──► Forecasting Service
    │                                    ──► Anomaly Detection
    │
    ├──► [Vector Index (Azure AI Search)] ──► RAG Chatbot
    │
    └──► [Azure SQL / Postgres read replica] ──► NL-to-SQL Engine
```

---

## 5. Security & Compliance

### 5.1 PHI/PII Controls

| Control | Implementation |
|---|---|
| PHI masking at ingest | Tokenization before data leaves raw zone |
| No patient-level data in RAG corpus | Only aggregated/anonymized summaries indexed |
| Chatbot input/output scanning | Regex + NER PHI classifier on both sides |
| Audit logging | All queries, LLM calls, and SQL executions logged with user, timestamp, input hash |
| Key management | Azure Key Vault; PHI token map encrypted at rest |
| Access control | Azure AD + RBAC; least-privilege DB users |

### 5.2 Data Retention

| Data Type | Retention Policy |
|---|---|
| Raw OData ingest (Parquet) | 24 months, then archive to cold tier |
| Feature store (historical features) | 36 months |
| Forecast outputs | Indefinite (append-only; low volume) |
| Chatbot query logs | 12 months |
| Audit logs | 7 years (compliance) |

---

## 6. Observability

| Signal | Tool | Alert Condition |
|---|---|---|
| Ingest pipeline failures | Azure Monitor + Airflow | Any nightly job failure |
| Feature store freshness | Custom metric | Features not updated by 07:00 |
| Forecast MAPE drift | MLflow + custom monitor | MAPE increases > 5pp over 7-day rolling window |
| API latency / errors | Azure Monitor | p95 > 3s or error rate > 1% |
| Chatbot answer quality | Manual eval + LLM-as-judge (weekly sample) | Accuracy drops below 80% |
| PHI scanner triggers | Audit log | Any PHI detection event → immediate alert to compliance |

---

## 7. Technology Stack Summary

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| API framework | FastAPI |
| Data ingestion | httpx, Pydantic v2, Great Expectations |
| Orchestration | Apache Airflow (or Azure Data Factory) |
| Storage | Azure Data Lake Storage Gen2 (Parquet), PostgreSQL |
| Feature store | Feast 0.36+ on PostgreSQL |
| ML / forecasting | Prophet, XGBoost, scikit-learn, PyTorch (TFT optional) |
| MLOps | MLflow (tracking + registry) |
| RAG framework | LangChain or LlamaIndex |
| Vector store | Azure AI Search |
| LLM | Azure OpenAI (GPT-4o) or Anthropic Claude API |
| NL-to-SQL | sqlglot (validation), custom prompt chain |
| Alerting | Slack Webhooks, Microsoft Teams Webhooks |
| Infrastructure | Azure Container Apps, Azure Key Vault, Azure Monitor |
| CI/CD | GitHub Actions or Azure DevOps |
| Containerization | Docker, Azure Container Registry |
