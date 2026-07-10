# A+ Audit Report

End-to-end read-only audit of the AI Financial Platform followed by a multi-agent
(heavier) fix pass. This document records the final per-dimension grades, what was
fixed, and — for every dimension still short of A+ — the specific remaining gap,
what was tried, why it did not reach A+ within the fix batches, and the concrete
next step to close it.

Audit ran against the true deployed code (`origin/main`), which was **5 commits
ahead of the local checkout** at session start (background-seed cold-start fix +
RBAC differentiation + admin MLOps panel). Those upstream commits already resolved
the original "blocking cold-start seed" performance finding; the audit was
re-baselined accordingly before any fixes were written.

## PRs merged during the loop

| PR | Lane | Summary |
|----|------|---------|
| (foundation) | CI/tooling | `.github/workflows/ci.yml` (ruff + pytest), `pyproject.toml`, `.coderabbit.yaml`, cleared all pre-existing lint |
| [#3](https://github.com/tomnguyen103/AI_Financial_Platform/pull/3) | Security | fail-closed secrets in prod, gate dev-token issuer, security headers, proxy-aware + bounded rate limiter, log-injection sanitize |
| [#5](https://github.com/tomnguyen103/AI_Financial_Platform/pull/5) | Back-end + data | `entity_type` `Literal`→422, typed `response_model`s, atomic forecast regen, signed `bias`, 5 indexes, NL2SQL preview cap |
| [#6](https://github.com/tomnguyen103/AI_Financial_Platform/pull/6) | ML + reliability | CUSUM baseline de-contamination, Slack error handling, `/alerts` pagination, LLM timeout + fallback, structured nightly logging, ingestion error detail |
| [#4](https://github.com/tomnguyen103/AI_Financial_Platform/pull/4) | Front-end | WCAG 2.2 AA (keyboard, labels, live regions, focus, no `alert()`) + design uplift (type scale, elevation, inline-SVG CI chart, skeletons, tabular-nums) |
| [#7](https://github.com/tomnguyen103/AI_Financial_Platform/pull/7) | Hardening | request-body length bounds on free-text fields |

Test count grew 28 → 63; CI (ruff + pytest) is green on `main`.

## Final grade table

| Dimension | Before | After | Round reached |
|---|---|---|---|
| Security | C | **A** | 1–2 |
| Correctness | B− | **A** | 1 |
| Back-End (framework) | B | **A** | 1–2 |
| Back-End (data layer) | B− | **B+** | 1 |
| Testing | C− | **A−** | 1 |
| DX / CI | D | **A** | foundation |
| Front-End (accessibility) | D+ | **A** | 1 |
| Front-End (design quality) | C+/B− | **A−** | 1 |
| Performance | C | **A−** | 1–2 |

No dimension remains below B+. Five dimensions are **A**, four are **A−/B+**. The
remaining gaps below A+ are documented next; each is a large refactor, a low-value
change, or subjective polish — the loop's stop condition ("remaining gaps are
nice-to-have, blocked, or out of scope; diminishing returns") is met.

## Dimensions still below A+

### Back-End — data layer — **B+**
- **Money stored as `REAL`** (`app/db.py` SCHEMA — `billed_amount`, `paid_amount`,
  `amount_collected`, `settlement_amount`, aging buckets). Float accumulation drifts
  on `SUM(...)` for a *financial* product.
  - *Why not A+ this pass:* correct fix is integer-cents (or fixed-point) across the
    schema **and** every read/write/aggregation site (`features/compute.py`,
    `llm/client.py`, seed generator, forecasting) plus presentation formatting — a
    cross-cutting refactor with real regression surface, beyond a surgical batch.
  - *Next step:* migrate monetary columns to `INTEGER` cents, add a
    `dollars()`/`cents()` boundary helper, convert at the API/CSV edge, backfill the
    seed generator, and add golden-value tests on aggregates.
- **No versioned migrations** (`CREATE TABLE IF NOT EXISTS` only). Any future column
  change is a silent no-op on an existing DB.
  - *Why not A+ this pass:* genuinely low value here — Render's disk is ephemeral and
    the app reseeds on boot, so there is no persistent schema to migrate; adding a
    migration runner is scaffolding with no current consumer.
  - *Next step:* add `PRAGMA user_version` + a `migrations/NNN_*.sql` runner **if/when**
    a persistent volume or real datastore is introduced.
- Minor: N+1 connection-per-entity in the anomaly driver-narrative path; no `CHECK`
  constraints on enum-like columns (`severity`, `status`, `stage`).

### Testing — **A−**
- Coverage is much wider (security, backend validation, data layer, anomaly, LLM
  resilience, ingestion) but `app/features/*` and parts of `app/ingestion/*` and the
  forecasting service still lack direct behavioral tests, and there is **no coverage
  gate**.
  - *Why not A+ this pass:* reaching a verified 80%+ floor across every module is a
    sustained test-writing effort; a hard coverage gate can't be added until the floor
    is actually met without turning CI red.
  - *Next step:* add `pytest-cov` with `--cov-fail-under`, raise the threshold
    incrementally as feature/ingestion tests land.
- The suite shares one writable `data/platform.db` (conftest `init_db()` against the
  real `DB_PATH`); pagination tests already work around it with unique IDs, but a
  `tmp_path`-scoped DB fixture would remove the shared-state flakiness risk entirely.

### Front-End — design quality — **A−**
- Type scale, elevation tokens, an inline-SVG 80% confidence-interval chart,
  skeleton loaders, and tabular-nums money all landed and are confirmed in code and
  the accessibility tree.
  - *Why not A+ this pass:* the in-app browser's **screenshot renderer hung**
    repeatedly this session, so the visual polish (the CI chart with real forecast
    data, dark/light, desktop/mobile pixel review) could not be eyeballed end-to-end.
    `read_page`/`get_page_text` confirmed structure and the a11y fixes, but agency-grade
    design is partly a subjective visual judgment that needs a real screenshot pass.
  - *Next step:* capture desktop + mobile screenshots (signed-in, with a live
    forecast so the CI SVG renders), review spacing/hierarchy, and iterate.

### Performance — **A−**
- **Query-embedding cache reverted.** A round-2 LRU cache was added, but CodeRabbit
  correctly flagged that `LLMClient.embed` returns a *stub* embedding on a transient
  provider failure, so the cache would poison a query with a degraded result that
  survives recovery. It was removed rather than shipped unsafe.
  - *Next step:* have `LLMClient` expose success-vs-fallback (e.g. `embed_one` that
    raises on failure); cache only real successes, fall back to stub uncached.
- Per-request `sqlite3.connect` re-runs `PRAGMA journal_mode=WAL` every call
  (WAL is a persistent DB-level setting — set it once in `init_db()`); 10k-row CSV
  export is materialized in memory (stream via `StreamingResponse`). Both are
  incremental, low-risk follow-ups.

### Security — **A** (below A+ only on defense-in-depth features)
- Audit log is append-only by convention, **not tamper-evident** — no hash chain or
  WORM sink. *Next step:* `prev_hash → hash(row + prev_hash)` chaining or ship to an
  append-only external sink; a real feature, not a bug.
- NL2SQL cost ceiling is enforced by row-cap + timeout + read-only connection, not by
  rejecting cross-joins-without-LIMIT at the AST level. *Next step:* AST inspection to
  reject cartesian products lacking an explicit `LIMIT`.
- CSP still allows `'unsafe-inline'` (the dashboard is inline HTML/CSS/JS). *Next
  step:* nonce-based CSP if the single-file dashboard is ever split into assets.

### Back-End framework — **A** (below A+ only on minors)
- `/forecasts/entities` is unpaginated (the list is naturally bounded by the small
  count of distinct facilities/attorneys/case types, so impact is negligible).
- Latent `threading.Timer` watchdog vs. `conn.close()` race in the NL2SQL executor at
  the exact timeout boundary — narrow window; a shared lock or `timer.join()` closes it.

## Recommended next steps (priority order)

1. Money → integer cents (moves data layer to A; highest-value real defect).
2. `pytest-cov` gate + feature/ingestion/forecasting-service tests (Testing → A+).
3. Success-signalled embedding cache + WAL-pragma-once + streamed CSV (Performance → A+).
4. Tamper-evident audit chain + AST-level NL2SQL cost limit (Security → A+).
5. Browser screenshot pass to finish the design review (Design → A/A+).
6. mypy typecheck in CI; `tmp_path` DB fixture; `CHECK` constraints; `/forecasts`
   pagination; executor watchdog lock.
