"""Nightly orchestrator (arch §3 batch pipeline; PRD §8 SLAs).

Runs the daily refresh in dependency order and reports per-stage status + timing
so an operator (or cron/Azure Data Factory in prod) can see what happened:

    python -m scripts.run_nightly

Each stage is isolated: a failure is recorded and the run continues where it can,
mirroring the spec's "degrade, don't crash" posture for the batch layer.
"""
from __future__ import annotations

import sys
import time

from app.anomaly.alerting import run_detection
from app.db import init_db
from app.features.compute import compute_all
from app.features.store import stale_groups
from app.forecasting.service import run_all as forecasting_run_all
from app.ingestion.pipeline import run_ingest
from app.logging_config import get_logger

log = get_logger(__name__)


def _stage(name: str, fn) -> dict:
    start = time.time()
    try:
        result = fn()
        status = "ok"
        err = None
    except Exception as e:  # noqa: BLE001
        # Full traceback goes to the structured log stream (not just str(e)) so
        # an operator can diagnose the failure; stdout keeps the human summary.
        log.exception("nightly stage failed", extra={"stage": name})
        result, status, err = None, "failed", str(e)
    elapsed = round(time.time() - start, 2)
    print(f"[{status:6}] {name:24} {elapsed:>6}s  {err or result}")
    return {"stage": name, "status": status, "seconds": elapsed, "error": err}


def main() -> int:
    print("=== nightly run ===")
    init_db()
    stages = [
        _stage("ingest", run_ingest),
        _stage("features", compute_all),
        _stage("forecasting", forecasting_run_all),
        _stage("anomaly_detection", run_detection),
    ]
    stale = stale_groups()
    if stale:
        print(f"[warn  ] stale feature groups past SLA: {stale}")
    failed = [s["stage"] for s in stages if s["status"] != "ok"]
    print("=== done ===", "FAILED:" + ",".join(failed) if failed else "all ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
