"""One-shot bootstrap: schema -> ingest synthetic data -> features -> train -> forecast -> detect.

Run once after install to get a fully populated local platform:

    python -m scripts.seed_data
"""
from __future__ import annotations

import sys

from app.anomaly.alerting import run_detection
from app.config import settings
from app.db import init_db
from app.features.compute import compute_all
from app.forecasting.service import run_all as forecasting_run_all
from app.ingestion.pipeline import run_ingest


def main() -> int:
    print("1/5 init schema ...")
    init_db()

    print(f"2/5 ingest [source={settings.ingest_source}] ...")
    ing = run_ingest()
    print("     ", ing)

    print("3/5 compute features ...")
    feats = compute_all()
    print("     ", feats)

    print("4/5 train + register + forecast ...")
    fc = forecasting_run_all()
    print("     ", fc)

    print("5/5 anomaly detection ...")
    det = run_detection()
    print("     ", det)

    print("\nSeed complete. Start the API with:  python -m app.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
