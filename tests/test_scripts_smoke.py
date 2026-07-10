"""Smoke tests for the batch orchestration scripts against the hermetic DB."""
from __future__ import annotations

from scripts import run_nightly, seed_data


def test_run_nightly_all_stages_ok(_hermetic_db):
    """The nightly orchestrator runs every stage in order and reports success."""
    rc = run_nightly.main()
    assert rc == 0  # 0 == all stages ok (1 would mean a stage failed)


def test_seed_data_main_succeeds(_hermetic_db):
    """The one-shot bootstrap seed completes end-to-end without error."""
    assert seed_data.main() == 0
