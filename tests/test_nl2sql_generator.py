from __future__ import annotations

from app.nl2sql.generator import generate_sql


def test_pending_payment_by_attorney_uses_attorney_aging():
    sql = generate_sql("total pending payment by attorney").lower()

    assert "from attorney_aging" in sql
    assert "attorney_id" in sql
    assert "facility_id" not in sql.split("from", 1)[0]
    assert "bucket_0_30" in sql
    assert "bucket_180_plus" in sql
