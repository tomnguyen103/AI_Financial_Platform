"""NL2SQL AST cartesian-product guard (defense-in-depth query-cost check)."""
from __future__ import annotations

from app.nl2sql.validator import validate_sql


def test_comma_join_without_predicate_is_rejected():
    ok, why = validate_sql("SELECT * FROM collections, visits")
    assert not ok
    assert "cartesian" in why.lower()


def test_cross_join_without_predicate_is_rejected():
    ok, why = validate_sql("SELECT * FROM collections c CROSS JOIN visits v")
    assert not ok
    assert "cartesian" in why.lower()


def test_explicit_join_on_predicate_is_accepted():
    ok, _ = validate_sql(
        "SELECT c.amount_collected FROM collections c "
        "JOIN visits v ON c.facility_id = v.facility_id"
    )
    assert ok


def test_comma_join_with_where_predicate_is_accepted():
    ok, _ = validate_sql(
        "SELECT * FROM collections, visits "
        "WHERE collections.facility_id = visits.facility_id"
    )
    assert ok


def test_single_table_query_is_accepted():
    ok, _ = validate_sql("SELECT * FROM collections LIMIT 5")
    assert ok


def test_three_way_join_with_predicates_is_accepted():
    ok, _ = validate_sql(
        "SELECT c.attorney_id FROM collections c "
        "JOIN visits v ON c.facility_id = v.facility_id "
        "JOIN lop l ON l.facility_id = c.facility_id"
    )
    assert ok


def test_subquery_reference_is_not_flagged_as_cartesian():
    # Two tables appear, but the second is inside a subquery — not a cross join.
    ok, _ = validate_sql(
        "SELECT * FROM collections "
        "WHERE facility_id IN (SELECT facility_id FROM visits)"
    )
    assert ok
