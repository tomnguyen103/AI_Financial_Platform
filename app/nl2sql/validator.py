"""SQL safety validation (data design §7.2).

Defense in depth before any generated SQL reaches the database:
  1. Parse with sqlglot (reject unparseable).
  2. Statement must be a single SELECT.
  3. No forbidden DML/DDL keywords anywhere in the text.
  4. Every referenced table must be in the whitelist.
  5. AST query-cost guard: reject a clear cartesian product (>=2 base tables in
     the top-level FROM with zero join predicates anywhere) — such a query can
     burn CPU up to the executor timeout even under the row cap.

Execution is *also* run on a read-only connection (db.get_readonly_conn), so this
is the first of two independent guards.
"""
from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp

from app.nl2sql.glossary import ALLOWED_TABLES

FORBIDDEN = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
             "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "ATTACH", "PRAGMA",
             "VACUUM"]

# Binary comparison ops that, when applied column-to-column, tie two tables
# together (i.e. count as a join predicate).
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def _has_column_to_column_predicate(where: exp.Expression | None) -> bool:
    """True if the WHERE tree contains a column-vs-column comparison.

    A column compared to a literal (e.g. ``a.x = 5``) is NOT a join predicate;
    only column-to-column comparisons link two tables.
    """
    if where is None:
        return False
    for cmp in where.find_all(_COMPARISONS):
        if isinstance(cmp.left, exp.Column) and isinstance(cmp.right, exp.Column):
            return True
    return False


def _cartesian_reason(select: exp.Select) -> str | None:
    """Return a rejection reason if `select` is a clear cartesian product.

    Conservative by design: only flags the unambiguous case — two or more base
    tables in the top-level FROM/JOINs with ZERO join conditions anywhere (no
    ON, no USING, and no column-to-column WHERE predicate). Legitimate
    single-table queries and any multi-table query that carries a join
    condition are left untouched. Nested subqueries are ignored (only the
    outermost FROM's table sources are counted).
    """
    from_node = select.args.get("from") or select.args.get("from_")
    if from_node is None:
        return None
    joins = select.args.get("joins") or []

    # Count only top-level base-table sources (a subquery source is not a base
    # table and is deliberately not treated as a cartesian risk here).
    sources = [
        node.this
        for node in [from_node, *joins]
        if isinstance(node.this, exp.Table)
    ]
    if len(sources) < 2:
        return None

    # Any explicit join predicate anywhere -> not a cartesian product.
    for join in joins:
        if join.args.get("on") is not None or join.args.get("using"):
            return None
    if _has_column_to_column_predicate(select.args.get("where")):
        return None

    names = ", ".join(sorted({t.name for t in sources}))
    return (
        f"Cartesian product rejected: tables ({names}) are combined with no join "
        "condition (missing ON/USING or a WHERE join predicate)."
    )


def validate_sql(sql: str) -> tuple[bool, str]:
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return False, "Empty SQL."

    try:
        # Reject multiple statements early (parse_one would only see the first).
        statements = [s for s in sqlglot.parse(sql) if s is not None]
        if len(statements) > 1:
            return False, "Only a single statement is permitted."
        parsed = sqlglot.parse_one(sql)
    except Exception as e:  # noqa: BLE001 - surface parse error to caller
        return False, f"SQL parse error: {e}"

    if not isinstance(parsed, exp.Select):
        return False, "Only SELECT statements are permitted."

    sql_upper = sql.upper()
    for keyword in FORBIDDEN:
        # word-boundary-ish check to avoid matching column names like "updated_at"
        if f" {keyword} " in f" {sql_upper} " or sql_upper.startswith(keyword + " "):
            return False, f"Statement contains forbidden keyword: {keyword}"

    referenced = {t.name for t in parsed.find_all(exp.Table)}
    for table in referenced:
        if table not in ALLOWED_TABLES:
            return False, f"Table not permitted: {table}"
    if not referenced:
        return False, "No permitted table referenced."

    cartesian = _cartesian_reason(parsed)
    if cartesian:
        return False, cartesian

    return True, "OK"
