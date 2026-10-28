"""The SQL guard: validates every query (template or LLM-written) before it
runs, against the real parsed SQL (sqlglot, Presto dialect) -- never
against the planner's own description of what it did.

Rules: a single SELECT, no writes, no SELECT *, every output column
aliased with unique aliases, only catalog tables (and never Athena and
Postgres tables in one query), aggregate functions only over columns whose
role allows them, only declared joins, and -- because Athena bills by bytes
scanned over 10-100M-row tables -- an aggregate over a large table needs a
date or geo-hierarchy filter (_check_bounding_filter). String comparisons
are made case-insensitive with LOWER(...) = LOWER(...).
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify

from gmr_agents.schema.catalog import AthenaCatalog
from gmr_agents.sql.dialect import DIALECT, has_aggregate
from gmr_agents.schema.table_metadata import SCHEMA_RE

_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Command,
)

_AGG_ROLE_RULES = (
    (exp.Sum, {"measure"}),
    (exp.Avg, {"measure"}),
    (exp.Max, {"measure", "date"}),
    (exp.Min, {"measure", "date"}),
)

# Geo-hierarchy columns present (with varying coverage) across both domains'
# tables -- an equality filter on any of these, OR a comparison on a "date"-
# role column, counts as a bounding filter for the large-table rule below.
_GEO_BOUNDING_COLUMNS = {"zone", "circle", "division", "sub_division", "discom_code", "cluster"}

# Tables that are a genuine CUBE over these columns (outermost to innermost)
# -- see _check_hierarchy_cube_scoping's own docstring for why an unpinned
# ancestor silently multiplies the true count instead of just returning 0 rows.
_HIERARCHY_CUBE_TABLES: dict[str, tuple[str, ...]] = {
    "hierarchy_summary_autoindexing": ("project", "discom", "zone", "circle", "division"),
}


# Tables whose count columns are RUNNING totals per place (each month's row already
# includes every earlier month). Verified live: "utilization in Kashi" as SUM over
# gold_inventory_monthly_balance gave 18.8M supplied / 92.03% instead of 1.38M / 98.12%.
_CUMULATIVE_TABLES: dict[str, set[str]] = {
    "gold_inventory_monthly_balance": {"inventory_count", "installed_count", "stock_consumed_count"},
}


def _check_cumulative_sums(root: exp.Expression, cte_names: set[str]) -> str | None:
    """A SUM/AVG over a running-total column read straight from its table adds every
    month's cumulative figure together. Only the latest row per place (picked first in
    a subquery/CTE with DISTINCT ON or ROW_NUMBER) may be summed."""
    for select in root.find_all(exp.Select):
        sources = [(select.args.get("from_") or select.args.get("from"))]
        sources += [j for j in select.args.get("joins") or []]
        direct = {s.this.name.lower() for s in sources
                  if s is not None and isinstance(s.this, exp.Table)} - cte_names
        for table in direct & set(_CUMULATIVE_TABLES):
            for agg in select.find_all(exp.Sum, exp.Avg):
                if isinstance(agg.parent, exp.Window) or agg.find_ancestor(exp.Select) is not select:
                    continue
                if {c.name.lower() for c in agg.find_all(exp.Column)} & _CUMULATIVE_TABLES[table]:
                    return (
                        f"{table} stores RUNNING (cumulative) totals per place, so summing its rows adds the "
                        f"same meters once per month. For supplied / installed / utilization / stock totals use "
                        f"public.gold_meter_inventory_event_day_agg (or _month_agg) with period_start <= the as-of "
                        f"date. Use {table} only for a balance as of a month: pick the LATEST row per (project, "
                        f"discom, zone, circle, division, sub_division, category, meter_type) with period_start <= "
                        f"that month in a subquery (DISTINCT ON / ROW_NUMBER), then SUM those rows."
                    )
    return None


def _single_row(select: exp.Expression, ctes: dict[str, exp.Expression], seen: frozenset = frozenset()) -> bool:
    """True when the SELECT can return at most one row: LIMIT 1, an aggregate without
    GROUP BY, or nothing but single-row sources (constants, one-row CTEs)."""
    if not isinstance(select, exp.Select):
        return False
    limit = select.args.get("limit")
    if limit is not None and isinstance(limit.expression, exp.Literal) and limit.expression.name == "1":
        return True
    if select.args.get("group"):
        return False
    if any(e.find(exp.AggFunc) and not e.find(exp.Window) for e in select.expressions):
        return True
    return all(_single_source(s, ctes, seen) for s in _sources(select))


def _sources(select: exp.Select) -> list[exp.Expression]:
    src = select.args.get("from_") or select.args.get("from")
    return ([src.this] if src is not None else []) + [j.this for j in select.args.get("joins") or []]


def _single_source(rel: exp.Expression, ctes: dict[str, exp.Expression], seen: frozenset) -> bool:
    if isinstance(rel, exp.Subquery):
        return _single_row(rel.this, ctes, seen)
    if isinstance(rel, exp.Table) and rel.name.lower() in ctes and rel.name.lower() not in seen:
        name = rel.name.lower()
        return _single_row(ctes[name], ctes, seen | {name})
    return False  # a real table, a function (generate_series), a UNION ...


def _check_cross_join_fanout(root: exp.Expression) -> str | None:
    """Two multi-row sets side by side with no join condition (FROM a, b / CROSS JOIN)
    pair every row with every row. Verified live: 'mi pace vs stock category wise' ran
    FROM agg, pace, days_elapsed -- 3 categories x 3 = 9 rows, each category shown with
    every other category's installs. A one-row set (params, a total, LIMIT 1) is fine."""
    ctes = {c.alias_or_name.lower(): c.this for c in root.find_all(exp.CTE)}
    for select in root.find_all(exp.Select):
        src = select.args.get("from_") or select.args.get("from")
        joins = select.args.get("joins") or []
        # CROSS JOIN LATERAL (VALUES ...) / UNNEST is an unpivot (one row per stage per zone),
        # not two independent sets -- seen live as a false rejection of 'journey per step zone wise'.
        loose = [j for j in joins if j.args.get("on") is None and not j.args.get("using")
                 and (j.args.get("kind") or "").upper() not in ("LATERAL",) and not j.args.get("side")
                 and not isinstance(j.this, (exp.Lateral, exp.Values, exp.Unnest))
                 and not (isinstance(j.this, exp.Subquery) and isinstance(j.this.this, exp.Values))]
        if src is None or not loose:
            continue
        multi = [r for r in [src.this, *(j.this for j in loose)] if not _single_source(r, ctes, frozenset())]
        # FROM a, b WHERE a.k = b.k is a join written in WHERE: link relations an equality connects.
        group = {r.alias_or_name.lower(): r.alias_or_name.lower() for r in multi}

        def top(a: str) -> str:
            while group[a] != a:
                a = group[a]
            return a
        where = select.args.get("where")
        for eq in (where.find_all(exp.EQ) if where is not None else []):
            left, right = eq.this, eq.expression
            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                lt, rt = left.table.lower(), right.table.lower()
                if not lt or not rt:  # unqualified a = b between two sets: cannot tell -- trust it
                    group = {k: "*" for k in group} | {"*": "*"}
                    break
                if lt in group and rt in group:
                    group[top(lt)] = top(rt)
        if len({top(r.alias_or_name.lower()) for r in multi}) >= 2:
            names = ", ".join(r.alias_or_name for r in multi)
            return (f"{names} are joined with no join condition (FROM a, b / CROSS JOIN) and each has several rows, "
                    f"so every row is paired with every other row and the result repeats each group with the "
                    f"other groups' numbers. Join them ON their shared key (e.g. USING (category)), or compute "
                    f"everything in one SELECT with GROUP BY.")
    return None


def _column_hint(error: str, referenced: set[str], schema: dict[str, dict[str, str]]) -> str:
    """Which tables of the query lack / have the unknown column -- the bare parser error sent
    two repairs in a row to the same zone filter on a table with no zone column."""
    m = re.search(r"Column '(\w+)'", error)
    if not m:
        return ""
    col = m.group(1).lower()
    lacking = sorted(t for t in referenced if t in schema and col not in schema[t])
    having = sorted(t for t in referenced if t in schema and col in schema[t])
    if not lacking:
        return ""
    hint = f" -- '{col}' is not a column of {', '.join(lacking)}"
    if having:
        hint += (f"; only {', '.join(having)} has it, so filter '{col}' inside that table's own SELECT (before "
                 f"any UNION), and answer the other table's part at the level it does have (e.g. its project)")
    return hint + "."


def _check_grouping_level(root: exp.Expression) -> str | None:
    """GROUPING(x) only works in the SELECT whose own GROUP BY has x. Seen live: a CTE did
    GROUP BY ROLLUP (category) and the outer SELECT called GROUPING(category) -- Postgres error,
    and the repair wrote the same SQL again."""
    for g in root.find_all(exp.Grouping):
        sel = g.find_ancestor(exp.Select)
        if sel is not None and not sel.args.get("group"):
            return ("GROUPING(...) is used in a SELECT that has no GROUP BY of its own -- it only works in the SELECT "
                    "that does the GROUP BY ROLLUP. Label the Total row there (CASE WHEN GROUPING(x) = 1 THEN 'Total' "
                    "...), or in an outer query test the column itself (x IS NULL) instead of GROUPING(x).")
    return None


# Gold tables holding three complete roll-up levels per row key (gmr-etl cb13e5d): category x meter_type rows,
# per-category rows (meter_type = 'ALL') and all-category rows (category = 'ALL', meter_type NULL).
_ROLLUP_TABLES = {"gold_mi_technician_productivity_day_agg", "gold_mi_technician_productivity_month_agg",
                  "gold_mi_productivity_trend_month_agg"}
_ROLLUP_FIX = ("Read exactly one level: all categories = category = 'ALL' (no meter_type filter); one category = "
               "category = 'DT' AND meter_type = 'ALL'; one meter type = category <> 'ALL' AND meter_type = "
               "'1PH-Consumer_meter'.")


_CATEGORY_TYPES = {"dt": {"ltct"}, "feeder": {"htct"}}


def _check_rollup_level(root: exp.Expression) -> str | None:
    """MI productivity tables: a SUM over more than one roll-up level counts every install twice (live, 23 Oct:
    Kashi Aug 2026 shown as 35,738 installs instead of 17,869), and category = 'ALL' AND meter_type = 'ALL'
    matches nothing (zero rows). The prompt rules alone did not hold -- the planner kept writing both."""
    sums = root.find(exp.Sum) is not None
    for sel in root.find_all(exp.Select):
        # This SELECT's own FROM / JOIN tables (a subquery's tables have a nearer SELECT ancestor).
        own = {t.name.lower() for t in sel.find_all(exp.Table) if t.find_ancestor(exp.Select) is sel}
        if not own & _ROLLUP_TABLES:
            continue
        where = sel.args.get("where")
        text = where.sql(dialect="postgres").lower() if where is not None else ""
        if " or " in text and "meter_type" in text:
            # Two levels on purpose: fine only when they stay apart -- GROUP BY category (one row per level) or a
            # FILTER per measure. Live k06: (category = 'ALL' OR meter_type = 'ALL') GROUP BY zone summed both (2x).
            group = sel.args.get("group")
            by_category = group is not None and any(c.name.lower() == "category" for c in group.find_all(exp.Column))
            if not (by_category or sel.find(exp.Filter) is not None) and sums:
                return ("(category = 'ALL' OR meter_type = 'ALL') reads two roll-up levels; without GROUP BY category or "
                        "a FILTER (WHERE category = 'ALL') on each SUM every install counts twice. " + _ROLLUP_FIX)
            continue
        # The trend cube holds project = 'ALL' rows beside every real project's rows: without a project filter both
        # are summed (live 26 Oct: discom-wise Sep 2026 = 135,306 installs, twice the real 67,653).
        if "gold_mi_productivity_trend_month_agg" in own:
            if not re.search(r"\bproject\)?\s*(?:=|in\b)", text):
                return ("gold_mi_productivity_trend_month_agg needs project = 'ALL' (program-wide, also with a zone / "
                        "circle / division scope) or LOWER(project) = 'x': it holds both, so without the filter every "
                        "install counts twice.")
            # Each install sits once per (filter scope x aggregate scope) pair: without both filters it is summed up
            # to 35 times (live 26 Oct: 'circle wise in Azamgarh zone' drafts left filter_level out).
            missing = [c for c in ("aggregate_level", "filter_level") if not re.search(rf"\b{c}\)?\s*(?:=|in\b)", text)]
            if missing:
                return (f"gold_mi_productivity_trend_month_agg needs {' and '.join(missing)}: every install is repeated "
                        "per scope. Program-wide = aggregate_level = 'all' AND filter_level = 'all'; inside a place = "
                        "filter_level = 'zone' AND filter_label = 'NAME'; 'X wise' = aggregate_level = 'x' GROUP BY label.")
        cat_all = re.search(r"\bcategory\)?\s*=\s*'all'", text)
        mt_all = re.search(r"\bmeter_type\)?\s*=\s*'all'", text)
        group = sel.args.get("group")
        by_category = group is not None and any(c.name.lower() == "category" for c in group.find_all(exp.Column))
        if (mt_all and sums and not by_category and sel.find(exp.Filter) is None
                and re.search(r"\bcategory\)?\s*(?:<>|!=)\s*'all'", text)):
            # Live 26 Oct: a division total from the three per-category roll-ups -> 5.71 instead of 5.73 (a technician
            # fitting consumer and DT meters on one day counted as two technician-days).
            return ("category <> 'ALL' AND meter_type = 'ALL' adds the three per-category roll-ups: a technician working "
                    "two categories on one day counts as two technician-days. For an all-category figure use "
                    "category = 'ALL'; keep the per-category rows only with GROUP BY category.")
        if cat_all and mt_all:
            return ("category = 'ALL' rows have meter_type NULL, so adding meter_type = 'ALL' returns no rows. "
                    + _ROLLUP_FIX)
        # Meter types live in fixed categories; live 23 Oct: '3PH productivity' -> category = 'DT' AND
        # meter_type = '3PH-Consumer_meter' -> zero rows reported as 'no data'.
        cat = re.search(r"\bcategory\)?\s*=\s*'(dt|feeder)'", text)
        mt = re.search(r"\bmeter_type\)?\s*=\s*'(1ph|3ph|ltct|htct)-consumer_meter'", text)
        if cat and mt and mt.group(1) not in _CATEGORY_TYPES[cat.group(1)]:
            return (f"No {cat.group(1).upper()} rows have meter_type '{mt.group(1).upper()}-Consumer_meter' (1PH / 3PH "
                    "are CONSUMER only; LTCT is CONSUMER or DT; HTCT is CONSUMER or FEEDER), so this returns no rows. "
                    "A meter type alone = category <> 'ALL' AND meter_type = 'X-Consumer_meter'.")
        if not sums or "meter_type" in text or cat_all:
            continue
        if re.search(r"\bcategory\)?\s*(?:=|in)\s*\(?\s*'(?:consumer|dt|feeder)'", text):
            return ("One category must also filter meter_type = 'ALL' (its roll-up row): without it the meter-type "
                    "rows and the roll-up are both summed and every install counts twice. " + _ROLLUP_FIX)
        if re.search(r"\bcategory\)?\s*(?:<>|!=)\s*'all'", text):
            return ("category <> 'ALL' alone sums the meter-type rows AND the per-category roll-ups -- every "
                    "install counts twice. " + _ROLLUP_FIX)
        if "category" not in text:
            return ("No category filter: the all-category, per-category and meter-type roll-up rows are all summed "
                    "(installs counted three times). " + _ROLLUP_FIX)
    return None


_PERIOD_BOUND = re.compile(
    r"\bperiod_start\)?\s*(?:(between)\s+(.+?)\s+and\s+(.+?)(?=\s+and\s|\s+or\s|\)|$)|(>=|<=|>|<|=)\s*(.+?)(?=\s+and\s|\s+or\s|\)\s*(?:and|or|$)|$))")
_DATE_LIT = re.compile(r"'(\d{4})-(\d{2})-(\d{2})'")


def _month_aligned(bound: str, op: str) -> bool:
    """Does this period_start bound fall on a month boundary (so a month row is wholly in or out)?"""
    if "current_date" in bound or "now()" in bound or "current_timestamp" in bound:
        # An upper bound at today is 'as of now' (the current month row is month-to-date); a today-relative
        # LOWER bound is a rolling window and must be month-truncated.
        return op in ("<=", "<") or "date_trunc('month'" in bound.replace('"', "'") or "date_trunc(month" in bound
    m = _DATE_LIT.search(bound)
    if not m:
        return True  # not a literal / today-relative bound -- leave it
    y, mo, d = (int(g) for g in m.groups())
    if op in (">=", "<", "="):
        return d == 1
    import calendar
    # '<=' / '>': a month end, or the 1st (rows are dated the 1st, so '<= 2026-09-01' = through September)
    return d == 1 or d == calendar.monthrange(y, mo)[1]


def _check_month_table_window(root: exp.Expression) -> str | None:
    """*_month_agg tables hold one row per whole month dated the 1st, so a mid-month or rolling bound
    keeps or drops WHOLE months. Live, 23 Oct: 'oem wise meters installed in last 2 months' filtered
    gold_mi_installation_month_agg with period_start BETWEEN CURRENT_DATE - 1 - INTERVAL '2 months' AND
    CURRENT_DATE - 1 -> only the Sep and Oct rows (140,175) instead of 22 Aug .. 22 Oct (156,382).
    The prompt rules and a verified day-table example alone did not hold."""
    for sel in root.find_all(exp.Select):
        sources = [sel.args.get("from_") or sel.args.get("from"), *(sel.args.get("joins") or [])]
        own = {t.name.lower() for s in sources if s is not None for t in s.find_all(exp.Table)
               if t.find_ancestor(exp.Select) is sel}
        month_tables = sorted(t for t in own if t.endswith("_month_agg"))
        where = sel.args.get("where")
        if not month_tables or where is None:
            continue
        text = where.sql(dialect="postgres").lower()
        for m in _PERIOD_BOUND.finditer(text):
            if m.group(1):
                bad = not (_month_aligned(m.group(2), ">=") and _month_aligned(m.group(3), "<="))
            else:
                bad = not _month_aligned(m.group(5), m.group(4))
            if bad:
                day = month_tables[0].replace("_month_agg", "_day_agg")
                return (f"{month_tables[0]} has ONE row per whole month (period_start = the 1st), so this "
                        f"period_start bound ({m.group(0).strip()}) counts or drops whole months. A range that does "
                        f"not start on the 1st and end on a month end (e.g. a rolling 'last N months' or a "
                        f"mid-month date) must read {day} with the same filters.")
    return None


def _check_meter_type_column(root: exp.Expression) -> str | None:
    """Consumer meter type on the installation tables is inventory_meter_type. progress_meter_type (the
    dashboard's narrow map) misses 6 raw codes, so its 4 types fall short of the consumer total (1,012 meters
    all time). Live, 26 Oct: the same kind of ask used either column from run to run -- 3PH / HTCT came
    back on the narrow map, 1PH / LTCT on the full one."""
    tables = {t.name.lower() for t in root.find_all(exp.Table)}
    if not any(t.startswith("gold_mi_installation_") for t in tables):
        return None
    if any(c.name.lower() == "progress_meter_type" for c in root.find_all(exp.Column)):
        return ("Use inventory_meter_type for consumer meter types (values '1PH-Consumer_meter', "
                "'3PH-Consumer_meter', 'LTCT-Consumer_meter', 'HTCT-Consumer_meter'), never progress_meter_type: "
                "the progress map leaves some meter codes out, so its types do not add up to the consumer total.")
    return None


def _fail(reason: str) -> dict:
    return {
        "ok": False, "reason": reason, "verification_required": "normal", "normalized_sql": None,
        "data_engine": None, "implicit_all_time_scope": False,
    }


def _case_insensitive_string_filters(root: exp.Select, bool_columns: set[str]) -> exp.Select:
    """Same motivation as sql_guard.py's own version (a filter's literal
    case doesn't reliably match the source data's case), rewritten for
    Presto: wraps both sides of a string `=`/`!=` in LOWER(...) instead of
    appending `COLLATE NOCASE` (SQLite-only syntax, not valid Presto SQL).

    `bool_columns` (bare column names known to be boolean-typed among the
    tables this query references) are skipped -- verified live: a plan
    correctly comparing `is_non_sat_pending = 'True'` (valid Postgres --
    a string literal implicitly casts to boolean in a boolean comparison)
    got rewritten into `LOWER(is_non_sat_pending) = LOWER('True')`, which
    Postgres rejects outright ("function lower(boolean) does not exist")
    -- LOWER() requires an explicit text argument, and Postgres does NOT
    implicitly cast a boolean COLUMN to text as a function argument (only
    the literal, and only in the narrower context of a direct comparison).
    The original, unwrapped comparison already works correctly as-is, so
    for a boolean column this only needs to skip the rewrite, not replace
    it with anything."""
    for node in root.find_all((exp.EQ, exp.NEQ)):
        left, right = node.this, node.expression
        if isinstance(left, exp.Column) and left.name in bool_columns:
            continue
        if isinstance(right, exp.Column) and right.name in bool_columns:
            continue
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal) and right.is_string:
            node.set("this", exp.Lower(this=left.copy()))
            node.set("expression", exp.Lower(this=right.copy()))
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal) and left.is_string:
            node.set("this", exp.Lower(this=left.copy()))
            node.set("expression", exp.Lower(this=right.copy()))
    return root


def _resolver(alias_map: dict[str, str], tables_meta: dict[str, object]):
    def resolve(col: exp.Column) -> tuple[str, str] | None:
        table = alias_map.get(col.table, col.table)
        table = table.lower() if table else table
        if table not in tables_meta:
            return None
        return table, col.name
    return resolve


# A comparison that can pin a column to a value (as opposed to IS NOT NULL / !=).
_NARROWING_NODES = (exp.EQ, exp.In, exp.Between, exp.GT, exp.GTE, exp.LT, exp.LTE)


def _is_literal_ish(o: exp.Expression | None) -> bool:
    # A date literal comes back from qualify() wrapped as CAST('2026-06-01' AS
    # DATE), not a bare Literal -- verified directly (BETWEEN DATE '...' AND
    # DATE '...' was being rejected as "not a value pin" before this check
    # unwrapped the cast).
    if isinstance(o, exp.Literal) or (isinstance(o, exp.Cast) and isinstance(o.this, exp.Literal)):
        return True
    # A scalar MAX/MIN subquery pins one value just as a literal does: the mapping audit's "latest month" is
    # data_date = (SELECT MAX(data_date) FROM hes_gold.mapping_audit_summary), so the month never goes stale.
    if isinstance(o, exp.Subquery) and isinstance(o.this, exp.Select):
        projections = o.this.expressions
        return len(projections) == 1 and isinstance(projections[0].unalias(), (exp.Max, exp.Min))
    return False


def _check_bounding_filter(qualified: exp.Select, referenced_tables: set[str], tables_meta: dict, resolve) -> str | None:
    """Returns a failure reason, or None if every LARGE table touched by an
    aggregate has a qualifying bounding filter. Checked independently of
    the dimension_name-pinning rule on the SQLite side (a different table
    shape, a different reason to require a filter) -- this is purely about
    Athena scan cost/latency, not correctness."""
    if not has_aggregate(qualified):
        return None

    large_tables = {name for name in referenced_tables if tables_meta[name].is_large}
    if not large_tables:
        return None

    # A table joined to any other referenced table is exempt from needing
    # its OWN direct bounding filter -- the join's equality condition
    # already narrows it to matching rows (e.g. a large dim_meter joined by
    # meter_no to an already date-filtered billing table never scans
    # dim_meter unfiltered in practice). Still requires the OTHER table in
    # that join to itself be safely scannable -- a two-large-table join
    # with NEITHER side filtered nor any third anchor is exactly the
    # unbounded case this rule exists to catch, so it's still rejected below
    # if nothing in the whole query ends up filtered at all.
    joined_tables: set[str] = set()
    for join in qualified.args.get("joins") or []:
        on = join.args.get("on")
        if on is None:
            continue
        for eq in on.find_all(exp.EQ):
            left, right = resolve(eq.this) if isinstance(eq.this, exp.Column) else None, \
                          resolve(eq.expression) if isinstance(eq.expression, exp.Column) else None
            if left and right:
                joined_tables.add(left[0])
                joined_tables.add(right[0])

    # Only the WHERE clause is checked here -- a JOIN's own ON condition is
    # a column-to-column equality (handled separately above, via
    # joined_tables), not a comparison against a fixed value, so it proves
    # nothing about scan cost on its own and is deliberately excluded from
    # this check to avoid a false "bounded" verdict from an ordinary
    # zone-to-zone join condition.
    where = qualified.args.get("where")

    # A column merely being MENTIONED in the WHERE clause proves nothing
    # about scan cost -- "zone IS NOT NULL" or "zone != 'x'" references
    # `zone` but narrows nothing close to the way "zone = 'x'" does.
    # Verified this distinction matters directly: a query using exactly
    # that shape passed an earlier version of this check (which scanned
    # for ANY Column reference anywhere in the condition tree) and then
    # ran an unfiltered COUNT(*) over the full unified_installations
    # table. Only counted as bounding now: a real narrowing comparison
    # against a LITERAL -- equality, IN, BETWEEN, or a range comparison
    # (>, >=, <, <=) -- never a comparison between two columns (that's a
    # join-shaped condition, not a value pin).
    bounded_tables: set[str] = set()
    any_direct_filter = False
    if where is not None:
        for node in where.find_all(_NARROWING_NODES):
            if isinstance(node, exp.In):
                operands = [node.this, *(node.args.get("expressions") or [])]
            elif isinstance(node, exp.Between):
                operands = [node.this, node.args.get("low"), node.args.get("high")]
            else:  # EQ, GT, GTE, LT, LTE all have this/expression
                operands = [node.this, node.args.get("expression")]
            cols = [o for o in operands if isinstance(o, exp.Column)]
            literals = [o for o in operands if _is_literal_ish(o)]
            if not cols or not literals:
                continue  # column-to-column (or otherwise non-literal) comparison -- not a value pin
            for col in cols:
                resolved = resolve(col)
                if resolved is None:
                    continue
                table, column = resolved
                col_meta = next((c for c in tables_meta[table].columns if c.column_name == column), None)
                if col_meta and (col_meta.role == "date" or column in _GEO_BOUNDING_COLUMNS):
                    bounded_tables.add(table)
                    any_direct_filter = True

    if joined_tables and any_direct_filter:
        # At least one table in the join graph is directly bounded -- treat
        # every table it's joined to as covered too (see comment above).
        bounded_tables |= joined_tables

    unbounded = sorted(large_tables - bounded_tables)
    if unbounded:
        return (
            f'{", ".join(unbounded)} {"is a large table" if len(unbounded) == 1 else "are large tables"} '
            f"(500K+ rows) with no bounding filter -- an aggregate over "
            f"{'it' if len(unbounded) == 1 else 'them'} with no WHERE/JOIN-ON filter on a date "
            f"column or a geo-hierarchy column ({', '.join(sorted(_GEO_BOUNDING_COLUMNS))}) would "
            f"scan the whole table. Add a date range or a zone/circle/division filter."
        )
    return None


def _detect_implicit_all_time_scope(qualified: exp.Select, referenced_tables: set[str], tables_meta: dict, resolve) -> bool:
    """True when the query aggregates (SUM/AVG/MAX/MIN/COUNT) over a table
    that HAS a date-role column, but that column is neither in GROUP BY nor
    pinned by a direct WHERE filter -- meaning the aggregate silently spans
    every recorded date for that table, not just one period.

    NOT a rejection -- verified live that this is often a perfectly
    legitimate, reliable way to answer a no-date ranking/single_value
    question (e.g. "which feeder has the worst SAIFI" -> MAX(saifi) GROUP
    BY feeder_code across every recorded date, naming that feeder's
    all-time peak). An earlier version of this rule REJECTED this shape,
    on the wrong assumption that it was an unaggregated arbitrary-row pick;
    it isn't, so this only flags it for narrator.py to disclose the
    "across all recorded dates" scope explicitly, never to block the
    query."""
    if not has_aggregate(qualified):
        return False

    group = qualified.args.get("group")
    grouped_cols: set[str] = set()
    for c in (group.expressions if group else []):
        if isinstance(c, exp.Column):
            resolved = resolve(c)
            if resolved:
                grouped_cols.add(resolved[1])

    where = qualified.args.get("where")
    filtered_cols: set[tuple[str, str]] = set()
    if where is not None:
        for node in where.find_all(_NARROWING_NODES):
            for col in node.find_all(exp.Column):
                resolved = resolve(col)
                if resolved:
                    filtered_cols.add(resolved)

    for table in referenced_tables:
        meta = tables_meta.get(table)
        if meta is None:
            continue
        for c in meta.columns:
            if c.role == "date" and c.column_name not in grouped_cols and (table, c.column_name) not in filtered_cols:
                return True
    return False


def _check_hierarchy_cube_scoping(qualified: exp.Select, referenced_tables: set[str], tables_meta: dict, resolve) -> str | None:
    """A table in _HIERARCHY_CUBE_TABLES is a genuine CUBE, not a
    hierarchical rollup: for one real leaf combination, EVERY combination of
    its ancestor columns being their real value or independently NULL'd out
    is ALSO its own stored row, carrying the same numbers duplicated.
    Verified live against hierarchy_summary_autoindexing: filtering only
    division='EDD Ballia Nagar', with no constraint on project/discom/zone/
    circle, matched 16 rows (every one of 2^4 ancestor NULL/real-value
    combinations), not 1 -- an aggregate or headline read off that silently
    multiplied the true figure by 16, and tripped the pipeline's own
    independent-recompute check into a false mismatch.

    A query is only correctly scoped when, reading the declared levels
    outermost-to-innermost, there is a single boundary index: every level up
    to and including it is pinned to a value (EQ/IN against a literal) or
    required "IS NOT NULL", and every level after it is required "IS NULL"
    -- no gap (an unconstrained level before the boundary), and nothing
    pinned/not-null after it. This intentionally requires an explicit "IS
    NULL" on every column below the scoped level even for a plain grand
    total (all 5 NULL) -- verified live, that's the actual stored shape of
    the grand-total row, not an assumption."""
    for table_name, levels in _HIERARCHY_CUBE_TABLES.items():
        if table_name not in referenced_tables:
            continue
        state: dict[str, str] = {level: "free" for level in levels}
        where = qualified.args.get("where")
        if where is not None:
            for node in where.find_all(exp.EQ):
                left, right = node.this, node.args.get("expression")
                left_cols = list(left.find_all(exp.Column)) if left is not None else []
                right_cols = list(right.find_all(exp.Column)) if right is not None else []
                left_has_literal = left is not None and next(left.find_all(exp.Literal), None) is not None
                right_has_literal = right is not None and next(right.find_all(exp.Literal), None) is not None
                cols = left_cols if (left_cols and right_has_literal) else (right_cols if (right_cols and left_has_literal) else [])
                for col in cols:
                    resolved = resolve(col)
                    if resolved and resolved[0] == table_name and resolved[1] in state:
                        state[resolved[1]] = "pinned"
            for node in where.find_all(exp.In):
                for col in node.this.find_all(exp.Column) if node.this is not None else []:
                    resolved = resolve(col)
                    if resolved and resolved[0] == table_name and resolved[1] in state:
                        state[resolved[1]] = "pinned"
            for node in where.find_all(exp.Is):
                col = node.this
                if not isinstance(col, exp.Column):
                    continue
                resolved = resolve(col)
                if not (resolved and resolved[0] == table_name and resolved[1] in state):
                    continue
                state[resolved[1]] = "not_null" if isinstance(node.parent, exp.Not) else "null"

        constrained_idxs = [i for i, lvl in enumerate(levels) if state[lvl] in ("pinned", "not_null")]
        boundary = max(constrained_idxs) if constrained_idxs else -1
        problems: list[str] = []
        for i, lvl in enumerate(levels):
            # "project" is NOT a strict ancestor of discom/zone/circle/division the
            # way those 4 nest inside each other -- verified live: the single
            # discom "PuVVNL" has real, DIFFERENT rows under BOTH project='Kashi'
            # AND project='Triveni' (Kashi's PuVVNL total + Triveni's PuVVNL total
            # == the project-IS-NULL PuVVNL row, exactly). So it gets its own rule
            # instead of the generic boundary check below: "pinned" (a specific
            # project named) and "null" (the real, already-computed combined-
            # across-projects total) are always fine; bare "IS NOT NULL" is only
            # fine when project ITSELF is the scoped/broken-down level (boundary
            # == 0) -- as an ancestor filler for some OTHER level (boundary > 0)
            # it's ambiguous, silently picking one of several different real
            # totals instead of either one specific project or the combined one.
            if lvl == "project":
                if state[lvl] in ("pinned", "null") or (state[lvl] == "not_null" and boundary == 0):
                    continue
                problems.append(
                    '"project" must be pinned to a specific value (e.g. project=\'Kashi\') or explicitly '
                    '"project IS NULL" here, not a bare "IS NOT NULL" -- a single discom/zone/circle/division '
                    "can span more than one project (verified live for discom \"PuVVNL\": it has separate real "
                    "totals under BOTH project='Kashi' and project='Triveni'), so leaving project merely "
                    "\"NOT NULL\" ambiguously picks one of several different real answers instead of either "
                    "one specific project's slice or the true project-agnostic combined total."
                )
                continue
            if i <= boundary and state[lvl] not in ("pinned", "not_null"):
                problems.append(f'"{lvl}" needs a value filter or "{lvl} IS NOT NULL" (currently unconstrained)')
            elif i > boundary and state[lvl] != "null":
                current = "unconstrained" if state[lvl] == "free" else f'"{state[lvl]}"'
                problems.append(f'"{lvl}" needs "{lvl} IS NULL" ({current} right now)')
        if problems:
            return (
                f"{table_name} is a CUBE -- reading project>discom>zone>circle>division, every level up to your "
                f"scoped one must be pinned to a value or IS NOT NULL with no gap, and every level after it must "
                f"be IS NULL, or duplicate rollup rows get counted on top of the real one. Fix: {'; '.join(problems)}."
            )
    return None


def check_sql_safety(sql: str, catalog: AthenaCatalog) -> dict:
    """Safety-only check for the LLM pipeline (gmr_agents/llm_pipeline.py):
    one read-only SELECT, no write/DDL nodes, only catalog tables and
    columns, one engine. It makes NO business decisions (no role rules, no
    bounding-filter or join policies) and never rewrites the query -- the
    LLM's SQL runs exactly as written. Same return shape as check_sql()."""
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        return _fail("Only a single statement is allowed.")
    tables_meta = {t.table_name: t for t in catalog.list_tables()}
    # Find the engine from the table names, then validate in that engine's dialect.
    # Postgres-only syntax (SUM(x) FILTER (WHERE ...), ::casts) does not parse as
    # Presto, so try both dialects just to read the table names.
    names = set()
    for probe in (DIALECT, "postgres"):
        try:
            statements = sqlglot.parse(stripped, dialect=probe)
        except Exception:
            continue
        if statements and statements[0] is not None:
            names = {t.name.lower() for t in statements[0].find_all(exp.Table)}
            if names:
                break
    if not names:
        # Neither dialect could parse it: still read the table names, so the error names the
        # right engine and says what is wrong.
        names = {m.lower() for m in re.findall(rf"\b{SCHEMA_RE}\.\"?([A-Za-z0-9_]+)", stripped)}
        if not re.match(r"\s*(select|with|\()", stripped, re.IGNORECASE):
            return _fail(f"The SQL must start with SELECT or WITH (it starts with '{stripped[:30]}').")
    engines = {tables_meta[n].engine for n in names if n in tables_meta}
    if len(engines) > 1:
        return _fail("This query mixes postgres and athena tables -- one query can use only one engine. Put the "
                     "postgres part in \"sql\" and the athena part in \"extra_queries\" (each on its own engine); "
                     "the answer combines their results.")
    engine = next(iter(engines), "athena")
    # No known table (a misspelt / invented name): the engine is unknown, so try both
    # dialects -- the repair hint then names the unknown table, not a misleading parse error.
    dialects = [("postgres" if engine == "postgres" else DIALECT)] if engines else [DIALECT, "postgres"]
    statements, parse_error = None, None
    for dialect in dialects:
        try:
            statements = sqlglot.parse(stripped, dialect=dialect)
            break
        except Exception as exc:
            parse_error = parse_error or exc
    if statements is None:
        return _fail(f"Could not parse SQL ({engine} syntax): {parse_error}")
    if len(statements) != 1 or statements[0] is None:
        return _fail("Only a single SELECT statement is allowed.")
    root = statements[0]
    if not isinstance(root, (exp.Select, exp.Union)):
        return _fail(f"Only SELECT is allowed (got {type(root).__name__}).")
    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return _fail(f"Disallowed statement type: {type(node).__name__}.")

    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    referenced = {t.name.lower() for t in root.find_all(exp.Table)} - cte_names
    unknown = referenced - set(tables_meta)
    if unknown:
        return _fail(f"Unknown table(s): {', '.join(sorted(unknown))}. Use only tables listed in SCHEMA.")
    schema = {name: {c.column_name: c.sql_type for c in t.columns} for name, t in tables_meta.items()}
    try:
        qualify(root.copy(), schema=schema, dialect=dialect)
    except Exception as exc:
        return _fail(f"Unknown or ambiguous column reference: {exc}{_column_hint(str(exc), referenced, schema)}")
    cumulative_reason = (_check_cumulative_sums(root, cte_names) or _check_cross_join_fanout(root)
                         or _check_grouping_level(root) or _check_rollup_level(root)
                         or _check_month_table_window(root) or _check_meter_type_column(root))
    if cumulative_reason:
        return _fail(cumulative_reason)
    return {
        "ok": True, "reason": None, "verification_required": "normal",
        "normalized_sql": stripped, "data_engine": engine, "implicit_all_time_scope": False,
    }

