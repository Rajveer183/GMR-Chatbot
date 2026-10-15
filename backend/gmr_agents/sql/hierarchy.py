"""Place hierarchy repair for Postgres gold queries (no LLM).

1. Orphans: a few gold rows have a blank parent level although their child level is
   known elsewhere in the same data -- sub-division '2614221-EDSD CHOMUA' with no
   division (it is 'EDD KOSI MATHURA' on every other row), division 'EDD JAUNPUR
   NAGAR' with no circle ('EDC I JAUNPUR'). A breakdown then showed them as
   'Unmapped' although they belong to a zone / circle / division. `refresh()` learns
   child -> parent from the data itself (only children with exactly ONE known
   parent), and `rewrite()` fills the blank parent before the query runs:
       division  ->  COALESCE(NULLIF(TRIM(division), ''), CASE LOWER(TRIM(sub_division)) WHEN ... END)
   in every SELECT that reads a gold table directly. If the rewritten SQL fails, the
   caller runs the original (sql/executor.py), so a rewrite can never break an answer.

2. Stock: meters still in stock have no zone / circle / division (a meter gets its place
   when it is installed) -- 119,589 meters on 2026-10-04, all of today's stock. On the
   inventory tables a blank place is therefore 'Not installed yet (in stock)', never
   'Unmapped' (`relabel_stock`).

3. Stock -> zone: stock sits in a warehouse, and every warehouse is named after a district
   whose circle sits in a zone (BALLIA_SWH -> 'edc ballia' -> Azamgarh, Varanasi_HW ->
   'edc varanasi' / 'eudc i varanasi' -> Varanasi I). `refresh()` learns warehouse -> zone
   from that name match, weighted by each zone's installs on the matching circles (a zone
   under 5% is dropped; Prayagraj_HW matches both Prayagraj zones and is split by share).
   A query that uses `zone` on gold_meter_inventory_event_day_agg / _month_agg then reads
   the table with its no-zone stock rows swapped for the same meters from
   gold_inventory_unutilized_stock_snapshot carrying their warehouse's zone -- verified
   2026-10-04: identical at (project, category, meter_type, date) grain, 0 mismatches,
   so totals never change; only zone-wise supplied / stock / utilization gain the stock.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict

import pandas as pd
import sqlglot
from sqlglot import exp

log = logging.getLogger(__name__)

# (parent, child): a blank parent is filled from the child
_LEVELS = [("zone", "circle"), ("circle", "division"), ("division", "sub_division")]
_REFRESH_SECONDS = 3600
_STATE: dict = {"map": {}, "stock_zones": [], "places": {}, "at": 0.0, "running": False}
_EVENT_TABLES = {"gold_meter_inventory_event_day_agg", "gold_meter_inventory_event_month_agg"}
_WAREHOUSE_SUFFIX_RE = re.compile(r"[\s_-]*(?:hw|zw|swh|wh|warehouse|store)$", re.I)
_MIN_ZONE_SHARE = 0.05
_FIRST_LOAD_WAIT_SECONDS = 30
_LOCK = threading.Lock()

IN_STOCK = "Not installed yet (in stock)"
_STOCK_TABLE_RE = re.compile(r"\bgold_(?:meter_inventory_event_\w+|inventory_monthly_balance)\b", re.I)
_PLACE_COL_RE = re.compile(r"(?:^|_)(?:zone|circle|division|sub_division|subdivision)(?:_name)?$", re.I)


def _learn(rows: list[tuple]) -> dict[tuple[str, str], dict[str, str]]:
    """rows = (zone, circle, division, sub_division), lower-cased, None for blank.
    Returns {(parent, child): {child value: parent value}} for orphan children only."""
    idx = {"zone": 0, "circle": 1, "division": 2, "sub_division": 3}
    out: dict[tuple[str, str], dict[str, str]] = {}
    for parent, child in _LEVELS:
        parents, orphans = defaultdict(set), set()
        for r in rows:
            c, p = r[idx[child]], r[idx[parent]]
            if c is None:
                continue
            (orphans.add(c) if p is None else parents[c].add(p))
        fixes = {c: next(iter(parents[c])) for c in orphans if len(parents.get(c, ())) == 1}
        if fixes:
            out[(parent, child)] = fixes
    return out


def _learn_stock_zones(warehouses: list[tuple], circles: list[tuple]) -> list[tuple]:
    """warehouses = (project, warehouse); circles = (project, discom, zone, circle, installs),
    all lower-cased. Returns (project, discom, warehouse, zone, lo, hi) rows: the warehouse's
    stock share (hi - lo) for that zone, the shares of one warehouse covering 0..1."""
    out = []
    project_discom = {p: discom for p, discom, *_ in circles}
    for project, warehouse in warehouses:
        town = _WAREHOUSE_SUFFIX_RE.sub("", warehouse).strip()
        by_zone: dict[str, float] = defaultdict(float)
        discoms: dict[str, str] = {}
        for p, discom, zone, circle, installs in circles:
            if town and p == project and town in re.split(r"[^a-z]+", circle):
                by_zone[zone] += float(installs or 0)
                discoms[zone] = discom
        total = sum(by_zone.values())
        kept = {z: v for z, v in by_zone.items() if total and v / total >= _MIN_ZONE_SHARE}
        if not kept:  # no circle of that name: the stock stays without a zone ('in stock')
            out.append((project, project_discom.get(project), warehouse, None, 0.0, 1.0))
            continue
        lo, kept_total = 0.0, sum(kept.values())
        for i, (zone, v) in enumerate(sorted(kept.items())):
            hi = 1.0 if i == len(kept) - 1 else lo + v / kept_total
            out.append((project, discoms[zone], warehouse, zone, round(lo, 6), round(hi, 6)))
            lo = hi
    return out


def refresh(force: bool = False) -> None:
    """Re-learn the orphan map from every gold table that has all four place levels
    (~15 s, run in the background at startup and hourly)."""
    with _LOCK:
        if _STATE["running"] or (not force and time.time() - _STATE["at"] < _REFRESH_SECONDS):
            return
        _STATE["running"] = True
    try:
        from gmr_agents.sql.connections import pooled_postgres_connection

        with pooled_postgres_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.columns WHERE table_schema = 'public' "
                        "AND table_name LIKE 'gold\\_%' AND column_name IN ('zone','circle','division','sub_division') "
                        "GROUP BY table_name HAVING COUNT(*) = 4")
            tables = [r[0] for r in cur.fetchall()]
            if not tables:
                return
            blank = "LOWER(NULLIF(TRIM({0}), ''))"
            cols = ", ".join(blank.format(c) for c in ("zone", "circle", "division", "sub_division"))
            cur.execute(" UNION ".join(f"SELECT {cols} FROM public.{t}" for t in tables))
            rows = cur.fetchall()
            mapping = _learn(rows)
            places = {v: lvl for r in rows for lvl, v in zip(("zone", "circle", "division"), r) if v}
            cur.execute("SELECT DISTINCT LOWER(discom) FROM public.gold_meter_inventory_event_month_agg WHERE discom IS NOT NULL")
            places.update({r[0]: "discom" for r in cur.fetchall()})
            _STATE["places"] = places
            cur.execute("SELECT DISTINCT LOWER(project), LOWER(warehouse) FROM public.gold_inventory_unutilized_stock_snapshot "
                        "WHERE warehouse IS NOT NULL")
            warehouses = cur.fetchall()
            cur.execute("SELECT LOWER(project), MIN(discom), LOWER(zone), LOWER(circle), SUM(installed_count) "
                        "FROM public.gold_meter_inventory_event_month_agg WHERE zone IS NOT NULL AND circle IS NOT NULL "
                        "GROUP BY 1, 3, 4")
            stock_zones = _learn_stock_zones(warehouses, cur.fetchall())
        _STATE.update(map=mapping, stock_zones=stock_zones, at=time.time())
        log.info("hierarchy: %d orphan place value(s) mapped to their parent level %s",
                 sum(len(v) for v in mapping.values()), {f"{p}<-{c}": v for (p, c), v in mapping.items()})
        log.info("hierarchy: warehouse -> zone %s", [(w, z, round(hi - lo, 3)) for _, _, w, z, lo, hi in stock_zones])
    except Exception:
        log.warning("hierarchy: refresh failed -- queries run unchanged", exc_info=True)
    finally:
        _STATE["running"] = False


def place_names() -> dict[str, str]:
    """Every stored zone / circle / division / discom name (lower-case) -> its level."""
    return dict(_STATE["places"])


def refresh_in_background() -> None:
    if time.time() - _STATE["at"] >= _REFRESH_SECONDS and not _STATE["running"]:
        threading.Thread(target=refresh, name="hierarchy-refresh", daemon=True).start()


def wait_first_load() -> None:
    """Query threads only (blocks). Until the first refresh lands, a zone answer would leave
    stock out -- and the cache pre-warmer asks right after startup, so it would be cached
    for the day."""
    if not _STATE["at"]:
        start = time.time()  # the first second covers a thread that has not set "running" yet
        while not _STATE["at"] and (_STATE["running"] or time.time() < start + 1) \
                and time.time() < start + _FIRST_LOAD_WAIT_SECONDS:
            time.sleep(0.2)


def _quote(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def _filled(col: exp.Column, parent: str, child: str, fixes: dict[str, str]) -> exp.Expression:
    q = f"{col.table}." if col.table else ""
    whens = " ".join(f"WHEN {_quote(c)} THEN {_quote(p)}" for c, p in sorted(fixes.items()))
    return sqlglot.parse_one(f"COALESCE(NULLIF(TRIM({q}{parent}), ''), CASE LOWER(TRIM({q}{child})) {whens} END)",
                             read="postgres")


def _stock_by_zone(table: str, stock_zones: list[tuple]) -> str:
    """The event table with its no-zone stock rows replaced by the same meters from the
    stock snapshot, each placed in its warehouse's zone (split ones by whole meters)."""
    cols = ("period_start, period_end, project, discom, zone, circle, division, sub_division, category, "
            "meter_type, inventory_count, installed_count, stock_consumed_count")
    if table.endswith("_month_agg"):
        start = "date_trunc('month', s.inventory_date)::date"
        end = "(date_trunc('month', s.inventory_date) + INTERVAL '1 month - 1 day')::date"
    else:
        start = end = "s.inventory_date"
    lit = lambda v: "NULL" if v is None else (_quote(v) if isinstance(v, str) else repr(float(v)))  # noqa: E731
    values = ", ".join(f"({', '.join(lit(v) for v in row)})" for row in stock_zones)
    return (
        f"SELECT {cols} FROM public.{table} WHERE zone IS NOT NULL "
        f"UNION ALL "
        f"SELECT {start}, {end}, LOWER(s.project), m.discom, m.zone, NULL, NULL, NULL, s.category, s.meter_type, "
        f"CAST(ROUND(s.stock_count * COALESCE(m.hi, 1)) - ROUND(s.stock_count * COALESCE(m.lo, 0)) AS BIGINT), "
        f"CAST(0 AS BIGINT), CAST(0 AS BIGINT) "
        f"FROM public.gold_inventory_unutilized_stock_snapshot AS s "
        f"LEFT JOIN (VALUES {values}) AS m(project, discom, warehouse, zone, lo, hi) "
        f"ON m.project = LOWER(s.project) AND m.warehouse = LOWER(s.warehouse)"
    )


_STAGE_TABLE = "gold_meter_installation_pending_stage_snapshot"


def _stages_by_zone(stock_zones: list[tuple]) -> str:
    """The pre-installation stage table with discom / zone from its warehouse (catalog
    VIRTUAL_COLUMNS) -- split warehouses by whole meters, like the stock."""
    lit = lambda v: "NULL" if v is None else (_quote(v) if isinstance(v, str) else repr(float(v)))  # noqa: E731
    values = ", ".join(f"({', '.join(lit(v) for v in row)})" for row in stock_zones)
    return (
        "SELECT s.id, s.project, s.warehouse, s.oem, s.category, s.meter_type, s.current_stage, "
        "CAST(ROUND(s.meter_count * COALESCE(m.hi, 1)) - ROUND(s.meter_count * COALESCE(m.lo, 0)) AS BIGINT) AS meter_count, "
        "m.discom, m.zone "
        f"FROM public.{_STAGE_TABLE} AS s "
        f"LEFT JOIN (VALUES {values}) AS m(project, discom, warehouse, zone, lo, hi) "
        "ON m.project = LOWER(s.project) AND m.warehouse = LOWER(s.warehouse)"
    )


def _place_stock(tree: exp.Expression, stock_zones: list[tuple]) -> bool:
    changed = False
    for t in list(tree.find_all(exp.Table)):
        name = t.name.lower()
        if name in _EVENT_TABLES:
            body = _stock_by_zone(name, stock_zones)
        elif name == _STAGE_TABLE:
            body = _stages_by_zone(stock_zones)
        else:
            continue
        alias = t.alias or name
        sub = sqlglot.parse_one(f"SELECT 1 FROM ({body}) AS {alias}", read="postgres")
        t.replace(sub.args["from_"].this)
        changed = True
    return changed


def rewrite(sql: str, mapping: dict | None = None, stock_zones: list[tuple] | None = None) -> str:
    """`sql` with (a) the event tables' stock placed in its warehouse's zone when the query
    uses `zone`, and (b) blank parent levels filled from the orphan map; unchanged when
    there is nothing to do. (b) touches only columns of a SELECT that reads a gold table
    directly (an outer query over a subquery has no child column to read)."""
    mapping = _STATE["map"] if mapping is None else mapping
    stock_zones = _STATE["stock_zones"] if stock_zones is None else stock_zones
    low = sql.lower()
    # The stage table has no zone / discom of its own: any use of them needs the rewrite.
    stage_place = _STAGE_TABLE in low and re.search(r"\b(zone|discom)\b", low)
    if not re.search(r"\b(zone|circle|division)\b", low) and not stage_place:
        return sql
    place_stock = bool(stock_zones) and (stage_place or (re.search(r"\bzone\b", low) and any(
        t in low for t in _EVENT_TABLES)))
    if not mapping and not place_stock:
        return sql
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return sql
    changed = _place_stock(tree, stock_zones) if place_stock else False
    by_parent = {p: (c, fixes) for (p, c), fixes in mapping.items()}
    for select in list(tree.find_all(exp.Select)):
        direct = [t.name.lower() for t in select.find_all(exp.Table) if t.find_ancestor(exp.Select) is select]
        reads_gold = any(n.startswith("gold_") for n in direct)
        # The placed-stock branch (_stock_by_zone) has its zone from the warehouse, no circle.
        if not reads_gold or "gold_inventory_unutilized_stock_snapshot" in direct or _STAGE_TABLE in direct:
            continue
        aliases = {e.alias.lower() for e in select.expressions if isinstance(e, exp.Alias)}
        for col in list(select.find_all(exp.Column)):
            name = col.name.lower()
            if name not in by_parent or col.find_ancestor(exp.Select) is not select:
                continue
            if isinstance(col.parent, exp.Ordered) and name in aliases:
                continue  # a bare ORDER BY <output alias> sorts the label, not the raw column
            child, fixes = by_parent[name]
            filled = _filled(col, name, child, fixes)
            # A bare projection ('SELECT division, ...') keeps its output name.
            col.replace(exp.alias_(filled, col.name) if col.parent is select else filled)
            changed = True
    return tree.sql(dialect="postgres") if changed else sql


# Stored meter_type -> the name the dashboard uses: 4 consumer types (+ 'Other', 710 meters,
# never in stock), DT = 3PLTCTSM, Feeder = HTCTPTSM.
METER_TYPE_LABELS = {
    "1ph-consumer_meter": "1PH", "3ph-consumer_meter": "3PH", "ltct-consumer_meter": "LTCT",
    "htct-consumer_meter": "HTCT", "other-consumer_meter": "Other (consumer)",
    "3pltctsm": "DTR", "htctptsm": "Feeder",
}


def _labels(df: pd.DataFrame) -> list:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def relabel_types(df: pd.DataFrame) -> pd.DataFrame:
    """Raw meter_type codes in any label column -> dashboard names ('3PLTCTSM' -> 'DTR')."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = None
    for c in _labels(df):
        s = df[c]
        if not s.map(lambda v: isinstance(v, str)).any():
            continue
        mapped = s.map(lambda v: METER_TYPE_LABELS.get(v.strip().lower(), v) if isinstance(v, str) else v)
        if not mapped.equals(s):
            out = df.copy() if out is None else out
            out[c] = mapped
    return df if out is None else out


def relabel_total(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """A ROLLUP grand-total row the query left unlabelled (every label blank) is named
    'Total' -- only when its numbers are the sum of the other rows. Verified live: 'remaining
    stock type wise' showed its total as a blank meter type with -31,379."""
    if not isinstance(df, pd.DataFrame) or len(df) < 3 or not re.search(r"\b(rollup|cube|grouping\s+sets)\b", sql or "", re.I):
        return df
    labels = _labels(df)
    nums = [c for c in df.columns if c not in labels and pd.api.types.is_numeric_dtype(df[c])]
    if not labels or not nums:
        return df
    blank = df[labels].apply(lambda s: s.isna() | s.astype(str).str.strip().isin({"", "None", "nan"})).all(axis=1)
    if blank.sum() != 1:
        return df
    i = blank.idxmax()
    rest = df.drop(index=i)[nums[0]].sum()
    if not pd.notna(df.at[i, nums[0]]) or abs(float(df.at[i, nums[0]]) - float(rest)) > 0.5:
        return df
    df = df.copy()
    df[labels[0]] = df[labels[0]].astype(object)
    df.at[i, labels[0]] = "Total"
    return df


def _natural_key(v) -> tuple:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(v)))


def natural_order(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the SQL sorted by a label AS TEXT ('SAT-1, SAT-10, SAT-11, SAT-2') are put in
    number order (SAT-1, SAT-2 ... SAT-12), same direction; a Total row keeps its place.
    Rows in any other order (ranked by a value) are left alone."""
    if not isinstance(df, pd.DataFrame) or len(df) < 3:
        return df
    for c in _labels(df):
        # Not astype(str): it keeps NaN as a float (a ROLLUP grand-total row's empty month), and
        # re.search below then crashed the whole query ('expected string ..., got float').
        vals = df[c].map(lambda v: "" if pd.isna(v) else str(v))
        is_total = vals.str.strip().str.lower().isin({"total", "grand total", "all"})
        body = vals[~is_total].tolist()
        if len(body) < 3 or sum(bool(re.search(r"\d", v)) for v in body) < len(body) - 1:
            continue
        natural = sorted(body, key=_natural_key)
        if natural == body or natural[::-1] == body:
            return df  # already in number order
        if body == sorted(body) or body == sorted(body, reverse=True):
            desc = body != sorted(body)
            rows = df[~is_total].copy()
            rows = rows.iloc[sorted(range(len(rows)), key=lambda i: _natural_key(rows[c].iloc[i]), reverse=desc)]
            totals = df[is_total]
            first = bool(is_total.iloc[0]) if len(is_total) else False
            return pd.concat([totals, rows] if first else [rows, totals]).reset_index(drop=True)
        return df  # first label column is ranked some other way
    return df


# current_stage (where a meter without realized revenue sits now) -> the dashboard's workflow
# stage names (Workflow Lead Time card): a meter with the agency is in 'Agency to Supervisor'.
STAGE_LABELS = {
    "inventory": "DI to Agency", "agency": "Agency to Supervisor", "supervisor": "Supervisor to Technician",
    "technician": "Technician to Installation", "installation_done_sat_pending": "Installation to SAT",
    "sat_done_revenue_pending": "SAT to Revenue",
}


def relabel_stages(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or "_pending_stage_snapshot" not in (sql or "").lower():
        return df
    out = df.copy()
    for c in _labels(df):
        out[c] = df[c].map(lambda v: STAGE_LABELS.get(v.strip().lower(), v) if isinstance(v, str) else v)
    return out


def relabel(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """Every Postgres result: unlabelled ROLLUP total -> 'Total' (first, so it is not taken
    for stock), blank places -> in stock, raw meter types / stages -> dashboard names."""
    return relabel_stages(sql, relabel_types(relabel_stock(sql, relabel_total(sql, df))))


def relabel_stock(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """On the inventory tables a blank / 'Unmapped' zone, circle or division is stock not
    installed yet (it has no place until it is installed)."""
    if not isinstance(df, pd.DataFrame) or df.empty or not _STOCK_TABLE_RE.search(sql or ""):
        return df
    cols = [c for c in df.columns if _PLACE_COL_RE.search(str(c).strip().lower().replace(" ", "_"))
            and not pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return df
    df = df.copy()
    for c in cols:
        s = df[c]
        blank = s.isna() | s.astype(str).str.strip().str.lower().isin({"", "unmapped", "unknown", "none", "null"})
        if blank.any():
            df[c] = s.astype(object).where(~blank, IN_STOCK)
    return df
