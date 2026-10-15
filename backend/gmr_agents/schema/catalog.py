"""The live schema catalog: one AthenaCatalog per domain, built from
Athena's (and, for WFM, Postgres') information_schema over the curated
table shortlist in data/table_metadata.yaml.

Per table: database, engine, row count and columns; per column: SQL type
and a role (date / dimension / identifier / measure / free_text) classified
from the column name and declared type (_classify_role) -- the tables are
far too large to profile values. Postgres dimension columns with few
distinct values also carry their sample values. Joins are DECLARED from
matching column names (_declared_join_candidates), all marked unverified.
schema_brief.py renders the LLM planner's compact schema from it.

Cached to gmr_agents/cache/<domain>_catalog.json. Startup serves the cache
whatever its age (warm_start) and main.py refreshes it in the background:
right away when stale, then every MAX_CACHE_AGE_SECONDS.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from gmr_agents.schema.table_metadata import (
    HEADER_LABEL_OVERRIDES, MDM_TABLES, TABLE_DESCRIPTIONS, WFM_POSTGRES_TABLES,
    WFM_TABLES,
)

log = logging.getLogger(__name__)

# GMR_CATALOG_CACHE_DIR overrides where the cache lives (the golden-snapshot
# eval points it at a frozen copy).
_CACHE_DIR = Path(os.environ.get("GMR_CATALOG_CACHE_DIR") or Path(__file__).resolve().parents[1] / "cache")
# Row counts are independent queries -- run a few at once during a refresh (per
# domain; both domains refresh together, and boto3 pools 10 connections).
_REFRESH_WORKERS = int(os.environ.get("GMR_CATALOG_REFRESH_WORKERS", 4))

# Row-count threshold above which validation/sql_guard.py requires an aggregate to
# carry a bounding filter (a date/data_date predicate, or a geo-hierarchy
# equality filter: zone/circle/division/sub_division/discom_code) before
# it's allowed to run -- Athena bills by bytes scanned, and every one of
# these tables is large enough that an unfiltered aggregate is both slow and
# a real, avoidable cost. 500K is comfortably below every table in both
# shortlists (data/table_metadata.yaml) except the two smallest (low_voltage_pockets ~6K,
# theft_risk_by_area ~7K rows -- verified small enough to scan freely).
LARGE_TABLE_ROW_THRESHOLD = 500_000

# How long a cached catalog snapshot (table list, row counts, sample
# values) is trusted before the NEXT process to call list_tables() re-pulls
# it live instead -- see _load_cache's own comment for the two real,
# observed failures this exists to catch (a dropped table, and every WFM
# Postgres table's row count going stale after the source data emptied
# out). 1 hour by design: long enough that active local development
# (uvicorn --reload restarting the whole process on every file save) keeps
# reusing a warm cache and starting in ~instantly, short enough that the
# app is never running on a day-old (or older) picture of what tables
# exist or how big they are. Override with the GMR_CATALOG_MAX_CACHE_AGE
# env var (seconds) if a different tradeoff is ever needed without a code
# change -- e.g. a much shorter value while actively debugging a live data
# issue, or a longer one to avoid a slow refresh during a demo.
MAX_CACHE_AGE_SECONDS = int(os.environ.get("GMR_CATALOG_MAX_CACHE_AGE", 3600))

# How long AthenaCatalog.list_tables() waits after a live refresh() failure
# (e.g. the WFM Postgres host unreachable) before it will try the live
# connection again, rather than re-attempting -- and re-eating
# connect_timeout -- on every single request during an outage. Deliberately
# much shorter than MAX_CACHE_AGE_SECONDS: this is "how long to back off a
# known-broken connection," not "how long a snapshot stays trustworthy."
_REFRESH_FAILURE_COOLDOWN_SECONDS = int(os.environ.get("GMR_CATALOG_REFRESH_COOLDOWN", 60))


# --------------------------------------------------------------------------- #
# Role classification (name + Athena data_type only -- see docstring point 1)
# --------------------------------------------------------------------------- #

_IDENTIFIER_HINTS = (
    "_no", "_id", "_code", "_key", "_hash", "sequence", "serial", "ticket",
    "number", "imei", "sim", "consumer_number", "session_id", "response_id",
)
_MEASURE_HINTS = (
    "count", "total", "score", "pct", "percent", "kwh", "kva", "kw", "wh",
    "vah", "vah_import", "vah_export", "load", "deviation", "voltage",
    "blocks", "intervals", "days_observed", "rating", "risk_score",
    # hierarchy_summary_autoindexing's per-status consumer counts
    # (confirmed_number, reassign_number, no_data_number, ...) -- bigint
    # counts that the "number" identifier hint would otherwise misclassify.
    "confirmed_", "reassign_", "review_", "no_data_", "concluded_", "pending_verification_",
)
_FREE_TEXT_HINTS = ("remarks", "description", "response", "note")
_DATE_TYPES = ("date",)
_TIMESTAMP_TYPES = ("timestamp",)


def _classify_role(column_name: str, data_type: str) -> str:
    name = column_name.lower()
    dtype = data_type.lower()

    if dtype in _DATE_TYPES or dtype.startswith(_TIMESTAMP_TYPES):
        return "date"
    if dtype == "boolean":
        return "dimension"  # a flag, not a quantity to sum

    # Postgres spells some of these differently from Athena
    # (information_schema.columns.data_type: "character varying", "text",
    # "smallint", "double precision") -- normalized here so one role
    # classifier serves both engines' catalogs.
    is_numeric_type = (
        dtype in ("bigint", "integer", "smallint", "double", "decimal", "double precision", "numeric", "real")
        or dtype.startswith("decimal") or dtype.startswith("numeric")
    )

    if any(hint in name for hint in _FREE_TEXT_HINTS):
        return "free_text"
    # Identifier hint wins over a numeric type -- e.g. "sequence"/"logical_
    # record_number" are bigint columns but are row identifiers, not
    # quantities to sum, unless the name ALSO carries a measure hint
    # (e.g. "risk_score" would still be a measure even if it matched "_no").
    if any(hint in name for hint in _IDENTIFIER_HINTS) and not any(hint in name for hint in _MEASURE_HINTS):
        return "identifier"
    if is_numeric_type:
        return "measure"
    return "dimension"


# Postgres-only (see _sample_dimension_values): only worth sampling a
# column whose real value set is small enough that showing it in full
# actually helps the planner pick the exact right literal/column -- above
# this, it's more likely a free-form name/ID than a fixed set of codes.
_MAX_SAMPLE_DISTINCT = 30


def _sample_dimension_values(conn, cur, table_name: str, column_name: str) -> list[str] | None:
    """Real distinct values for a low-cardinality dimension/identifier
    column -- Postgres only (cheap: every WFM_POSTGRES_TABLES entry is a
    small, pre-aggregated gold_* table). Exists because the planner
    otherwise has to GUESS a categorical filter's exact stored spelling
    from the column name alone -- verified live to fail two different ways
    on real questions: guessing a plausible-but-wrong literal ('120+ days'
    when the real stored value is 'age_120_plus'), and picking the wrong
    COLUMN entirely for a value that happens to exist somewhere else ('DT'
    is a real progress_category value, not a progress_meter_type one, but
    nothing told the planner that). Returns None (not sampled / too high
    cardinality) rather than an empty list when no fixed value set makes
    sense to show -- a genuinely free-form column (a name, a remark)
    shouldn't get dumped into every prompt.

    Rolls back on any failure before returning: this connection is NOT in
    autocommit mode, so a failed query here would otherwise leave the
    transaction aborted, silently breaking every subsequent table's
    introspection in the same refresh() loop with a confusing "current
    transaction is aborted" error that looks unrelated to this function."""
    try:
        cur.execute(f'SELECT COUNT(DISTINCT "{column_name}") FROM "{table_name}"')
        distinct_count = cur.fetchone()[0]
        if not distinct_count or distinct_count > _MAX_SAMPLE_DISTINCT:
            return None
        cur.execute(
            f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
            f'WHERE "{column_name}" IS NOT NULL ORDER BY "{column_name}" LIMIT {_MAX_SAMPLE_DISTINCT}'
        )
        return [str(row[0]) for row in cur.fetchall()]
    except Exception:
        log.warning("Could not sample distinct values for %s.%s -- continuing without them.",
                     table_name, column_name, exc_info=True)
        conn.rollback()
        return None


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

# Columns the query layer adds to a table (sql/hierarchy.py rewrites the table into a subquery
# that has them): meters still before installation sit in a warehouse, and each warehouse
# belongs to a zone -- the same rule as zone-wise stock (user's choice, 2026-10-04).
VIRTUAL_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "gold_meter_installation_pending_stage_snapshot": [("discom", "text"), ("zone", "text")],
}


def _with_virtual(table_name: str, columns: list) -> list:
    have = {c.column_name for c in columns}
    return columns + [ColumnMeta(table_name=table_name, column_name=name, role="dimension", sql_type=typ)
                      for name, typ in VIRTUAL_COLUMNS.get(table_name, []) if name not in have]


@dataclass
class ColumnMeta:
    table_name: str
    column_name: str
    role: str
    sql_type: str
    header_label: str = ""
    # Real distinct values for a low-cardinality dimension/identifier
    # column -- Postgres tables only, see refresh()'s _sample_values. None
    # means "not sampled" (an Athena column, or a Postgres column that
    # wasn't dimension-role or had too many distinct values), NOT "this
    # column has no values."
    sample_values: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.header_label:
            self.header_label = self.column_name.replace("_", " ").title()


@dataclass
class JoinCandidate:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    is_reliable: bool = False  # always False here -- see module docstring point 2
    format_quirk: str | None = None


@dataclass
class TableMeta:
    table_name: str  # unqualified, e.g. "dim_meter" or "gold_om_open_ageing_snapshot"
    database: str  # Athena/Glue database (e.g. "hes_gold") or Postgres database (DB_NAME)
    domain: str  # "mdm" | "wfm"
    row_count: int
    engine: str = "athena"  # "athena" | "postgres" -- which connector/dialect this table needs
    columns: list[ColumnMeta] = field(default_factory=list)
    join_candidates: list[JoinCandidate] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.database}.{self.table_name}"

    @property
    def sql_reference(self) -> str:
        """How this table must actually be written in a FROM/JOIN clause --
        NOT always the same as qualified_name. Postgres has no cross-database
        qualification the way Athena/Presto does (the connection is already
        scoped to one database), and this database's own name contains a
        hyphen, which is not valid inside an unquoted SQL identifier -- the LLM
        planner is shown exactly this (schema/schema_brief.py)."""
        return f"public.{self.table_name}" if self.engine == "postgres" else self.qualified_name

    @property
    def description(self) -> str | None:
        return TABLE_DESCRIPTIONS.get(self.table_name)

    @property
    def is_large(self) -> bool:
        # Postgres's gold_ tables are already pre-aggregated (day_agg/
        # month_agg/snapshot) and run 1.6M rows at most (verified directly)
        # -- Postgres also has no Athena-style bytes-scanned billing model,
        # so the bounding-filter rule this flag exists for (validation/sql_guard.py)
        # is Athena-specific and never applies to a Postgres table.
        return self.engine == "athena" and self.row_count >= LARGE_TABLE_ROW_THRESHOLD


# --------------------------------------------------------------------------- #
# Declared (unverified) cross-table join candidates per domain
# --------------------------------------------------------------------------- #

def _declared_join_candidates(domain: str) -> list[JoinCandidate]:
    if domain == "mdm":
        fact_tables = ["energy_theft_suspect", "theft_risk_by_area", "voltage_imbalance",
                       "low_voltage_pockets", "voltage_deviation", "billing", "daily_usage"]
        return [
            JoinCandidate(left_table=t, left_column="meter_no", right_table="dim_meter", right_column="meter_no")
            for t in fact_tables
        ]
    if domain == "wfm":
        pairs = [
            ("unified_active_complaints", "unified_complaints"),
            ("unified_installations", "unified_smart_meter_asset"),
            ("unified_smart_meter_inventory", "unified_smart_meter_asset"),
        ]
        candidates = [
            JoinCandidate(left_table=a, left_column="consumer_number", right_table=b, right_column="consumer_number")
            for a, b in pairs
        ]
        # Geo-hierarchy joins, useful when two WFM tables need combining by
        # location rather than by a specific consumer/meter. NOTE: these are
        # all Athena-side (gmr_wfm_silver) tables -- a join candidate is
        # never declared between an Athena table and a Postgres one, since
        # neither connector can run a single query spanning both engines
        # (see validation/sql_guard.py's engine-consistency check).
        geo_tables = ["unified_complaints", "unified_active_complaints", "unified_installations"]
        for i, a in enumerate(geo_tables):
            for b in geo_tables[i + 1:]:
                candidates.append(JoinCandidate(left_table=a, left_column="feeder_code", right_table=b, right_column="feeder_code"))

        # sat_meter_mappings -> sat_milestones: the only path from a METER to
        # its SAT completion date (sat_meter_mappings has no date column of
        # its own; sat_milestones has no meter column). Declared as TWO
        # separate candidates (project, sat_number) rather than one, because
        # sat_number alone is NOT globally unique -- see sat_milestones'
        # TABLE_DESCRIPTIONS hint -- so a correct join always needs BOTH.
        candidates.append(JoinCandidate(left_table="sat_meter_mappings", left_column="project", right_table="sat_milestones", right_column="project"))
        candidates.append(JoinCandidate(left_table="sat_meter_mappings", left_column="sat_number", right_table="sat_milestones", right_column="sat_number"))

        # unified_installations -> sat_meter_mappings: the bridge from a
        # CONSUMER (consumer_number, this table's own key) to that meter's
        # SAT date -- verified live, ~3.75M overlapping meter serial numbers
        # between the two tables. Without this, "when was this consumer
        # installed AND when was its SAT done" (mi_date lives here, sat_date
        # only reachable via sat_meter_mappings/sat_milestones) had no
        # declared path from a consumer_number question to the SAT tables
        # at all.
        candidates.append(JoinCandidate(left_table="unified_installations", left_column="new_meter_serial_number", right_table="sat_meter_mappings", right_column="meter_serial_number"))

        # Postgres gold_ tables: deliberately NO all-pairs geo-join
        # declaration here, unlike the Athena geo_tables loop above. It
        # existed at the original 7-table shortlist (a manageable 21
        # pairs) but the same all-pairs approach at 21 tables would emit
        # 210 speculative "X.zone = Y.zone" candidates into every prompt --
        # a real, direct cost (token bloat on every single question) for a
        # declaration that was never based on a verified real relationship
        # beyond "both tables happen to have a zone column," and that no
        # real question so far has ever needed (every WFM Postgres eval
        # case queries a single table). validation/sql_guard.py rejects any join
        # that isn't a declared candidate, so this only removes an
        # unused, unverified capability -- add a specific, justified
        # candidate here if a real cross-table Postgres join need
        # actually comes up, rather than a blanket one "just in case."
        return candidates
    return []


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #

class AthenaCatalog:
    """One instance per domain ("mdm" or "wfm"). Builds (or loads from cache)
    a TableMeta list for that domain's curated table shortlist."""

    def __init__(self, domain: str) -> None:
        if domain not in ("mdm", "wfm"):
            raise ValueError(f"Unknown domain: {domain!r}")
        self.domain = domain
        self._tables_by_db = MDM_TABLES if domain == "mdm" else WFM_TABLES
        self._cache_path = _CACHE_DIR / f"{domain}_catalog.json"
        self._tables: list[TableMeta] | None = None
        # Set the moment refresh() raises (e.g. the WFM Postgres host is
        # unreachable) -- see list_tables()'s own comment for why this
        # exists: without it, EVERY request during an outage re-attempts
        # the full live connection and eats its connect_timeout, rather
        # than just the first one.
        self._last_refresh_failure: float | None = None
        # One refresh at a time: the startup refresh, the hourly one and a
        # request that finds no catalog must never rebuild it concurrently.
        self._refresh_lock = threading.Lock()

    def _load_cache(self, *, ignore_age: bool = False) -> list[TableMeta] | None:
        if not self._cache_path.exists():
            return None
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        # Verified live, not hypothetical: this cache went stale in exactly
        # the ways that matter -- a table (load_diversion) silently dropped
        # from the live data lake while the cache still listed it as
        # queryable, and separately every WFM Postgres table's real row
        # count fell to 0 while the cache still held its old, populated
        # count. Neither was ever caught automatically before this check
        # existed -- the cache is only ever invalidated by hand (deleting
        # the file) or by calling refresh() directly. A cache older than
        # MAX_CACHE_AGE_SECONDS is now treated as absent, so the NEXT
        # list_tables() call re-pulls live from Athena/Postgres instead of
        # trusting a snapshot that could be hours or days stale.
        # ignore_age=True is list_tables()'s own last-resort fallback when a
        # live refresh() just failed -- a stale-but-real cache degrades the
        # ANSWER (structure/row-counts may be a bit old) far less than
        # having no WFM catalog at all for the whole outage.
        built_at = raw.get("built_at")
        if not ignore_age and (built_at is None or (time.time() - built_at) > MAX_CACHE_AGE_SECONDS):
            log.info(
                "catalog[%s]: cache is stale (built_at=%s) -- will refresh live instead of using it.",
                self.domain, built_at,
            )
            return None

        tables = []
        for t in raw.get("tables", []):
            tables.append(TableMeta(
                table_name=t["table_name"], database=t["database"], domain=t["domain"],
                row_count=t["row_count"], engine=t.get("engine", "athena"),
                columns=_with_virtual(t["table_name"], [ColumnMeta(**c) for c in t["columns"]]),
                join_candidates=[JoinCandidate(**j) for j in t["join_candidates"]],
            ))
        return tables

    def _save_cache(self, tables: list[TableMeta]) -> None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "built_at": time.time(),
            "tables": [
                {
                    "table_name": t.table_name, "database": t.database, "domain": t.domain,
                    "row_count": t.row_count, "engine": t.engine,
                    "columns": [
                        {"table_name": c.table_name, "column_name": c.column_name, "role": c.role,
                         "sql_type": c.sql_type, "header_label": c.header_label,
                         "sample_values": c.sample_values}
                        for c in t.columns
                    ],
                    "join_candidates": [
                        {"left_table": j.left_table, "left_column": j.left_column,
                         "right_table": j.right_table, "right_column": j.right_column,
                         "is_reliable": j.is_reliable, "format_quirk": j.format_quirk}
                        for j in t.join_candidates
                    ],
                }
                for t in tables
            ],
        }
        # Write-then-rename, so a reader never sees a half-written file.
        tmp = self._cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._cache_path)

    def warm_start(self) -> bool:
        """Serve the cached catalog right away, whatever its age -- startup
        must not wait minutes for a live rebuild. Returns True when that
        cache is stale (or missing) and the caller should refresh() in the
        background. Only with no cache at all does this block on a live
        build (there is nothing else to serve)."""
        if self._tables is not None:
            return False
        fresh = self._load_cache()
        if fresh is not None:
            self._tables = fresh
            return False
        stale = self._load_cache(ignore_age=True)
        if stale is not None:
            self._tables = stale
            log.info("catalog[%s]: serving the cached catalog (%d tables) while it refreshes in the background.",
                     self.domain, len(stale))
            return True
        self.list_tables()
        return False

    def refresh(self) -> list[TableMeta]:
        """Rebuilds the catalog from live Athena (and, for "wfm", also
        Postgres) information_schema and overwrites the cache. Runs in the
        background after startup and every MAX_CACHE_AGE_SECONDS; call it
        directly after changing a table shortlist above."""
        with self._refresh_lock:
            return self._refresh()

    def _athena_tables(self, join_candidates: list[JoinCandidate]) -> list[TableMeta]:
        """One information_schema query per database for every shortlisted
        table's columns, and the per-table row counts run concurrently --
        the same metadata as one query pair per table, in a fraction of the
        time."""
        from gmr_agents.sql.connections import athena_query

        tables: list[TableMeta] = []
        with ThreadPoolExecutor(max_workers=_REFRESH_WORKERS) as pool:
            for database, table_names in self._tables_by_db.items():
                names_sql = ", ".join(f"'{t}'" for t in table_names)
                counts = {t: pool.submit(athena_query, f"SELECT COUNT(*) AS n FROM {t}", database=database)
                          for t in table_names}
                cols_df = athena_query(
                    "SELECT table_name, column_name, data_type FROM information_schema.columns "
                    f"WHERE table_schema='{database}' AND table_name IN ({names_sql}) "
                    "ORDER BY table_name, ordinal_position",
                    database=database,
                )
                by_table: dict[str, list[tuple[str, str]]] = {}
                for _, row in cols_df.iterrows():
                    by_table.setdefault(row["table_name"], []).append((row["column_name"], row["data_type"]))
                for table_name in table_names:
                    columns = [
                        ColumnMeta(
                            table_name=table_name, column_name=column_name,
                            role=_classify_role(column_name, data_type), sql_type=data_type,
                            header_label=HEADER_LABEL_OVERRIDES.get((table_name, column_name), ""),
                        )
                        for column_name, data_type in by_table.get(table_name, [])
                    ]
                    row_count = int(counts[table_name].result().iloc[0]["n"])
                    tables.append(TableMeta(
                        table_name=table_name, database=database, domain=self.domain, engine="athena",
                        row_count=row_count, columns=columns,
                        join_candidates=[j for j in join_candidates if table_name in (j.left_table, j.right_table)],
                    ))
                    log.info("catalog[%s]: athena %s.%s -- %d cols, %d rows", self.domain, database, table_name,
                             len(columns), row_count)
        return tables

    def _refresh(self) -> list[TableMeta]:
        join_candidates = _declared_join_candidates(self.domain)

        def _table_joins(table_name: str) -> list[JoinCandidate]:
            return [j for j in join_candidates if table_name in (j.left_table, j.right_table)]

        tables = self._athena_tables(join_candidates)

        if self.domain == "wfm" and WFM_POSTGRES_TABLES:
            from gmr_agents.sql.connections import postgres_connection

            athena_tables = list(tables)
            try:
                # Verified live: a single 5 s connect timeout (a network blip)
                # during the hourly refresh dropped every gold_* table for the
                # next hour. Retry a few times before treating it as an outage.
                conn = None
                for attempt in range(3):
                    try:
                        conn = postgres_connection()
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        log.warning("catalog[wfm]: Postgres connect attempt %d failed -- retrying.", attempt + 1)
                        time.sleep(3)
                try:
                    cur = conn.cursor()
                    for table_name in WFM_POSTGRES_TABLES:
                        cur.execute(
                            "SELECT column_name, data_type FROM information_schema.columns "
                            "WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position",
                            (table_name,),
                        )
                        cols = cur.fetchall()
                        cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                        row_count = cur.fetchone()[0]
                        columns = []
                        for col_name, data_type in cols:
                            role = _classify_role(col_name, data_type)
                            sample_values = (
                                _sample_dimension_values(conn, cur, table_name, col_name)
                                if role in ("dimension", "identifier") else None
                            )
                            columns.append(ColumnMeta(
                                table_name=table_name, column_name=col_name, role=role,
                                sql_type=data_type, sample_values=sample_values,
                            ))
                        tables.append(TableMeta(
                            table_name=table_name, database=os.environ.get("DB_NAME") or "postgres", domain=self.domain, engine="postgres",
                            row_count=int(row_count), columns=_with_virtual(table_name, columns),
                            join_candidates=_table_joins(table_name),
                        ))
                        log.info("catalog[wfm]: postgres public.%s -- %d cols, %d rows",
                                 table_name, len(columns), row_count)
                finally:
                    conn.close()
            except Exception:
                # Verified live: a Postgres-side outage/timeout here used to
                # propagate out of refresh() entirely, discarding the Athena
                # tables already built above (unified_*/sat_* -- a real,
                # working catalog) along with the genuinely-unreachable
                # gold_* Postgres tables. One connector being down should
                # degrade to "gold_* tables temporarily unavailable," not
                # wipe out an otherwise-healthy Athena-side WFM catalog --
                # especially since list_tables()'s own except-branch has no
                # cache to fall back to right after a fresh cache-invalidate.
                log.warning(
                    "catalog[wfm]: Postgres refresh failed -- serving Athena-only WFM tables "
                    "(gold_* tables unavailable) instead of discarding the whole catalog.",
                    exc_info=True,
                )
                # Keep the gold_* tables we already knew (in memory, else the
                # last cache): their shape does not change between refreshes,
                # and queries reach Postgres again as soon as it is back --
                # dropping them sent every dashboard KPI question to the LLM
                # against raw Silver (e.g. "consumers installed" counted
                # consumer numbers). Only when none are known is WFM served
                # Athena-only.
                known = [t for t in (self._tables or self._load_cache(ignore_age=True) or [])
                         if t.engine == "postgres"]
                tables = athena_tables + known
                if known:
                    log.warning("catalog[wfm]: keeping %d previously loaded gold_* table(s) until Postgres answers.",
                                len(known))
                # Serve it, but don't cache it: a cached degraded catalog
                # counts as fresh for MAX_CACHE_AGE_SECONDS, so every restart
                # in that window kept the gold_* tables missing long after
                # Postgres was back.
                self._tables = tables
                return tables

        self._tables = tables
        self._save_cache(tables)
        return tables

    def list_tables(self) -> list[TableMeta]:
        if self._tables is not None:
            return self._tables
        cached = self._load_cache()
        if cached is not None:
            self._tables = cached
            return self._tables

        # Live refresh needed (no fresh cache). If the LAST attempt failed
        # recently, don't pay a second connect_timeout on this request too
        # -- verified live: a Postgres outage was otherwise re-attempted on
        # every single question, each one silently eating the full
        # connect_timeout before the request could even start routing.
        # Serve a stale cache (better degraded data than none) or an empty
        # list instead, and only try the live connection again once the
        # cooldown has passed.
        if (
            self._last_refresh_failure is not None
            and (time.time() - self._last_refresh_failure) < _REFRESH_FAILURE_COOLDOWN_SECONDS
        ):
            return self._load_cache(ignore_age=True) or []

        try:
            with self._refresh_lock:
                if self._tables is not None:  # another thread just built it
                    return self._tables
                self._tables = self._refresh()
        except Exception:
            self._last_refresh_failure = time.time()
            log.warning(
                "catalog[%s]: live refresh failed -- serving a stale cache (if any) for the next "
                "%ds instead of retrying the live connection on every request.",
                self.domain, _REFRESH_FAILURE_COOLDOWN_SECONDS, exc_info=True,
            )
            return self._load_cache(ignore_age=True) or []
        return self._tables

    def get_table(self, table_name: str) -> TableMeta | None:
        return next((t for t in self.list_tables() if t.table_name == table_name), None)

    def all_join_candidates(self) -> list[JoinCandidate]:
        seen: set[tuple] = set()
        out: list[JoinCandidate] = []
        for t in self.list_tables():
            for j in t.join_candidates:
                key = (j.left_table, j.left_column, j.right_table, j.right_column)
                if key not in seen:
                    seen.add(key)
                    out.append(j)
        return out


_CATALOGS: dict[str, AthenaCatalog] = {}


def get_catalog(domain: str) -> AthenaCatalog:
    if domain not in _CATALOGS:
        _CATALOGS[domain] = AthenaCatalog(domain)
    return _CATALOGS[domain]
