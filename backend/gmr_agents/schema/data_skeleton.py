"""Data skeleton: a small, READ-ONLY picture of every table's real data, so the
LLM (via retrieval, schema/knowledge.py) knows where each thing lives and how it
is spelled. Only SELECT queries run; nothing in any table is changed.

Per table (all ~40 catalog tables):
    purpose      table_metadata.yaml description
    engine/rows  postgres | athena, row count (catalog)
    dates        each date column's real MIN..MAX (Postgres; small Athena tables)
    dimensions   every real value of each low-cardinality column (catalog samples)
    keys         the grain (columns that make a row unique) and declared join keys
    combos       the dimension-value combinations that actually occur (Postgres,
                 SELECT DISTINCT, capped) -- "covering all possibilities"
    measures     numeric columns (what to SUM)
    sample       a ~DATA_SKELETON_SAMPLE_PCT % random sample (TABLESAMPLE), capped at
                 DATA_SKELETON_SAMPLE_ROWS rows; Athena: a cheap LIMIT sample
Plus every geography path (project > discom > zone > circle > division > sub_division).

Saved in the local JSON knowledge store (outputs/knowledge.json, artifact "data_skeleton"); rebuilt in a background thread when older
than DATA_SKELETON_MAX_AGE seconds (never on the question path).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

from shared import config

log = logging.getLogger(__name__)

PATH = os.path.join(config.OUTPUT_DIR, "data_skeleton.json")
_GEO = ["project", "discom", "zone", "circle", "division", "sub_division"]
_NUMERIC = ("int", "numeric", "double", "real", "decimal", "float", "bigint", "smallint")
_STATE = {"data": None, "mtime": None, "building": False}


def _pg(sql: str) -> list[dict]:
    from gmr_agents.sql.connections import pooled_postgres_connection

    with pooled_postgres_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, (str(v) if v is not None else None for v in row))) for row in cur.fetchall()]


def _athena(sql: str, database: str) -> list[dict]:
    from gmr_agents.sql.connections import athena_query

    df = athena_query(sql, database=database)
    return [{k: (None if v is None else str(v)) for k, v in r.items()} for r in df.to_dict("records")]


def _table_card(t) -> dict:
    is_pg = t.engine == "postgres"
    ref = f"public.{t.table_name}" if is_pg else f"{t.database}.{t.table_name}"
    run = _pg if is_pg else (lambda sql: _athena(sql, t.database))
    dims = {c.column_name: list(c.sample_values) for c in t.columns if c.sample_values}
    dates = [c.column_name for c in t.columns if any(k in c.sql_type.lower() for k in ("date", "timestamp"))]
    measures = [c.column_name for c in t.columns
                if any(k in c.sql_type.lower() for k in _NUMERIC) and c.column_name not in dims
                and not c.column_name.endswith("_id")]
    # Keys: the columns that make one row unique (dimensions + date) -- the table's grain --
    # and the declared join keys to other tables.
    grain = [c.column_name for c in t.columns if c.column_name in dims or c.column_name in dates
             or c.role in ("identifier", "dimension", "date")][:12]
    joins = [f"{j.left_table}.{j.left_column} = {j.right_table}.{j.right_column}"
             for j in getattr(t, "join_candidates", [])][:6]
    card = {"table": t.table_name, "engine": t.engine, "ref": ref, "rows": int(t.row_count or 0),
            "keys": grain, "joins": joins,
            "dimensions": dims, "measures": measures[:20], "dates": {}, "combos": [], "sample": []}
    small_athena = not is_pg and int(t.row_count or 0) <= config.DATA_SKELETON_ATHENA_MAX_ROWS
    for col in dates[:3]:
        if is_pg or small_athena:
            try:
                r = run(f"SELECT MIN({col}) AS lo, MAX({col}) AS hi FROM {ref}")[0]
                card["dates"][col] = [r.get("lo"), r.get("hi")]
            except Exception as exc:
                log.debug("skeleton: dates for %s.%s failed: %s", t.table_name, col, exc)
    combo_cols = [c for c in dims if c not in _GEO][:4]
    if is_pg and combo_cols:
        try:
            card["combos"] = run(f"SELECT DISTINCT {', '.join(combo_cols)} FROM {ref} LIMIT {config.DATA_SKELETON_MAX_COMBOS}")
        except Exception as exc:
            log.debug("skeleton: combos for %s failed: %s", t.table_name, exc)
    try:
        if is_pg:
            rows = run(f"SELECT * FROM {ref} TABLESAMPLE SYSTEM ({config.DATA_SKELETON_SAMPLE_PCT}) "
                       f"LIMIT {config.DATA_SKELETON_SAMPLE_ROWS}")
            if not rows:  # tiny tables: a block sample can be empty
                rows = run(f"SELECT * FROM {ref} LIMIT {config.DATA_SKELETON_SAMPLE_ROWS}")
        else:
            rows = run(f"SELECT * FROM {ref} LIMIT {config.DATA_SKELETON_SAMPLE_ROWS}")
        card["sample"] = [{k: v for k, v in r.items() if v not in (None, "", "None")} for r in rows]
    except Exception as exc:
        log.debug("skeleton: sample for %s failed: %s", t.table_name, exc)
    return card


def build() -> dict:
    """Query every table read-only and save the skeleton (takes a minute or two)."""
    from gmr_agents.schema.catalog import get_catalog

    t0 = time.perf_counter()
    cards = []
    for domain in ("wfm", "mdm"):
        for t in get_catalog(domain).list_tables():
            try:
                cards.append({**_table_card(t), "domain": domain})
            except Exception:
                log.warning("skeleton: %s skipped", t.table_name, exc_info=True)
    geography: list[dict] = []
    try:
        geography = _pg("SELECT DISTINCT " + ", ".join(_GEO) + " FROM public.gold_mi_installation_day_agg "
                        "WHERE project IS NOT NULL ORDER BY 1, 2, 3, 4, 5, 6")
    except Exception:
        log.warning("skeleton: geography paths unavailable", exc_info=True)
    data = {"built_at": datetime.now(timezone.utc).isoformat(), "tables": cards, "geography": geography}
    from gmr_agents.schema import knowledge_store

    knowledge_store.save_artifact("data_skeleton", data)
    _STATE.update(data=data, mtime=time.time())
    log.info("skeleton: %d table cards, %d geography paths in %.0fs -> knowledge DB",
             len(cards), len(geography), time.perf_counter() - t0)
    return data


def load() -> dict:
    """The saved skeleton (knowledge DB artifact 'data_skeleton'); {} if not built yet."""
    if _STATE["data"] is None:
        from gmr_agents.schema import knowledge_store

        data, built = knowledge_store.load_artifact("data_skeleton")
        _STATE.update(data=data, mtime=built)
    return _STATE["data"] or {}


def stale() -> bool:
    load()
    return _STATE["mtime"] is None or time.time() - _STATE["mtime"] > config.DATA_SKELETON_MAX_AGE


def build_if_stale() -> bool:
    """Build when missing / too old (called from a background thread). True if built."""
    if _STATE["building"] or not stale():
        return False
    _STATE["building"] = True
    try:
        build()
        return True
    finally:
        _STATE["building"] = False
