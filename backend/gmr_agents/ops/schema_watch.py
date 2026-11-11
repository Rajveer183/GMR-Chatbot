"""Schema-change and data-quality detection on every background catalog refresh.

The catalog refresh (main.py _refresh_one_catalog, hourly, in a background thread) already
reads every table's columns and row count. This compares the catalog before and after that
refresh -- in memory, no extra query -- and logs what changed to
outputs/logs/schema_changes.jsonl:

    tables_added / tables_removed      a table appeared in / vanished from the source
    columns_added / columns_removed    per table
    type_changes                       a column's SQL type changed
    emptied                            row count went from > 0 to 0 (a failed load)
    shrunk                             row count fell by more than SHRINK_RATIO (a partial load)

GET /api/data/schema-changes lists them; ops/alerts.py raises an alert for recent ones.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from shared.logging_config import append_jsonl

log = logging.getLogger(__name__)
_IST = timezone(timedelta(hours=5, minutes=30))
SHRINK_RATIO = 0.5


def snapshot(tables) -> dict:
    """{table: {"rows": int|None, "columns": {name: type}}} from a list of TableMeta.
    Never raises: {} (= nothing to compare) on anything unexpected."""
    try:
        return {t.qualified_name: {"rows": t.row_count,
                                   "columns": {c.column_name: str(c.sql_type or "") for c in t.columns}}
                for t in tables or []}
    except Exception:
        log.debug("schema snapshot failed", exc_info=True)
        return {}


def diff(before: dict, after: dict) -> dict:
    out: dict = {}
    if not before or not after:     # first build / failed refresh: nothing to compare
        return out
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    if added:
        out["tables_added"] = added
    if removed:
        out["tables_removed"] = removed
    cols_added, cols_removed, types, emptied, shrunk = {}, {}, {}, [], []
    for name in sorted(set(before) & set(after)):
        b, a = before[name], after[name]
        bc, ac = b["columns"], a["columns"]
        if set(ac) - set(bc):
            cols_added[name] = sorted(set(ac) - set(bc))
        if set(bc) - set(ac):
            cols_removed[name] = sorted(set(bc) - set(ac))
        changed = {c: f"{bc[c]} -> {ac[c]}" for c in set(bc) & set(ac) if bc[c] and ac[c] and bc[c] != ac[c]}
        if changed:
            types[name] = changed
        br, ar = b.get("rows"), a.get("rows")
        if isinstance(br, int) and isinstance(ar, int) and br > 0:
            if ar == 0:
                emptied.append(name)
            elif ar < br * SHRINK_RATIO:
                shrunk.append({"table": name, "before": br, "after": ar})
    for key, val in (("columns_added", cols_added), ("columns_removed", cols_removed), ("type_changes", types),
                     ("emptied", emptied), ("shrunk", shrunk)):
        if val:
            out[key] = val
    return out


def record(domain: str, before: dict, after: dict) -> dict:
    """Diff and log; never raises (a refresh must not fail because of this)."""
    try:
        changes = diff(before, after)
        if changes:
            append_jsonl("schema_changes", {"at": datetime.now(_IST).strftime("%Y-%m-%dT%H:%M:%S"),
                                            "domain": domain, **changes})
            log.warning("Schema / data change in %s catalog: %s", domain, ", ".join(sorted(changes)))
        return changes
    except Exception:
        log.debug("schema diff failed", exc_info=True)
        return {}


def recent(hours: float = 168.0) -> list[dict]:
    from gmr_agents.ops.metrics import _read

    since = (datetime.now(_IST) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    return list(reversed(_read("schema_changes", since)))
