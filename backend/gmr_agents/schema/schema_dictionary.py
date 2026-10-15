"""Schema dictionary: every table fully explained, generated -- never hand-copied --
from what the system already knows, so it stays true as tables and rules change:

    live catalog          columns, types, roles, row counts, declared joins
    table_metadata.yaml   business purpose of each table, meaning of each column
    data skeleton         real values, date ranges, grain (keys), sample rows
    business_rules.yaml   KPI topics using the table, its rules / important filters
    examples.yaml         verified example questions answered from the table

Per table: name, SQL reference, domain, engine, rows, business purpose, business
areas, primary key (grain), foreign keys / relationships, joins allowed, date
columns (+ real range), columns (type, role, description, values), important
filters & business rules, example questions.

Outputs: the local JSON knowledge store (outputs/knowledge.json, artifact "schema_dictionary": indexed for retrieval,
compact headers in the SQL prompt) and document/GMR_Schema_Dictionary.md (for people).
"""

from __future__ import annotations

import re

import json
import logging
import os
from datetime import datetime, timezone

from shared import config

log = logging.getLogger(__name__)

PATH = os.path.join(config.OUTPUT_DIR, "schema_dictionary.json")
DOC_PATH = os.path.join(os.path.dirname(config.PROJECT_ROOT), "document", "GMR_Schema_Dictionary.md")
_SHARED_DIMS = ["project", "discom", "zone", "circle", "division", "sub_division"]
_JOIN_POLICY = ("No JOIN between gold tables (combine them with scalar subqueries or UNION ALL); one query = one "
                "engine (never mix Postgres and Athena). Silver (Athena) tables join only on the declared keys below.")
_STATE = {"data": None, "mtime": None}


def build() -> dict:
    from gmr_agents.schema import business_rules, data_skeleton
    from gmr_agents.schema.catalog import get_catalog
    from gmr_agents.schema.table_metadata import COLUMN_HINTS, TABLE_DESCRIPTIONS

    cards = {c["table"]: c for c in data_skeleton.load().get("tables") or []}
    topics = business_rules.topics()
    examples = business_rules.examples()
    global_rules = business_rules.global_rules()
    tables: dict[str, dict] = {}
    for domain in ("wfm", "mdm"):
        for t in get_catalog(domain).list_tables():
            name = t.table_name
            card = cards.get(name, {})
            ref = f"public.{name}" if t.engine == "postgres" else f"{t.database}.{name}"
            areas, rules = [], []
            for key, topic in topics.items():
                used = name in (topic.get("tables") or [])
                lines = [r for r in topic.get("rules") or [] if name in str(r)]
                if used or lines:
                    areas.append(f"{key}: {topic.get('title', key)}")
                    rules.extend(lines or [r for r in (topic.get("rules") or [])[:2]])
            rel = [f"{j.left_table}.{j.left_column} = {j.right_table}.{j.right_column}"
                   for j in getattr(t, "join_candidates", [])]
            cols = {c.column_name for c in t.columns}
            shared = [d for d in _SHARED_DIMS if d in cols]
            if shared and name != "geography_nodes":
                rel.append(f"geography: {', '.join(shared)} -> geography_nodes (same names, LOWER(col) compare)")
            if "project" in cols and name != "sat_eligibility_config":
                rel.append("project -> sat_eligibility_config.project (SAT eligibility cutoff)")
            tables[name] = {
                "table": name, "sql_reference": ref, "domain": domain, "engine": t.engine,
                "rows": int(t.row_count or 0),
                "purpose": TABLE_DESCRIPTIONS.get(name, ""),
                "business_areas": areas,
                "primary_key": card.get("keys") or [],
                "relationships": rel,
                "joins_allowed": _JOIN_POLICY,
                "date_columns": card.get("dates") or {c.column_name: None for c in t.columns
                                                       if any(k in c.sql_type.lower() for k in ("date", "timestamp"))},
                "columns": [{"name": c.column_name, "type": c.sql_type, "role": c.role,
                             "description": COLUMN_HINTS.get((name, c.column_name), c.header_label),
                             **({"values": list(c.sample_values)[:15]} if c.sample_values else {})}
                            for c in t.columns],
                "important_filters": [r for r in rules if any(w in str(r).lower() for w in ("where", "filter", "always", "never", "only", "="))][:6],
                "business_rules": rules[:10],
                "example_questions": [ex["q"][0] if isinstance(ex.get("q"), list) else ex.get("q")
                                      for ex in examples if name in str(ex.get("sql", ""))][:6],
                "sample_rows": card.get("sample", [])[:3],
            }
    data = {"built_at": datetime.now(timezone.utc).isoformat(), "global_rules": global_rules, "tables": tables}
    from gmr_agents.schema import knowledge_store

    knowledge_store.save_artifact("schema_dictionary", data)
    _STATE["data"] = data
    _write_markdown(data)
    log.info("schema dictionary: %d tables -> knowledge DB + %s", len(tables), DOC_PATH)
    return data


def load() -> dict:
    """The schema dictionary (knowledge DB artifact 'schema_dictionary')."""
    if _STATE["data"] is None:
        from gmr_agents.schema import knowledge_store

        _STATE["data"] = knowledge_store.load_artifact("schema_dictionary")[0]
    return _STATE["data"] or {}


def entry(name: str) -> dict:
    return (load().get("tables") or {}).get(name, {})


_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:[+-]\d{2}:?\d{2}|Z)?")


def _date_only(value) -> str:
    """'2026-08-31 20:11:06.482743+00:00' -> '2026-08-31' (the time adds tokens, not meaning)."""
    return _TS_RE.sub(r"\1", str(value))


def _clip(text: str, limit: int) -> str:
    """Shorten at a clause boundary ('; ', '. ', ', ') -- never mid-word."""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    cut = max(text.rfind(sep, 0, limit) for sep in ("; ", ". ", ", ", " -- "))
    if cut < limit // 2:
        cut = text.rfind(" ", 0, limit)
    return text[:cut].rstrip(" ,;.-") + " ..."


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def header(name: str, seen: set[str] | None = None, known_text: str = "") -> str:
    """Compact per-table explanation for the SQL prompt (purpose, grain, dates,
    relationships, key filters, an example question). With `seen` (shared across
    the tables of one prompt) and `known_text` (e.g. the rules already in that
    prompt), a relationship or key filter already shown is not repeated."""
    e = entry(name)
    if not e:
        return ""
    seen = set() if seen is None else seen
    known = _norm(known_text)

    def fresh(items, limit):
        out = []
        for r in items:
            key = _norm(r)
            if key in seen or (len(key) > 40 and key[:120] in known):
                continue
            seen.add(key)
            out.append(_clip(r, limit))
        return out

    bits = [f"### {e['sql_reference']} -- {e['purpose']}".rstrip(" -")]
    if e.get("primary_key"):
        bits.append(f"  grain (one row per): {', '.join(e['primary_key'][:8])}")
    dates = {c: v for c, v in (e.get("date_columns") or {}).items() if v}
    if dates:
        bits.append("  dates: " + "; ".join(f"{c} {_date_only(v[0])}..{_date_only(v[1])}" for c, v in dates.items()))
    rels = fresh(e.get("relationships") or [], 240)[:3]
    if rels:
        bits.append("  relationships: " + " | ".join(rels))
    filters = fresh([str(r) for r in e.get("important_filters") or []], 220)[:2]
    if filters:
        bits.append("  key filters: " + " | ".join(filters))
    if e.get("example_questions") and "table_eg" not in config.PROMPT_COMPACT:
        bits.append(f"  e.g. \"{e['example_questions'][0]}\"")
    return "\n".join(bits)


def text_for_index(name: str) -> list[str]:
    """Texts for the retrieval 'tables' index: purpose + areas + rules + example questions."""
    e = entry(name)
    if not e:
        return []
    return [f"{name.replace('_', ' ')}: {e['purpose']} areas: {'; '.join(e['business_areas'])}",
            " ".join(e.get("example_questions") or [])[:800],
            " ".join(str(r) for r in e.get("important_filters") or [])[:800]]


def _write_markdown(data: dict) -> None:
    out = ["# GMR Chatbot -- Schema Dictionary", "",
           f"_Generated {data['built_at']} from the live catalog, table metadata, data skeleton, business rules "
           f"and verified examples (gmr_agents/schema/schema_dictionary.py). Do not edit by hand._", "",
           "## Global rules", *[f"- {r}" for r in data.get("global_rules") or []], ""]
    for name, e in sorted(data["tables"].items(), key=lambda kv: (kv[1]["domain"], kv[1]["engine"], kv[0])):
        out += [f"## {name}", "",
                f"- **SQL reference:** `{e['sql_reference']}`  |  **domain:** {e['domain']}  |  **engine:** {e['engine']}  |  **rows:** {e['rows']:,}",
                f"- **Business purpose:** {e['purpose'] or '-'}",
                f"- **Business areas:** {'; '.join(e['business_areas']) or '-'}",
                f"- **Primary key (grain):** {', '.join(e['primary_key']) or '-'}",
                f"- **Relationships / foreign keys:** {'; '.join(e['relationships']) or '-'}",
                f"- **Joins allowed:** {e['joins_allowed']}",
                "- **Date columns:** " + ("; ".join(f"`{c}` {v[0]} .. {v[1]}" if v else f"`{c}`"
                                                   for c, v in (e['date_columns'] or {}).items()) or "-"), ""]
        if e.get("important_filters"):
            out += ["**Important filters / business rules:**", *[f"- {r}" for r in e["business_rules"]], ""]
        if e.get("example_questions"):
            out += ["**Example questions:**", *[f"- {q}" for q in e["example_questions"]], ""]
        out += ["| Column | Type | Role | Description | Values |", "|---|---|---|---|---|"]
        for c in e["columns"]:
            vals = ", ".join(map(str, c.get("values", [])[:8]))
            desc = str(c["description"]).replace("|", "/")
            out.append(f"| `{c['name']}` | {c['type']} | {c['role']} | {desc} | {vals.replace('|', '/')} |")
        out.append("")
    os.makedirs(os.path.dirname(DOC_PATH), exist_ok=True)
    with open(DOC_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
