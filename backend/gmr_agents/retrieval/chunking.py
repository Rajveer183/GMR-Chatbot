"""Data-aware chunking of structured knowledge into retrievable logical units.

Who controls it: application code; it runs when the retrieval index is (re)built.

The knowledge here is structured YAML, not free text, so chunks follow the data's
own boundaries instead of every-N-tokens windows:

    kpi_definitions.yaml   one PARENT chunk per KPI (title, card, aliases, summary)
                           + one CHILD chunk per section: logic, formulas, caveats,
                             sql_recipe, filters, response, example, tables
                           -- a list section is split only BETWEEN items when it is
                             over max_tokens (a logic step / caveat is never cut);
                             SQL, formulas and worked examples are atomic
                           -- tiny sections are merged into a neighbour so no chunk
                             is too thin to retrieve on its own
    gold_tables            one chunk per table (name, builder, sources, notes)
    pipeline_steps         one chunk for the whole pipeline (it is read as a unit)

Every chunk keeps metadata: doc, kpi_code / table, section, parent id, previous /
next sibling ids, token estimate, type -- so retrieval can expand a precise child
chunk with its parent overview (see expand()).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from shared.llm_gateway import estimate_tokens

log = logging.getLogger(__name__)

DOC = "kpi_definitions.yaml"
# Sections that must stay whole whatever their size (structure carries the meaning).
ATOMIC = {"sql_recipe", "formulas", "example"}
SECTION_TITLES = {"logic": "How it is computed", "formulas": "Formulas", "caveats": "Caveats / known issues",
                  "sql_recipe": "SQL recipe", "filters": "Filters", "response": "API response shape",
                  "example": "Worked example", "tables": "Tables used"}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    kind: str                         # kpi_overview | kpi_section | gold_table | pipeline
    section: str
    doc: str = DOC
    parent: str | None = None
    prev: str | None = None
    next: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


# --------------------------------------------------------------------------- #
# rendering a YAML value as readable text (structure preserved)
# --------------------------------------------------------------------------- #

def _render_item(item: Any) -> str:
    if isinstance(item, dict):
        if set(item) <= {"step", "detail"}:
            return f"- {item.get('step', '')}: {item.get('detail', '')}".strip()
        return "- " + "; ".join(f"{k}: {v}" for k, v in item.items())
    return f"- {item}"


def _render(value: Any) -> list[str]:
    """A section value as a list of indivisible text items."""
    if isinstance(value, list):
        return [_render_item(v) for v in value]
    if isinstance(value, dict):
        return [f"- {k}: {v}" for k, v in value.items()]
    return [str(value).strip()]


def _group(items: list[str], max_tokens: int) -> list[list[str]]:
    """Consecutive items packed into groups of at most max_tokens; an item larger
    than max_tokens forms its own group (never cut)."""
    groups: list[list[str]] = [[]]
    for it in items:
        if groups[-1] and estimate_tokens("\n".join(groups[-1] + [it])) > max_tokens:
            groups.append([])
        groups[-1].append(it)
    return [g for g in groups if g]


# --------------------------------------------------------------------------- #
# KPI entries
# --------------------------------------------------------------------------- #

def chunk_kpi(kpi: dict, *, min_tokens: int, max_tokens: int, code: str | None = None) -> list[Chunk]:
    code = code or str(kpi.get("kpi_code") or kpi.get("number"))
    title = str(kpi.get("title") or code)
    head = f"KPI {kpi.get('number', '')} {title} ({kpi.get('section', '')})".strip()
    parent_id = f"kpi:{code}"
    overview = "\n".join(filter(None, [
        head, str(kpi.get("card") or ""), "Also called: " + ", ".join(map(str, kpi.get("aliases") or [])),
        str(kpi.get("summary") or "")]))
    chunks = [Chunk(parent_id, overview, "kpi_overview", "overview", meta={"kpi_code": code, "title": title})]

    sections: list[tuple[str, list[str]]] = []
    tables = [f"- gold: {t.get('table')} ({t.get('columns', '')})" for t in kpi.get("gold_tables") or []] + \
             [f"- silver: {t.get('table')} ({t.get('columns', '')}; from {t.get('source', '')})" for t in kpi.get("silver_tables") or []]
    if tables:
        sections.append(("tables", tables))
    # No "example": worked examples are a dated run ('25 Sep 2026: 17,763 closed tickets ...').
    # The data refreshes every day, so an explanation must never quote them as today's figures.
    for key in ("logic", "formulas", "caveats", "sql_recipe", "filters", "response"):
        if kpi.get(key):
            sections.append((key, _render(kpi[key])))

    pieces: list[tuple[str, list[str]]] = []
    for key, items in sections:
        groups = [items] if key in ATOMIC else _group(items, max_tokens)
        for n, g in enumerate(groups, 1):
            pieces.append((f"{key}" if len(groups) == 1 else f"{key}#{n}", g))

    # Merge a too-small piece into the previous one of the SAME section family or,
    # failing that, the previous piece -- unless that would break the max (atomic stays alone).
    merged: list[tuple[str, list[str]]] = []
    for name, items in pieces:
        small = estimate_tokens("\n".join(items)) < min_tokens
        if merged and small and merged[-1][0].split("#")[0] not in ATOMIC and name.split("#")[0] not in ATOMIC \
                and estimate_tokens("\n".join(merged[-1][1] + items)) <= max_tokens:
            prev_name, prev_items = merged[-1]
            merged[-1] = (f"{prev_name}+{name}", prev_items + [f"[{SECTION_TITLES.get(name.split('#')[0], name)}]"] + items)
        else:
            merged.append((name, items))

    for name, items in merged:
        label = " + ".join(SECTION_TITLES.get(n.split("#")[0], n) for n in name.split("+"))
        chunks.append(Chunk(f"kpi:{code}:{name}", f"{head} -- {label}\n" + "\n".join(items), "kpi_section",
                            name, parent=parent_id, meta={"kpi_code": code, "title": title}))
    siblings = chunks[1:]
    for a, b in zip(siblings, siblings[1:]):
        a.next, b.prev = b.chunk_id, a.chunk_id
    return chunks


def chunk_gold_tables(gold: dict) -> list[Chunk]:
    out = []
    for name, t in (gold or {}).items():
        body = "\n".join(_render(t) if isinstance(t, (list, str)) else [f"- {k}: {v}" for k, v in t.items()])
        out.append(Chunk(f"table:{name}", f"Gold table {name} (how it is built)\n{body}", "gold_table", "gold_tables",
                         meta={"table": name}))
    return out


def chunk_document(data: dict, *, min_tokens: int, max_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    kpis = data.get("kpis") or []
    # Two KPIs can share a backend kpi_code (e.g. MI vs SAT and the funnel): ids stay unique.
    codes = Counter(str(k.get("kpi_code")) for k in kpis)
    for kpi in kpis:
        code = str(kpi.get("kpi_code") or kpi.get("number"))
        if codes[code] > 1:
            code = f"{code}-{kpi.get('number')}"
        chunks.extend(chunk_kpi(kpi, min_tokens=min_tokens, max_tokens=max_tokens, code=code))
    chunks.extend(chunk_gold_tables(data.get("gold_tables") or {}))
    if data.get("pipeline_steps"):
        chunks.append(Chunk("pipeline", "Data pipeline (SmartEMS -> Bronze -> Silver -> Gold -> Postgres -> dashboard)\n"
                            + "\n".join(_render(data["pipeline_steps"])), "pipeline", "pipeline_steps"))
    stats = Counter(c.kind for c in chunks)
    toks = sorted(c.tokens for c in chunks)
    if toks:
        log.info("chunking: %s -> %d chunks %s, tokens min %d / median %d / max %d", DOC, len(chunks), dict(stats),
                 toks[0], toks[len(toks) // 2], toks[-1])
    return chunks


def expand(chunk_ids: list[str], by_id: dict[str, Chunk]) -> list[str]:
    """Selected chunks plus each child's parent overview (once), parents first --
    precise retrieval, with the context needed to read a section on its own."""
    out: list[str] = []
    for cid in chunk_ids:
        c = by_id.get(cid)
        if c is None:
            continue
        if c.parent and c.parent not in out:
            out.append(c.parent)
        if cid not in out:
            out.append(cid)
    return out
