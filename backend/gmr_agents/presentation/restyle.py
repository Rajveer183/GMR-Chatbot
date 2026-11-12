"""Natural-language chart changes and conversation recaps on the previous answer -- code only.

    'make that a bar chart' / 'show it as a pie' / 'as a line chart sorted descending'
        -> the previous answer's own chart redrawn as that type (same rows: no LLM call, no
           query). Before this the follow-up went through the pipeline again and lost the
           previous filter -- live 11 Nov: '6 months of installations' -> 'make that a bar
           chart' came back with all 30 months.
    'summarise what we have found so far' / 'recap this conversation'
        -> each earlier answer's question and lead sentence, in order (no LLM call).

Only a message that is NOTHING but the change matches (a new measure, place or period
goes through the pipeline as before).
"""

from __future__ import annotations

import copy
import re

from gmr_agents.presentation import blocks

_TYPES = [
    (r"horizontal bar|bar ?h|horizontal", "horizontal_bar"), (r"stacked bar|stacked", "stacked_bar"),
    (r"grouped bar|grouped|side[- ]by[- ]side", "grouped_bar"), (r"pie", "pie"), (r"do(?:ugh)?nut", "donut"),
    (r"line|trend ?line", "line"), (r"area", "area"), (r"bar|column", "bar"), (r"table", "table")]
_TYPE_ALT = "|".join(p for p, _ in _TYPES)
RESTYLE_RE = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+|now\s+|ok\s+|okay\s+)*(?:(?:make|turn|change|convert|switch|show|draw|display|plot|redraw|put|give)"
    r"(?:\s+(?:it|that|this|them|the same|same|the chart|the graph|this chart|that chart|the data|this data))?"
    r"(?:\s+(?:in|into|to|as|with|in the form of))?|(?:as|in))(?:\s+(?:a|an))?\s*"
    rf"(?P<type>{_TYPE_ALT})(?:\s+(?:chart|graph|plot|view|format))?(?:\s+instead)?"
    r"(?:\s*,?\s*(?:and\s+)?(?:sorted|sort(?: it)?|ordered|order(?: it)?)\s*(?:by value\s*)?(?P<sort>desc\w*|asc\w*|high(?:est)? to low\w*|low(?:est)? to high\w*|largest first|smallest first))?"
    r"\s*(?:please)?\s*[.!?]*\s*$", re.I)

RECAP_RE = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+)*(?:summar(?:ise|ize)|recap|sum up|give me a (?:summary|recap)( of)?|what have we (?:found|learned|learnt|covered))\b"
    r"(?:.*\b(?:so far|this (?:conversation|chat|session|discussion)|our (?:conversation|chat|discussion)|we (?:have )?(?:found|discussed|covered|looked at|seen)|until now|till now))?\s*[.!?]*\s*$",
    re.I)
_RECAP_SCOPE = re.compile(r"so far|conversation|chat|session|discussion|we (?:have )?(?:found|discussed|covered|looked|seen)|until now|till now|what have we", re.I)


def restyle_request(question: str) -> tuple[str, str | None] | None:
    """(chart type, 'desc' / 'asc' / None) when the message only asks to redraw the last chart."""
    m = RESTYLE_RE.match(question or "")
    if not m:
        return None
    word = m.group("type").lower()
    ctype = next(t for p, t in _TYPES if re.fullmatch(p, word))
    s = (m.group("sort") or "").lower()
    sort = "desc" if s.startswith(("desc", "high", "largest")) else "asc" if s else None
    return ctype, sort


def is_recap(question: str) -> bool:
    q = question or ""
    return bool(RECAP_RE.match(q) and _RECAP_SCOPE.search(q))


_LABELS = {"horizontal_bar": "horizontal bar chart", "stacked_bar": "stacked bar chart", "grouped_bar": "grouped bar chart",
           "pie": "pie chart", "donut": "donut chart", "line": "line chart", "area": "area chart", "bar": "bar chart",
           "table": "table"}


def restyle(previous: dict, ctype: str, sort: str | None) -> dict | None:
    """The previous response with its primary chart redrawn (None when it had no chart or table)."""
    resp = copy.deepcopy(previous)
    charts = resp.get("charts") or []
    if ctype == "table":
        if not resp.get("tables"):
            return None
        resp["charts"] = []
        resp["summary"] = "Here is the same data as a table."
        return resp
    if not charts:
        return None
    c = charts[0]
    cats = [str(x) for x in c.get("x") or []]
    series = copy.deepcopy(c.get("series") or [{"name": c.get("title") or "Value", "data": c.get("y") or []}])
    series = [{"name": s.get("name"), "data": list(s.get("data") or [])} for s in series]
    if not cats or not series or any(len(s["data"]) != len(cats) for s in series):
        return None
    if sort:
        order = sorted(range(len(cats)), key=lambda i: (series[0]["data"][i] or 0), reverse=(sort == "desc"))
        cats = [cats[i] for i in order]
        for s in series:
            s["data"] = [s["data"][i] for i in order]
    final, why = blocks._chart_fallback(ctype, cats, series, 0)
    new = {"id": c.get("id") or "primary_chart", "title": c.get("title"), "type": final, "x": cats,
           "y": series[0]["data"], **({"series": series} if len(series) > 1 else {}),
           **({"unit": c["unit"]} if c.get("unit") else {})}
    if why:
        new["chosen_type"], new["fallback_reason"] = ctype, why
    resp["charts"] = [new, *charts[1:]]
    note = f"Here is the same data as a {_LABELS.get(final, final.replace('_', ' '))}"
    if sort:
        note += ", sorted " + ("highest first" if sort == "desc" else "lowest first")
    note += "."
    if why:
        note += f" (A {_LABELS.get(ctype, ctype)} cannot show it well -- {why} -- so it is drawn as a {_LABELS.get(final, final)}.)"
    resp["summary"] = note
    resp["suggested_followups"] = previous.get("suggested_followups") or []
    return resp


def _lead(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", re.sub(r"[*_#`]", "", str(text or ""))).strip()
    m = re.match(r"(.+?[.!?])(?:\s|$)", text)
    first = m.group(1) if m and len(m.group(1)) >= 25 else text
    return first if len(first) <= limit else first[: limit - 3].rsplit(" ", 1)[0] + "..."


def recap(messages: list[dict]) -> dict:
    """Earlier answers of this session: question + lead sentence each, newest last."""
    turns, question = [], None
    for m in messages:
        if m.get("role") == "user":
            question = m.get("content")
        elif m.get("role") == "assistant" and question:
            r = m.get("response") or {}
            dbg = r.get("debug") or {}
            if not r.get("refused") and not dbg.get("recap") and not dbg.get("chart_restyle"):
                turns.append((question, _lead(m.get("content") or r.get("summary") or "")))
            question = None
    if not turns:
        return {"summary": "Nothing has been answered in this conversation yet.", "insights": [], "tables": []}
    cols = ["#", "You asked", "What we found"]
    rows = [dict(zip(cols, [i, q, a])) for i, (q, a) in enumerate(turns, 1)]
    summary = f"So far in this conversation ({len(turns)} answer{'s' if len(turns) != 1 else ''}): " + " ".join(
        f"({i}) {a}" for i, (_, a) in enumerate(turns[-6:], max(1, len(turns) - 5)))
    table = {"id": "recap", "title": "This conversation so far", "heading": None, "columns": cols, "rows": rows,
             "page": 1, "page_size": len(rows), "total_rows": len(rows), "has_next": False, "has_previous": False}
    return {"summary": summary, "insights": [], "tables": [table]}
