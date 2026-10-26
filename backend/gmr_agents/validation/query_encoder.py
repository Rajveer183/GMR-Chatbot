"""Deterministic query encoder -- the user's question as structured slots, no LLM.

Who controls it: application code. The LLM understanding step (triage) is the
semantic encoder (intent, follow-up resolution, topics, tables); this adds the
exact, checkable slots a regex can read reliably, so the SQL writer sees them
explicitly and logs show what was asked:

    projects / oems / categories / sat_stages   named values (same tables as entity_check)
    years / months / dates / relative_period    the period asked for
    top_n / direction                           'top 5' -> 5, desc; 'bottom 3' / 'lowest' -> asc
    signals                                     compare, share, change, what_if_pct, trend, average,
                                                ranking, count, breakdown_by

It never decides anything on its own -- it informs the planner and the trace.
"""

from __future__ import annotations

import re

from gmr_agents.validation.entity_check import _CATEGORIES, _NUM_WORDS, _OEMS, _PROJECTS, _named, _words, meter_types

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_MONTH_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.IGNORECASE)
_RELATIVE = ["today", "yesterday", "last week", "this week", "last month", "this month", "last year", "this year",
             "last 7 days", "last 30 days", "last 3 months", "last 6 months", "last 12 months", "so far", "till date",
             "kal", "aaj"]
_LEVELS = ["project", "discom", "zone", "circle", "division", "sub_division", "substation", "feeder", "oem",
           "category", "month", "week", "day", "technician"]
# Other words users put before 'wise' / after 'by' for a level. Live, 26 Oct: 'make wise installation in Kashi
# last month' was not tagged as an OEM breakdown and came back as one total.
_LEVEL_ALIASES = {"oem": ["make", "vendor", "brand", "manufacturer", "company"], "project": ["cluster"],
                  "sub_division": ["sub division", "sub-division", "subdivision"]}


def encode(question: str) -> dict:
    q = str(question or "")
    low = q.lower()
    words = _words(q)
    out: dict = {}

    for key, table in (("projects", _PROJECTS), ("oems", _OEMS)):
        found = sorted(_named(q, table))
        if found:
            out[key] = found
    cats = [c for c, aliases in _CATEGORIES.items() if any(f" {a} " in words for a in aliases)]
    if cats:
        out["categories"] = cats
    types = meter_types(q)
    if types:
        out["meter_types"] = types
    stages = sorted(set(re.findall(r"\bsat[\s-]?(\d{1,2})\b", low)), key=int)
    if stages:
        out["sat_stages"] = [f"SAT-{s}" for s in stages]

    years = sorted(set(re.findall(r"\b(20\d\d)\b", q)))
    if years:
        out["years"] = years
    months = []
    for m in _MONTH_RE.findall(q):
        full = next(x for x in _MONTHS if x.startswith(m.lower()[:3]))
        if full not in months:
            months.append(full)
    if months:
        out["months"] = months
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
                       q, re.IGNORECASE)
    if dates:
        out["dates"] = dates
    rel = [r for r in _RELATIVE if re.search(rf"\b{re.escape(r)}\b", low)]
    if rel:
        out["relative_period"] = rel
    if re.search(r"\bas (of|on)\b", low):
        out["as_of"] = True

    top = re.search(r"\b(top|best|worst|bottom|lowest|highest|first|last)\s+(\d{1,3}|three|five|ten|fifteen|twenty)\b", low)
    if top:
        n = top.group(2)
        out["top_n"] = int(n) if n.isdigit() else _NUM_WORDS.get(n)
    # Only words whose sort order does not depend on the metric: 'best' / 'worst' / 'most'
    # do ('most reliable OEM' = LOWEST defect rate), so the planner decides those.
    if re.search(r"\b(bottom|lowest|least|fewest|minimum|smallest)\b", low):
        out["direction"] = "asc"
    elif re.search(r"\b(top|highest|maximum|largest|biggest)\b", low):
        out["direction"] = "desc"

    signals = []
    for name, pattern in (("compare", r"\b(compare|comparison|vs\.?|versus|difference|gap|how many more|than)\b"),
                          ("share", r"\b(share|percent(age)? of|proportion|contribution|% of)\b"),
                          ("change", r"\b(change|growth|grew|increase|decrease|decline|drop|rise|yoy|year on year|mom|month on month)\b"),
                          ("trend", r"\b(trend|over time|month-?wise|monthly|weekly|daily)\b"),
                          ("average", r"\b(average|avg|mean|per day|per month)\b"),
                          ("ranking", r"\b(rank|ranking|top|bottom|best|worst|highest|lowest|most|least|fewest)\b"),
                          ("count", r"\b(how many|count|number of|total)\b")):
        if re.search(pattern, low):
            signals.append(name)
    pct = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*%", q)
    if pct and re.search(r"\b(if|cut|reduce|clear|increase|grow|drop)\b", low):
        signals.append("what_if")
        out["what_if_pct"] = float(pct.group(1))
    if signals:
        out["signals"] = signals
    by = [lvl for lvl in _LEVELS
          if any(re.search(rf"\b(by|per|each|every|across|wise)\s+{re.escape(w)}s?\b|\b{re.escape(w)}[- ]?wise\b", low)
                 for w in [lvl, *_LEVEL_ALIASES.get(lvl, [])])]
    if "sub_division" in by and "division" in by and not re.search(r"(?<!sub )(?<!sub-)\bdivision[- ]?wise\b|\bby\s+division", low):
        by.remove("division")  # 'sub division wise' contains 'division wise'
    if by:
        out["breakdown_by"] = by
    return out


def prompt_line(encoded: dict) -> str:
    """One compact line for the SQL writer; empty when nothing was parsed."""
    if not encoded:
        return ""
    parts = [f"{k}={','.join(map(str, v)) if isinstance(v, list) else v}" for k, v in encoded.items()]
    return "PARSED FROM THE QUESTION (by code -- exact values to honour): " + "; ".join(parts)
