"""Conversation place scope (no LLM): which project / discom / zone / circle / division a
message is about, decided the same way every time.

Seen live (2026-10-04, one chat):
    1. 'how much stock is older than 90 days in agra'  -> Agra
    2. 'how many unused meters are 0-30 days old'      -> ALL projects (complete sentence: new question)
    3. 'stock between 100 and 200 days old'            -> Agra again (taken from turn 1, skipping turn 2)
The understanding model judged 'follow-up or not' differently each time. The rule here:
  * the message names a place, or asks for everything ('overall', 'all projects', 'project
    wise') -> the message decides, nothing is added;
  * otherwise the LATEST turn's place is kept (carry) for any question in the same data area
    (WFM / MDM) -- topic changes included: 'non sat ageing for dtr meters in agra' then
    'feeder meters without sat' (another topic) must still be Agra (user, 2026-10-04);
  * a place in the resolved question that is in neither the message nor the latest turn's
    question / answer is removed -- an earlier turn never leaks back in (strip).
The resolved question is what the user sees as 'Interpreted as', so a carried place is visible.
"""

from __future__ import annotations

import re

from gmr_agents.validation.entity_check import _PROJECTS, _words

_RESET_RE = re.compile(
    r"\b(overall|all (projects|zones|discoms|circles|divisions|places)|whole (programme|program|project)|"
    r"programme|program[- ]wide|across (all|projects|the program)|everywhere|total (programme|program)|"
    r"(project|discom)[- ]?wise|by (project|discom)|each (project|discom)|per (project|discom)|sab|poore)\b",
    re.IGNORECASE)
_LEVEL_WORD = {"zone": "zone", "circle": "circle", "division": "division", "discom": "discom", "project": ""}
_ROMAN = {"i", "ii", "iii", "iv", "v"}
_UPPER = {"edd", "edc", "eudc", "eudd", "edsd", "puvvnl", "dvvnl", "mvvnl", "pvvnl", "kesco"}


def _vocab(places: dict[str, str]) -> list[tuple[str, str, str]]:
    """(text to match, level, canonical name); longest first, a project before a same-named zone."""
    out = [(alias, "project", key) for key, aliases in _PROJECTS.items() for alias in aliases]
    out += [(re.sub(r"[^a-z0-9]+", " ", name).strip(), lvl, name) for name, lvl in places.items()]
    return sorted({o for o in out if o[0]}, key=lambda o: (-len(o[0]), o[1] != "project"))


def found(text: str, places: dict[str, str]) -> dict[str, str]:
    """{canonical place: level} named in the text (a matched span is not matched again,
    so 'Varanasi I zone' is the zone, not also the Kashi project's alias 'varanasi')."""
    w = _words(text)
    out: dict[str, str] = {}
    for alias, lvl, name in _vocab(places):
        if f" {alias} " in w:
            out.setdefault(name, lvl)
            w = w.replace(f" {alias} ", "  ")
    return out


def display(name: str, level: str) -> str:
    words = [p.upper() if p in _ROMAN or p in _UPPER else p.capitalize() for p in re.split(r"\s+", name.strip())]
    text = " ".join(words).replace("Puvvnl", "PuVVNL").replace("PUVVNL", "PuVVNL").replace("DVVNL", "DVVNL")
    return f"{text} {_LEVEL_WORD[level]}".strip()


def _strip(question: str, name: str, level: str) -> str:
    alias = r"\s+".join(map(re.escape, re.sub(r"[^a-z0-9]+", " ", name).split()))
    aliases = [alias] + ([re.escape(a) for a in _PROJECTS.get(name, [])] if level == "project" else [])
    for a in aliases:
        question = re.sub(rf"\s*\b(in|for|of|at|across|within)\s+(the\s+)?{a}(\s+(project|zone|discom|circle|division))?\b",
                          "", question, flags=re.IGNORECASE)
        question = re.sub(rf"\b{a}'s\s+", "", question, flags=re.IGNORECASE)
    return re.sub(r"\s+([?.,])", r"\1", re.sub(r"\s{2,}", " ", question)).strip()


# Coarse -> fine. A question that COMPARES a level ('which project ...', 'zone wise', 'top 5
# divisions') is answered across ALL places of that level: memory may keep only a coarser place
# (Kashi for 'which zone is worst'), never one at that level or below. Seen live: 'Which project
# has the highest PMPM invoice percentage?' after a Kashi turn was answered for Kashi alone.
_RANK = {"project": 0, "discom": 0, "zone": 1, "circle": 2, "division": 3, "sub_division": 4}
_LEVELS_RE = r"(projects?|discoms?|zones?|circles?|divisions?|sub[- ]?divisions?)"
_COMPARE_RE = re.compile(
    rf"\b(?:which|what|each|every|all|per|by|top|bottom|compare|comparing|across|among|between|rank|ranking)\s+"
    rf"(?:\w+\s+){{0,2}}?{_LEVELS_RE}\b|\b{_LEVELS_RE}[- ]?wise\b|"
    rf"\b{_LEVELS_RE}\s+(?:has|have|is|are|with|had)\s+(?:the\s+)?(?:highest|lowest|best|worst|most|least|max|min|top)",
    re.IGNORECASE)


def compared_rank(message: str) -> int | None:
    """The coarsest place level the message compares across (0 = project), or None."""
    ranks = [_RANK[re.sub(r"s$", "", w.lower()).replace("-", "_").replace(" ", "_").replace("subdivision", "sub_division")]
             for m in _COMPARE_RE.finditer(message) for w in m.groups() if w]
    return min(ranks) if ranks else None


def apply(message: str, resolved: str, latest: dict | None, same_area: bool,
          places: dict[str, str]) -> tuple[str, dict]:
    """(resolved question with the place scope fixed, what changed {carried, stripped})."""
    if not latest or found(message, places):
        return resolved, {}
    if _RESET_RE.search(message):
        # 'overall feeder meters without sat' after an Agra turn was resolved as Agra's figure.
        changes: dict = {}
        for name, lvl in found(resolved, places).items():
            stripped = _strip(resolved, name, lvl)
            if name not in found(stripped, places):
                resolved = stripped
                changes.setdefault("stripped", []).append(name)
        return resolved, changes
    rank = compared_rank(message)
    latest_q = str(latest.get("resolved_question") or latest.get("question") or "")
    in_latest_q = {n: lvl for n, lvl in found(latest_q, places).items() if rank is None or _RANK[lvl] < rank}
    allowed = {**in_latest_q, **{n: lvl for n, lvl in found(str(latest.get("summary") or ""), places).items()
                                 if rank is None or _RANK[lvl] < rank}}
    changes: dict = {}
    for name, lvl in found(resolved, places).items():
        if name not in allowed:
            stripped = _strip(resolved, name, lvl)
            if name not in found(stripped, places):
                resolved = stripped
                changes.setdefault("stripped", []).append(name)
    if in_latest_q and same_area:
        missing = [(n, lvl) for n, lvl in in_latest_q.items() if n not in found(resolved, places)]
        if missing:
            where = " and ".join(display(n, lvl) for n, lvl in missing)
            resolved = re.sub(r"\s*([?.])?\s*$", "", resolved) + f" in {where}" + ("?" if resolved.rstrip().endswith("?") else "")
            changes["carried"] = [n for n, _ in missing]
    return resolved, changes
