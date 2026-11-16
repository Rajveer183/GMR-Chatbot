"""Festival / holiday names as dates ('since Diwali', 'after Holi', 'before Republic Day').

16 Nov 2026: 'How many meters were installed since Diwali?' was answered from Diwali 2025 (20 Oct 2025, 2.4
million meters) when Diwali 2026 was eight days earlier. The model does not know the lunar-calendar dates, so
code pins them: the most recent occurrence on or before today, or the next one when the question looks ahead.
"""

from __future__ import annotations

import re
from datetime import date
from gmr_agents.execution.ist_today import today as _ist_today  # IST date on any machine

# Main festival day as observed in Uttar Pradesh. Fixed-date holidays are generated per year below.
_LUNAR = {
    "diwali": ["2024-11-01", "2025-10-20", "2026-11-08", "2027-10-29"],
    "dhanteras": ["2025-10-18", "2026-11-06", "2027-10-27"],
    "holi": ["2025-03-14", "2026-03-04", "2027-03-22"],
    "dussehra": ["2024-10-12", "2025-10-02", "2026-10-20", "2027-10-09"],
    "chhath": ["2025-10-28", "2026-11-15", "2027-11-04"],
    "raksha bandhan": ["2025-08-09", "2026-08-28", "2027-08-17"],
    "janmashtami": ["2025-08-16", "2026-09-04", "2027-08-25"],
}
_FIXED = {"republic day": (1, 26), "independence day": (8, 15), "gandhi jayanti": (10, 2), "christmas": (12, 25),
          "makar sankranti": (1, 14), "new year": (1, 1)}
_ALIASES = {"deepawali": "diwali", "deepavali": "diwali", "dipawali": "diwali", "diwaali": "diwali", "दिवाली": "diwali",
            "dasara": "dussehra", "dussera": "dussehra", "dasehra": "dussehra", "vijayadashami": "dussehra",
            "होली": "holi", "chhath puja": "chhath", "chath": "chhath", "rakhi": "raksha bandhan",
            "janmashtmi": "janmashtami", "xmas": "christmas", "15 august": "independence day",
            "26 january": "republic day", "new year's day": "new year"}
_NAMES = sorted({*_LUNAR, *_FIXED, *_ALIASES}, key=len, reverse=True)
_NAME_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in _NAMES) + r")(?!\w)", re.I)
_AHEAD_RE = re.compile(r"\b(?:next|upcoming|coming|till|until|by|before)\b", re.I)
_YEAR_RE = re.compile(r"\b(20\d\d)\b")


def _dates(name: str) -> list[date]:
    if name in _LUNAR:
        return [date.fromisoformat(d) for d in _LUNAR[name]]
    m, d = _FIXED[name]
    return [date(y, m, d) for y in range(2024, 2029)]


def resolve(question: str, today: date | None = None) -> list[tuple[str, date]]:
    """(name as written, date) for each festival named; [] when none (or no known date)."""
    today = today or _ist_today()
    out = []
    for m in _NAME_RE.finditer(question or ""):
        written = m.group(1)
        if written.lower() in ("15 august", "26 january"):
            continue  # already a date the planner reads
        name = _ALIASES.get(written.lower(), written.lower())
        days = _dates(name)
        tail = question[m.end():m.end() + 8]
        year = _YEAR_RE.search(tail) or _YEAR_RE.search(question[max(0, m.start() - 8):m.start()])
        if year:
            pick = [d for d in days if d.year == int(year.group(1))]
        elif _AHEAD_RE.search(question[max(0, m.start() - 25):m.start()]) and not re.search(
                r"\b(?:since|after|from|last)\b", question[max(0, m.start() - 25):m.start()], re.I):
            pick = [d for d in days if d > today][:1] or [d for d in days if d <= today][-1:]
        else:
            pick = [d for d in days if d <= today][-1:]
        if pick:
            out.append((written, pick[0]))
    return out


def annotate(question: str, today: date | None = None) -> str | None:
    """The question with '(Diwali = 08 Nov 2026)' added, or None when nothing to add."""
    got = resolve(question, today)
    if not got:
        return None
    note = "; ".join(f"{w} = {d.strftime('%d %b %Y')}" for w, d in got)
    return re.sub(r"[\s?.]*$", "", question) + f" ({note})?"
