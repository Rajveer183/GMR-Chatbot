"""Wrong years in an answer ('October 2025' when the data is 2026), fixed by code.

number_check skips years on purpose (1990..2100 would otherwise need to be result values).
That let 11 Nov's 'SAT completion kyun kam hua pichle mahine?' answer say 'Pichle mahine
(October 2025)' -- the SQL used CURRENT_DATE (2026) and no 2025 appeared anywhere.

Checked: a month name followed by a year ('Oct 2025', 'October, 2025'). Supported years:
any year in the result rows, in the question / context text, the current year, and the
previous one when the question says last year. A month-year that fails is rewritten to the
year the result has for that month, else that month's most recent occurrence up to today --
no LLM call. A bare year without a month is left alone (too ambiguous to fix safely).
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MONTH_YEAR = re.compile(
    r"\b(?P<m>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?,?\s+(?P<y>(?:19|20)\d{2})\b", re.I)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_LAST_YEAR = re.compile(r"last year|previous year|pichl[ea] saal|year before|\byoy\b|year[- ]on[- ]year", re.I)


def _frame_years(frames: list) -> dict[int, set[int]]:
    """{month: years} seen in the rows' dates, plus key 0 = every year seen at all."""
    out: dict[int, set[int]] = {0: set()}
    for f in frames or []:
        if not isinstance(f, pd.DataFrame) or f.empty:
            continue
        for c in f.columns:
            s = f[c].head(500)
            if pd.api.types.is_numeric_dtype(s):
                if re.search(r"year", str(c), re.I):
                    out[0] |= {int(v) for v in s.dropna() if 1990 <= v <= 2100}
                continue
            text = s.astype(str)
            for v in text:
                for y in _YEAR.findall(v):
                    out[0].add(int(y))
            d = pd.to_datetime(text, errors="coerce", format="mixed") if text.str.contains(r"\d{4}-\d{2}").any() else None
            if d is not None:
                for t in d.dropna():
                    out.setdefault(t.month, set()).add(t.year)
            for v in text:
                for m in _MONTH_YEAR.finditer(v):
                    out.setdefault(_MONTHS[m.group("m")[:3].lower()], set()).add(int(m.group("y")))
    return out


def fix(text: str, frames: list, given: str, today: date | None = None) -> tuple[str, list[str]]:
    """(text with unsupported month-years corrected, the 'Month YYYY' strings that were wrong)."""
    if not text or not _MONTH_YEAR.search(text):
        return text, []
    today = today or date.today()
    seen = _frame_years(frames)
    ok = set(seen[0]) | {int(y) for y in _YEAR.findall(given or "")} | {today.year}
    if _LAST_YEAR.search(given or ""):
        ok.add(today.year - 1)
    wrong: list[str] = []

    def swap(m: re.Match) -> str:
        y, month = int(m.group("y")), _MONTHS[m.group("m")[:3].lower()]
        if y in ok or y in seen.get(month, set()):
            return m.group(0)
        have = seen.get(month, set())
        right = max(have) if have else (today.year if month <= today.month else today.year - 1)
        if right == y:
            return m.group(0)
        wrong.append(m.group(0))
        return m.group(0)[: m.start("y") - m.start()] + str(right)

    return _MONTH_YEAR.sub(swap, text), wrong


# Weekdays. 11 Nov: 'Kal (10 Nov 2026) par 2,342 meters lage ... Kal Sunday tha, isliye technicians kam the' --
# 10 Nov 2026 was a Tuesday, and the 'reason' was built on the wrong day. A sentence naming a weekday that does
# not match the one date the answer is about is dropped (fixing the word would keep the invented reasoning).
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY_RE = re.compile(r"\b(?:" + "|".join(_WEEKDAYS) + r"|somvar|mangalvar|budhvar|guruvar|shukravar|shanivar|"
                         r"ravivar|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार)\b", re.I)
_HINDI_DAYS = {"somvar": 0, "mangalvar": 1, "budhvar": 2, "guruvar": 3, "shukravar": 4, "shanivar": 5, "ravivar": 6,
               "सोमवार": 0, "मंगलवार": 1, "बुधवार": 2, "गुरुवार": 3, "शुक्रवार": 4, "शनिवार": 5, "रविवार": 6}
_DAY_DATE = re.compile(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+((?:19|20)\d{2})\b"
                       r"|\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", re.I)


def _dates_in(text: str) -> set[date]:
    out = set()
    for m in _DAY_DATE.finditer(text):
        try:
            if m.group(1):
                out.add(date(int(m.group(3)), _MONTHS[m.group(2)[:3].lower()], int(m.group(1))))
            else:
                out.add(date(int(m.group(4)), int(m.group(5)), int(m.group(6))))
        except ValueError:
            continue
    return out


def drop_wrong_weekdays(text: str) -> tuple[str, list[str]]:
    """(text without sentences whose weekday contradicts the single date the text names, those sentences)."""
    if not text or not _WEEKDAY_RE.search(text):
        return text, []
    days = _dates_in(text)
    if len(days) != 1:
        return text, []
    right = next(iter(days)).weekday()
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    keep, dropped = [], []
    for s in sentences:
        named = {(_HINDI_DAYS.get(w.lower()) if w.lower() in _HINDI_DAYS else _WEEKDAYS.index(w.lower()))
                 for w in _WEEKDAY_RE.findall(s)}
        (dropped if named and right not in named else keep).append(s)
    return (" ".join(keep) if dropped else text), dropped
