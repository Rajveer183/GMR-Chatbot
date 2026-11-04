"""Number check for written answers: every figure in the summary / insights must
come from the query result.

The answer LLM is told to copy numbers digit-for-digit, but small models still
slip (1,277,180 written as 1,127,180; 3,326,409 as 332,6409). This check finds
each number in the text and accepts it when it matches -- after rounding, or in
lakh / crore / million -- a value in the result, or a simple sum / difference /
share / ratio of two values (the prompt allows those). Anything else is reported
so the pipeline can ask the LLM to correct it, and `repair` fixes what is still
wrong afterwards: a near-miss (same length, at most two digits different) is
replaced by the result's value, otherwise the sentence carrying it is dropped.
"""

from __future__ import annotations

import bisect
import itertools
import math
import re

import pandas as pd

_UNIT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "l": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7,
         "million": 1e6, "mn": 1e6, "m": 1e6, "k": 1e3, "thousand": 1e3, "billion": 1e9, "bn": 1e9}
# A number (Indian or western grouping, optional decimals) and an optional scale word.
_NUM = re.compile(r"(?<![\w.])(-?\d+(?:,\d+)*(?:\.\d+)?)\s*(%|lakhs?|lac|crores?|cr|million|mn|thousand|billion|bn|[lLkKmM])?(?![\w])")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}:\d{2}\b")


def _grid(df: pd.DataFrame) -> list[list[float | None]]:
    rows = []
    num = df.apply(lambda c: pd.to_numeric(c, errors="coerce")).head(200)
    for r in num.itertuples(index=False):
        rows.append([None if pd.isna(v) or not math.isfinite(float(v)) else float(v) for v in r])
    return rows


def _values(frames: list[pd.DataFrame]) -> list[float]:
    out = {v for df in frames if isinstance(df, pd.DataFrame) and not df.empty
           for row in _grid(df) for v in row if v is not None}
    return sorted(out)


_PCT_COL = re.compile(r"pct|percent|rate|share|efficiency|ratio", re.I)
_MONEY_BEFORE = re.compile(r"(?:rs\.?|inr|₹)\s*$", re.I)


class Allowed(set):
    """The values an answer may state; `.money` leaves out percentage columns, so
    'Rs 65.83 crore' is not accepted just because 65.83 is a % in the result."""
    money: set


def allowed_values(frames: list[pd.DataFrame], extra_text: str = "") -> Allowed:
    """Every value an answer may state: result cells and column totals, plus the
    sum, difference, share (a / b * 100) and ratio of two values in the SAME row
    or the SAME column (Feb vs Jan, SAT / MI, project / total), and a share's
    complement (100 - x); plus the numbers in `extra_text` -- everything the answer
    writer was GIVEN besides the result: the question, the earlier answers it was
    shown, the KPI notes."""
    out = Allowed(_allowed(frames, extra_text, keep_pct=True))
    out.money = _allowed(frames, extra_text, keep_pct=False)
    return out


def _allowed(frames: list[pd.DataFrame], extra_text: str, keep_pct: bool) -> set[float]:
    allowed: set[float] = {abs(n) for n, _u in _numbers(extra_text)}

    def pairs(vals: list[float]) -> None:
        vals = vals[:60]
        allowed.update(vals)
        if keep_pct:  # the complement of a share: 97.56% installed -> 2.44% not installed
            allowed.update(100 - v for v in vals if 0 <= v <= 100)
        for x, y in itertools.permutations(vals, 2):
            allowed.update((x + y, abs(x - y)))
            if y and keep_pct:  # a ratio / share is never a money amount
                allowed.update((x / y * 100, x / y, (x - y) / y * 100, abs(x - y) / abs(y) * 100))

    for df in frames:
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        # Numbers inside label cells ('91-120 days', '120+ days', 'SAT-10') are stated by the
        # result too: '120' was once flagged and the repair cut the answer to "shows 10,".
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                for v in df[c].dropna().astype(str).head(200):
                    allowed.update(abs(n) for n, _u in _numbers(v))
        if not keep_pct:
            df = df[[c for c in df.columns if not _PCT_COL.search(str(c))]]
            if df.empty or not len(df.columns):
                continue
        grid = _grid(df)
        for row in grid:
            pairs([v for v in row if v is not None])
        for c in range(len(df.columns)):
            col = [row[c] for row in grid if row[c] is not None]
            if col:
                allowed.add(sum(col))
                pairs(col + ([sum(col)] if len(col) > 1 else []))
    return {abs(v) for v in allowed if math.isfinite(v)}


def _numbers(text: str) -> list[tuple[float, str]]:
    text = _DATE.sub(" ", str(text or ""))
    out = []
    for raw, unit in _NUM.findall(text):
        try:
            out.append((float(raw.replace(",", "")), (unit or "").lower()))
        except ValueError:
            continue
    return out


def _tolerance(raw: str, scale: float) -> float:
    """A decimal may be rounded OR truncated: one unit of its last digit ('10.72%' for
    10.7259 -- measured as a frequent false flag, each costing a correction call;
    '42.15 lakh' -> 1,000, '86.7' -> 0.1). A whole number stays exact (half a unit: '363' -> 0.5)."""
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return (1.0 if decimals else 0.5) * 10 ** -decimals * scale + 1e-9


def _supported(raw: str, unit: str, allowed: set[float], sorted_allowed: list[float]) -> bool:
    value = abs(float(raw.replace(",", "")))
    options = [(value, 1.0)]
    if unit in _UNIT:
        options.append((value * _UNIT[unit], _UNIT[unit]))  # 42.15 lakh -> 4,215,000
    for v, scale in options:
        tol = _tolerance(raw, scale)
        i = bisect.bisect_left(sorted_allowed, v - tol)
        if i < len(sorted_allowed) and sorted_allowed[i] <= v + tol:
            return True
    return False


_GROUPED = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?|-?\d{1,2}(,\d{2})*,\d{3}(\.\d+)?")


def unsupported(text: str, allowed: set[float]) -> list[str]:
    """The numbers in `text` (as written) that no allowed value supports. Small whole
    numbers (3 risks, 30 days, SAT-13) and years are not checked; every figure from
    100 up and every decimal / percentage / lakh-crore figure is."""
    sorted_allowed = sorted(allowed)
    sorted_money = sorted(getattr(allowed, "money", allowed))
    bad = []
    clean = _DATE.sub(" ", str(text or ""))
    for m in _NUM.finditer(clean):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if not (unit in _UNIT or unit == "%" or "." in raw or abs(value) >= 100):
            continue
        if 1990 <= value <= 2100 and "," not in raw and "." not in raw:
            continue  # years
        if "," in raw and not _GROUPED.fullmatch(raw):
            bad.append(m.group(0).strip())  # malformed grouping like 332,6409
        elif not _supported(raw, unit, allowed, sorted_money if _is_money(clean, m, unit) else sorted_allowed):
            bad.append(m.group(0).strip())
    return bad


def _is_money(text: str, m: re.Match, unit: str) -> bool:
    return unit in ("crore", "crores", "cr") or bool(_MONEY_BEFORE.search(text[max(0, m.start() - 6):m.start()]))


def _near_miss(raw: str, values: list[float]) -> float | None:
    """The result value written with the same number of digits and at most two of
    them different (a transcription slip), when exactly one such value exists."""
    digits = re.sub(r"\D", "", raw.split(".")[0])
    if len(digits) < 4:
        return None
    hits = set()
    for v in values:
        vd = str(int(round(abs(v))))
        if len(vd) == len(digits) and sum(a != b for a, b in zip(vd, digits)) <= 2:
            hits.add(int(round(abs(v))))
    return hits.pop() if len(hits) == 1 else None


def _closest(raw: str, values: list[float]) -> str | None:
    """A figure within 0.5% of exactly one result value (561.37 when the result
    says 561.0) -> that value, written like the result (same decimals)."""
    value = abs(float(raw.replace(",", "")))
    near = {v for v in values if v and abs(abs(v) - value) / abs(v) <= 0.005}
    if len(near) != 1:
        return None
    v = abs(near.pop())
    if float(v).is_integer():
        return _fmt_like(raw, int(v)) if "," in raw or v >= 1000 else str(int(v))
    return f"{v:,.4f}".rstrip("0").rstrip(".")


def _fmt_like(raw: str, value: int) -> str:
    if "," not in raw:
        return str(value)
    if re.search(r"\d,\d{2},\d{3}", raw):  # Indian grouping 12,77,180
        s = str(value)
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        return ",".join(parts + [tail]) if parts else tail
    return f"{value:,}"


def repair(text: str, allowed: set[float], frames: list[pd.DataFrame]) -> tuple[str, list[str]]:
    """Fix what is still unsupported: replace near-miss figures with the result's
    value; drop sentences whose figure has no match. Returns (text, problems left)."""
    values = _values(frames)
    for bad in unsupported(text, allowed):
        raw = re.match(r"-?\d[\d,]*(?:\.\d+)?", bad)
        fix = _near_miss(raw.group(0), values) if raw else None
        if fix is not None:
            text = text.replace(raw.group(0), _fmt_like(raw.group(0), fix), 1)
            continue
        snap = _closest(raw.group(0), values) if raw else None
        if snap is not None:
            text = text.replace(raw.group(0), snap, 1)
    # Smallest cut first: a bracketed aside or a comma / dash clause holding the bad figure
    # ("= 505,761.36 (Rs 505.76 lakh)") -- then whole sentences, if others remain.
    for bad in unsupported(text, allowed):
        aside = re.search(r"\s*[(\[][^()\[\]]*" + re.escape(bad) + r"[^()\[\]]*[)\]]", text)
        if aside is None:
            aside = re.search(r"\s*[,;–—-][^,;.–—]*" + re.escape(bad) + r"[^,;.–—]*(?=[,;.]|$)", text)
        if aside is not None:
            text = (text[:aside.start()] + text[aside.end():]).strip()
    left = unsupported(text, allowed)
    if left:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = [s for s in sentences if not unsupported(s, allowed)]
        if kept:
            text = " ".join(kept)
    return text, unsupported(text, allowed)
