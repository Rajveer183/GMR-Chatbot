"""Highest / lowest check for written answers (no LLM).

The number check (number_check.py) accepts any figure that is IN the result, so
'the lowest feeder count (48)' passes although February's 29 is lower. This check
reads each superlative claim -- highest / lowest / peak / most / fewest ... followed
by a number in the same clause -- finds the rows holding that number, and accepts
the claim only when the number is the max (or min) of its column, compared within
the same category when the result is broken down (month x category). Total rows
and the current partial period are left out, as in the FACTS the writer was given.
Claims whose number is not in the result are left to the number check.
"""

from __future__ import annotations

import re

import pandas as pd

_MAX = r"highest|peak(?:ed|s)?|most|maximum|max|largest|biggest|top"
_MIN = r"lowest|least|fewest|minimum|min|smallest|bottom"
# superlative, then up to 80 chars without a sentence end or another superlative, then the number
_CLAIM = re.compile(rf"\b(?P<word>{_MAX}|{_MIN})\b(?P<gap>(?:(?!\b(?:{_MAX}|{_MIN})\b)[^.;!?\d]){{0,80}}?)"
                    r"(?P<num>-?\d[\d,]*(?:\.\d+)?)(?!\s*%?\s*(?:of|out of)\b)", re.I)
_MIN_RE = re.compile(rf"^(?:{_MIN})$", re.I)
# Dates are not the claimed figure: 'lowest in May 2026 (51,946)' claims 51,946.
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATES = re.compile(rf"\d{{4}}-\d{{2}}(?:-\d{{2}})?|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?={_MON}\b)|"
                    rf"(?<![\w,])(?:19|20)\d\d(?![\d,])|\b{_MON}[-' ]\d{{2}}\b", re.I)


def _mask_dates(text: str) -> str:
    return _DATES.sub(lambda m: re.sub(r"\d", "x", m.group()), text)


def _label_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def _is_total(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"total", "grand total"})


def _levels(df: pd.DataFrame) -> pd.Series:
    """Rollup level of each row: which label columns read 'Total'. A flat result has one level."""
    labels = _label_cols(df)
    if not labels:
        return pd.Series("", index=df.index)
    return pd.concat([_is_total(df[c]) for c in labels], axis=1).astype(int).astype(str).agg("".join, axis=1)


def _body(df: pd.DataFrame, period_col: str | None, partial: set[str]) -> pd.DataFrame:
    """Rows a claim is checked against: no grand-total rows (first label 'Total') and no partial
    period. A multi-level ROLLUP keeps its subtotal rows -- each is compared only with rows of its
    own level (_levels), so 'AGRA ... Total ... Total' is a project row, not a skipped total.
    Seen live: ROLLUP(project, discom, zone, ...) dropped every project row and 'highest closed
    tickets (2,906)' for AGRA passed although KASHI had 13,128."""
    labels = _label_cols(df)
    keep = pd.Series(True, index=df.index)
    if labels:
        keep &= ~_is_total(df[labels[0]])
        keep &= ~pd.concat([_is_total(df[c]) for c in labels], axis=1).all(axis=1)
    if period_col and partial:
        keep &= ~df[period_col].astype(str).str.strip().isin(partial)
    return df[keep]


def wrong_claims(text: str, df: pd.DataFrame | None, period_col: str | None = None,
                 partial: set[str] | None = None) -> list[str]:
    """Superlative claims in `text` that the result contradicts, e.g.
    "lowest ... (48): FEEDER's lowest is 29 (2026-02-01)"."""
    if not isinstance(df, pd.DataFrame) or len(df) < 2 or not text:
        return []
    body = _body(df, period_col, partial or set())
    if len(body) < 2:
        return []
    labels = _label_cols(df)
    # Month x category: compare within the category, never across categories.
    group_col = next((c for c in labels if c != period_col), None) if period_col and len(labels) > 1 else None
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    level = _levels(body)
    problems = []
    for m in _CLAIM.finditer(_mask_dates(text)):
        raw = m.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        want_min = bool(_MIN_RE.match(m.group("word")))
        verdicts = []  # (ok, explanation) per place the number appears
        for col in num_cols:
            vals = pd.to_numeric(body[col], errors="coerce")
            hits = vals[(vals.round(decimals) - value).abs() < 10 ** -decimals / 2 + 1e-9]
            for idx in hits.index:
                scope = vals if group_col is None else vals[body[group_col] == body.at[idx, group_col]]
                scope = scope[level[scope.index] == level[idx]].dropna()
                if len(scope) < 2:
                    continue
                best = scope.min() if want_min else scope.max()
                ok = round(best, decimals) == round(value, decimals)
                who = f"{body.at[idx, group_col]}'s " if group_col else ""
                label_col = period_col or labels[0] if labels else None
                at = f" ({body.at[scope.idxmin() if want_min else scope.idxmax(), label_col]})" if label_col else ""
                fmt = f"{best:,.{decimals}f}"
                verdicts.append((ok, f"'{m.group('word')} ... {m.group('num')}' is wrong: {who}{col} "
                                     f"{'lowest' if want_min else 'highest'} is {fmt}{at}"))
        if verdicts and not any(ok for ok, _ in verdicts):
            problems.append(verdicts[0][1])
    return problems


def drop_sentences(text: str, df: pd.DataFrame | None, period_col: str | None = None,
                   partial: set[str] | None = None) -> str:
    """`text` without the sentences that carry a wrong superlative claim."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(p for p in parts if not wrong_claims(p, df, period_col, partial)).strip()
