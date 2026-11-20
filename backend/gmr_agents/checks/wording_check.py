"""Wording checks on the written answer -- code only, no LLM call.

leaks(text)
    The answer prompt forbids pipeline words, yet models still write them. Live 11 Nov: 'RESULT contains
    200 rows of meter complaint data; ...', 'June data is not available in RESULT'. Upper-case pipeline
    labels (RESULT, RESULT 2, FACTS, DRIVERS ...) are rewritten into plain words.

unsupported_causes(text, vocabulary)
    A causal claim ('driven by X', 'due to X', 'caused by X', 'because of X') whose X the answer was never
    given: most of X's content words appear nowhere in the question, the rows (column names and values) or
    the code-computed blocks. Live 11 Nov: 'unusual installation days' -> 'These anomalies are driven by
    irregular batch cycles and feeder audit schedules' -- nothing in the data said so. The sentence is
    dropped (never the only sentence of the summary).
"""

from __future__ import annotations

import re

import pandas as pd

_LEAKS = [
    # 'RESULT contains 200 rows of meter complaint data; the total ...' -> 'The total ...' (a row count is pipeline talk)
    (re.compile(r"\b(?:the\s+)?RESULT(?:\s+\d+)?\s+(?:contains|has|holds|shows|lists)\s+\d[\d,]*\s+rows?\b[^;.]*[;.]\s*"), ""),
    (re.compile(r"\b(?:in|from|within)\s+(?:the\s+)?(?:RESULT|FACTS|DRIVERS|INVESTIGATION|STATS)(?:\s+\d+)?\b"), "in the data"),
    (re.compile(r"\b(?:the\s+)?(?:RESULT|FACTS)(?:\s+\d+)?\s+(?:shows?|indicates?|lists?|contains?|has)\b"), "the data shows"),
    # 12 Nov: '... with 115,910 tickets, excluding the ROLLUP Total row.' -- query mechanics, not business talk
    (re.compile(r"\s*[,;(]?\s*\b(?:excluding|ignoring|not counting|without|except)\s+(?:the\s+)?(?:ROLLUP\s+)?"
                r"(?:grand[- ])?total\s+rows?\b\)?", re.I), ""),
    (re.compile(r"\b(?:the\s+)?ROLLUP\s+(?:grand[- ])?total(\s+rows?)?\b", re.I), "the overall total"),
    (re.compile(r"\bRESULT(?:\s+\d+)?\b"), "the data"),
    (re.compile(r"\bFACTS\b"), "the figures"),
]

_CAUSE_RE = re.compile(r"\b(?:driven by|due to|caused by|because of|owing to|attributed to|attributable to|a result of|"
                       r"stem(?:s|ming)? from|explained by)\s+([^.;:!?]{3,140})", re.I)
_STOP = set("""the a an of and or to in on at for by with from this that these those its their our your than then rather
which who whom whose what when where while as is are was were be been being it they them there here more most less
least very much many some any all each every other such only also not no nor but if so into over under within across
between about after before during per via new old high higher low lower large larger small smaller""".split())


def leaks(text: str) -> str:
    out = text or ""
    for pat, rep in _LEAKS:
        out = pat.sub(rep, out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out[:1].upper() + out[1:] if out != (text or "").strip() else out


def vocabulary(frames: list, given: str) -> set[str]:
    words: set[str] = set()
    for df in frames:
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for c in df.columns:
            words |= set(re.findall(r"[a-z]{3,}", str(c).lower().replace("_", " ")))
        for c in df.columns:
            col = df[c]
            if isinstance(col, pd.DataFrame):          # duplicate column names: first one
                col = col.iloc[:, 0]
            if not pd.api.types.is_numeric_dtype(col):
                for v in col.head(300).tolist():
                    words |= set(re.findall(r"[a-z]{3,}", str(v).lower()))
    words |= set(re.findall(r"[a-z]{3,}", (given or "").lower().replace("_", " ")))
    return words


def _stem(w: str) -> str:
    return re.sub(r"(?:ies|es|s|ing|ed|al)$", "", w)


def unsupported_causes(text: str, vocab: set[str]) -> list[str]:
    """The causal phrases in text whose content words are mostly not in vocab."""
    stems = {_stem(w) for w in vocab}
    bad = []
    for m in _CAUSE_RE.finditer(text or ""):
        words = [w for w in re.findall(r"[a-z]{3,}", m.group(1).lower()) if w not in _STOP]
        if len(words) < 2:
            continue
        known = sum(1 for w in words if w in vocab or _stem(w) in stems)
        if known / len(words) < 0.5:
            bad.append(m.group(0).strip())
    return bad


# Three or more figures chained by dashes: a day-by-day series squashed into one token. Live 18 Nov: 'AGRA
# leading with 22-620-975 consumer installs ... DT installs (6-10-10-10-10-10-10-10-10-10)'. Every figure was in
# the result, so the number check passed it. Dates (2026-11-08, 08-11-2026) are not runs.
_DASH = r"\s?[-‐‑‒–—]\s?"
_FIG = r"\d+(?:,\d+)*(?:\.\d+)?"
_RUN_RE = re.compile(rf"(?<![\w.,]){_FIG}(?:{_DASH}{_FIG}){{2,}}(?![\w])")
_DATE_LIKE_RE = re.compile(r"^(?:\d{4}[-‑]\d{1,2}[-‑]\d{1,2}|\d{1,2}[-‑]\d{1,2}[-‑]\d{2,4})$")


def squashed_runs(text: str) -> list[str]:
    """Runs of 3+ figures joined by dashes ('22-620-975'), dates excluded."""
    return [m.group(0) for m in _RUN_RE.finditer(text or "") if not _DATE_LIKE_RE.match(m.group(0))]


# 'Not available' with no figure at all. Live 18 Nov: 'Kashi zone data is not available in the current snapshot.'
# for a query that had already kept only Kashi rows (21 rows, zones VARANASI I / II, AZAMGARH) -- 'Kashi' is the
# filter, not a column value, and the model read it as missing.
_EMPTY_CLAIM_RE = re.compile(
    r"\b(?:not available|unavailable|no data|no records?|not found|could ?n[o']t (?:find|locate)|"
    r"(?:is|are|was|were) not (?:present|recorded|reported|captured)|does not (?:contain|include|have)|"
    r"do not have|no (?:installations?|figures?|rows?|results?|entries) (?:were |was )?(?:found|recorded|available))\b",
    re.I)


def false_empty(text: str, df) -> bool:
    """The answer says there is no data, quotes no figure, and the result has non-zero figures outside Total rows."""
    if not text or re.search(r"\d", text) or not _EMPTY_CLAIM_RE.search(text) or "AI summary is unavailable" in text:
        return False   # (the last: the code's own 'AI busy' line, never the model's claim)
    if df is None or not hasattr(df, "select_dtypes") or df.empty:
        return False
    labels = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    rows = df
    if labels:
        rows = df[~df[labels].apply(lambda c: c.astype(str).str.strip().str.lower().isin({"total", "grand total"})).any(axis=1)]
    nums = rows.select_dtypes("number")
    return not nums.empty and bool((nums.fillna(0) != 0).to_numpy().any())


def quotes_nothing(text: str, df) -> bool:
    """The answer quotes nothing at all -- no figure (0 and 1 aside) and no name / code from RESULT's first rows --
    while RESULT has rows: only a caveat or a method note. 19 Nov: 'Feeders with the highest loss in July in Kashi'
    -> 'Feeder input carries the feeder meter's multiplying factor ... rankings are usable.' (10 rows held the
    feeders and their losses). Any written figure passes (whether it is right is number_check's job), and a
    'no data' claim is false_empty's job -- so a normal answer never costs an extra call here."""
    if not text or "AI summary is unavailable" in text or df is None or not hasattr(df, "select_dtypes") or df.empty:
        return False
    if _EMPTY_CLAIM_RE.search(text) or df.select_dtypes("number").empty:
        return False
    for raw in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", text):
        try:
            if float(raw.replace(",", "")) not in (0.0, 1.0):
                return False
        except ValueError:
            continue
    head = df.head(20)
    low = text.lower()
    for c in [c for c in head.columns if not pd.api.types.is_numeric_dtype(head[c])]:
        for v in head[c].astype(str).str.strip():
            if len(v) >= 3 and v.lower() not in ("total", "grand total", "none", "nan") and v.lower() in low:
                return False
    return True


def sql_filters(sql: str | None) -> list[str]:
    """The literal filters a query applied ('project = kashi', 'category = DT'): rows already inside them need not
    name them."""
    out = []
    for col, val in re.findall(r"(?:lower|upper|trim)?\(?\s*\b([a-z_]+)\)?\s*=\s*'([^']{1,40})'", sql or "", re.I):
        item = f"{col.lower()} = {val}"
        if item not in out and col.lower() not in ("then", "else", "when"):
            out.append(item)
    return out[:6]


# A share named against a base the result never had. Live 18 Nov: 'SAT-done 3,866,783 (87.68% of installed, 87.68% of
# eligible)' -- RESULT had no eligible column; 87.68 is the share of installed, the share of eligible is 94.40. The
# number check passed it (87.68 is in RESULT). Only bases that are separate KPIs in this programme are checked.
_BASES = ("eligible", "supplied", "received", "target", "invoiced", "inventory")
_PCT_OF_RE = re.compile(r"\d+(?:\.\d+)?\s?%\s+of\s+(?:the\s+|all\s+|total\s+)?(" + "|".join(_BASES) + r")\b[\w -]{0,40}?"
                        r"(?=[),.;:]|\s+(?:and|while|with|but)\b|$)", re.I)


def unsupported_bases(text: str, frames: list, question: str = "") -> list[str]:
    """'87.68% of eligible' when no result column / label and not the question names that base."""
    names = " ".join(str(c).lower().replace("_", " ") for df in frames if hasattr(df, "columns") for c in df.columns)
    labels = " ".join(str(v).lower() for df in frames if hasattr(df, "columns")
                      for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) for v in df[c].head(200))
    known = f"{names} {labels} {question.lower()}"
    # Stems: a 'total_supplies' / 'inventory_count' column supports '% of supplied' (receipts are the supply).
    stems = {"eligible": ("eligib",), "supplied": ("suppl", "invent", "receiv"), "received": ("receiv", "suppl", "invent"),
             "target": ("target",), "invoiced": ("invoic",), "inventory": ("invent", "suppl", "receiv")}
    return [m.group(0).strip() for m in _PCT_OF_RE.finditer(text or "")
            if not any(s in known for s in stems[m.group(1).lower()])]


def strip_phrases(text: str, phrases: list[str]) -> str:
    """text without these clauses ('(87.68% of installed, 87.68% of eligible)' -> '(87.68% of installed)');
    a clause that cannot be cut cleanly takes its sentence with it (never the whole text)."""
    out = text
    for ph in phrases:
        esc = re.escape(ph)
        new = re.sub(rf",\s*{esc}|{esc}\s*,\s*|\s*\(\s*{esc}\s*\)", "", out, count=1)
        out = new if new != out else drop_sentences(out, [ph])
    return out


def drop_sentences(text: str, phrases: list[str]) -> str:
    """text without the sentences holding these phrases (unchanged if that would empty it)."""
    if not phrases:
        return text
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z*(\d])", text or "")
    keep = [p for p in parts if not any(ph in p for ph in phrases)]
    return " ".join(keep).strip() if keep else text
