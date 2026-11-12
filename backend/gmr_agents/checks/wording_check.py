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


def drop_sentences(text: str, phrases: list[str]) -> str:
    """text without the sentences holding these phrases (unchanged if that would empty it)."""
    if not phrases:
        return text
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z*(\d])", text or "")
    keep = [p for p in parts if not any(ph in p for ph in phrases)]
    return " ".join(keep).strip() if keep else text
