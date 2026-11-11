"""Several questions in one message -> the separate questions, each answered in turn.

Before: 'How many meters were installed yesterday? And what is the SAT % by project?' ran as one
question and only the first part was answered (the planner picked one KPI). Now each part runs
as its own turn (main.answer_message), so a later part can lean on an earlier one ('... in Kashi?
And by zone?') through the normal follow-up handling, and the answers are shown together.

Split only on clear separators -- never on a bare 'and' / 'aur' / 'vs', which join the parts of
ONE question ('installations and SAT by project', 'compare Agra and Kashi'):
    numbered parts        '1. ... 2. ...', '1) ... 2) ...', 'Q1 ... Q2 ...'
    separate lines        two or more lines that each read as a question / request
    question marks        '...? ...?' with words on both sides
    explicit add-ons      '; also ...', 'and also ...', 'also tell me ...', 'plus, ...', 'aur ye bhi ...'
A part must have at least 3 words (or be a short follow-up such as 'And by zone?'). At most
MAX_PARTS run; the rest are returned separately so the answer can say they were not answered.
"""

from __future__ import annotations

import re

MAX_PARTS = 3

_NUMBERED = re.compile(r"(?:^|\s)(?:\(?\d{1,2}[.)]|q\d{1,2}[.:)]?)\s+(?=\S)", re.I)
_ADDON = re.compile(
    r"\s*(?:;\s*(?:and\s+)?also\b|,?\s+and\s+also\b|,?\s+also\s+(?:tell|show|give|share|list|let me know)\b|"
    r"[,;]\s*plus,?\s+|,?\s+aur\s+(?:ye|yeh|saath)\s+(?:hi|bhi|mein)\b)\s*", re.I)
_ASK_START = re.compile(
    r"^(?:and\s+|also\s+|aur\s+)?(?:how|what|which|who|when|where|why|is|are|was|were|do|does|did|can|could|show|list|"
    r"give|tell|share|compare|find|get|display|plot|draw|kitne|kitna|kitni|kya|kaun|kab|kahan|kyun|kyon|batao|"
    r"dikhao|top|total|count|number|percentage|%|sat|install|stock|revenue)\b", re.I)
_FOLLOWUP_TAIL = re.compile(r"^(?:and|also|aur)?\s*(?:by|for|in|per|only|just|what about|how about|ab|sirf)\b", re.I)


def _words(s: str) -> int:
    return len(re.findall(r"\w+", s))


def _ok(part: str) -> bool:
    return _words(part) >= 3 or bool(_FOLLOWUP_TAIL.search(part) and part.rstrip().endswith("?") and _words(part) >= 2)


def _clean(part: str) -> str:
    part = re.sub(r"^\s*(?:and|also|aur|plus)\b[\s,]*", "", part.strip(" \t;,"), flags=re.I).strip()
    return part[:1].upper() + part[1:] if part else part


def _accept(parts: list[str]) -> list[str] | None:
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) >= 2 and all(_ok(p) for p in parts):
        return [_clean(p) for p in parts]
    return None


def split(message: str) -> tuple[list[str], list[str]]:
    """(questions to answer, extra questions beyond MAX_PARTS). One question -> ([message], [])."""
    text = (message or "").strip()
    if _words(text) < 6:
        return [text], []
    found = None
    # Numbered parts
    marks = list(_NUMBERED.finditer(text))
    if len(marks) >= 2 and marks[0].start() <= 2:
        found = _accept([text[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(text))]
                         for i, m in enumerate(marks)])
    # Separate lines that each read as a question / request
    if not found:
        lines = [l.strip(" -*•\t") for l in text.splitlines() if l.strip(" -*•\t")]
        if len(lines) >= 2 and all(l.endswith("?") or _ASK_START.search(l) for l in lines):
            found = _accept(lines)
    # Question marks with words on both sides
    if not found and (text.count("?") >= 2 or re.search(r"\?\s+\S", text)):
        parts = [p for p in re.split(r"(?<=\?)\s+", text) if p.strip()]
        if len(parts) >= 2:
            found = _accept(parts)
    # Explicit add-ons ('; also', 'and also', 'also tell me', 'aur ye bhi')
    if not found:
        parts = _ADDON.split(text)
        if len(parts) >= 2 and all(_ASK_START.search(_clean(p)) or _words(p) >= 4 for p in parts[1:]):
            found = _accept(parts)
    if not found:
        return [text], []
    return found[:MAX_PARTS], found[MAX_PARTS:]
