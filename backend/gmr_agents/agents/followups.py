"""Next-Question Suggester: follow-up options from verified questions and the pick parser
shared by the narrator and knowledge agents.

Moved from gmr_agents/llm_pipeline.py unchanged.
"""

from __future__ import annotations
from typing import Any
from gmr_agents.catalog import business_rules
from gmr_agents.retrieval import knowledge


_FOLLOWUP_OPTIONS = 10


def _followup_options(triage: dict) -> list[str]:
    """Verified questions (examples.yaml `suggest`) most related to this one -- the
    only follow-ups ever offered, so every suggestion has a checked answer. The
    example this answer came from (and anything worded like the question) is left out."""
    offered = business_rules.suggestions()
    # Same data source only: an MDM (hes_gold) answer was offered WFM installation questions -- seen live
    # 10 Nov on 'low power factor meters in September'. No same-source suggestion = no follow-ups.
    if triage.get("domain") in {"wfm", "mdm"}:
        sqls = {str(i): str(ex.get("sql") or "") for i, ex in enumerate(business_rules.examples())}
        mdm = lambda sql: "hes_gold." in sql or "hes_silver." in sql  # noqa: E731
        offered = {i: q for i, q in offered.items() if mdm(sqls.get(i, "")) == (triage["domain"] == "mdm")}
    if not offered:
        return []
    asked = " ".join(str(triage.get("question") or "").lower().split()).rstrip(" ?.")
    used = {str(e).lstrip("e") for e in triage.get("examples") or []}
    hits = knowledge.search("examples", triage.get("question") or "", len(offered) + 20)
    used |= {item for item, score in hits if score >= 10}  # near-exact wording match = this question
    out: list[str] = []
    for item, _score in hits:
        q = offered.get(item)
        if q and item not in used and " ".join(q.lower().split()).rstrip(" ?.") != asked and q not in out:
            out.append(q)
        if len(out) == _FOLLOWUP_OPTIONS:
            break
    return out


def _options_text(options: list[str]) -> str:
    return ("FOLLOW-UP OPTIONS (verified questions -- answer `followups` with their numbers):\n"
            + "\n".join(f"{i}. {q}" for i, q in enumerate(options, 1))) if options else ""


def _pick_followups(raw: Any, options: list[str]) -> list[str]:
    """The LLM's picks mapped back to the verified options (by number, or by exact
    text); anything else is dropped and the list is topped up in relevance order."""
    picked: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        text = str(item).strip().rstrip(".")
        if text.isdigit() and 1 <= int(text) <= len(options):
            q = options[int(text) - 1]
        else:
            q = next((o for o in options if o.lower().rstrip(" ?.") == text.lower().rstrip(" ?.")), None)
        if q and q not in picked:
            picked.append(q)
    for q in options:
        if len(picked) >= 3:
            break
        if q not in picked:
            picked.append(q)
    return picked[:3]
