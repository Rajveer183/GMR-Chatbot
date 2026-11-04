"""Helpers every agent uses: trace steps, streamed events, the final answer record, shared limits.

Moved from gmr_agents/llm_pipeline.py unchanged.
"""

from __future__ import annotations
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from shared import config, llm_gateway
from gmr_agents.catalog.table_metadata import SCHEMA_RE
import logging

log = logging.getLogger("gmr_agents.llm_pipeline")  # same logger name as before the split

_IST = timezone(timedelta(hours=5, minutes=30))

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_SCHEMA_TOKEN_BUDGET = config.SCHEMA_TOKEN_BUDGET

_ANSWER_ROWS = config.ANSWER_PREVIEW_ROWS

_LAST_CONTEXT: dict[str, dict] = {}  # question -> context metrics of its last planner prompt

_TABLE_ROWS = config.TABLE_MAX_ROWS

_CHART_POINTS = config.CHART_MAX_POINTS

_BUSY = ("All AI models are busy right now (free-tier rate limit reached). "
         "Please ask again in about a minute.")


def _today() -> str:
    now = datetime.now(_IST)
    return f"{now:%Y-%m-%d} ({now:%A}), Asia/Kolkata"


def _emit(state: dict, kind: str, **data: Any) -> None:
    """Send a progress event to a streaming client (no-op for the plain endpoint).
    The first CONTENT event (numbers / chart / table, or the first words of the
    answer) sets the question's time to first token (ttft_ms, logged per question)."""
    if kind in ("partial", "summary_delta") and "ttft_ms" not in state and state.get("t_start"):
        state["ttft_ms"] = round((time.perf_counter() - state["t_start"]) * 1000, 1)
    emit = state.get("emit")
    if emit is not None:
        try:
            emit({"type": kind, **data})
        except Exception:
            log.debug("emit failed", exc_info=True)


def _step(state: dict, name: str, t0: float, **detail: Any) -> None:
    ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info("[%s] %-9s %7.1fms  %s", state["trace_id"], name, ms,
             " ".join(f"{k}={v}" for k, v in detail.items() if v not in (None, "", [])))
    state["trace"].append({"step": name, "ms": ms, **detail})


def _conversation_focus(prior_turns: list[dict]) -> str:
    """What the conversation has been dwelling on (for deep-dive follow-ups): the
    projects / OEMs / categories / SAT stages / years named in recent turns and the
    tables their SQL used. Context for the LLM -- it decides what still applies."""
    from gmr_agents.checks.entity_check import _CATEGORIES, _OEMS, _PROJECTS, _named

    text = " ".join(str(t.get("resolved_question") or t.get("question") or "") for t in prior_turns)
    bits = []
    # Most recent first: alphabetical order put the FIRST turn's project ahead of the latest
    # one, and follow-ups ('break it down by OEM') were resolved against it (seen live).
    for label, table in (("projects", _PROJECTS), ("OEMs", _OEMS)):
        found: list[str] = []
        for t in reversed(prior_turns):
            for name in sorted(_named(str(t.get("resolved_question") or t.get("question") or ""), table)):
                if name not in found:
                    found.append(name)
        if found:
            bits.append(f"{label} (latest first): {', '.join(found)}")
    cats = [c for c, aliases in _CATEGORIES.items() if any(a in text.lower() for a in aliases)]
    if cats:
        bits.append(f"categories: {', '.join(cats)}")
    stages = sorted(set(re.findall(r"sat[\s-]?(\d{1,2})", text.lower())))
    if stages:
        bits.append(f"SAT stages: {', '.join(stages)}")
    years = sorted(set(re.findall(r"\b(20\d\d)\b", text)))
    if years:
        bits.append(f"years: {', '.join(years)}")
    tables = []
    for t in prior_turns:
        for name in re.findall(rf"\b{SCHEMA_RE}\.([a-z0-9_]+)", str((t.get("context") or {}).get("sql") or "")):
            if name not in tables:
                tables.append(name)
    if tables:
        bits.append(f"tables used: {', '.join(tables[-4:])}")
    return "; ".join(bits)


def _history_block(prior_turns: list[dict] | None) -> str:
    """Conversation memory: older turns as their questions, the latest
    MEMORY_DETAILED_TURNS in full (question, short answer, key numbers, result
    keys), and the conversation's current focus."""
    if not prior_turns:
        return "(none)"
    turns = prior_turns[-config.MEMORY_TURNS:]
    older, recent = turns[:-config.MEMORY_DETAILED_TURNS], turns[-config.MEMORY_DETAILED_TURNS:]
    lines = []
    if older:
        lines.append("Earlier: " + " | ".join(str(t.get("resolved_question") or t.get("question")) for t in older))
    for i, t in enumerate(recent):
        ctx = t.get("context") or {}
        if i == len(recent) - 1:
            # Named explicitly: without it the model often continued the FIRST turn (seen live).
            lines.append("LATEST TURN (an incomplete follow-up continues THIS one -- its topic, measure and filters):")
        lines.append(f"Q: {t.get('resolved_question') or t.get('question')}")
        lines.append(f"A: {str(t.get('summary') or '')[:config.MEMORY_ANSWER_CHARS]}")
        if ctx.get("kpis"):
            lines.append("   numbers: " + "; ".join(map(str, ctx["kpis"][:4])))
        if ctx.get("keys"):
            lines.append(f"   rows by {ctx.get('key_column', 'key')}: " + ", ".join(map(str, ctx["keys"][:8])))
    focus = _conversation_focus(turns)
    if focus:
        lines.append(f"Conversation focus so far -- {focus}")
    text = "\n".join(lines)
    # Token budget: drop the oldest 'Earlier' questions first, then shorten the answers
    # (the extracted focus -- entities, filters, tables -- is always kept).
    if older and llm_gateway.estimate_tokens(text) > config.MAX_HISTORY_TOKENS:
        return _history_block(prior_turns[-config.MEMORY_DETAILED_TURNS:])
    while llm_gateway.estimate_tokens(text) > config.MAX_HISTORY_TOKENS and "A: " in text:
        text = "\n".join(l[: max(40, len(l) // 2)] if l.startswith("A: ") else l for l in text.split("\n"))
        if all(len(l) <= 43 for l in text.split("\n") if l.startswith("A: ")):
            break
    return text


_PAST_DATE_RE = re.compile(r"\bas of\b|\bas on\b|\b\d{1,2}\s*(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
                           r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b",
                           re.IGNORECASE)

_ALL_TIME_LEAD_RE = re.compile(r"^\s*Assuming you mean [^.;:]*?\(?all[- ]time\)?\s*[,;:-]\s*", re.IGNORECASE)

_PERIOD_RE = re.compile(r"\b(19|20)\d{2}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b(today|"
                        r"yesterday|week|month|year|quarter|days?|since|last|this|kal|aaj)\b", re.IGNORECASE)

_METER_SERIAL_RE = re.compile(r"\b(?=[A-Za-z]{1,4}\d)[A-Za-z]{1,4}\d{6,10}\b")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def termination_reason(state: dict) -> str:
    """Why a run ended, as a stable code for logs / dashboards (never shown to users)."""
    graph = state.get("graph") or {}
    status = graph.get("status")
    errors = graph.get("errors") or []
    calls = state.get("llm_calls") or []
    if status == "step_limit":
        return "MAX_ITERATIONS_EXCEEDED"
    if state.get("limit_hit") or any("COST_LIMIT" in str(c.get("error") or "") for c in calls):
        return "COST_LIMIT_EXCEEDED"
    if state.get("narration_source") == "error":
        return "TOOL_EXECUTION_FAILED"
    if not state.get("refused"):
        return "COMPLETED"
    if status == "busy" or state.get("narration_source") == "none (AI busy)":
        return "LLM_UNAVAILABLE"
    if errors and errors[-1] == "duplicate":
        return "DUPLICATE_ACTION"
    if errors and errors[-1] in ("validation", "not_allowed", "invalid_args"):
        return "VALIDATION_FAILED"
    if graph.get("loop_detected"):
        return "LOOP_DETECTED"
    if errors and errors[-1] == "db_query_limit":
        return "DB_QUERY_LIMIT_EXCEEDED"
    if errors and errors[-1] == "connectivity":
        return "TIMEOUT_OR_DB_UNAVAILABLE"
    if state.get("narration_source") == "grounding":
        return "ABSTAINED_NO_EVIDENCE"
    return "REFUSED"


def _finish(state: dict, summary: str, *, refused: bool, source: str, followups: list[str] | None = None,
            insights: list[str] | None = None, agent: str | None = None) -> None:
    state.update(summary=summary, insights=insights or [], followups=followups or [], kpis=[], charts=[],
                 tables=[], refused=refused, narration_source=source, agent=agent)
