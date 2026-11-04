"""Supervisor entry point: run one question through the agent graph and log it.

Moved from gmr_agents/llm_pipeline.py unchanged.
"""

from __future__ import annotations
import time
from datetime import datetime
from typing import Any
from shared import config, llm_gateway
from shared.logging_config import append_jsonl, new_trace_id
from gmr_agents.resolvers import language
from gmr_agents.guards import injection
import logging
from gmr_agents.agents.common import _IST, _finish, termination_reason

log = logging.getLogger("gmr_agents.llm_pipeline")  # same logger name as before the split


def run_agent(question: str, *, prior_questions: list[str] | None = None, prior_turns: list[dict] | None = None,
              trace_id: str | None = None, session_id: str | None = None, emit=None,
              response_language: str = language.AUTO) -> dict:
    """Never raises: a failure becomes an honest error answer, and is logged.
    emit(event) -- optional: receives progress events for a streaming client.
    response_language -- the dropdown preference ('auto' = answer in the user's language)."""
    tid = trace_id or new_trace_id()
    t_start = time.perf_counter()
    state: dict[str, Any] = {"question": question, "trace_id": tid, "trace": [], "warnings": [],
                             "prior_questions": prior_questions or [], "emit": emit, "t_start": t_start,
                             "language_pref": language.normalize_preference(response_language),
                             "forced_language": language.forced(question, response_language)}
    log.info("[%s] QUERY  %r", tid, question)
    signals = injection.scan(question)
    if signals:
        # A signal only: the triage guardrail refuses "ignore your instructions" requests and
        # nothing the LLM writes can run unless the allowlist, SQL guard and read-only DB allow it.
        log.warning("[%s] security: injection pattern(s) in the user message %s", tid, signals[:3])
        state["security"] = {"injection_signals": signals[:5]}
    error: str | None = None
    with llm_gateway.question_scope(tid, config.QUESTION_LLM_BUDGET_SECONDS) as scope:
        try:
            from gmr_agents.supervisor.graph import run_graph  # LangGraph orchestration (gmr_agents/graph/)

            run_graph(state, prior_turns, session_id)
        except Exception as exc:
            log.exception("[%s] question failed", tid)
            error = f"{type(exc).__name__}: {exc}"
            _finish(state, "Something went wrong while answering this question. It has been logged -- please "
                           "try again or rephrase.", refused=True, source="error")
        calls = list(scope.calls)
        if scope.limit_hit:
            state["limit_hit"] = scope.limit_hit
    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    ok_models = sorted({c["model"] for c in calls if c["status"].startswith("ok")})
    state.update(llm_calls=calls, total_ms=total_ms)
    trace = state.get("trace") or []
    # Execution summary (API + logs): what ran, what it cost -- never the model's reasoning.
    state["execution"] = {
        "trace_id": tid, "steps": (state.get("graph") or {}).get("steps"), "latency_ms": total_ms,
        "ttft_ms": state.get("ttft_ms") or total_ms,  # first numbers / words shown (whole answer if none streamed)
        "llm_calls": sum(1 for c in calls if c["status"].startswith("ok")), "llm_attempts": len(calls),
        "tool_calls": sum(1 for s in trace if s.get("step") == "execute" or str(s.get("step", "")).startswith("extra")),
        "db_queries": sum(1 for s in trace if (s.get("step") == "execute" and s.get("tool") != "calculator"
                                                and not s.get("reused") and "rows" in s)
                          or str(s.get("step", "")).startswith("extra")),
        "input_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
        "output_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
        "cached_tokens": sum(c.get("cached_tokens") or 0 for c in calls),
        "estimated_cost_usd": round(sum(c.get("cost_usd") or 0 for c in calls), 6),
        "termination": termination_reason(state),
        # Context size and what went into it (KV-cache-friendly = small, relevant, non-redundant).
        "context_tokens": next((s.get("tokens") for s in trace if s.get("step") == "context"), None),
        "context_components": next((s.get("components") for s in trace if s.get("step") == "context"), None),
        "encoded": next((s.get("encoded") for s in trace if s.get("step") == "context"), None),
        "schema_tables": len(state.get("schema_tables") or []),
        "retrieved_examples": len((state.get("triage") or {}).get("examples") or []),
        "result_rows": state.get("row_count"),
        "result_preview_tokens": state.get("result_preview_tokens"),
    }
    plan = state.get("plan") or {}
    triage = state.get("triage") or {}
    append_jsonl("questions", {
        "at": datetime.now(_IST).strftime("%Y-%m-%dT%H:%M:%S"), "trace_id": tid, "session_id": session_id,
        "question": question, "resolved_question": state.get("resolved_question"), "kind": triage.get("kind"),
        "language": state.get("target_language"),
        "domain": state.get("agent"), "topics": triage.get("topics"), "tables": plan.get("tables"),
        "sql": plan.get("sql"), "extra_sql": plan.get("extra_sql"), "reasoning": plan.get("reasoning"),
        "rows": state.get("row_count"),
        "refused": bool(state.get("refused")), "summary": (state.get("summary") or "")[:500], "error": error,
        "total_ms": total_ms, "models": ok_models, "llm_calls": len(calls),
        "llm_ms": round(sum(c["ms"] for c in calls), 1),
        "tokens": sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0) for c in calls),
        "cached_tokens": sum(c.get("cached_tokens") or 0 for c in calls),
        "steps": state["trace"], "graph": state.get("graph"), "security": state.get("security"),
        "termination": termination_reason(state), "execution": state.get("execution"),
    })
    log.info("[%s] DONE   %7.1fms  agent=%s refused=%s models=%s", tid, total_ms, state.get("agent"),
             state.get("refused"), ",".join(ok_models))
    return state
