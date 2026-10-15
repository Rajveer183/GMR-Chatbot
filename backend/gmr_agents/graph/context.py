"""Per-run context for values that must not go into checkpointed graph state.

Who controls it: application code. LangGraph checkpoints the graph state after
every node, so it must stay JSON-serializable; query results (pandas DataFrames),
the prompt text, the legacy pipeline state dict (trace steps, streaming `emit`
callback) live here instead, keyed by trace_id, and are dropped when the run ends.
Nodes find their context through config["configurable"]["trace_id"].
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

_LOCK = threading.Lock()
_RUNS: dict[str, "RunContext"] = {}


@dataclass
class RunContext:
    pipeline_state: dict[str, Any]                       # what run_agent returns (trace, plan, kpis, emit ...)
    base_prompt: str = ""                                # planner prompt built once per question
    repair_prompt: str = ""                              # what repair calls start from (lean when PROMPT_COMPACT has lean_repair)
    raw_plan: dict = field(default_factory=dict)         # the controller's last JSON (display, labels, extras)
    plan: dict | None = None                             # executor plan for the current tool call
    result: dict | None = None                           # executor result of the accepted run
    extras: list[dict] = field(default_factory=list)
    fallback: dict | None = None                         # a zero-row run kept in case retries run out
    widened: bool = False                                # schema already widened with the topic's tables
    results_by_key: dict[str, dict] = field(default_factory=dict)  # idempotency: key -> executor result
    tool_ctx: Any = None                                 # app-side tool inputs (catalog, verdict, plan)
    prepared: Any = None                                 # the validated ToolCall (tools/calling.py)


def open_run(trace_id: str, pipeline_state: dict[str, Any]) -> RunContext:
    ctx = RunContext(pipeline_state=pipeline_state)
    with _LOCK:
        _RUNS[trace_id] = ctx
    return ctx


def get(config) -> RunContext:
    return _RUNS[config["configurable"]["trace_id"]]


def close_run(trace_id: str) -> None:
    with _LOCK:
        _RUNS.pop(trace_id, None)
