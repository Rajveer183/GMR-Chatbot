"""Graph state -- what LangGraph carries from node to node and checkpoints.

Who controls it: LangGraph owns it (merges each node's returned update, using the
reducers below, and saves it after every node); nodes only *return* changes.

Three kinds of memory, kept separate on purpose:
  * graph state (this file)  -- one question's working memory: the request, the
    planner's tool call, validation verdicts, results, errors, step count, answer.
    Checkpointed per session thread (graph.py), so a run can be inspected/resumed.
  * short-term session memory -- the conversation so far (previous questions,
    resolved questions, their SQL and summaries), stored by shared/session_store.py
    in outputs/gmr_chat_sessions.json and passed in as `prior_turns`.
  * long-term memory -- verified knowledge that outlives sessions: examples.yaml
    (question -> SQL patterns), business_rules.yaml, kpi_definitions.yaml, and the
    retrieval indexes + embeddings in outputs/knowledge.json.
No separate "memory agent": retrieval of the relevant long-term memory happens
inside the controller's prompt building, where it is used.

Values that cannot be serialized (DataFrames, the streaming callback) are not
put in the state: they live in RunContext (context.py), keyed by trace_id.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict


def append_list(left: list | None, right: list | None) -> list:
    """Reducer: audit lists accumulate across nodes within one question. `None`
    resets -- run_graph() starts each question with None, because the session's
    thread checkpoint still holds the previous question's lists."""
    return [] if right is None else [*(left or []), *right]


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer: idempotency records accumulate across nodes (later keys win); `None` resets."""
    return {} if right is None else {**(left or {}), **right}


class ToolCall(TypedDict, total=False):
    tool: str                 # must be a key of tools/registry.py TOOLS
    args: dict[str, Any]      # {"sql": ..., "extra_queries": [...]}
    attempt: int              # which controller attempt produced it
    model: str                # which LLM wrote it


class ToolError(TypedDict, total=False):
    kind: str                 # validation | entity_mismatch | duplicate | database_error | zero_rows | connectivity | ...
    message: str              # shown to the controller verbatim on the next attempt
    sql: str


Route = Literal["data", "explain", "chitchat", "out_of_scope", "busy"]


class AgentState(TypedDict, total=False):
    # --- request ---------------------------------------------------------------
    question: str                         # the user's message as typed
    session_id: str | None
    trace_id: str
    prior_turns: list[dict]               # short-term session memory (read-only here)

    # --- understanding (controller, first call) ------------------------------------
    route: Route
    triage: dict                          # resolved question, topics, tables, language, examples ...

    # --- planning loop -------------------------------------------------------------
    attempt: int                          # controller (planner) calls so far
    step_count: int                       # nodes run so far -- the step guard reads this
    tool_call: ToolCall | None            # the controller's current request (None = no tool needed)
    tool_calls: Annotated[list[ToolCall], append_list]       # every request made (audit)
    validated: dict | None                # validator verdict for tool_call (normalized SQL, engine)
    entities_checked: bool                # the entity verifier ran once already
    zero_rows_retried: bool               # an empty result was sent back once already
    executed: Annotated[dict[str, dict], merge_dicts]          # idempotency: request key -> {ok, rows, error}
    last_error: ToolError | None          # structured error for the controller's next attempt
    errors: Annotated[list[ToolError], append_list]           # every error (audit)
    has_result: bool                      # a usable result is in RunContext
    has_fallback: bool                    # a zero-row result is kept in case retries run out

    # --- outcome -------------------------------------------------------------------
    status: Literal["running", "answered", "refused", "step_limit", "busy", "error"]
    summary: str
    insights: list[str]
    followups: list[str]
    refused: bool
