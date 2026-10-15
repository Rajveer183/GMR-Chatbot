"""Idempotency guard -- the same operation is never executed twice needlessly.

Who controls it: application code. The LLM is not trusted to remember what it
already ran; the key is computed here from what the tool would actually do:

    key = sha1(tool name | normalized arguments | context that changes the result)

    tool name   "run_sql"
    arguments   the guard-normalized SQL (whitespace collapsed, lower-cased keywords
                come from sqlglot), so cosmetic rewrites map to the same key
    context     the data domain (wfm / mdm) -- results differ per database

Two levels:
  * within one question (graph state `executed`): a repeat of an already-run SQL is
    not executed again. A repeat of a SUCCESSFUL call reuses its result; a repeat of
    a FAILED call is turned straight into a structured error ("you already ran this
    and it failed with ...") so a retry loop cannot spin on the same query.
  * across questions: gmr_agents/schema/knowledge_store.py keeps materialised results
    (outputs/sql_results.json) keyed by the same SQL + the data version, used by the
    executor -- a repeated query since the last data refresh never hits the database.
"""

from __future__ import annotations

import hashlib
import json


def request_key(tool: str, args: dict, context: dict) -> str:
    normalized = {k: " ".join(str(v).split()) if isinstance(v, str) else v for k, v in sorted(args.items())}
    payload = json.dumps({"tool": tool, "args": normalized, "context": context}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
