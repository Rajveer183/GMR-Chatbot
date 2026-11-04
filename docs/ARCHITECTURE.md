# Chatbot architecture: agents and code stages

The chatbot uses **4 LLM agents**. Only these 4 make LLM calls, and each makes one call per question, plus a bounded SQL repair. Every other step runs as **deterministic code**. The [LangGraph](../backend/gmr_agents/supervisor/graph.py) supervisor wires them together.

Why only 4: each extra LLM hop costs about 2–5 s and 2–7k tokens. A code stage costs milliseconds and 0 tokens, and gives the same result every time. That split is why answers match the dashboard.

```
question → Cache Keeper (ops/answer_cache) ─ hit → answer
         └ Input Guard (guards/injection) → Router agent ─┬ chitchat / out of scope → reply
                                                          ├ explain → Knowledge agent → citation check → answer
                                                          └ data → SQL Writer agent ⟲ repair → Narrator agent → answer
```

## LLM agents (`gmr_agents/agents/`)

| Agent | Node (`agent.py`) | Prompt and logic | Roles it covers |
|---|---|---|---|
| Router | `understand_node`, `reply_node` | `router/triage.py`: a code router for verified wordings (no LLM), otherwise the triage call; scope, measure and wording fixes are done by code | Intent Router, Question Refiner, Clarifier (stated assumption) |
| SQL Writer | `controller_node`, `validate_node`, `execute_node` | `sql_writer/planner.py`: planner prompt, plan parsing, repair hints, parallel extra queries | SQL Writer, Query Debugger, Query Planner (extra queries) |
| Narrator | `answer_node` | `narrator/writer.py`: answer call, number and extreme checks with code repair first | Answer Narrator, Fact Checker |
| Knowledge (RAG) | `explain_node` | `knowledge/explain.py`: grounded explanation from retrieved chunks | Glossary Keeper, Doc Librarian, Catalog Navigator |

`agents/followups.py` builds the follow-up suggestions (Next-Question Suggester) for the Narrator and Knowledge agents. `agents/common.py` holds the trace, event and final-answer helpers that every agent uses.

## Code stages (no LLM)

| Folder | Roles |
|---|---|
| `supervisor/` | Graph, conditional edges, step / retry / loop guards, termination nodes, `run.py` (`run_agent`) |
| `retrieval/` | Schema Scout, Example Finder, Value Matcher, glossary chunks: one hybrid index (BM25 + local embeddings, RRF) |
| `catalog/` | Live table catalog, Join Mapper (declared joins), Metric Gatekeeper (`data/business_rules.yaml`), verified examples (`data/examples.yaml`) |
| `resolvers/` | Entity / Time Resolver (place scope, question slots), response language and translation |
| `guards/` | Input Guard (injection), Syntax Checker / Access Warden (`sql_guard`, `sql_validator`), Budget Guard (`limits`), idempotency |
| `checks/` | Logic Critic (`entity_check`), Fact Checker (`number_check`, `extreme_check`), citation grounding |
| `execution/` | Query Runner, Dialect Adapter, place-hierarchy repair (Result Auditor), `tools/` allowlist (`run_sql`, `calculator`) |
| `presentation/` | Analyst facts (`analyst.py`), Chart Maker (`blocks.py`, `dashboard_charts.py`) |
| `ops/` | Cache Keeper (`answer_cache`, `prewarm`), prompt token budgets (`context_manager`), metrics |

## Verified-SQL fast path (`SQL_VERIFIED_DIRECT`, off by default)

When a question is a verified example's own wording, that example's hand-checked SQL runs without a SQL Writer call. "Own wording" means the first `q` of the example, ignoring case and punctuation. The fast path is used only when **all** of these hold:
- The question is not a follow-up. The router already sends follow-ups to triage.
- The example's other wordings aren't involved. They are patterns the planner adapts, for example one Kashi SQL also lists "Agra mein SAT kitna pending hai".
- Code decides the presentation: the rows belong to a dashboard card, or the result is a single-row figure.
- The SQL has no literal date under a relative-time wording.

The guard, the entity check, execution and the number checks all still run. A rejection or zero rows goes to the SQL Writer with its full retry budget. Code lives in `agents/sql_writer/agent.py` (`_verified_direct_plan`).

Evidence from logs (28 Oct – 4 Nov, 847 questions) and a read-only database comparison (4 Nov):
- 247 questions (29%) qualify, saving about 5.9 s and about 6.7k tokens each.
- On 181 of them the planner wrote the same SQL.
- On the 64 where it deviated, the verified SQL returned the same numbers, a superset of them, or the correct figure where the planner was wrong. In one planner answer the "not yet due" counts were negative.

**Open question before enabling:** the verified "Is the burnt-meter count rising month on month?" SQL counts today's partial day, while the planner stopped at yesterday.

## Compatibility

The old module paths still import:
- `gmr_agents.llm_pipeline` and `gmr_agents.graph.nodes` are facades that re-export the moved names.
- `schema/`, `sql/`, `tools/`, `validation/`, `graph/` and the old top-level modules are aliases to the new modules.

The data files moved from `schema/data/` to `catalog/data/`.

## Refactor gate

`backend/eval/question_bank/replay_prompts.py` replays logged questions offline, with the LLM and database stubbed from the logs. It records a hash of every prompt and every answer. A restructure must reproduce the earlier run exactly:

```
python eval/question_bank/replay_prompts.py --out before.jsonl
python eval/question_bank/replay_prompts.py --out after.jsonl --compare before.jsonl
```
