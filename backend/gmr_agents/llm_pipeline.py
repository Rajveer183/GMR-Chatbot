"""LLM-only question answering: every decision is made by an LLM.

There are no answer templates and no keyword/regex decision rules on this
path. The business knowledge the old hard-coded templates carried lives in
catalog/data/business_rules.yaml and is handed to the LLM as rules.

    1. triage  (fast model)   what kind of question (data / explain / chitchat /
                              out_of_scope), the standalone English question
                              (translation, spelling, follow-up resolved), the
                              domain, the rule topics and the tables it needs
    2. plan    (SQL model)    one SQL query from the chosen topics' rules + the
                              chosen tables' schema, plus how to present the
                              result (number cards / chart type / table)
       safety check           validation/sql_guard.check_sql_safety: read-only,
                              known tables/columns, one engine -- never rewrites
       execute                the SQL runs exactly as the LLM wrote it
       repair  (SQL model)    on a safety rejection, a database error or zero
                              rows, the failed SQL + the exact error go back to
                              the LLM (max config.LLM_MAX_REPAIRS)
    3. answer  (fast model)   summary / insights / follow-ups from the result rows

If no model is available (free-tier limits) the user is told so honestly --
nothing is guessed by code. Model choice, retries and per-call logging are in
shared/llm_gateway.py; one record per question goes to
outputs/logs/questions.jsonl.

The code now lives in the agent modules below (see docs/ARCHITECTURE.md); this module
re-exports every name so existing imports (main.py, prewarm.py, eval/, tests/) keep working.
"""

from __future__ import annotations

from shared import config, llm_gateway  # noqa: F401  (tests patch llm_gateway through here)
from gmr_agents.agents.common import (  # noqa: F401
    _IST,
    _ANSI_RE,
    _SCHEMA_TOKEN_BUDGET,
    _ANSWER_ROWS,
    _LAST_CONTEXT,
    _TABLE_ROWS,
    _CHART_POINTS,
    _BUSY,
    _today,
    _emit,
    _step,
    _conversation_focus,
    _history_block,
    _PAST_DATE_RE,
    _ALL_TIME_LEAD_RE,
    _PERIOD_RE,
    _METER_SERIAL_RE,
    termination_reason,
    _finish,
)
from gmr_agents.agents.router.triage import (  # noqa: F401
    _TRIAGE_SYSTEM,
    _table_menu,
    _triage,
    _consistent_assumption,
    _parse_triage,
    _CONSUMER_WORD_RE,
    _MAPPING_RE,
    _WFM_WORDS_RE,
    _FOLLOWUP_WORDS,
    _route,
    fix_scope,
    _FRAME_WORDS,
    _content_words,
    fix_measure,
    fix_question,
    _PROJECT_LEVEL_WISE_RE,
    _JOURNEY_LEVELS_RE,
    _JOURNEY_STAGES,
)
from gmr_agents.agents.sql_writer.planner import (  # noqa: F401
    _PLAN_SYSTEM,
    _plan_prompt,
    _REPAIR_PROMPT,
    repair_base,
    _dedupe_hint_values,
    _SQL_START_RE,
    _EDGE_ORDER,
    _REMINDER,
    _edges_order,
    _clean_sql,
    _to_plan,
    _TABLE_REF_RE,
    _repair_hint,
    _FILTER_LITERAL_RE,
    _DATE_LITERAL_RE,
    _filter_literals,
    _zero_is_genuine,
    _table_columns,
    _is_empty,
    _run_one_extra,
    _run_extras,
    _MISSING_DATA_RE,
    _widened_tables,
)
from gmr_agents.presentation.blocks import (  # noqa: F401
    _label,
    _number,
    _cell,
    _TS_RE,
    _tidy_ts,
    _RAW_COL_RE,
    raw_cols,
    _MULTI_SERIES,
    _SINGLE_SERIES,
    _STACKED,
    _WHOLE_SERIES_RE,
    _chart_fallback,
    _distinct_labels,
    _is_subtotal,
    _one_rollup_level,
    _build_chart,
    _blocks,
)
from gmr_agents.agents.narrator.writer import (  # noqa: F401
    _ANSWER_SYSTEM,
    _write_answer,
)
from gmr_agents.agents.followups import (  # noqa: F401
    _FOLLOWUP_OPTIONS,
    _followup_options,
    _options_text,
    _pick_followups,
)
from gmr_agents.presentation.analyst import (  # noqa: F401
    _partial_periods,
    _WEEK_LABEL_RE,
    _short_weeks,
    _DATE_COLS,
    _date_window,
    _result_facts,
    _RATE_COL_RE,
    _result_preview,
)
from gmr_agents.agents.knowledge.explain import (  # noqa: F401
    _EXPLAIN_SYSTEM,
    _explain,
)
from gmr_agents.supervisor.run import (  # noqa: F401
    run_agent,
)
