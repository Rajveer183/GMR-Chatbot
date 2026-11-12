"""Answer Narrator agent: the answer LLM call, follow-up picks, and the number / highest-lowest
checks (Fact Checker) with code repair first.

Moved from gmr_agents/llm_pipeline.py unchanged.
"""

from __future__ import annotations
import json
import re
import time
from shared import config, llm_gateway
from gmr_agents.catalog import business_rules
from gmr_agents.ops import answer_cache
from gmr_agents.resolvers import language
from gmr_agents.guards import injection
from gmr_agents.checks import extreme_check, freshness, grounding, number_check, wording_check, year_check
import logging
from gmr_agents.agents.common import _ALL_TIME_LEAD_RE, _ANSWER_ROWS, _PAST_DATE_RE, _PERIOD_RE, _emit, _step, _today
from gmr_agents.agents.followups import _followup_options, _options_text, _pick_followups
from gmr_agents.presentation.analyst import _date_window, _partial_periods, _result_facts, _result_preview, _short_weeks
from gmr_agents.presentation.blocks import _tidy_ts
from gmr_agents.presentation import drivers as _drivers, investigate as _investigate, stats as _stats

log = logging.getLogger("gmr_agents.llm_pipeline")  # same logger name as before the split


# --------------------------------------------------------------------------- #
# 3. Answer writing
# --------------------------------------------------------------------------- #

_ANSWER_SYSTEM = """You are a sharp data analyst for GMR's smart-meter programme (Uttar Pradesh; projects Agra, Kashi, Triveni). Answer the user's question from the RESULT rows only.
Terms: SAT = Site Acceptance Test (a batch acceptance of installed meters, SAT-1..SAT-13); MI = meter installation; eligible MI = installed on/before the project's SAT cutoff; non-SAT = eligible meters not yet in a SAT batch; DT/DTR = distribution transformer meter; PMPM = per meter per month payment; lumpsum = one-time payment per meter after SAT; OEM = meter maker (AEW, CAP, HPL, SEIPL); auto-indexing = consumer-to-feeder tagging audit (mapping audit), where MI = consumers declared to a feeder and CI = consumers in the field indexing register (not meter installation).
- First sentence: the direct answer with the key number(s) and units. Copy numbers digit-for-digit from RESULT, with commas (1,234,567); add lakh/million in brackets when asked. IDs (meter serials, consumer numbers, SAT numbers) are written as-is, never with commas. Money columns are in crore: write 'Rs 12.34 crore'.
- Never invent numbers: every number must come from RESULT (simple sums/differences/shares/products are fine).
- Never compute an overall rate / average / productivity yourself (e.g. total installs / summed technicians): technicians are not technician-days, and per-row rates do not average that way. Never add up a per-row count of people (active technicians) across days/weeks -- the same person is counted again. Quote the Total row's value; if RESULT has no Total row, give the range (lowest to highest) instead.
- Answer exactly what was asked, in its shape: 'how many ... monthwise / weekly / daywise' = the total for the whole period (from FACTS) only -- never list the per-period figures in the summary, the table below the answer shows them (peak/low months go in insights); 'which month' = that month and its number; 'compare' = both values and the gap. No filler.
- Talk like an analyst to a manager, about the business -- never about the pipeline. Never write the words RESULT, FACTS, PERIODS, NOTES, query, SQL, rows, dataset, result set, table, 'the data shows/provided', and never explain how a number was calculated ('calculated by summing ...') unless the user asked how.
- Insights are business observations (a trend, the driver, the outlier, a share) -- never caveats about method or what the data lacks. If something the user asked for is not available, say once, plainly, in the summary: '<item> is not available' (no mention of RESULT).
- Comparisons must match the numbers: before writing higher/lower/most/least/leads/lags, compare the two values; never call a smaller number higher. A percentage is not money and money is not a percentage: keep each number's own unit from its column (…_pct = %, …_cr = Rs crore).
- Change columns (change, increase, pct_change, growth) are measured FROM the earlier/base value TO the later one: write 'fell/rose N% from <earlier> (<value>) to <later> (<value>)'. Never turn it around ('earlier is N% lower than later' is wrong when later fell).
- Rankings: rank only the real items; a 'Total' row is the overall figure -- mention it separately if useful, never as a position in the ranking.
- A breakdown (one row per zone / category / division / project ...) is summarised ACROSS its rows: the OVERALL figure from FACTS (the Total row), then the highest and lowest item for the key measure(s). Never describe only one row as if it were the whole answer, and never present a breakdown row that shares a name with the scope (the AGRA zone inside the Agra project) as the overall figure.
- Name measures in words ('DI to Agency', 'installation to SAT'), never column names like di_to_agency.
- State only orders and values that are in RESULT; never 'implied' / guessed rankings from earlier answers.
- A list whose 'total available' is larger than the rows shown (a correction queue, ticket list ...): the count is 'total available' -- say e.g. '155,824 consumers; the strongest-evidence ones first' -- never the number of rows shown.
- Figures quoted in USER MESSAGE ('about 1.68 lakh', 'only 59 defective', '4.28L') are the user's, often from an older refresh: when RESULT differs, open with today's figure from RESULT and say the quoted one is out of date -- never explain, reuse or subtract the quoted figure as if it were true.
- Threshold asks ('less than one week', 'below 95%', 'more than 120 days'): only rows whose value is past the threshold qualify -- check each value; if none qualify say so plainly and name the closest ones with their values. Never count every row as qualifying.
- 'Biggest / steepest / largest drop, gap or loss' = the row with the highest value of that loss column (see FACTS) -- compare all of them before naming one.
- A Yes / No flag column in RESULT (focus_project, biggest_drop, under_one_week, same_rank_order ...) was computed exactly by code: the row marked 'Yes' IS the answer -- name that row and quote ITS figures; never pick a different row by your own reading. Dates in RESULT (e.g. eta_after_1_batch) are copied, never recomputed.
- When RESULT splits the headline number (e.g. eligible pending vs not yet due, consumer/DTR/feeder, per project), mention the split briefly.
- If an ASSUMPTION is given, state it in the first sentence ('Assuming you mean ...').
- Snapshot data (SAT, non-SAT, stock, current stage, receivables) has no history: if the question asks for a past date ('as of 20 Sep') and RESULT is the current snapshot, say the figure is as on the latest refresh and that day-by-day history is not stored -- never present it as the past date's value.
- 'why', 'should we', 'which to focus on', what-if: reason from the numbers -- compare, name the driver, give a clear verdict. Summaries/drafts/messages: write them fully (bullets allowed), up to ~10 lines.
- If rows were cut off, answer from what is shown and point to the full table below the answer -- never say a month or item is 'missing' because it was cut off. If the result is empty, say no matching data was found and suggest why.
- Write summary and insights in LANGUAGE (follow its description). In every language, numbers, dates, IDs and amounts are copied exactly as in RESULT, in Western digits 0-9 -- never in words.
Reply with JSON only:
{"summary": "the answer (1-3 sentences, longer only for summaries/drafts/why)", "insights": ["0-2 short, specific observations from the data"], "followups": [3 numbers from FOLLOW-UP OPTIONS: the most useful next questions after this answer, best first]}"""


def _write_answer(state: dict, triage: dict, plan: dict, result: dict, extras: list[dict] | None = None) -> dict:
    t0 = time.perf_counter()
    preview, n = _result_preview(result.get("result"))
    state["result_preview_tokens"] = llm_gateway.estimate_tokens(preview)  # tool-result size fed back to the LLM
    more = f", first {_ANSWER_ROWS} shown" if n > _ANSWER_ROWS else ""
    memory = state.get("memory_turns") or []
    writing = language.writing_language(triage.get("language", "en"))  # other targets: English, then translated
    parts = [f"Today: {_today()}", f"LANGUAGE: {language.prompt_line(writing)}",
             *(["CONVERSATION SO FAR (build on it naturally when relevant): " + " || ".join(
                 f"Q: {t.get('resolved_question') or t.get('question')} -> A: {str(t.get('summary') or '')[:160]}"
                 for t in memory[-2:])] if memory else []),
             f"USER MESSAGE: {state['question']}", f"QUESTION: {triage['question']}"]
    if len(triage.get("parts") or []) > 1:
        parts.append("PARTS (answer each one, in order, from the results):\n"
                     + "\n".join(f"{i}. {p}" for i, p in enumerate(triage["parts"], 1)))
    if triage.get("assumption"):
        parts.append(f"ASSUMPTION: {triage['assumption']}")
    notes = business_rules.answer_notes(triage["topics"][:2])
    if notes:
        parts.append(f"NOTES (how these KPIs work):\n{notes}")
    parts.append(f"HOW IT WAS COMPUTED: {plan.get('reasoning', '')}")
    tables = plan.get("tables") or []
    if tables and all(t.endswith("_snapshot") for t in tables) and _PAST_DATE_RE.search(state["question"]):
        parts.append("DATA NOTE: every table used is a CURRENT snapshot (latest refresh); it keeps no day-by-day "
                     "history, so the date in the question could not be applied. Say this in the first sentence "
                     "and give the figure as on the latest refresh.")
    facts = _result_facts(result.get("result"))
    parts.append(f"RESULT ({n} row(s){more}; total available: {result.get('group_count')}):\n"
                 + injection.wrap(preview, "database"))
    options = _followup_options(triage)
    if facts:
        parts.append("FACTS (computed by code from RESULT, Total rows excluded; partial period left out of highest / "
                     "lowest -- use these for totals and highest / lowest / lagging / leading claims):\n" + facts)
    # Analytical asks (correlation, spread, forecast range, Monte Carlo, PCA ...): exact figures computed by
    # code, plus the named methods that were NOT run -- nothing for ordinary questions.
    analysis = _stats.analyse(state["question"] + " || " + str(triage["question"]), result.get("result"),
                              extras=[ex.get("result") for ex in extras or []])
    lined = ""
    if triage.get("cross_domain") and extras:
        lined = _stats.line_up(result.get("result"), [ex.get("result") for ex in extras])
        if lined:
            parts.append(lined)
    if analysis:
        parts.append(analysis)
        _step(state, "stats", time.perf_counter(), methods=sorted(_stats.wanted(state["question"] + " " + str(triage["question"]))),
              not_run=_stats.not_run(state["question"]))
    # 'Why did X rise / fall': where the change came from, per group, computed by code (presentation/drivers.py).
    # Investigation (presentation/investigate.py): the same split along every result's dimension, standard
    # explanations tested (coverage / localised / broad), splits reconciled, confidence graded. Replaces DRIVERS.
    inv_text = ""
    try:
        inv_q = state["question"] + " || " + str(triage["question"])
        splits = [(ex.get("purpose") or "", ex.get("result")) for ex in extras or []]
        if _investigate.asked(inv_q) and not splits:
            # The planner wrote no other breakdown: code re-runs the main query split by other dimensions.
            t_split = time.perf_counter()
            splits = _investigate.auto_splits(plan)
            _step(state, "auto_splits", t_split, splits=[p for p, _ in splits])
        inv_text, inv = _investigate.assess(inv_q, result.get("result"), splits, tables=plan.get("tables"))
    except Exception:
        log.warning("answer: investigation failed", exc_info=True)
        inv = None
    if inv:
        state["investigation"] = inv
        _step(state, "investigate", time.perf_counter(), splits=len(inv["dimensions"]),
              confidence=inv["confidence"]["level"], contradictions=len(inv["contradictions"]))
    try:
        why = "" if inv_text else _drivers.breakdown(state["question"] + " || " + str(triage["question"]), result.get("result"))
    except Exception:
        log.warning("answer: driver breakdown failed", exc_info=True)
        why = ""
    why = why or inv_text
    if why:
        parts.append(why)
        _step(state, "drivers", time.perf_counter(), lines=why.count("\n"))
    periods = _partial_periods(result.get("result"))[2]
    if periods:
        parts.append(periods)
    window = _date_window(plan, result.get("result"))
    if window:
        parts.append(window)
    short = _short_weeks(result.get("result"))[1]
    if short:
        parts.append("SHORT WEEKS (checked by code): " + ", ".join(f"{k} has only {v} day(s)" for k, v in short.items())
                     + " -- compare its counts per day, never call its total the lowest week or a drop.")
    for i, ex in enumerate(extras or [], 2):
        ex_preview, ex_n = _result_preview(ex.get("result"), config.MAX_TOOL_RESULT_TOKENS // 2)
        parts.append(f"RESULT {i} -- {ex['purpose']} ({ex_n} row(s)):\n" + injection.wrap(ex_preview, "database"))
    if options:
        parts.append(_options_text(options))
    # Lost-in-the-middle: the question is restated last, where long prompts are attended best.
    parts.append(f"ANSWER THIS QUESTION FROM RESULT (and FACTS): {triage['question']}")
    prompt = "\n".join(parts)
    _emit(state, "status", message="Writing the answer")
    if state.get("emit") is not None and writing == triage.get("language", "en"):
        # Streaming client: pass the summary on word by word as it is written (not when it is
        # written in English for a later translation -- the user would see the English first).
        res = llm_gateway.chat_json_stream("answer", prompt, system=_ANSWER_SYSTEM,
                                           on_text=lambda text: _emit(state, "summary_delta", text=text),
                                           max_tokens=config.LLM_MAX_TOKENS_ANSWER)
    else:
        res = llm_gateway.chat_json("answer", prompt, system=_ANSWER_SYSTEM, max_tokens=config.LLM_MAX_TOKENS_ANSWER)
    _step(state, "answer", t0, model=res.model if res else "none available")
    if res is None or not str(res.data.get("summary") or "").strip():
        return {"summary": "Here is the data for your question. (The AI summary is unavailable right now -- "
                           "free-tier rate limit -- ask again in a minute for a written answer.)",
                "insights": [], "followups": [], "source": "none (AI busy)"}
    d = res.data
    # Native-script digits (४२) as 0-9, so the number check below reads every figure.
    summary = language.ascii_digits(str(d["summary"])).strip()
    insights = [language.ascii_digits(str(i)) for i in d.get("insights") or [] if str(i).strip()][:3]
    # A bare label is not an answer. Live 10 Nov: 'Which project is the laggard on pending SAT?' came back as
    # 'AGRA' -- right project, no figure, while RESULT held 96,372 of 263,243. One rewrite asks for the figures.
    df0 = result.get("result")
    has_numbers = df0 is not None and hasattr(df0, "select_dtypes") and not df0.empty and not df0.select_dtypes("number").empty
    if has_numbers and len(summary.split()) < 6 and not re.search(r"\d", summary):
        log.warning("answer: bare summary %r -- asking for the figures", summary)
        redo = llm_gateway.chat_json(
            "answer", prompt + "\n\nYOUR PREVIOUS ANSWER:\n" + json.dumps({"summary": summary}, ensure_ascii=False)
            + "\nThat is not a full answer. Answer in one or two sentences that quote the figures from RESULT that "
              "answer the question (e.g. the top item with its value and the total). Same JSON format.",
            system=_ANSWER_SYSTEM, max_tokens=config.LLM_MAX_TOKENS_ANSWER)
        if redo is not None and str(redo.data.get("summary") or "").strip():
            d = redo.data
            summary = language.ascii_digits(str(d["summary"])).strip()
            insights = [language.ascii_digits(str(i)) for i in d.get("insights") or [] if str(i).strip()][:3] or insights

    # Number check: every figure written must come from the results (or a simple
    # sum / difference / share within a row or column). A slip goes back to the LLM
    # once with the exact numbers named; what is still wrong is repaired in code.
    frames = [result.get("result")] + [ex.get("result") for ex in extras or []]
    # Numbers the writer was given besides RESULT are not invented: the question, the earlier
    # answers in CONVERSATION SO FAR ('and in Agra?' -> 'vs 23,150 in Kashi'), the KPI notes
    # ('120 days', 'per 1,000 meters'). Measured: most number-check flags were these, each
    # costing a correction call -- or a code 'repair' that rewrote a correct figure.
    # FACTS too: its per-category totals are computed in code ('CONSUMER total across the 8 periods').
    # STATS / LINED UP are computed by code from these results too: their figures (correlations, P10-P90, scenario
    # values) are supported. 11 Nov: without them every Monte Carlo / projection figure was 'unsupported' and the
    # code repair cut it -- 'The Monte Carlo simulation of the next 30 days. For a single day ...'.
    given = " ".join([state["question"], triage["question"], triage.get("assumption") or "", notes or "", facts,
                      analysis or "", lined or "", why or "",
                      *(str(t.get("summary") or "")[:160] for t in memory[-2:])])
    allowed = number_check.allowed_values(frames, given)
    bad = number_check.unsupported(" ".join([summary, *insights]), allowed)
    if bad:
        _step(state, "number_check", time.perf_counter(), unsupported=bad)
        # Code repair first (no extra LLM call): near-miss digits fixed, invented figures
        # snapped to the result's value, a wrong aside removed. Only when that cannot fix
        # the summary cleanly is the LLM asked to rewrite it (one call).
        fixed, left = number_check.repair(summary, allowed, frames)
        if left or len(fixed) < 0.6 * len(summary):
            log.warning("answer: numbers not in the result %s -- asking for a correction", bad)
            fix = llm_gateway.chat_json(
                "answer", prompt + "\n\nYOUR PREVIOUS ANSWER:\n"
                + json.dumps({"summary": summary, "insights": insights}, ensure_ascii=False)
                + f"\nTHESE NUMBERS ARE NOT IN RESULT: {', '.join(bad)}. Rewrite the answer copying every number "
                  "digit-for-digit from RESULT (or computing it exactly from RESULT values). Same JSON format.",
                system=_ANSWER_SYSTEM, max_tokens=config.LLM_MAX_TOKENS_ANSWER)
            if fix is not None and str(fix.data.get("summary") or "").strip():
                summary = language.ascii_digits(str(fix.data["summary"])).strip()
                insights = [language.ascii_digits(str(i)) for i in fix.data.get("insights") or [] if str(i).strip()][:3]
            fixed, left = number_check.repair(summary, allowed, frames)
        else:
            log.info("answer: unsupported numbers %s repaired in code", bad)
        summary = fixed
        repaired = [number_check.repair(i, allowed, frames) for i in insights]
        insights = [text for text, problems in repaired if not problems]  # an insight with a wrong figure is dropped
        if left:
            log.warning("answer: summary still has unsupported numbers %s", left)
        if state.get("emit") is not None:
            _emit(state, "summary_replace", text=summary)
    # Month-years the result / question never had ('October 2025' for 2026 data): corrected in code.
    summary, wrong_years = year_check.fix(summary, frames, given)
    insights = [year_check.fix(i, frames, given)[0] for i in insights]
    summary, wrong_days = year_check.drop_wrong_weekdays(summary)
    insights = [i for i in insights if not year_check.drop_wrong_weekdays(f"{summary} {i}")[1]]
    if wrong_days:
        _step(state, "weekday_check", time.perf_counter(), dropped=wrong_days)
        wrong_years = [*wrong_years, *wrong_days]
    if wrong_years:
        _step(state, "year_check", time.perf_counter(), fixed=wrong_years)
        if state.get("emit") is not None:
            _emit(state, "summary_replace", text=summary)
    # Insights are observations about THIS result: a number from an earlier answer is not allowed
    # there. Seen live: 'Installation to SAT for LTCT' (105.39) got the insight 'shorter than HTCT
    # (406.82) and 3PH (301.12)' -- their TOTAL journey from the previous turn, so the comparison
    # was false (HTCT 87.50, 3PH 82.51 installation-to-SAT). The summary may still quote them.
    own = number_check.allowed_values(frames, " ".join([state["question"], triage["question"], notes or "",
                                                         analysis or "", lined or "", why or ""]))
    insights = [i for i in insights if not number_check.unsupported(i, own)]
    # Highest / lowest check: a figure can be IN the result yet not be the max / min it is
    # called ('lowest feeder count (48)' when Feb had 29). Wrong insight = dropped; a wrong
    # summary gets one correction with the right figures, then the bad sentence is cut.
    period_col, partial, _ = _partial_periods(result.get("result"))
    wrong = extreme_check.wrong_claims(summary, result.get("result"), period_col, partial)
    insights = [i for i in insights if not extreme_check.wrong_claims(i, result.get("result"), period_col, partial)]
    if wrong:
        _step(state, "extreme_check", time.perf_counter(), wrong=wrong)
        log.warning("answer: wrong highest/lowest claims %s -- asking for a correction", wrong)
        fix = llm_gateway.chat_json(
            "answer", prompt + "\n\nYOUR PREVIOUS ANSWER:\n"
            + json.dumps({"summary": summary, "insights": insights}, ensure_ascii=False)
            + f"\nTHESE HIGHEST / LOWEST CLAIMS ARE WRONG: {'; '.join(wrong)}. Rewrite the answer using FACTS "
              "for every highest / lowest claim. Same JSON format.",
            system=_ANSWER_SYSTEM, max_tokens=config.LLM_MAX_TOKENS_ANSWER)
        if fix is not None and str(fix.data.get("summary") or "").strip():
            new = language.ascii_digits(str(fix.data["summary"])).strip()
            if not number_check.unsupported(new, allowed):
                summary = new
        summary = extreme_check.drop_sentences(summary, result.get("result"), period_col, partial) or summary
        if state.get("emit") is not None:
            _emit(state, "summary_replace", text=summary)
    if _PERIOD_RE.search(f"{state['question']} {triage['question']}") and _ALL_TIME_LEAD_RE.match(summary):
        # The user named a period: an 'all time' assumption contradicts the question -- drop the clause.
        # The English question too: 'ਪਿਛਲੇ ਮਹੀਨੇ' (last month) never matched the English period words.
        rest = _ALL_TIME_LEAD_RE.sub("", summary, count=1)
        summary = rest[:1].upper() + rest[1:]
    # Pipeline words ('RESULT contains 200 rows ...') rewritten, and causes the answer was never given dropped
    # (checks/wording_check.py; code only).
    # A failure here must never cost the answer (12 Nov: a float cell crashed vocabulary() -> 'Something went wrong').
    cleaned, causes = summary, []
    try:
        cleaned = wording_check.leaks(summary)
        vocab = wording_check.vocabulary(frames, given)
        causes = wording_check.unsupported_causes(cleaned, vocab)
        cleaned = wording_check.drop_sentences(cleaned, causes)
        insights = [wording_check.leaks(i) for i in insights]
        insights = [i for i in insights if not wording_check.unsupported_causes(i, vocab)]
    except Exception:
        log.warning("answer: wording check failed", exc_info=True)
        cleaned = summary
    if cleaned != summary:
        _step(state, "wording_check", time.perf_counter(), dropped_causes=causes[:3])
        summary = cleaned
        if state.get("emit") is not None:
            _emit(state, "summary_replace", text=summary)
    # Sources of a data answer: built by code from what actually ran (never by the LLM).
    # The refresh time is the WFM gold snapshot's -- it says nothing about MDM tables.
    refreshed = answer_cache.current_version().split("|")[0] if state.get("agent") != "mdm" else ""
    try:
        fresh = freshness.check(plan.get("tables") or [], plan.get("sql"))
    except Exception:
        log.warning("answer: freshness check failed", exc_info=True)
        fresh = []
    state["citations"] = grounding.data_citations(plan, triage, f"{_tidy_ts(refreshed)} IST" if refreshed else None, fresh)
    stale = freshness.warnings(fresh)
    if stale:
        _step(state, "freshness", time.perf_counter(), stale=stale)
        state["warnings"] = [*(state.get("warnings") or []), *stale]
    # What the answer is really written in (models skip the LANGUAGE line now and then); a
    # mismatch with the target is translated afterwards (main._in_language).
    state["answer_language"] = language.written_in(" ".join([summary, *insights]), writing)
    return {"summary": summary, "insights": insights,
            "followups": _pick_followups(d.get("followups"), options), "source": res.model}

_ANSWER_SYSTEM += "\n" + injection.UNTRUSTED_RULE
