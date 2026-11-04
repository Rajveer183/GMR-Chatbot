"""Knowledge agent (RAG): explains a KPI / term / data source from retrieved chunks, with
citations checked by checks/grounding.py.

Moved from gmr_agents/llm_pipeline.py unchanged.
"""

from __future__ import annotations
import time
from shared import config, llm_gateway
from gmr_agents.resolvers import language
from gmr_agents.checks import grounding
from gmr_agents.agents.common import _BUSY, _step
from gmr_agents.agents.followups import _followup_options, _options_text, _pick_followups


_EXPLAIN_SYSTEM = """You explain GMR smart-meter programme KPIs, dashboard labels and data terms to a business user (SAT = Site Acceptance Test). Answer using ONLY the RETRIEVED CONTEXT chunks -- they are the exact definitions the dashboard uses; explain in business words, not SQL/column names.
Rules: do not use outside knowledge for facts; every fact must come from a chunk, and list the chunk_ids you used in "citations" (only ids shown in RETRIEVED CONTEXT); copy numbers (thresholds, days, %) exactly from the chunks; if the chunks do not answer the question, set "abstain": true and say what is missing instead of guessing; if two chunks disagree, do not pick one -- describe both in "conflicts".
Write in LANGUAGE (follow its description); numbers stay exactly as in the chunks, in Western digits 0-9.
Reply with JSON only: {"summary": "clear explanation, 2-6 sentences, include the formula in words", "insights": ["0-3 useful details/caveats"], "citations": ["chunk ids used"], "supported": true, "abstain": false, "conflicts": [], "followups": [3 numbers from FOLLOW-UP OPTIONS: the most useful next questions, best first]}"""


def _explain(state: dict, triage: dict) -> dict:
    t0 = time.perf_counter()
    # Grounding (validation/grounding.py): only rule / glossary chunks relevant enough to
    # the question are shown; with none, abstain without calling the LLM.
    evidence = grounding.collect_explain_evidence(triage["question"], triage["topics"], k=config.GROUNDING_TOP_K)
    writing = language.writing_language(triage.get("language", "en"))  # other targets: English, then translated
    if not evidence:
        _step(state, "explain", t0, outcome="abstained: no relevant evidence")
        state["grounding"] = {"abstained": True, "confidence": 0.0, "problems": ["no evidence above threshold"]}
        return {"summary": grounding.ABSTAIN_NO_EVIDENCE, "insights": [], "followups": _followup_options(triage)[:3],
                "refused": True, "source": "grounding"}
    prompt = ("RETRIEVED CONTEXT:\n" + grounding.context_block(evidence, config.GROUNDING_MAX_CONTEXT_CHARS)
              + f"\n\nLANGUAGE: {language.prompt_line(writing)}\nUSER QUERY: {triage['question']}")
    options = _followup_options(triage)
    if options:
        prompt += "\n\n" + _options_text(options)
    res = llm_gateway.chat_json("explain", prompt, system=_EXPLAIN_SYSTEM, max_tokens=config.LLM_MAX_TOKENS_EXPLAIN)
    _step(state, "explain", t0, model=res.model if res else "none available")
    if res is None or not str(res.data.get("summary") or "").strip():
        return {"summary": _BUSY, "insights": [], "followups": [], "refused": True, "source": "none (AI busy)"}
    d = res.data
    d["summary"] = language.ascii_digits(str(d.get("summary") or ""))
    d["insights"] = [language.ascii_digits(str(i)) for i in d.get("insights") or []]
    checked = grounding.verify(d, evidence)
    if not checked.abstained:
        state["answer_language"] = language.written_in(" ".join([checked.summary, *checked.insights]), writing)
    state["citations"] = checked.citations
    state["grounding"] = {"abstained": checked.abstained, "confidence": checked.confidence,
                          "problems": checked.problems, "conflicts": checked.conflicts,
                          "evidence": [(c.chunk_id, round(c.similarity, 3)) for c in evidence]}
    _step(state, "grounding", time.perf_counter(), abstained=checked.abstained, confidence=checked.confidence,
          problems=checked.problems[:3] or None)
    insights = checked.insights[:3] + [f"Sources disagree: {c}" for c in checked.conflicts][:2]
    return {"summary": checked.summary, "insights": insights,
            "followups": _pick_followups(d.get("followups"), options), "refused": checked.abstained,
            "source": res.model}
