"""Answer-text checks of 18 Nov (checks/wording_check.py + narrator/writer.py), no LLM quota used:
a 'no data' claim while RESULT has figures, and a day series squashed into '22-620-975'."""
import pandas as pd
import pytest

from gmr_agents.agents.narrator import writer
from gmr_agents.checks import wording_check as W
from shared import llm_gateway

KASHI = pd.DataFrame({"day": ["Total", "2026-11-13", "2026-11-13", "2026-11-14"],
                      "zone": ["Total", "VARANASI II", "AZAMGARH", "VARANASI II"],
                      "meters_installed": [1346, 488, 269, 589]})
SQL = "SELECT ... FROM public.gold_mi_installation_day_agg WHERE LOWER(project) = 'kashi' GROUP BY ROLLUP (period_start, zone)"


def test_squashed_runs_skip_dates_and_bands():
    assert W.squashed_runs("AGRA 22‑620‑975 installs") == ["22‑620‑975"]
    assert W.squashed_runs("from 2026-11-08 to 2026-11-17, bands 0-30 and 31-60 days") == []


def test_false_empty_needs_no_figure_and_real_rows():
    assert W.false_empty("Kashi zone data is not available in the current snapshot.", KASHI)
    assert not W.false_empty("No data for feeder; DT 157.", KASHI)              # quotes a figure
    assert not W.false_empty("No data was found.", KASHI.assign(meters_installed=0))
    assert W.sql_filters(SQL) == ["project = kashi"]


@pytest.fixture()
def scripted(monkeypatch):
    replies, prompts = [], []

    def fake(task, prompt, **_kw):
        prompts.append(prompt)
        return llm_gateway.LLMResult(replies.pop(0), "scripted", "scripted", None, None, 0.0)
    monkeypatch.setattr(llm_gateway, "chat_json", fake)
    monkeypatch.setattr(writer.answer_cache, "current_version", lambda: "")
    monkeypatch.setattr(writer.freshness, "check", lambda *a, **k: [])
    return replies, prompts


def _write(question):
    state = {"question": question, "trace": [], "agent": "wfm", "trace_id": "test"}
    triage = {"question": question, "topics": ["installation"], "language": "en", "parts": []}
    plan = {"sql": SQL, "tables": ["gold_mi_installation_day_agg"], "reasoning": "scripted"}
    return writer._write_answer(state, triage, plan, {"result": KASHI, "group_count": len(KASHI)})


def test_no_data_claim_is_rewritten_with_the_filter_named(scripted):
    replies, prompts = scripted
    replies += [{"summary": "Kashi zone data is not available in the current snapshot.", "insights": []},
                {"summary": "Kashi installed 1,346 meters on 13-14 Nov; VARANASI II led with 488 on 13 Nov.", "insights": []}]
    out = _write("Meters installed per day in Kashi zone wise")
    assert "1,346" in out["summary"] and len(prompts) == 2
    assert "project = kashi" in prompts[1]


def test_squashed_series_is_rewritten(scripted):
    replies, prompts = scripted
    replies += [{"summary": "VARANASI II leads with 488‑269‑589 installs.", "insights": []},
                {"summary": "VARANASI II leads with 1,077 installs over the two days.", "insights": []}]
    out = _write("Meters installed per day in Kashi zone wise")
    assert "‑" not in out["summary"] and len(prompts) == 2


def test_clean_answer_costs_no_extra_call(scripted):
    replies, prompts = scripted
    replies += [{"summary": "Kashi installed 1,346 meters; VARANASI II led with 488 on 13 Nov.", "insights": []}]
    _write("Meters installed per day in Kashi zone wise")
    assert len(prompts) == 1


# entity_check: 'category wise (consumer, DT, feeder)' must not run as progress_category = 'CONSUMER' only (18 Nov).
from gmr_agents.checks import entity_check as E  # noqa: E402

_ONE = ("SELECT period_start, project, SUM(meter_count) FROM public.gold_mi_installation_day_agg "
        "WHERE has_meter_serial AND progress_category = 'CONSUMER' GROUP BY ROLLUP (period_start, project)")
_SPLIT = ("SELECT period_start, project, progress_category, SUM(meter_count) FROM public.gold_mi_installation_day_agg "
          "WHERE has_meter_serial AND progress_category IS NOT NULL GROUP BY ROLLUP (period_start, project, progress_category)")


def _cat_flag(question, sql):
    return any("category-wise answer" in p for p in E.check(question, [sql]))


def test_category_wise_catches_a_single_prefixed_category_filter():
    assert _cat_flag("Daily MI progress by project, category wise (consumer, DT, feeder)", _ONE)
    assert _cat_flag("Consumer, DT and feeder meters installed per day in Kashi", _ONE)
    assert not _cat_flag("Daily MI progress by project, category wise (consumer, DT, feeder)", _SPLIT)


def test_category_rule_leaves_single_category_asks_alone():
    assert not _cat_flag("How many consumer meters were installed?", _ONE)
    assert not _cat_flag("A burnt DT meter leads to how many affected consumers?", _ONE.replace("CONSUMER", "DT"))


# 18 Nov 25-question check: a share against a base RESULT never had, and a band Total under ROLLUP.
def test_share_of_a_missing_base_is_cut():
    df = pd.DataFrame({"project": ["Total"], "installed": [4410095], "sat_done": [3866783]})
    text = "SAT-done 3,866,783 (87.68% of installed, 87.68% of eligible); lumpsum 3,326,409."
    bases = W.unsupported_bases(text, [df])
    assert bases == ["87.68% of eligible"]
    assert W.strip_phrases(text, bases) == "SAT-done 3,866,783 (87.68% of installed); lumpsum 3,326,409."
    assert W.unsupported_bases("97.51% of supplied meters.", [df.assign(total_supplies=1)]) == []


def test_band_total_under_rollup_is_sent_back():
    sql = ("SELECT p.project, b.label, SUM(n.meter_count) FROM (VALUES ('agra', 1)) AS p(project, o) CROSS JOIN "
           "(VALUES (1, 'age_0_30', '0-30 days'), (6, NULL, 'Total')) AS b(ord, age_key, label) LEFT JOIN "
           "public.gold_non_sat_ageing_snapshot n ON n.project = p.project AND (n.age_key = b.age_key OR b.age_key IS NULL) "
           "GROUP BY ROLLUP (p.project, (b.ord, b.label))")
    assert any("band list" in p for p in E.check("Non-SAT ageing by project", [sql]))
    assert not any("band list" in p for p in E.check("Non-SAT ageing", [sql.replace("ROLLUP (p.project, (b.ord, b.label))", "b.ord, b.label")]))


def test_band_total_rollup_is_flattened_in_code():
    from gmr_agents.agents.sql_writer import planner as PL
    sql = ("SELECT p.project, b.label FROM (VALUES (1, 'age_0_30', '0-30 days'), (6, NULL, 'Total')) AS b(ord, age_key, label) "
           "GROUP BY ROLLUP (p.p_ord, p.project, (b.ord, b.label)) ORDER BY 1")
    assert "GROUP BY p.p_ord, p.project, b.ord, b.label ORDER BY 1" in PL._clean_sql(sql)
    plain = "SELECT zone, SUM(x) FROM t GROUP BY ROLLUP (zone)"           # no band Total row: untouched
    assert PL._clean_sql(plain) == plain
