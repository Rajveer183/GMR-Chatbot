"""19 Nov UI chat: 'how many tickets are closed on 18 november' -> 'tell about 2025' was resolved as the whole of 2025
(gpt-oss-120b). A follow-up naming only a year keeps the latest turn's day / month."""

from gmr_agents.agents.router.triage import carry_partial_date

DAY_TURN = [{"resolved_question": "How many O&M tickets were closed on 18 November 2026?"}]
MONTH_TURN = [{"resolved_question": "How many meters were installed in March 2026?"}]


def _run(asked, resolved, prior):
    state = {"question": asked, "trace_id": "test", "trace": []}
    triage = {"kind": "data", "question": resolved, "followup": False}
    changed = carry_partial_date(state, triage, prior)
    return changed, triage


def test_bare_year_keeps_day_and_month():
    changed, t = _run("tell about 2025", "How many O&M tickets were closed in 2025?", DAY_TURN)
    assert changed and t["question"] == "How many O&M tickets were closed on 18 November 2025?" and t["followup"]
    changed, t = _run("and 2025?", "How many O&M tickets were closed in the year 2025?", DAY_TURN)
    assert t["question"] == "How many O&M tickets were closed on 18 November 2025?"


def test_bare_year_keeps_month():
    changed, t = _run("and 2025?", "How many meters were installed in 2025?", MONTH_TURN)
    assert changed and t["question"] == "How many meters were installed in March 2025?"


def test_whole_year_and_comparisons_unchanged():
    for asked in ("whole year 2025", "tell about entire 2025", "2025 vs 2026", "month-wise 2025", "2025 trend"):
        changed, t = _run(asked, "How many O&M tickets were closed in 2025?", DAY_TURN)
        assert not changed, asked


def test_own_period_or_no_dated_turn_unchanged():
    assert not _run("november 2025", "How many O&M tickets were closed in November 2025?", DAY_TURN)[0]
    assert not _run("last 7 days of 2025", "How many O&M tickets were closed in 2025?", DAY_TURN)[0]
    assert not _run("tell about 2025", "How many O&M tickets were closed in 2025?",
                    [{"resolved_question": "How many O&M tickets were closed in 2026?"}])[0]
    assert not _run("tell about 2025", "How many O&M tickets were closed in 2025?", None)[0]


def test_model_already_kept_the_date():
    changed, t = _run("tell about 2025", "How many O&M tickets were closed on 18 November 2025?", DAY_TURN)
    assert not changed and t["question"].endswith("18 November 2025?")
