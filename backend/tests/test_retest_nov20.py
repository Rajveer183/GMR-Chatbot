"""20 Nov retest: the follow-ups still wrong after the 19-20 Nov fixes."""

from datetime import date
from unittest import mock

from gmr_agents.checks import entity_check
from gmr_agents.presentation import feeder_compare


def _compare_turn(month="2026-08", months=("2026-06", "2026-07", "2026-08")):
    return {"question": "Compare it with its feeder.", "resolved_question": "Compare it with its feeder.",
            "summary": "**Meter AL1157049** (consumer 1 phase) -- substation **ETMADPUR SS_22238**, Agra zone, "
                       "mapping audit Aug 2026.",
            "context": {"domain": "mdm", "answer_kind": "feeder_compare", "meter_no": "AL1157049",
                        "audit_month": month, "audit_months": list(months)}}


# ---- feeder comparison: a bare period follow-up keeps the consumer ------------------------------------------

def test_last_month_after_feeder_compare_is_previous_audit_month():
    req = feeder_compare.request("What about last month?", [_compare_turn()], today=date(2026, 11, 20))
    assert req == {"meter": "AL1157049", "consumer_id": None, "month": date(2026, 7, 1)}


def test_named_month_after_feeder_compare():
    for q in ("And June?", "for June 2026", "What about Jun?"):
        req = feeder_compare.request(q, [_compare_turn()], today=date(2026, 11, 20))
        assert req and req["meter"] == "AL1157049" and req["month"] == date(2026, 6, 1), q


def test_last_month_skips_unaudited_months():
    req = feeder_compare.request("previous month", [_compare_turn(months=("2026-06", "2026-08"))],
                                 today=date(2026, 11, 20))
    assert req["month"] == date(2026, 6, 1)


def test_last_month_falls_back_to_summary_month():
    turn = _compare_turn()
    turn["context"] = {k: v for k, v in turn["context"].items() if not k.startswith("audit")}
    req = feeder_compare.request("What about last month?", [turn], today=date(2026, 11, 20))
    assert req["month"] == date(2026, 7, 1)


def test_period_followup_needs_a_feeder_compare_turn():
    plain = {"question": "Installations by project", "summary": "...", "context": {"domain": "wfm"}}
    assert feeder_compare.request("What about last month?", [plain]) is None
    # a period plus another ask is not a bare period follow-up
    assert feeder_compare.request("What about installations last month?", [_compare_turn()]) is None
    assert feeder_compare.request("What about last month?", []) is None


# ---- mapping audit: a literal calendar 'last month' is caught too ------------------------------------------

def test_mapping_last_month_literal_calendar_pin_flagged():
    sql = ("SELECT verdict, SUM(consumer_count) FROM hes_gold.mapping_audit_summary "
           "WHERE data_date = DATE '2026-10-01' GROUP BY verdict")
    with mock.patch.object(entity_check, "_ist_today", return_value=date(2026, 11, 20)):
        problems = entity_check._mapping_problems("What about last month?", [sql])
    assert any("audit month before the latest" in p for p in problems)


def test_mapping_last_month_subquery_and_named_month_pass():
    good = ("SELECT verdict, SUM(consumer_count) FROM hes_gold.mapping_audit_summary WHERE data_date = (SELECT "
            "MAX(data_date) FROM hes_gold.mapping_audit_summary WHERE data_date < (SELECT MAX(data_date) FROM "
            "hes_gold.mapping_audit_summary)) GROUP BY verdict")
    july = "SELECT verdict FROM hes_gold.mapping_audit_summary WHERE data_date = DATE '2026-07-01'"
    with mock.patch.object(entity_check, "_ist_today", return_value=date(2026, 11, 20)):
        assert not any("audit month before" in p for p in entity_check._mapping_problems("What about last month?", [good]))
        assert not any("audit month before" in p for p in entity_check._mapping_problems("What about last month?", [july]))


def test_calendar_last_month_pin_january():
    with mock.patch.object(entity_check, "_ist_today", return_value=date(2027, 1, 5)):
        assert entity_check._calendar_last_month_pin("where data_date = date '2026-12-01'")
        assert not entity_check._calendar_last_month_pin("where data_date = date '2026-11-01'")


# ---- is_non_sat_pending labelled as eligible pending --------------------------------------------------------

_MONTHWISE = ("SELECT TO_CHAR(period_start, 'YYYY-MM') AS month, SUM(meter_count) AS eligible_pending_sat "
              "FROM public.gold_mi_installation_day_agg WHERE LOWER(project) = 'kashi' AND is_non_sat_pending = TRUE "
              "GROUP BY 1")


def test_non_sat_pending_called_eligible_flagged():
    assert entity_check._non_sat_label_problems(_MONTHWISE.lower())


def test_non_sat_pending_ok_when_labelled_or_cut_off():
    assert not entity_check._non_sat_label_problems(_MONTHWISE.replace("eligible_pending_sat", "installed_not_sat").lower())
    cut = _MONTHWISE.replace("WHERE", "JOIN public.sat_eligibility_config c ON LOWER(c.project) = LOWER(project) "
                                      "WHERE period_start <= c.eligibility_cutoff_date AND")
    assert not entity_check._non_sat_label_problems(cut.lower())


def test_not_sat_by_month_with_install_filters_flagged():
    sql = ("SELECT date_trunc('month', period_start)::date AS month, SUM(meter_count) AS installed_not_sat "
           "FROM public.gold_mi_installation_day_agg WHERE LOWER(project) = 'kashi' AND sat_stage IS NULL "
           "AND has_meter_serial AND progress_category IS NOT NULL AND NOT is_replacement GROUP BY 1 ORDER BY 1")
    assert any("add up to the SAT snapshot" in p for p in entity_check._non_sat_label_problems(sql.lower()))
    plain = ("SELECT date_trunc('month', period_start)::date AS month, SUM(meter_count) AS installed_not_sat "
             "FROM public.gold_mi_installation_day_agg WHERE LOWER(project) = 'kashi' AND sat_stage IS NULL "
             "GROUP BY 1 ORDER BY 1")
    assert not entity_check._non_sat_label_problems(plain.lower())


def test_not_sat_filter_column_and_other_subquery_untouched():
    progress = ("SELECT zone, SUM(meter_count) FILTER (WHERE is_non_sat_pending) AS sat_pending, SUM(meter_count) AS mi "
                "FROM public.gold_mi_installation_day_agg WHERE has_meter_serial AND progress_category IS NOT NULL "
                "AND NOT is_replacement GROUP BY zone")
    pace = ("WITH p AS (SELECT SUM(meter_count) / 30.0 AS pace FROM public.gold_mi_installation_day_agg "
            "WHERE has_meter_serial AND NOT is_replacement AND period_start >= CURRENT_DATE - 30) "
            "SELECT p.pace, (SELECT SUM(meter_count) FROM public.gold_mi_sat_invoice_snapshot WHERE sat_stage IS NULL) "
            "AS not_sat FROM p")
    for sql in (progress, pace):
        assert not any("add up to" in p for p in entity_check._non_sat_label_problems(sql.lower()))


# ---- serial filtered on consumer_id -> meter_no, by code ---------------------------------------------------

def test_serial_on_consumer_id_rewritten():
    from gmr_agents.guards.serial_lookup import fix
    got = fix("SELECT * FROM hes_gold.dim_meter WHERE regexp_replace(TRIM(consumer_id), '^0+', '') = "
              "regexp_replace('AL1157049','^0+', '') ORDER BY meter_no")
    assert got == "SELECT * FROM hes_gold.dim_meter WHERE meter_no = 'AL1157049' ORDER BY meter_no"
    assert fix("SELECT * FROM hes_gold.dim_meter d WHERE (TRIM(d.consumer_id) = 'al1157049') AND d.zone = 'AGRA'") \
        == "SELECT * FROM hes_gold.dim_meter d WHERE (d.meter_no = 'AL1157049') AND d.zone = 'AGRA'"


def test_digit_consumer_and_other_tables_untouched():
    from gmr_agents.guards.serial_lookup import fix
    for sql in ("SELECT * FROM hes_gold.dim_meter WHERE consumer_id = '5583636000'",
                "SELECT * FROM hes_gold.dim_meter WHERE regexp_replace(TRIM(consumer_id), '^0+', '') = "
                "regexp_replace('05583636000','^0+', '')",
                "SELECT * FROM public.other WHERE consumer_id = 'AL1157049'"):
        assert fix(sql) == sql


# ---- column names in the answer text -> the table's words --------------------------------------------------

def test_plain_columns():
    import pandas as pd
    from gmr_agents.checks.wording_check import plain_columns
    df = pd.DataFrame(columns=["project", "installed_not_sat", "eligible_pending_sat", "loss_pct"])
    assert plain_columns("KASHI: installed_not_sat 184,657, eligible_pending_sat 75,067.", [df]) \
        == "KASHI: installed not SAT 184,657, eligible pending SAT 75,067."
    assert plain_columns("Installed_not_sat counts", [df]) == "Installed not SAT counts"
    assert plain_columns("loss_pct 5", [df], {"loss_pct": "Loss %"}) == "Loss % 5"
    assert plain_columns("see hes_gold.installed_not_sat", [df]) == "see hes_gold.installed_not_sat"
    assert plain_columns("no columns here", [df]) == "no columns here"


# ---- month-column dates in the answer text -> 'Oct 2026' ---------------------------------------------------

def test_month_words_in_summary():
    import datetime as dt
    import pandas as pd
    from gmr_agents.presentation.blocks import month_words
    m = pd.DataFrame({"month": [dt.date(2024, 6, 1), dt.date(2026, 10, 1)], "n": [15, 45648]})
    assert month_words("peak 2026-10-01 (45,648), low 2024-06-01", [m]) == "peak Oct 2026 (45,648), low Jun 2024"
    s = pd.DataFrame({"month": ["2024-06", "2026-10"], "n": [1, 2]})
    assert month_words("2024\u201106: 15, 2026-10: 2", [s]) == "Jun 2024: 15, Oct 2026: 2"
    assert month_words("not 2026-10-15 or 2026-10-01T10", [s]) == "not 2026-10-15 or 2026-10-01T10"


def test_month_words_leaves_day_columns():
    import datetime as dt
    import pandas as pd
    from gmr_agents.presentation.blocks import month_words
    lone = pd.DataFrame({"install_date": [dt.date(2026, 10, 1)], "n": [1]})
    days = pd.DataFrame({"day": [dt.date(2026, 11, 1), dt.date(2026, 11, 2)], "n": [1, 2]})
    assert month_words("installed on 2026-10-01", [lone]) == "installed on 2026-10-01"
    assert month_words("peak on 2026-11-01", [days]) == "peak on 2026-11-01"


def test_group_digits_year_range_counts():
    import pandas as pd
    from gmr_agents.checks.number_check import group_digits
    df = pd.DataFrame({"month": ["2025-06", "2025-07"], "n": [1958, 2025]})
    assert group_digits("Jun 2025: 1958, Jul 2025: 1,637", [df]) == "Jun 2025: 1,958, Jul 2025: 1,637"
    assert group_digits("1958 meters were pending", [df]) == "1,958 meters were pending"
    assert group_digits("in 2025 the count rose", [df]) == "in 2025 the count rose"
    assert group_digits("since Jun 2025 it rose", [df]) == "since Jun 2025 it rose"
