"""11 Nov 2026 advanced-question retest: generic fixes (NULL band sums, receivables latest row, repair hints,
'Total ...' rows in FACTS)."""
import pandas as pd

from gmr_agents.agents.sql_writer.planner import _repair_hint
from gmr_agents.checks.entity_check import _null_band_sum_problems, check
from gmr_agents.presentation.analyst import _result_facts


def test_null_filter_sum_in_arithmetic_is_sent_back():
    bad = ("with ns as (select lower(project) as project, sum(meter_count) filter (where age_key = 'age_91_120') as a91, "
           "sum(meter_count) filter (where age_key = 'age_120_plus') as a120 from t group by 1) "
           "select (ns.a91 + ns.a120) as older_than_90 from ns")
    assert _null_band_sum_problems(bad)
    ok = ("select coalesce(sum(n) filter (where k = 'a'), 0) as a, coalesce(sum(n) filter (where k = 'b'), 0) as b, "
          "a + b as c from t")
    assert _null_band_sum_problems(ok) == []
    plain = "select sum(n) filter (where e) as eligible, round(sum(x) * 100.0 / nullif(sum(y), 0), 2) as pct from t"
    assert _null_band_sum_problems(plain) == []


def test_receivables_table_wide_max_is_one_projects_row():
    sql = ("select p.project, r.lumpsum_invoiced from bands p join public.revenue_receivables r on lower(r.project) = p.project "
           "where r.created_at = (select max(created_at) from public.revenue_receivables)")
    assert any("each project's latest row" in p for p in check("revenue impact of non-SAT meters", [sql]))
    good = ("select project, lumpsum_invoiced from (select r.*, row_number() over (partition by project order by created_at desc, "
            "id desc) as rn from public.revenue_receivables r) x where rn = 1")
    assert not any("each project's latest row" in p for p in check("revenue impact of non-SAT meters", [good]))


def test_repair_hints_for_reserved_alias_and_ambiguous_column():
    assert "rename it" in _repair_hint("select 1 as drop", "write / DDL keyword(s) not allowed: DROP")
    assert "qualify it" in _repair_hint("select project from a join b using (x)",
                                        'column reference "project" is ambiguous')


def test_total_journey_row_is_not_a_stage():
    df = pd.DataFrame({"stage": ["DI to Agency", "Installation to SAT", "SAT to Revenue", "Total Journey"],
                       "avg_days": [25.51, 86.2, 73.66, 186.76]})
    facts = _result_facts(df)
    assert "highest Installation to SAT (86.2)" in facts
    assert "Total Journey (186.76)" not in facts.split("highest")[-1]


def test_generate_series_allowed_other_table_functions_refused():
    from gmr_agents.catalog.catalog import get_catalog
    from gmr_agents.guards.sql_guard import check_sql_safety
    c = get_catalog("wfm")
    cal = ("select d::date as day, coalesce(sum(i.meter_count), 0) as n from generate_series(current_date - 7, "
           "current_date - 1, interval '1 day') as d left join public.gold_mi_installation_day_agg i on i.period_start = d::date "
           "group by 1")
    assert check_sql_safety(cal, c)["ok"]
    assert not check_sql_safety("select * from pg_read_file('/etc/passwd')", c)["ok"]


def test_defect_rate_needs_installed_base():
    from gmr_agents.checks.entity_check import _defect_rate_base_problems as f
    counts_only = "select meter_type, sum(defect_count) from public.gold_meter_defect_event_day_agg group by 1"
    assert f("Segment consumers by meter type and show defect rate per segment", counts_only)
    with_base = ("select (select sum(defect_rate_numerator_count) from public.gold_meter_defect_event_day_agg) * 100.0 / "
                 "(select sum(meter_count) from public.gold_mi_installation_month_agg)")
    assert f("defect rate by oem", with_base) == []
    assert f("how many burnt meters last month", counts_only) == []


def test_rollup_coalesce_label_without_grouping_is_sent_back():
    from gmr_agents.checks.entity_check import _rollup_coalesce_problems as f
    bad = ("SELECT COALESCE(project, 'Not placed') AS project, SUM(meter_count) AS n FROM public.gold_mi_installation_month_agg "
           "GROUP BY ROLLUP (project)")
    assert f([bad])
    good = ("SELECT CASE WHEN GROUPING(project) = 1 THEN 'Total' ELSE COALESCE(project, 'Not placed') END AS project, "
            "SUM(meter_count) AS n FROM t GROUP BY ROLLUP (project)")
    assert f([good]) == []
    assert f(["SELECT COALESCE(project, 'x') AS p, SUM(n) FROM t GROUP BY project"]) == []


def test_warnings_right_now_need_data():
    from gmr_agents.agents.router.triage import _ACT_NOW_RE
    assert _ACT_NOW_RE.search("What early warning signs should we watch in the O&M backlog right now?")
    assert _ACT_NOW_RE.search("Give me alerts: which numbers crossed a danger threshold this week?")
    assert not _ACT_NOW_RE.search("What does the alert threshold mean?")


def test_athena_default_month_and_cast_slips():
    from gmr_agents.checks.entity_check import _athena_slip_problems as f
    own_table = ("WITH m AS (SELECT MAX(data_date) AS d FROM hes_gold.feeder_loss GROUP BY data_date "
                 "HAVING COUNT(DISTINCT data_date) >= 10) SELECT circle FROM hes_gold.feeder_loss WHERE data_date = (SELECT d FROM m)")
    assert any("dt_load_profile" in p for p in f([own_table]))
    right = ("SELECT circle FROM hes_gold.feeder_loss WHERE data_date = (SELECT MAX(month_date) FROM (SELECT month_date FROM "
             "hes_gold.dt_load_profile WHERE month_date >= DATE '2026-06-01' GROUP BY month_date HAVING COUNT(DISTINCT data_date) >= 10) c)")
    assert f([right]) == []
    assert any("::" in p for p in f(["SELECT x::double FROM hes_gold.energy_theft_suspect"]))
    assert f(["SELECT x::numeric FROM public.gold_mi_installation_day_agg"]) == []
    assert f(["SELECT 'a::b' AS t FROM hes_gold.dim_meter"]) == []


def test_unsupported_method_on_programme_data_is_answered_with_data():
    from gmr_agents.agents.router.triage import _method_on_programme_data as f
    assert f("Using reinforcement learning, optimise technician allocation")
    assert f("Estimate time-to-failure for meters using survival analysis")
    assert not f("Explain reinforcement learning")


def test_athena_timeout_cancels_and_is_repairable(monkeypatch):
    import db_connection.athena as A
    from gmr_agents.execution.executor import _is_connectivity_error
    stopped = []

    class Fake:
        def start_query_execution(self, **kw):
            return {"QueryExecutionId": "q1"}

        def get_query_execution(self, QueryExecutionId):
            return {"QueryExecution": {"Status": {"State": "RUNNING"}}}

        def stop_query_execution(self, QueryExecutionId):
            stopped.append(QueryExecutionId)

    monkeypatch.setattr(A, "_athena", Fake())
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    import pytest
    with pytest.raises(RuntimeError) as e:
        A._execute("SELECT 1", "hes_gold", 0.01, max_seconds=0.0001)
    assert stopped == ["q1"] and "timed out" in str(e.value)
    assert not _is_connectivity_error(e.value)


def test_analytic_asks_are_data_definitions_stay_explain():
    from gmr_agents.agents.router.triage import _analytic_ask as f
    for q in ["What threshold for non-SAT ageing should trigger escalation?", "Run PCA on zone-level KPIs",
              "How do installation, SAT, invoicing and collection KPIs depend on each other causally?",
              "Build an enterprise-wide operational risk model"]:
        assert f(q), q
    for q in ["tell me: What does feeder confidence mean?", "What is SAIDI?", "How is SAT % calculated?",
              "What does non-SAT ageing mean?", "Hello"]:
        assert not f(q), q
