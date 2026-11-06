"""Technical-card SQL checks (6 Nov 2026 test): one row per meter per month, so a query must pin ONE month --
the dashboard's default month when none is asked, never MAX(data_date) (August 2026 holds only 2-3 days)."""

from gmr_agents.checks.entity_check import TECHNICAL_DEFAULT_MONTH, _technical_problems

PINNED = f"data_date = {TECHNICAL_DEFAULT_MONTH}"


def test_default_month_passes():
    sql = f"SELECT COUNT(*) FROM hes_gold.voltage_deviation_index WHERE {PINNED}"
    assert _technical_problems("What is the average VDI?", [sql]) == []


def test_unpinned_month_is_rejected():
    # Live 6 Nov: 'How many meters are Poor on VDI?' counted 11.3M meter-months (June + July + August).
    sql = "SELECT status, COUNT(*) FROM hes_gold.voltage_deviation_index GROUP BY status"
    assert any("pin one month" in p for p in _technical_problems("How many meters are Poor on VDI?", [sql]))


def test_max_data_date_is_not_the_default():
    sql = ("SELECT COUNT_IF(status = 'Poor') FROM hes_gold.voltage_unbalance_index WHERE blocks_3ph > 0 AND "
           "data_date = (SELECT MAX(data_date) FROM hes_gold.voltage_unbalance_index)")
    assert any("not MAX(data_date)" in p for p in _technical_problems("Poor voltage unbalance in Kashi", [sql]))


def test_month_wise_needs_no_pin_and_named_month_must_match():
    sql = "SELECT data_date, COUNT(*) FROM hes_gold.voltage_drop GROUP BY data_date"
    assert _technical_problems("Voltage drop month-wise", [sql]) == []
    sql = f"SELECT COUNT(*) FROM hes_gold.dt_peak_loading WHERE {PINNED}"
    assert any("August" in p for p in _technical_problems("Overloaded DTs in August", [sql]))


def test_three_phase_cards_need_the_signal_filter():
    sql = f"SELECT AVG(vui_mean) FROM hes_gold.voltage_unbalance_index WHERE {PINNED}"
    assert any("blocks_3ph" in p for p in _technical_problems("Average VUI", [sql]))


def test_ranking_skips_the_blank_area():
    sql = f"SELECT circle, SUM(oob_blocks) FROM hes_gold.pct_time_out_of_band WHERE {PINNED} GROUP BY circle ORDER BY 2 DESC"
    assert any("IS NOT NULL" in p for p in _technical_problems("Which circle is worst on voltage band?", [sql]))


def test_and_or_without_brackets_is_rejected():
    # Live 6 Nov: band = 'Critical' AND zone IN (Agra) OR zone IN (Kashi) -- the OR side dropped the month.
    sql = (f"SELECT COUNT(DISTINCT meter_no) FROM hes_gold.low_voltage_pockets_meters WHERE {PINNED} "
           "AND band = 'Critical' AND UPPER(zone) IN ('AGRA') OR UPPER(zone) IN ('AZAMGARH', 'VARANASI I')")
    probs = _technical_problems("How many critical low voltage meters are there in Kashi?", [sql])
    assert any("brackets" in p for p in probs)
    assert any("Kashi only" in p for p in probs)
    ok = sql.replace("IN ('AGRA') OR", "IN ('AZAMGARH') AND (zone IS NULL OR").replace("'VARANASI I')", "'VARANASI I'))")
    assert not any("brackets" in p for p in _technical_problems("critical low voltage meters", [ok]))


def test_rollup_on_an_expression_and_under_voltage_measure():
    sql = (f"SELECT UPPER(circle) c, SUM(oob_blocks) FROM hes_gold.pct_time_out_of_band WHERE {PINNED} "
           "GROUP BY ROLLUP (UPPER(circle))")
    probs = _technical_problems("% time under voltage by circle", [sql])
    assert any("ROLLUP" in p for p in probs) and any("uv_blocks" in p for p in probs)
