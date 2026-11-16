"""'Today' = the IST date in the database too (execution/ist_today.py).

Between 00:00 and 05:30 IST the UTC database sessions' CURRENT_DATE was still yesterday. The swap runs only
near the IST day boundary; the rest of the day the SQL sent is the very same string.
"""

from datetime import datetime, timezone

import pytest

from gmr_agents.execution import executor, ist_today

PG = "(NOW() AT TIME ZONE 'Asia/Kolkata')::date"
ATHENA = "CAST(current_timestamp AT TIME ZONE 'Asia/Kolkata' AS date)"
NIGHT = datetime(2026, 11, 16, 20, 0, tzinfo=timezone.utc)    # 01:30 IST on 17 Nov
DAY = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)       # 11:30 IST
SQL = "SELECT COUNT(*) FROM t WHERE d BETWEEN CURRENT_DATE - 7 AND current_date - INTERVAL '1 day'"


@pytest.mark.parametrize("utc, inside", [
    ((18, 24), False), ((18, 25), True), ((20, 0), True), ((23, 59), True), ((0, 4), True), ((0, 5), False),
    ((6, 0), False), ((12, 0), False),
])
def test_window(utc, inside):
    now = datetime(2026, 11, 16, *utc, tzinfo=timezone.utc)
    assert ist_today.near_day_boundary(now) is inside


def test_daytime_sql_is_the_same_object():
    assert ist_today.anchor(SQL, "postgres", now=DAY) is SQL
    assert ist_today.anchor(SQL, "athena", now=DAY) is SQL


def test_night_swaps_every_bare_current_date():
    assert ist_today.anchor(SQL, "postgres", now=NIGHT) == SQL.replace("CURRENT_DATE", PG).replace("current_date", PG)
    assert ist_today.anchor(SQL, "athena", now=NIGHT).count(ATHENA) == 2


@pytest.mark.parametrize("sql", [
    "SELECT 'current_date' AS label FROM t",
    'SELECT "current_date" AS window_end FROM t',
    "SELECT t.current_date FROM t",
    "SELECT my_current_date FROM t",
    "SELECT 1 -- current_date\nFROM t",
    "SELECT 1 /* current_date */ FROM t",
])
def test_literals_identifiers_and_comments_untouched(sql):
    assert ist_today.anchor(sql, "postgres", now=NIGHT) == sql


def test_no_current_date_unchanged():
    sql = "SELECT * FROM t WHERE d < (NOW() AT TIME ZONE 'Asia/Kolkata')::date"
    assert ist_today.anchor(sql, "postgres", now=NIGHT) is sql


def test_error_text_maps_back():
    exc = RuntimeError(f"Execution failed on sql 'SELECT {PG} - x': operator does not exist")
    assert "CURRENT_DATE - x" in str(ist_today.unanchor_error(exc, "postgres"))


def test_today_is_ist(monkeypatch):
    assert ist_today.today() == datetime.now(ist_today.IST).date()


class _Conn:
    def __init__(self):
        self.sent = []

    def cursor(self):
        conn = self

        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql):
                conn.sent.append(sql)
        return _Cur()


def test_executor_sends_ist_date_at_night_and_logs_original(monkeypatch):
    sent = []
    monkeypatch.setattr(ist_today, "near_day_boundary", lambda now=None: True)
    monkeypatch.setattr(executor.pd, "read_sql_query", lambda sql, conn: sent.append(sql) or executor.pd.DataFrame())
    executor._read_with_timeout(SQL, _Conn())
    assert sent == [SQL.replace("CURRENT_DATE", PG).replace("current_date", PG)]


def test_executor_sends_same_sql_by_day(monkeypatch):
    sent = []
    monkeypatch.setattr(ist_today, "near_day_boundary", lambda now=None: False)
    monkeypatch.setattr(executor.pd, "read_sql_query", lambda sql, conn: sent.append(sql) or executor.pd.DataFrame())
    executor._read_with_timeout(SQL, _Conn())
    assert sent == [SQL] and sent[0] is SQL


def test_executor_error_quotes_current_date(monkeypatch):
    monkeypatch.setattr(ist_today, "near_day_boundary", lambda now=None: True)

    def boom(sql, conn):
        raise RuntimeError(f"Execution failed on sql '{sql}': syntax error")
    monkeypatch.setattr(executor.pd, "read_sql_query", boom)
    with pytest.raises(RuntimeError) as info:
        executor._read_with_timeout(SQL, _Conn())
    assert PG not in str(info.value) and "CURRENT_DATE - 7" in str(info.value)
