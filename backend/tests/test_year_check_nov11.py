"""checks/year_check.py: month-years the data never had are corrected in code."""
from datetime import date

import pandas as pd

from gmr_agents.checks import year_check

TODAY = date(2026, 11, 11)


def test_invented_year_fixed_to_current():
    df = pd.DataFrame({"project": ["agra"], "period": ["last_month"], "total_sat": [0]})
    text, wrong = year_check.fix("Pichle mahine (October 2025) SAT 0% raha.", [df], "SAT kyun kam hua pichle mahine?", TODAY)
    assert text == "Pichle mahine (October 2026) SAT 0% raha." and wrong == ["October 2025"]


def test_year_in_data_kept():
    df = pd.DataFrame({"month": ["2025-10-01", "2026-10-01"], "n": [1, 2]})
    assert year_check.fix("Oct 2025 vs Oct 2026", [df], "", TODAY)[1] == []


def test_year_from_data_for_that_month():
    df = pd.DataFrame({"month": ["2024-12-01", "2025-01-01"], "n": [1, 2]})
    text, wrong = year_check.fix("December 2023 was low", [df], "", TODAY)
    assert text == "December 2024 was low" and wrong


def test_last_year_question_allows_previous_year():
    assert year_check.fix("In March 2025 ...", [], "installs last year", TODAY)[1] == []


def test_year_named_in_question_kept_and_no_month_left_alone():
    assert year_check.fix("June 2019 figures", [], "what were installs in June 2019", TODAY)[1] == []
    assert year_check.fix("since 2019 the count rose", [], "", TODAY)[1] == []


def test_future_month_without_data_goes_to_last_year():
    text, _ = year_check.fix("December 2021 closures", [], "", TODAY)
    assert text == "December 2025 closures"


def test_wrong_weekday_sentence_dropped():
    text = ("Kal (10 Nov 2026) par 2,342 smart meters lagaye gaye the. Kal Sunday tha, isliye technicians kam the, "
            "lekin installs normal the. Yeh 2,335 consumer meters the.")
    out, dropped = year_check.drop_wrong_weekdays(text)
    assert out == "Kal (10 Nov 2026) par 2,342 smart meters lagaye gaye the. Yeh 2,335 consumer meters the."
    assert len(dropped) == 1


def test_right_weekday_or_several_dates_kept():
    ok = "On 10 Nov 2026 (a Tuesday) 2,342 meters were installed."
    assert year_check.drop_wrong_weekdays(ok) == (ok, [])
    two = "Sunday 9 Nov 2026 had 100 and 10 Nov 2026 had 200."
    assert year_check.drop_wrong_weekdays(two) == (two, [])
