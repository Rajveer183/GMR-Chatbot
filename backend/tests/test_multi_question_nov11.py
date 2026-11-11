"""resolvers/multi_question.py: split only on clear separators."""
import pytest

from gmr_agents.resolvers.multi_question import split


@pytest.mark.parametrize("msg,want", [
    ("How many meters were installed yesterday? And what is the SAT % by project?",
     ["How many meters were installed yesterday?", "What is the SAT % by project?"]),
    ("1. installations in Kashi last month 2. SAT pending in Agra 3. open O&M complaints",
     ["Installations in Kashi last month", "SAT pending in Agra", "Open O&M complaints"]),
    ("How many meters were installed in Kashi last month? And by zone?",
     ["How many meters were installed in Kashi last month?", "By zone?"]),
    ("Show stock by warehouse\nWhat is the SAT % by project?",
     ["Show stock by warehouse", "What is the SAT % by project?"]),
    ("Give me installations by project and also tell me the remaining stock in Triveni",
     ["Give me installations by project", "Tell me the remaining stock in Triveni"]),
    ("kitne meter lage kal, aur ye bhi batao ki SAT kitna pending hai Agra mein",
     ["Kitne meter lage kal", "Batao ki SAT kitna pending hai Agra mein"]),
])
def test_splits(msg, want):
    assert split(msg)[0] == want


@pytest.mark.parametrize("msg", [
    "Compare installations and SAT completion for Agra and Kashi",
    "How many meters were installed and how many got SAT in October?",
    "Show installations by zone; only consumer meters in Kashi please",
    "What is the SAT % by project?",
    "Which zone has the highest installs vs last month?",
    "installs?",
    "Why did meter installations rise in October compared to September, by zone?",
])
def test_one_question_kept(msg):
    assert split(msg) == ([msg], [])


def test_more_than_three_parts_reported():
    asked, extra = split("1. installs in Kashi today 2. SAT in Agra now 3. stock in Triveni now 4. open complaints by zone")
    assert len(asked) == 3 and extra == ["Open complaints by zone"]


def test_answer_message_runs_each_part_and_combines(monkeypatch):
    from gmr_agents import main

    calls, events = [], []

    def fake(question, session_id, emit=None, **kw):
        calls.append((question, session_id, kw.get("prior_questions")))
        n = len(calls)
        if emit:
            emit({"type": "summary_delta", "text": f"ans{n}"})
            emit({"type": "partial", "kpis": [{"label": f"k{n}"}], "charts": [], "tables": []})
        return {"question": question, "summary": f"answer {n}", "session_id": "s1", "refused": False,
                "kpis": [{"label": f"k{n}"}], "charts": [{"id": "chart-0"}],
                "tables": [{"id": "t0", "columns": [], "rows": []}], "insights": [f"i{n}"],
                "warnings": ["same warning"], "citations": [{"id": "sql", "source": "Live query", "location": f"t{n}"}],
                "suggested_followups": [f"f{n}"], "debug": {"trace_id": f"tr{n}"}}

    monkeypatch.setattr(main, "answer_question", fake)
    r = main.answer_message("How many installed yesterday? And what is the SAT % by project?", None,
                            emit=events.append, prior_questions=["old"])
    assert [c[0] for c in calls] == ["How many installed yesterday?", "What is the SAT % by project?"]
    assert calls[1][1] == "s1" and calls[1][2] is None          # 2nd part: same session, history from the store
    assert r["summary"].startswith("**1. How many installed yesterday?**\nanswer 1\n\n**2. What is the SAT")
    assert [c["id"] for c in r["charts"]] == ["q1-chart-0", "q2-chart-0"]
    assert [(t["trace_id"], t["table_index"]) for t in r["tables"]] == [("tr1", 0), ("tr2", 0)]
    assert r["warnings"] == ["same warning"] and r["suggested_followups"] == ["f2"]
    assert [p["trace_id"] for p in r["debug"]["parts"]] == ["tr1", "tr2"]
    live = [e for e in events if e["type"] == "summary_replace"]
    assert live[-1]["text"].startswith("**1. How many installed yesterday?**\nanswer 1\n\n**2.")  # part 1 kept
    assert [k["label"] for k in [e for e in events if e["type"] == "partial"][-1]["kpis"]] == ["k1", "k2"]


def test_single_question_untouched(monkeypatch):
    from gmr_agents import main

    monkeypatch.setattr(main, "answer_question", lambda q, s, **kw: {"summary": "x", "q": q})
    assert main.answer_message("What is the SAT % by project?", None) == {"summary": "x", "q": "What is the SAT % by project?"}
