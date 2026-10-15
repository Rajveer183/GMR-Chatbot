"""Replay real follow-up failures (a follow-up resolved against an EARLIER turn instead of
the latest one) through the understanding step, several times each.

    python eval/question_bank/eval_followups.py        # 3 runs per case

Cases are real sessions from outputs/gmr_chat_sessions.json (2026-10-04). A run passes when
the standalone question keeps the latest turn's project / topic word and adds nothing
from earlier turns that the latest turn does not have.
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import logging  # noqa: E402

logging.disable(logging.WARNING)

from gmr_agents import llm_pipeline as P  # noqa: E402
from shared import config  # noqa: E402
from shared.session_store import SessionStore  # noqa: E402

_T1 = {"question": "how much stock is older than 90 days in agra", "resolved_question": "How much stock is older than 90 days in Agra?",
       "summary": "The stock older than 90 days in Agra is 47,900 meters.", "context": {"topics": ["stock_ageing"]}}
_T2 = {"question": "how many unused meters are 0-30 days old in all projects", "resolved_question": "How many unused meters are 0-30 days old?",
       "summary": "21,170 unused meters are 0-30 days old.", "context": {"topics": ["stock_ageing"]}}

_T3 = {"question": "non sat ageing for dtr meters in agra", "resolved_question": "Non-SAT ageing for DTR meters in Agra",
       "summary": "11,296 non-SAT DTR meters in Agra, 10,615 of them 120+ days old.",
       "context": {"topics": ["non_sat_ageing"], "domain": "wfm"}}

_T4 = {"question": "supply utilization in varanasi i zone", "resolved_question": "Supply utilization in Varanasi I zone",
       "summary": "Varanasi I zone utilization is 97.07%.", "context": {"topics": ["inventory_utilization"], "domain": "wfm"}}

_T5 = {"question": "pmpm invoice percentage in kashi", "resolved_question": "What is the PMPM invoice percentage in Kashi?",
       "summary": "Kashi PMPM invoice % is 87.82%.", "context": {"topics": ["sat_progress"], "domain": "wfm"}}

_T6 = {"question": "Which project has the highest PMPM invoice percentage?",
       "resolved_question": "Which project has the highest PMPM invoice percentage?",
       "summary": "AGRA has the highest PMPM invoice percentage.", "context": {"topics": ["sat_progress"], "domain": "wfm"}}

# (session or fixed turns, turns before the follow-up, message, words the answer must contain, words it must not)
CASES = [
    ("5891aaa98fbe445e", 2, "what about agra", ["agra", "sat"], ["kashi", "zone"]),
    ("5e4e86360ff34b2b", 2, "Break it down by OEM.", ["agra", "burnt", "oem"], ["triveni"]),
    # complete on its own: all projects or Agra are both fine -- never Kashi from turn 1
    ("10e2d982ae1f4559", 2, "zone wise SAT progress", ["zone", "sat"], ["kashi"]),
    # same topic, complete sentence: keeps the latest turn's place (validation/scope.py)
    # (fixed turns, not a stored session: that chat (1148ad16fa3f4e4e, 2026-10-04) kept growing)
    ([_T1], 1, "how many unused meters are 0-30 days old", ["agra", "30"], ["kashi", "triveni"]),
    # latest turn was all projects: turn 1's Agra must not come back
    ([_T1, _T2], 2, "stock between 100 and 200 days old", ["100", "200"], ["agra"]),
    # another topic, same conversation: the place still carries (user, 2026-10-04)
    ([_T3], 1, "feeder meters without sat", ["agra", "feeder"], ["kashi", "triveni"]),
    # 'overall' resets the place
    ([_T3], 1, "overall feeder meters without sat", ["feeder"], ["agra"]),
    # one named meter: never a carried place
    ([_T4], 1, "how long has meter HP8500592 been waiting for sat", ["hp8500592"], ["varanasi", "zone"]),
    # comparing projects = all projects, never the remembered one (user, 2026-10-04)
    ([_T5], 1, "Which project has the highest PMPM invoice percentage?", ["project", "pmpm"], ["kashi"]),
    # comparing zones keeps the coarser remembered project
    ([_T5], 1, "which zone has the lowest PMPM invoice percentage", ["zone", "kashi"], ["agra", "triveni"]),
    # no measure in the message: the latest turn's measure continues (user, 2026-10-04)
    ([_T6], 1, "provide a comparison between all 3 projects", ["pmpm"], ["installed meters"]),
]


def main(runs: int) -> None:
    store = SessionStore(os.path.join(config.OUTPUT_DIR, "gmr_chat_sessions.json"))
    ok = n = 0
    for sid, upto, msg, must, never in CASES:
        prior = (sid if isinstance(sid, list) else store.recent_turns(sid, 50))[:upto]
        for _ in range(runs):
            st = {"question": msg, "trace": [], "trace_id": "eval", "warnings": []}
            t = P._triage(st, prior)
            if t:
                P.fix_question(st, t, prior)  # code-side measure + place fixes, as in the pipeline
            q = (t or {}).get("question", "").lower()
            good = bool(t) and all(w in q for w in must) and not any(w in q for w in never)
            ok += good; n += 1
            print(f"{'ok' if good else 'XX'}  {msg!r:28} -> {q}")
            time.sleep(1)
    print(f"\n{ok}/{n} follow-ups resolved against the latest turn")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
