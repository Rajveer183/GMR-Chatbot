"""One regression command for the backend -- run it before committing a change.

    python eval/regression.py                       # unit tests + replay gate vs the saved baseline
    python eval/regression.py --record              # (re)record the replay baseline from the CURRENT code
    python eval/regression.py --db                  # also the live read-only checks (no LLM calls)

Stages (each one prints PASS / FAIL; the exit code is non-zero if any failed):
  1. pytest            every tests/test_*.py (offline: no LLM, no database)
  2. replay gate       eval/question_bank/replay_prompts.py: the logged questions replayed with the LLM and the
                       database stubbed; every prompt sent to the LLM and every final answer is hashed. IDENTICAL
                       = same prompts (so the same tokens) and the same answers as the baseline.
                       Record the baseline from the code BEFORE a change (git stash; --record; git stash pop).
  3. --db only         eval/question_bank/verify_examples.py (every verified SQL still runs) and
                       eval/question_bank/parity_audit.py (logged live SQL vs verified SQL, read-only; a report to read, not pass/fail).

The replay can drift with no code change when the data skeleton or catalog cache refreshes between runs
(see the replay docstring); on a replay diff, re-record the baseline on the old code right away and rerun
before blaming the change.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QB = os.path.join(BACKEND, "eval", "question_bank")
BASELINE = os.path.join(QB, "results", "replay_baseline.jsonl")
CURRENT = os.path.join(QB, "results", "replay_current.jsonl")


def stage(name: str, cmd: list[str]) -> bool:
    print(f"\n=== {name}: {' '.join(os.path.relpath(c, BACKEND) if os.path.isabs(c) else c for c in cmd[1:])}",
          flush=True)
    ok = subprocess.run(cmd, cwd=BACKEND).returncode == 0
    print(f"=== {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="record the replay baseline from the current code and stop")
    ap.add_argument("--db", action="store_true", help="also run the live read-only database checks")
    ap.add_argument("--limit", type=int, default=400, help="logged questions to replay (default 400)")
    args = ap.parse_args()
    py = sys.executable
    os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
    replay = os.path.join(QB, "replay_prompts.py")

    if args.record:
        return 0 if stage("replay baseline", [py, replay, "--out", BASELINE, "--limit", str(args.limit)]) else 1

    results = {"pytest": stage("pytest", [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"])}
    if os.path.exists(BASELINE):
        results["replay gate"] = stage("replay gate", [py, replay, "--out", CURRENT, "--limit", str(args.limit),
                                                       "--compare", BASELINE])
    else:
        print(f"\n=== replay gate: SKIPPED -- no baseline yet (python eval/regression.py --record)")
    if args.db:
        results["verified SQL runs"] = stage("verified SQL runs", [py, os.path.join(QB, "verify_examples.py")])
        # A report, not a pass/fail: logged answers older than an example's last edit show up as DIFF too.
        stage("parity audit (report)", [py, os.path.join(QB, "parity_audit.py")])

    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
