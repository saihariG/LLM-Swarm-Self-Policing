#!/usr/bin/env python
"""List swarms over the pre-registered error limit, and move them aside for a rerun with --apply.

A swarm is rerun with the same seed only if over 5% of its LLM calls ended in an error.
Moved folders stay in <block>/_excluded, and every move is logged there.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

LIMIT = 0.05


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("block", type=Path)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    over = []
    for run in sorted(a.block.iterdir()):
        if not (run / "summary.json").exists():
            continue
        events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
        calls = [e for e in events if e["type"] in ("assistant", "debrief")]
        errors = sum(1 for e in calls if e.get("error"))
        if calls and errors / len(calls) > LIMIT:
            over.append((run, errors, len(calls)))
    for run, errors, calls in over:
        print(f"{run.name}: {errors} of {calls} calls failed")
    print(len(over), "swarms over the limit")
    if a.apply and over:
        dest = a.block / "_excluded"
        dest.mkdir(exist_ok=True)
        with open(dest / "log.txt", "a") as log:
            for run, errors, calls in over:
                shutil.move(str(run), str(dest / f"{run.name}_{int(time.time())}"))
                log.write(f"{time.ctime()} {run.name} {errors}/{calls}\n")


if __name__ == "__main__":
    main()
