#!/usr/bin/env python
"""When removals happen in enforcement swarms: the round of the first removal and whether both seeds go.

Usage: python analysis/removal_timing.py
Reads the E1 swarms of main_A, main_B, main_C, ext_dose1 and ext_dose2 and writes notes/removal_timing.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ("main_A", "main_B", "main_C", "ext_dose1", "ext_dose2")


def block_timing(block: Path) -> dict:
    first, both_gone = Counter(), Counter()
    swarms = both = 0
    for run in sorted(block.iterdir()):
        if not (run / "summary.json").exists():
            continue
        if not json.loads((run / "summary.json").read_text())["cell"].endswith("E1"):
            continue
        events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
        removals = [e for e in events if e["type"] == "removal"]
        swarms += 1
        if removals:
            first[min(e["round"] for e in removals)] += 1
        seeds = [e for e in removals if e["target_role"].startswith("seed")]
        if len({e["target"] for e in seeds}) == 2:
            both += 1
            both_gone[max(e["round"] for e in seeds)] += 1
    return {"swarms": swarms, "first_removal_round": dict(sorted(first.items())),
            "both_seeds_removed": both, "round_both_seeds_gone": dict(sorted(both_gone.items()))}


def main() -> None:
    out = {b: block_timing(ROOT / "runs" / b) for b in BLOCKS if (ROOT / "runs" / b).exists()}
    (ROOT / "notes" / "removal_timing.json").write_text(json.dumps(out, indent=1))
    for b, d in out.items():
        print(b, d)


if __name__ == "__main__":
    main()
