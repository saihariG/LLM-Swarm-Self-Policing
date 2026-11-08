#!/usr/bin/env python
"""Exploratory check named in PREREG_AMENDMENTS.md: repeat the main tests without swarms that lost an agent.

An agent is lost when a call cut off in the middle of a tool call leaves invalid JSON in its
history, because every later call then fails. This check is outside the Holm family of the
main report, and it changes no pre-registered number.
Usage: python analysis/sensitivity.py runs/main_A --joiners-only runs/main_B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.outcomes import block_table, load_events  # noqa: E402
from analysis.report import analyse  # noqa: E402

MARK = "Invalid tool call"


def lost_agents(run_dir: Path) -> dict[str, dict]:
    """Agents with a refused history, with the round it started and the calls it failed."""
    lost: dict[str, dict] = {}
    for e in load_events(run_dir):
        if e["type"] in ("assistant", "debrief") and MARK in (e.get("error") or ""):
            row = lost.setdefault(e["agent"], {"first_round": e["round"], "failed_calls": 0})
            row["failed_calls"] += 1
    return lost


def affected_runs(block: Path, runs: list[str]) -> dict[str, dict]:
    found = {run: lost_agents(block / run) for run in runs}
    return {run: agents for run, agents in found.items() if agents}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("block", type=Path)
    p.add_argument("--per-cell", type=int, default=6)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--joiners-only", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "notes" / "sensitivity_report.json")
    a = p.parse_args()
    df = block_table(a.block)
    pure = block_table(a.joiners_only) if a.joiners_only else None
    affected = affected_runs(a.block, df["run"].tolist())
    kept = df[~df["run"].isin(affected)]
    result = analyse(kept, a.per_cell, a.n_perm, a.n_boot, joiners_only=pure)
    report = {"note": "Exploratory, repeats the main tests without swarms that lost an agent.",
              "swarms_total": len(df), "swarms_affected": len(affected), "affected": affected,
              "lost_agents": sum(len(v) for v in affected.values()), "result": result}
    a.out.write_text(json.dumps(report, indent=1, default=float))
    print(f"swarms {len(df)}, affected {len(affected)}, kept {len(kept)}")
    for key, row in result["confirmatory"].items():
        est = row.get("estimate", row.get("diff"))
        print(f"{key:28s} {est:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}]  p={row['p']:.4f}")


if __name__ == "__main__":
    main()
