#!/usr/bin/env python
"""Choose which judge labels the analysis reads.

judged_v1.jsonl and judged_v2.jsonl hold each rubric's labels. judged.jsonl, the file that
analysis/outcomes.py reads, becomes a copy of the chosen one, and each block records its choice
in judge_version.txt. The first switch saves the original judged.jsonl as judged_v1.jsonl.
Usage: python scripts/use_rubric.py v2 runs/main_A runs/main_B
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def active(block: Path) -> str:
    path = block / "judge_version.txt"
    return path.read_text().strip() if path.exists() else "v1"


def use(version: str, block: Path) -> int:
    runs = [r for r in sorted(block.iterdir()) if (r / "judged.jsonl").exists() or (r / "judged_v1.jsonl").exists()]
    missing = [r.name for r in runs if not (r / f"judged_{version}.jsonl").exists()
               and not (version == "v1" and active(block) == "v1")]
    if missing:
        raise SystemExit(f"{block}: {len(missing)} runs have no {version} labels, e.g. {missing[:3]}. Nothing changed.")
    if active(block) == "v1":
        for run in runs:
            if (run / "judged.jsonl").exists() and not (run / "judged_v1.jsonl").exists():
                shutil.copy2(run / "judged.jsonl", run / "judged_v1.jsonl")
    for run in runs:
        shutil.copy2(run / f"judged_{version}.jsonl", run / "judged.jsonl")
    (block / "judge_version.txt").write_text(version + "\n")
    return len(runs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("version", choices=["v1", "v2"])
    p.add_argument("blocks", type=Path, nargs="+")
    a = p.parse_args()
    for block in a.blocks:
        print(f"{block}: {use(a.version, block)} runs now read {a.version} labels")


if __name__ == "__main__":
    main()
