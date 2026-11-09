#!/usr/bin/env python
"""Exploratory: two more coders for the fresh hand-check sheet, both LLMs from a family used in no swarm.

Two fresh Claude instances (Opus 5.5 and Sonnet 5.5) labeled notes/hand_check_2.csv blind. Each saw only the
judge's rubric v2 and the sheet (message type and text), and no run, cell or other label. This is not a human
check. It asks whether other models, given the same rubric, reproduce the judge's labels.
For each coder against the judge's revised labels (v2), its first labels (v1), the keyword rule and the name
rule, and for the two coders against each other: raw agreement, Cohen's kappa over the four labels, and kappa
for call_out against the rest. Written on 2026-11-08, after the main outcomes were read.
Usage: python analysis/llm_coders.py notes/hand_check_2.csv notes/hand_check_2_claude_opus.csv
       notes/hand_check_2_claude_sonnet.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import outcomes  # noqa: E402
from analysis.kappa import cohen_kappa  # noqa: E402
from analysis.label_robustness import name_labels  # noqa: E402


def agreement(a: list[str], b: list[str]) -> dict:
    binary = lambda xs: ["call_out" if x == "call_out" else "not" for x in xs]  # noqa: E731
    return {"n": len(a), "agreement": sum(x == y for x, y in zip(a, b)) / len(a),
            "kappa": cohen_kappa(a, b), "call_out_kappa": cohen_kappa(binary(a), binary(b))}


def reference_labels(sheet: Path) -> dict[str, dict[str, str]]:
    """For each item: the judge's v2 and v1 labels, the keyword rule and the name rule."""
    key = {r["item"]: r for r in csv.DictReader(open(sheet.with_suffix(".key.csv"), newline=""))}
    text = {r["item"]: r["text"] for r in csv.DictReader(open(sheet, newline=""))}
    out: dict[str, dict[str, str]] = {}
    cache: dict[Path, tuple[dict, dict]] = {}
    for item, r in key.items():
        run = ROOT / "runs" / r["block"] / r["run"]
        if run not in cache:
            v1 = {j["seq"]: j["label"] for j in map(json.loads, (run / "judged_v1.jsonl").read_text().splitlines())}
            cache[run] = (v1, name_labels(run))
        v1, names = cache[run]
        seq = int(r["seq"])
        out[item] = {"v2": r["judge_label"], "v1": v1[seq], "keyword": outcomes.heuristic_label(text[item]),
                     "names": names[seq]["label"]}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sheet", type=Path)
    p.add_argument("coders", type=Path, nargs="+", help="CSV files with item,label")
    p.add_argument("--out", type=Path, default=ROOT / "notes/llm_coders.json")
    a = p.parse_args()
    ref = reference_labels(a.sheet)
    items = sorted(ref, key=int)
    coders = {}
    for path in a.coders:
        labels = {r["item"]: r["label"].strip() for r in csv.DictReader(open(path, newline=""))}
        missing = [i for i in items if i not in labels]
        if missing:
            raise SystemExit(f"{path} lacks {len(missing)} items, e.g. {missing[:3]}")
        coders[path.stem.split("_")[-1]] = [labels[i] for i in items]
    out: dict = {"n_items": len(items), "label_counts": {}, "against": {}}
    for name, labels in coders.items():
        out["label_counts"][name] = dict(Counter(labels))
        out["against"][name] = {src: agreement(labels, [ref[i][src] for i in items])
                                for src in ("v2", "v1", "keyword", "names")}
    for src in ("v2", "v1", "keyword", "names"):
        out["label_counts"][src] = dict(Counter(ref[i][src] for i in items))
    names = list(coders)
    if len(names) == 2:
        out["between_coders"] = agreement(coders[names[0]], coders[names[1]])
    a.out.write_text(json.dumps(out, indent=1))
    for name, rows in out["against"].items():
        print(name, {src: {k: round(v, 3) for k, v in r.items() if k != "n"} for src, r in rows.items()})
    if "between_coders" in out:
        print("between coders", {k: round(v, 3) for k, v in out["between_coders"].items() if k != "n"})
    print("label counts", out["label_counts"])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
