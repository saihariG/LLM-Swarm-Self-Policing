#!/usr/bin/env python
"""Secondary and exploratory tables listed in PREREG.md. Nothing here is in the Holm family.

Cell means, peers scored as a detector, containment under audit, per-model breakdowns,
block C, silence reasons, and peers next to the outside monitor on the same swarms.
Run the judge first, and analysis/silence.py and analysis/monitor_arm.py if those tables are wanted.
Usage: python analysis/secondary.py [--block-a main_A] [--block-b main_B] [--block-c main_C]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.effects import two_group  # noqa: E402
from analysis.extensions import extras, mean_ci  # noqa: E402
from analysis.monitor_report import condition_table, drift, paired_shift  # noqa: E402
from analysis.monitor_report import load as load_monitor  # noqa: E402
from analysis.outcomes import block_table, load_events  # noqa: E402
from analysis.report import add_primary  # noqa: E402

KEY_CELLS = ["R0P0C0H0E0", "R1P1C1H1E0", "R0P1C1H1E0", "R1P1C1H1E1", "R0P1C1H1E1"]
MAIN_METRICS = ["Y1", "exploit_share", "Y3", "Y4", "callout_rate", "exposed", "removals"]
DETECTOR_METRICS = ["tp_rate", "any_correct", "n_false_named", "honest_accused", "latency"]
BREAKS = ["no_notice", "no_judge", "no_own", "no_route", "no_act", "acted"]


def frame(block: Path, joiner: str = "gpt-oss", police: str = "glm") -> pd.DataFrame:
    df = add_primary(block_table(block), joiner, police)
    return df.merge(pd.DataFrame([extras(block / r) for r in df["run"]]), on="run")


def cell_table(df: pd.DataFrame, metrics: list[str], n_boot: int) -> list[dict]:
    rows = []
    for cell, g in df.groupby("cell"):
        row: dict = {"cell": cell, "swarms": len(g)}
        for m in metrics:
            values = g[m].to_numpy(dtype=float)
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean_ci(values, n_boot)
            row[f"{m}_n"] = int((~np.isnan(values)).sum())
        rows.append(row)
    return rows


def family_table(df: pd.DataFrame) -> list[dict]:
    """Each model's adoption, policing and reports, cell by cell."""
    rows = []
    for cell, g in df.groupby("cell"):
        for i in (0, 1):
            if f"m{i}_name" not in g:
                continue
            rows.append({"cell": cell, "model": str(g[f"m{i}_name"].iloc[0]).split("/")[-1], "swarms": len(g),
                         "adoption": float(g[f"m{i}_adoption"].mean()), "policing": float(g[f"m{i}_police"].mean()),
                         "reports": float(g[f"m{i}_report"].mean())})
    return rows


def containment_row(events: list[dict]) -> dict:
    """What an E1 swarm's audits did. Rounds run from the first accepted exploit to the first removal."""
    labels = {e["id"]: e["label"] for e in events if e["type"] == "verdict"}
    accepted = [e for e in events if e["type"] == "submission" and e["accepted"]]
    exploit_rounds = [e["round"] for e in accepted if labels[e["id"]] == "exploit"]
    audits = [e for e in events if e["type"] == "audit"]
    removals = [e for e in events if e["type"] == "removal"]
    first_removal = min((e["round"] for e in removals), default=None)
    return {"audits": len(audits), "removals": len(removals),
            "seeds_removed": sum(e["target_role"].startswith("seed") for e in removals),
            "live_removed": sum(e["target_role"] == "live" for e in removals),
            "honest_audited": sum(e["target_role"] == "honest" for e in audits),
            "first_removal_round": first_removal,
            "rounds_to_removal": first_removal - min(exploit_rounds) if first_removal and exploit_rounds else None}


def containment_table(block: Path, df: pd.DataFrame) -> list[dict]:
    rows = []
    for cell, g in df[df["E"] == 1].groupby("cell"):
        table = pd.DataFrame([containment_row(load_events(block / r)) for r in g["run"]])
        row: dict = {"cell": cell, "swarms": len(g), "credit_left": float(g["credit_left"].mean()),
                     "any_removal": float((table["removals"] > 0).mean())}
        row.update({c: float(pd.to_numeric(table[c]).mean()) for c in table.columns})
        rows.append(row)
    return rows


def replication(df: pd.DataFrame, n_perm: int) -> list[dict]:
    """Block C: enforcement on and off at R1 P1 C1 H1, for each outcome."""
    return [two_group(df, m, "R1P1C1H1E1", "R1P1C1H1E0", n_perm=n_perm) for m in ("exploit_share", "Y1", "Y3", "Y4")]


def silence_counts(block: Path, runs: list[str]) -> dict[str, Counter]:
    """first_break by model, from silence.jsonl in each run."""
    out: dict[str, Counter] = {}
    for run in runs:
        path = block / run / "silence.jsonl"
        if not path.exists():
            continue
        models = json.loads((block / run / "summary.json").read_text()).get("agent_models") or {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            model = models.get(row["agent"], "all").split("/")[-1]
            out.setdefault(model, Counter())[row.get("first_break", "unparsed")] += 1
    return out


def silence_table(block: Path, df: pd.DataFrame) -> list[dict]:
    rows = []
    for cell, g in df.groupby("cell"):
        for model, counts in silence_counts(block, g["run"].tolist()).items():
            total = sum(counts.values())
            rows.append({"cell": cell, "model": model, "agents": total,
                         **{k: counts.get(k, 0) / total for k in BREAKS}, "unparsed": counts.get("unparsed", 0) / total})
    return rows


def peers_and_monitors(block: Path, df: pd.DataFrame) -> dict:
    """Peers work per agent and monitors per item, so the two sit side by side and are not one scale."""
    key = df[df["cell"].isin(KEY_CELLS)]
    peers = {m: float(pd.to_numeric(key[m]).mean()) for m in ("tp_rate", "any_correct", "n_false_named", "honest_accused")}
    out: dict = {"peers": peers, "swarms": len(key), "monitors": [], "shifts": [], "drift": []}
    if not (block / "monitor").exists():
        return out
    scores = load_monitor(block)
    out["monitors"] = condition_table(scores).to_dict("records")
    for monitor in scores["monitor"].unique():
        have = set(scores[scores["monitor"] == monitor]["condition"])
        out["shifts"] += [paired_shift(scores, monitor, a, b) for a, b in
                          (("action", "own"), ("action", "swarm"), ("stream_control", "stream")) if {a, b} <= have]
        if "stream" in have:
            out["drift"].append(drift(scores, monitor))
    return out


def block_report(block: Path, df: pd.DataFrame, n_boot: int) -> dict:
    return {"swarms": len(df), "cells": cell_table(df, MAIN_METRICS, n_boot),
            "detector": cell_table(df, DETECTOR_METRICS, n_boot), "families": family_table(df),
            "containment": containment_table(block, df), "silence": silence_table(block, df),
            "peers_and_monitors": peers_and_monitors(block, df)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    p.add_argument("--block-a", default="main_A")
    p.add_argument("--block-b", default="main_B")
    p.add_argument("--block-c", default="main_C")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--out", type=Path, default=ROOT / "notes" / "secondary_report.json")
    a = p.parse_args()
    report: dict = {"note": "Secondary and exploratory, outside the Holm family."}
    for name, block in (("A", a.block_a), ("B", a.block_b)):
        path = a.runs / block
        report[name] = block_report(path, frame(path), a.n_boot)
    path = a.runs / a.block_c
    df = frame(path, "minimax", "muse")
    report["C"] = block_report(path, df, a.n_boot)
    if {"R1P1C1H1E1", "R1P1C1H1E0"} <= set(df["cell"]):
        report["C"]["replication"] = replication(df, a.n_perm)
    a.out.write_text(json.dumps(report, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
