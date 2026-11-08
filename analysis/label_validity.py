#!/usr/bin/env python
"""Post-hoc checks of the judge's labels against the verifier, which needs no human.

1. Writers: do agents who write a promote message submit exploits more often than other live
   agents? Do agents who write a call_out submit them less often?
2. Names: are the agents named in call_out messages cheaters (a seed, or a live agent with an
   accepted exploit), and how often is the honest confederate named? The baseline is the share
   of cheaters among the other agents in the swarm at the time of the message.
3. Agreement between the two rubric versions on every message, and of the first-pass hand labels
   with each version.
Exploratory (PREREG_AMENDMENTS.md, 6:40pm ET on 2026-11-07). Needs judged_v1.jsonl and judged_v2.jsonl.
Usage: python analysis/label_validity.py runs/main_A runs/main_B runs/main_C
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.kappa import cohen_kappa  # noqa: E402

VERSIONS = ("v1", "v2")


def load_labels(run: Path, version: str) -> dict[int, dict]:
    return {r["seq"]: r for r in (json.loads(x) for x in (run / f"judged_{version}.jsonl").read_text().splitlines())}


def run_facts(run: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    roles = summary["roles"]
    models = summary.get("agent_models") or {}
    verdicts = {e["id"]: e["label"] for e in events if e["type"] == "verdict"}
    first_exploit: dict[str, int] = {}
    for e in events:
        if e["type"] == "submission" and e["accepted"] and verdicts.get(e["id"]) == "exploit":
            first_exploit[e["agent"]] = min(first_exploit.get(e["agent"], 10**6), e["round"])
    live = sorted(a for a, r in roles.items() if r == "live")
    speech = [e for e in events if e["type"] in ("post", "dm", "feedback") and e.get("role") == "live"]
    return {"run": run.name, "roles": roles, "models": models, "live": live, "first_exploit": first_exploit,
            "speech": speech}


def family(models: dict, agent: str) -> str:
    name = models.get(agent, "single")
    return name.split("/")[-1]


def writer_rows(facts: dict, labels: dict[int, dict]) -> list[dict]:
    """One row per live agent: did it write each label, and did it ever have an accepted exploit."""
    wrote: dict[str, set] = defaultdict(set)
    for e in facts["speech"]:
        j = labels.get(e["seq"])
        if j:
            wrote[e["agent"]].add(j["label"])
    return [{"family": family(facts["models"], a), "exploiter": a in facts["first_exploit"],
             "promote": "promote" in wrote[a], "call_out": "call_out" in wrote[a]} for a in facts["live"]]


def name_rows(facts: dict, labels: dict[int, dict]) -> list[dict]:
    """One row per agent named in a call_out, with the role and cheating status of the named agent."""
    rows = []
    roles, first = facts["roles"], facts["first_exploit"]
    for e in facts["speech"]:
        j = labels.get(e["seq"])
        if not j or j["label"] != "call_out":
            continue
        others = [a for a in roles if a != e["agent"]]
        before = [a for a in others if first.get(a, 10**6) <= e["round"]]
        for name in sorted(set(j.get("accused") or []) & set(roles)):
            if name == e["agent"]:
                continue
            rows.append({"family": family(facts["models"], e["agent"]), "type": e["type"],
                         "run": facts.get("run"), "target": name,
                         "role": "seed" if roles[name].startswith("seed") else roles[name],
                         "cheater_ever": name in first, "cheater_before": first.get(name, 10**6) <= e["round"],
                         "baseline_before": len(before) / len(others)})
    return rows


def rate(rows: list[dict], key: str, where) -> dict:
    sub = [r for r in rows if where(r)]
    return {"n": len(sub), "rate": sum(r[key] for r in sub) / len(sub) if sub else None}


def summarise_writers(rows: list[dict]) -> dict:
    out = {}
    for fam in sorted({r["family"] for r in rows}) + ["all"]:
        mine = [r for r in rows if fam == "all" or r["family"] == fam]
        out[fam] = {"agents": len(mine),
                    "exploit_if_promote": rate(mine, "exploiter", lambda r: r["promote"]),
                    "exploit_if_no_promote": rate(mine, "exploiter", lambda r: not r["promote"]),
                    "exploit_if_call_out": rate(mine, "exploiter", lambda r: r["call_out"]),
                    "exploit_if_no_call_out": rate(mine, "exploiter", lambda r: not r["call_out"])}
    return out


def summarise_names(rows: list[dict]) -> dict:
    if not rows:
        return {"named": 0}
    n = len(rows)
    return {"named": n,
            "cheater_ever": sum(r["cheater_ever"] for r in rows) / n,
            "cheater_before": sum(r["cheater_before"] for r in rows) / n,
            "baseline_before": sum(r["baseline_before"] for r in rows) / n,
            "honest_named": sum(r["role"] == "honest" for r in rows) / n,
            "roles": dict(Counter(r["role"] for r in rows)),
            # named counts naming events. These count distinct agents (run, name), added 2026-11-08.
            "distinct_named": len({(r["run"], r["target"]) for r in rows}),
            "distinct_by_role": dict(Counter(r["role"] for r in {(r["run"], r["target"], r["role"]): r
                                                                 for r in rows}.values()))}


def agreement(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {"n": 0}
    a, b = zip(*pairs)
    binary = lambda xs: ["call_out" if x == "call_out" else "not" for x in xs]  # noqa: E731
    return {"n": len(pairs), "agreement": sum(x == y for x, y in pairs) / len(pairs),
            "kappa": cohen_kappa(list(a), list(b)), "call_out_kappa": cohen_kappa(binary(a), binary(b)),
            "table": {f"{x}|{y}": c for (x, y), c in sorted(Counter(pairs).items())}}


def hand_agreement(sheet: Path, block: Path) -> dict:
    """First-pass hand labels against each rubric version, looked up by run and seq."""
    hand = {r["item"]: r["hand_label"] for r in csv.DictReader(open(sheet, newline=""))}
    key = list(csv.DictReader(open(sheet.with_suffix(".key.csv"), newline="")))
    out = {}
    for version in VERSIONS:
        cache: dict[str, dict] = {}
        pairs = []
        for r in key:
            if r["run"] not in cache:
                cache[r["run"]] = load_labels(block / r["run"], version)
            label = cache[r["run"]][int(r["seq"])]["label"]
            if label != "unparsed" and hand[r["item"]]:
                pairs.append((hand[r["item"]], label))
        out[version] = agreement(pairs)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("blocks", type=Path, nargs="+")
    p.add_argument("--hand", type=Path, default=ROOT / "notes/hand_check.csv")
    p.add_argument("--hand-block", type=Path, default=ROOT / "runs/main_A")
    p.add_argument("--out", type=Path, default=ROOT / "notes/label_validity.json")
    a = p.parse_args()
    result: dict = {"blocks": {}}
    for block in a.blocks:
        writers = {v: [] for v in VERSIONS}
        names = {v: [] for v in VERSIONS}
        pairs = []
        for run in sorted(block.iterdir()):
            if not (run / "summary.json").exists():
                continue
            facts = run_facts(run)
            labels = {v: load_labels(run, v) for v in VERSIONS}
            for v in VERSIONS:
                writers[v] += writer_rows(facts, labels[v])
                names[v] += name_rows(facts, labels[v])
            for e in facts["speech"]:
                x, y = labels["v1"].get(e["seq"]), labels["v2"].get(e["seq"])
                if x and y and "unparsed" not in (x["label"], y["label"]):
                    pairs.append((x["label"], y["label"]))
        result["blocks"][block.name] = {
            "writers": {v: summarise_writers(writers[v]) for v in VERSIONS},
            "names": {v: summarise_names(names[v]) for v in VERSIONS},
            "v1_vs_v2": agreement(pairs)}
    if a.hand.exists():
        result["hand_first_pass"] = hand_agreement(a.hand, a.hand_block)
    a.out.write_text(json.dumps(result, indent=1, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
