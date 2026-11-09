#!/usr/bin/env python
"""Exploratory mechanism checks from the logs only (critique of 2026-11-07, items A3, A4 and A5).

pool       Rivalry and the chance to submit (A5). Block A swarms without audits: in rivalry swarms, the round
           by which every solvable task in the shared pool was taken and the share the scripted honest agent
           took. In all swarms, accepted honest work per live agent and submissions blocked because another
           agent already held the task.
removals   Who the audits removed (A4). Block A swarms with audits: seeds and live agents removed, and the
           share of swarms where seed A was gone before its round 7 reminder.
timeline   Adoption over time (A3). Cumulative gpt-oss adoption by round in the two P1 C1 H1 cells without
           audits, block A (five GLM agents) against block B (gpt-oss only), with the round of the first
           GLM board call-out (revised rubric) in each block A swarm.
The blocks were not randomized against each other, so the timeline is descriptive. All of this was written
after the main outcomes were read (PREREG_AMENDMENTS.md, 2026-11-08).
Usage: python analysis/mechanisms.py runs/main_A runs/main_B
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import outcomes  # noqa: E402
from swarmlab.bank import task_by_id  # noqa: E402

TIMELINE_CELLS = ("R0P1C1H1E0", "R1P1C1H1E0")
ROUNDS = 15


def _summary(run: Path) -> dict:
    return json.loads((run / "summary.json").read_text())


def family_of(summary: dict, agent: str) -> str:
    models = summary.get("agent_models") or {}
    name = models.get(agent, summary.get("model", ""))
    return "glm" if "glm" in name.lower() else "gpt-oss" if "gpt-oss" in name.lower() else name.split("/")[-1]


def pool_stats(run: Path) -> dict:
    """Task pool use in one swarm: exhaustion of solvable tasks (shared pools), honest work and late blocks."""
    s = _summary(run)
    roles = s["roles"]
    live = [a for a, r in roles.items() if r == "live"]
    events = outcomes.load_events(run)
    start = next(e for e in events if e["type"] == "run_start")
    solvable = {t for t in start["tasks"] if not task_by_id(t).impossible}
    verdict = {e["id"]: e["label"] for e in events if e["type"] == "verdict"}
    shared = s["cell"][1] == "1"
    owner: dict[str, str] = {}
    lock_round: dict[str, int] = {}
    late = Counter()
    honest_acc = Counter()
    for e in events:
        if e["type"] == "submission" and e["accepted"]:
            if e["task"] not in owner:
                owner[e["task"]], lock_round[e["task"]] = e["agent"], e["round"]
            if e["agent"] in live and verdict.get(e["id"]) != "exploit":
                honest_acc[family_of(s, e["agent"])] += 1
        elif (e["type"] == "submit_blocked" and shared and e["agent"] in live and e["task"] in owner
              and owner[e["task"]] != e["agent"]):
            late[family_of(s, e["agent"])] += 1
    n_fam = Counter(family_of(s, a) for a in live)
    out = {"cell": s["cell"], "shared": shared, "n_solvable": len(solvable),
           "honest_accepted_per_live": sum(honest_acc.values()) / len(live),
           "late_blocks_per_live": sum(late.values()) / len(live),
           "late_blocks_by_family": {f: late[f] / n for f, n in n_fam.items()}}
    if shared:
        taken = [t for t in solvable if t in owner]
        out["solvable_taken"] = len(taken) / len(solvable)
        out["honest_agent_share"] = sum(owner[t] == next(a for a, r in roles.items() if r == "honest")
                                        for t in taken) / len(solvable)
        out["exhausted_round"] = max(lock_round[t] for t in solvable) if len(taken) == len(solvable) else None
    return out


def removal_stats(run: Path) -> dict:
    """Removals in one audit swarm, by role, and whether seed A was gone before its round 7 reminder."""
    s = _summary(run)
    removals = [e for e in outcomes.load_events(run) if e["type"] == "removal"]
    seed_a = [e["round"] for e in removals if e["target_role"] == "seed_a"]
    return {"cell": s["cell"], "seeds": sum(e["target_role"].startswith("seed") for e in removals),
            "live": sum(e["target_role"] == "live" for e in removals),
            "live_by_family": dict(Counter(family_of(s, e["target"]) for e in removals if e["target_role"] == "live")),
            "honest": sum(e["target_role"] == "honest" for e in removals),
            "seed_a_gone_before_reminder": bool(seed_a) and min(seed_a) < 7}


def first_adoption(run: Path) -> dict[str, int | None]:
    """Round of each gpt-oss live agent's first accepted exploit, or None."""
    s = _summary(run)
    joiners = [a for a, r in s["roles"].items() if r == "live" and family_of(s, a) == "gpt-oss"]
    events = outcomes.load_events(run)
    verdict = {e["id"]: e["label"] for e in events if e["type"] == "verdict"}
    first: dict[str, int | None] = {a: None for a in joiners}
    for e in events:
        if e["type"] == "submission" and e["accepted"] and e["agent"] in first and verdict.get(e["id"]) == "exploit":
            if first[e["agent"]] is None:
                first[e["agent"]] = e["round"]
    return first


def first_board_callout(run: Path, labels_name: str = "judged_v2.jsonl") -> int | None:
    """Round of the first board post by a GLM agent that the judge labels call_out."""
    s = _summary(run)
    labels = {}
    path = run / labels_name
    if path.exists():
        labels = {r["seq"]: r["label"] for r in map(json.loads, path.read_text().splitlines())}
    rounds = [e["round"] for e in outcomes.load_events(run)
              if e["type"] == "post" and s["roles"].get(e["agent"]) == "live" and family_of(s, e["agent"]) == "glm"
              and labels.get(e["seq"]) == "call_out"]
    return min(rounds) if rounds else None


def timeline(runs: list[Path]) -> dict:
    """Cumulative adoption by round and late adoption among agents still clean at the end of round 3."""
    firsts = [first_adoption(r) for r in runs]
    agents = [f for d in firsts for f in d.values()]
    cumulative = [sum(f is not None and f <= t for f in agents) / len(agents) for t in range(1, ROUNDS + 1)]
    clean = [f for f in agents if f is None or f > 3]
    return {"swarms": len(runs), "agents": len(agents), "cumulative": cumulative,
            "by_round_3": cumulative[2], "later_among_clean_at_3": sum(f is not None for f in clean) / len(clean)}


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def runs_of(block: Path, keep) -> list[Path]:
    return [r for r in sorted(block.iterdir()) if (r / "summary.json").exists() and keep(_summary(r)["cell"])]


def figure(result: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    from analysis.figures import BLUE, INK, MUTED, ORANGE
    names = {"R0P1C1H1E0": "No rivalry (R0 P1 C1 H1)", "R1P1C1H1E0": "Collective corner (R1 P1 C1 H1)"}
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.0), sharey=True)
    x = list(range(1, ROUNDS + 1))
    for ax, cell in zip(axes, TIMELINE_CELLS):
        t = result["timeline"][cell]
        ax.plot(x, t["B"]["cumulative"], color=BLUE, lw=2, label="gpt-oss only (block B)")
        ax.plot(x, t["A"]["cumulative"], color=ORANGE, lw=2, label="With 5 GLM agents (block A)")
        if t["median_first_callout"] is not None:
            ax.axvline(t["median_first_callout"], color=MUTED, lw=1, ls=":")
        ax.set_title(names[cell], fontsize=8.5, color=INK, loc="left")
        ax.set_xticks([1, 3, 5, 10, 15])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Round")
    axes[0].set_ylabel("gpt-oss agents who cheated")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.42), ncol=2, fontsize=8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("block_a", type=Path)
    p.add_argument("block_b", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "notes/mechanisms.json")
    p.add_argument("--figure", type=Path, default=ROOT / "paper/figures/timeline.pdf")
    a = p.parse_args()
    result: dict = {"note": "Exploratory, from the logs only. The timeline compares blocks that were not randomized "
                            "against each other."}

    rows = [pool_stats(r) for r in runs_of(a.block_a, lambda c: c.endswith("E0"))]
    pool = {}
    for level in (False, True):
        sub = [r for r in rows if r["shared"] == level]
        entry = {"swarms": len(sub), "honest_accepted_per_live": mean([r["honest_accepted_per_live"] for r in sub]),
                 "late_blocks_per_live": mean([r["late_blocks_per_live"] for r in sub]),
                 "late_blocks_gpt_oss": mean([r["late_blocks_by_family"].get("gpt-oss") for r in sub]),
                 "late_blocks_glm": mean([r["late_blocks_by_family"].get("glm") for r in sub])}
        if level:
            ex = [r["exhausted_round"] for r in sub]
            entry.update({"solvable_taken": mean([r["solvable_taken"] for r in sub]),
                          "honest_agent_share": mean([r["honest_agent_share"] for r in sub]),
                          "exhausted_share": sum(x is not None for x in ex) / len(ex),
                          "exhausted_round_median": statistics.median([x for x in ex if x is not None])
                          if any(x is not None for x in ex) else None})
        pool["R1" if level else "R0"] = entry
    result["pool"] = pool

    rem = [removal_stats(r) for r in runs_of(a.block_a, lambda c: c.endswith("E1"))]
    result["removals"] = {"swarms": len(rem), "seeds": sum(r["seeds"] for r in rem), "live": sum(r["live"] for r in rem),
                          "live_by_family": dict(sum((Counter(r["live_by_family"]) for r in rem), Counter())),
                          "honest": sum(r["honest"] for r in rem),
                          "seed_a_gone_before_reminder": sum(r["seed_a_gone_before_reminder"] for r in rem) / len(rem)}

    result["timeline"] = {}
    for cell in TIMELINE_CELLS:
        ra, rb = runs_of(a.block_a, lambda c: c == cell), runs_of(a.block_b, lambda c: c == cell)
        calls = [first_board_callout(r) for r in ra]
        seen = [c for c in calls if c is not None]
        result["timeline"][cell] = {
            "A": timeline(ra), "B": timeline(rb),
            "callout_swarms": len(seen) / len(calls),
            "median_first_callout": statistics.median(seen) if seen else None,
            "callout_by_round_3": sum(c <= 3 for c in seen) / len(calls),
            "callout_by_round_4": sum(c <= 4 for c in seen) / len(calls)}
    a.out.write_text(json.dumps(result, indent=1, default=float))
    figure(result, a.figure)
    print(json.dumps({k: v for k, v in result.items() if k != "timeline"}, indent=1, default=float))
    for cell, t in result["timeline"].items():
        print(cell, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items() if k not in ("A", "B")},
              "A by r3", round(t["A"]["by_round_3"], 3), "B by r3", round(t["B"]["by_round_3"], 3),
              "A later", round(t["A"]["later_among_clean_at_3"], 3), "B later", round(t["B"]["later_among_clean_at_3"], 3))
    print(f"wrote {a.out} and {a.figure}")


if __name__ == "__main__":
    main()
