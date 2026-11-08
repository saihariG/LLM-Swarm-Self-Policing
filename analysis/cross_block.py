#!/usr/bin/env python
"""Exploratory comparisons across blocks. The blocks were not randomized against each other, so none of
these is a randomized test (PREREG_AMENDMENTS.md, 2026-11-08). Blocks A and B ran in the same window on
the same server.

adoption   gpt-oss adoption (Y1) with five GLM agents present (block A) against swarms of ten gpt-oss
           agents (block B), in every cell both blocks ran. A minus B, with a bootstrap interval and a
           permutation p that shuffles the block label within the cell.
block_b    block B without and with enforcement: adoption, exploit credit left and removals per swarm.
h7_joiner  H7 again with exploit share among gpt-oss submissions only, so both blocks share a denominator.
regex      the trick's signature (def __eq__) in accepted live submissions, a baseline for the monitor arm.
Usage: python analysis/cross_block.py runs/main_A runs/main_B runs/main_C
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from analysis import outcomes  # noqa: E402
from analysis.effects import interaction, two_group  # noqa: E402
from analysis.report import ENF_CELLS, add_primary  # noqa: E402

TRICK = re.compile(r"def __eq__")


def joiner_exploit_share(run: Path, joiner: str = "gpt-oss") -> float:
    """Exploit share among the joiner family's accepted submissions (all live agents in a one-model swarm)."""
    summary = json.loads((run / "summary.json").read_text())
    models = summary.get("agent_models") or {}
    live = {a for a, r in summary["roles"].items() if r == "live"}
    group = {a for a in live if joiner in models.get(a, "")} if models else live
    events = outcomes.load_events(run)
    verdict = {e["id"]: e["label"] for e in events if e["type"] == "verdict"}
    acc = [e for e in events if e["type"] == "submission" and e["accepted"] and e["agent"] in group]
    return sum(verdict.get(e["id"]) == "exploit" for e in acc) / len(acc) if acc else 0.0


def signature_counts(block: Path) -> dict:
    """Accepted live submissions by verifier label, and how many carry the trick's signature."""
    counts = {"exploit": [0, 0], "honest": [0, 0]}  # [with signature, total]
    for run in sorted(block.iterdir()):
        if not (run / "summary.json").exists():
            continue
        roles = json.loads((run / "summary.json").read_text())["roles"]
        events = outcomes.load_events(run)
        verdict = {e["id"]: e["label"] for e in events if e["type"] == "verdict"}
        for e in events:
            if e["type"] == "submission" and e["accepted"] and roles.get(e["agent"]) == "live":
                kind = "exploit" if verdict.get(e["id"]) == "exploit" else "honest"
                counts[kind][1] += 1
                counts[kind][0] += bool(TRICK.search(e.get("code") or ""))
    return {k: {"with_signature": w, "total": t, "share": w / t if t else None} for k, (w, t) in counts.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("block_a", type=Path)
    p.add_argument("block_b", type=Path)
    p.add_argument("more", type=Path, nargs="*", help="other blocks for the regex baseline")
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--out", type=Path, default=ROOT / "notes/cross_block.json")
    a = p.parse_args()
    dfa, dfb = add_primary(outcomes.block_table(a.block_a)), add_primary(outcomes.block_table(a.block_b))
    out: dict = {"note": "Exploratory, across blocks that were not randomized against each other."}

    out["adoption"] = []
    for cell in sorted(set(dfa["cell"]) & set(dfb["cell"])):
        both = pd.concat([dfa[dfa["cell"] == cell].assign(cell="A"), dfb[dfb["cell"] == cell].assign(cell="B")])
        res = two_group(both, "Y1", "A", "B", n_perm=a.n_perm)
        out["adoption"].append({"cell": cell, **res})
        print(f"adoption {cell}: A {res['mean_a']:.3f} (n={res['n_a']})  B {res['mean_b']:.3f} (n={res['n_b']})  "
              f"diff {res['diff']:+.3f} [{res['lo']:+.3f}, {res['hi']:+.3f}] p={res['p']:.4f}")

    out["block_b"] = (dfb.groupby("cell")[["Y1", "exploit_credit_left", "removals"]].mean().round(4)
                      .to_dict(orient="index"))
    for cell, v in out["block_b"].items():
        print(f"block B {cell}: adoption {v['Y1']:.3f}  credit left {v['exploit_credit_left']:.2f}  "
              f"removals {v['removals']:.3f}")

    enf_a = dfa[dfa["cell"].isin(ENF_CELLS)].copy()
    enf_b = dfb[dfb["cell"].isin(ENF_CELLS)].copy()
    enf_a["joiner_share"] = [joiner_exploit_share(a.block_a / r) for r in enf_a["run"]]
    enf_b["joiner_share"] = [joiner_exploit_share(a.block_b / r) for r in enf_b["run"]]
    both = pd.concat([enf_a.assign(M=1), enf_b.assign(M=0)], ignore_index=True)
    h7 = interaction(both, "joiner_share", "E", "M", n_perm=a.n_perm, covariates=("R",))
    means = both.groupby(["M", "cell"])["joiner_share"].mean().round(4)
    out["h7_joiner"] = {**h7, "cell_means": {f"{'A' if m else 'B'} {c}": v for (m, c), v in means.items()}}
    print(f"H7 on gpt-oss-only exploit share: {h7['estimate']:+.3f} [{h7['lo']:+.3f}, {h7['hi']:+.3f}] p={h7['p']:.4f}")

    out["regex"] = {b.name: signature_counts(b) for b in [a.block_a, a.block_b, *a.more]}
    for name, v in out["regex"].items():
        print(f"regex {name}: exploits {v['exploit']['with_signature']}/{v['exploit']['total']}, "
              f"honest {v['honest']['with_signature']}/{v['honest']['total']}")
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
