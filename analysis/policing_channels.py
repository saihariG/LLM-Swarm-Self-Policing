#!/usr/bin/env python
"""Exploratory: where GLM agents police, in public or in private. Block A's balanced factorial.

public   share of GLM agents with at least one board post labelled call_out
report   share of GLM agents with at least one feedback message labelled call_out (this is Y4)
Each is computed under the revised rubric (v2, the primary labels) and under the source that needs
no LLM (names, see analysis/label_robustness.py). Main effects of R, P, C and H use the
pre-registered estimators (factor_row in analysis/report.py).
PREREG.md asks that H3a show how much of policing the route adds. This script was written after
the main outcomes were read, so it is exploratory (PREREG_AMENDMENTS.md, 2026-11-08).
Usage: python analysis/policing_channels.py runs/main_A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import outcomes  # noqa: E402
from analysis.effects import balanced  # noqa: E402
from analysis.label_robustness import SOURCES, loader  # noqa: E402
from analysis.report import factor_row  # noqa: E402

USED = ("v2", "names")
FACTORS = ("R", "P", "C", "H")


def glm_channels(run: Path, labels: dict[int, dict], police: str = "glm") -> dict:
    """Share of the police family's live agents who called out in a board post, and in feedback."""
    summary = json.loads((run / "summary.json").read_text())
    models = summary.get("agent_models") or {}
    group = {a for a, r in summary["roles"].items() if r == "live" and police in models.get(a, "").lower()}
    if not group:
        return {"public": float("nan"), "report": float("nan")}
    public, report = set(), set()
    for e in outcomes.load_events(run):
        if e["type"] not in ("post", "feedback") or e["agent"] not in group:
            continue
        j = labels.get(e["seq"])
        if j and j["label"] == "call_out":
            (public if e["type"] == "post" else report).add(e["agent"])
    return {"public": len(public) / len(group), "report": len(report) / len(group)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("block", type=Path)
    p.add_argument("--per-cell", type=int, default=6)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--out", type=Path, default=ROOT / "notes/policing_channels.json")
    a = p.parse_args()
    base = outcomes.block_table(a.block)[["run", "cell", "R", "P", "C", "H", "E", "replicate"]].copy()
    cols = []
    for src in USED:
        load = loader(SOURCES[src])
        vals = {r: glm_channels(a.block / r, load(a.block / r)) for r in base["run"]}
        for kind in ("public", "report"):
            base[f"{kind}_{src}"] = [vals[r][kind] for r in base["run"]]
            cols.append(f"{kind}_{src}")
    bal = balanced(base, a.per_cell)
    out = {"note": "Exploratory, outside the Holm family. Block A, GLM agents, balanced factorial (E0, replicate below "
                   f"{a.per_cell}). public: board post labelled call_out. report: feedback labelled call_out.",
           "n_swarms": len(bal), "effects": {}, "cell_means": {}}
    for col in cols:
        out["effects"][col] = {f: factor_row(bal, col, f, a.n_perm, a.n_boot) for f in FACTORS}
        row = "  ".join(f"{f} {v['estimate']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] p={v['p']:.4f}"
                        for f, v in out["effects"][col].items())
        print(f"{col:15s} {row}")
    out["cell_means"] = base.groupby("cell")[cols].mean().round(4).to_dict(orient="index")
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
