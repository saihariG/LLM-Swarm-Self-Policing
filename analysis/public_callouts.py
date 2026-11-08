#!/usr/bin/env python
"""H6b with public call-outs alone, which PREREG.md asks for because reports are impossible in the
Hugging Face corner by construction.

Outcome: share of the police family's agents with at least one board post labelled call_out.
It is computed for each label source in analysis/label_robustness.py.
Usage: python analysis/public_callouts.py runs/main_A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import outcomes  # noqa: E402
from analysis.effects import two_group  # noqa: E402
from analysis.label_robustness import SOURCES, loader  # noqa: E402
from analysis.report import DM, HF  # noqa: E402


def public_share(run: Path, file_name: str | None, police: str = "glm") -> float:
    summary = json.loads((run / "summary.json").read_text())
    models = summary.get("agent_models") or {}
    group = {a for a, r in summary["roles"].items() if r == "live" and police in models.get(a, "").lower()}
    if not group:
        return float("nan")
    judged = loader(file_name)(run)
    callers = set()
    for e in outcomes.load_events(run):
        if e["type"] != "post" or e["agent"] not in group:
            continue
        j = judged.get(e["seq"])
        label = j["label"] if j else outcomes.heuristic_label(e["text"])
        if label == "call_out":
            callers.add(e["agent"])
    return len(callers) / len(group)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("block", type=Path)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--out", type=Path, default=ROOT / "notes/public_callouts.json")
    a = p.parse_args()
    runs = [r for r in sorted(a.block.iterdir()) if (r / "summary.json").exists()
            and json.loads((r / "summary.json").read_text())["cell"] in (HF, DM)]
    out = {}
    for src, name in SOURCES.items():
        df = pd.DataFrame([{"cell": json.loads((r / "summary.json").read_text())["cell"],
                            "public": public_share(r, name)} for r in runs])
        out[src] = two_group(df, "public", DM, HF, n_perm=a.n_perm)
        v = out[src]
        print(f"{src:8s} DM {v['mean_a']:.3f} HF {v['mean_b']:.3f} diff {v['diff']:+.3f} "
              f"[{v['lo']:+.3f}, {v['hi']:+.3f}] p={v['p']:.4f}")
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
