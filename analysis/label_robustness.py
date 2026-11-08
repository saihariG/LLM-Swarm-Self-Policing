#!/usr/bin/env python
"""The pre-registered tests, rerun with each set of call_out labels.

No human check of the judge passed the bar (PREREG_AMENDMENTS.md, 6:40pm ET on 2026-11-07),
so each hypothesis that uses Y3 is run with three label sources:
  v2       the judge with the revised rubric, the primary labels
  v1       the judge with the first rubric
  keyword  outcomes.heuristic_label, a keyword rule written before any run
A policing result counts as robust only if all three agree on its sign and on whether its
Holm-adjusted p is below 0.05. Tests that do not use labels come out the same under all three.

A fourth source, added after the main outcomes were read (PREREG_AMENDMENTS.md, 2026-11-08), needs no LLM:
  names    a message counts as a call_out if it names another agent that already had an accepted
           exploit at that point in the log
It is exploratory. It does not change the pre-registered robustness rule, and it is reported next to it.
A name is not always an accusation, and the rivalry leaderboard and the public library put names in
front of agents, so R and P are confounded on this source.
Usage: python analysis/label_robustness.py runs/main_A --joiners-only runs/main_B [--sources names]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from analysis import outcomes  # noqa: E402
from analysis.report import add_primary, analyse  # noqa: E402

SOURCES = {"v2": "judged_v2.jsonl", "v1": "judged_v1.jsonl", "keyword": None, "names": "names"}
PREREGISTERED = ("v2", "v1", "keyword")
POLICING = ("H1 P on policing", "H2b R on policing", "H3a H on policing", "H3b C on policing",
            "H6b corners on policing")


def name_labels(run_dir: Path) -> dict[int, dict]:
    """Labels with no LLM: call_out if a message names another agent that already had an accepted exploit."""
    events = outcomes.load_events(run_dir)
    verdict = {e["id"]: e["label"] for e in events if e["type"] == "verdict"}
    cheaters: set[str] = set()
    out: dict[int, dict] = {}
    for e in events:  # the log is in time order
        if e["type"] == "submission" and e["accepted"] and verdict.get(e["id"]) == "exploit":
            cheaters.add(e["agent"])
        elif e["type"] in ("post", "dm", "feedback"):
            hit = sorted((set(outcomes.NAME_RE.findall(e.get("text") or "")) - {e["agent"]}) & cheaters)
            out[e["seq"]] = {"seq": e["seq"], "label": "call_out" if hit else "other", "accused": hit}
    return out


def loader(file_name: str | None):
    """A stand-in for outcomes.load_judged that reads one label file, none for the keyword rule,
    or builds the labels that need no LLM."""
    if file_name == "names":
        return name_labels

    def load(run_dir: Path) -> dict[int, dict]:
        if file_name is None:
            return {}
        path = run_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing, so this source would fall back to keywords")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return {r["seq"]: r for r in rows}
    return load


def tables(block: Path, joiners: Path | None, file_name: str | None):
    original = outcomes.load_judged
    outcomes.load_judged = loader(file_name)
    try:
        df = outcomes.block_table(block)
        pure = outcomes.block_table(joiners) if joiners else None
    finally:
        outcomes.load_judged = original
    return df, pure


def verdict(rows: dict[str, dict]) -> dict[str, dict]:
    """Per policing hypothesis: the sign and the Holm call under each source, and whether they agree.

    robust uses the pre-registered sources only. names_agrees says whether the source that needs no LLM
    has the same sign and Holm call as the primary labels (v2)."""
    out = {}
    for key in POLICING:
        got = {src: r["confirmatory"].get(key) for src, r in rows.items()}
        if any(v is None for v in got.values()):
            continue
        est = {src: float(v.get("estimate", v.get("diff"))) for src, v in got.items()}
        sig = {src: bool(float(v["p_holm"]) < 0.05) for src, v in got.items()}
        sign = {src: int(np.sign(e)) for src, e in est.items()}
        pre = [s for s in PREREGISTERED if s in got]
        out[key] = {"estimate": est, "p_holm": {s: float(v["p_holm"]) for s, v in got.items()}, "significant": sig,
                    "robust": len({sign[s] for s in pre}) == 1 and len({sig[s] for s in pre}) == 1}
        if "names" in got and "v2" in got:
            out[key]["names_agrees"] = sign["names"] == sign["v2"] and sig["names"] == sig["v2"]
    return out


def one_source(job: tuple) -> tuple[str, dict, dict]:
    """Build the tables and run the pre-registered tests for one label source (runs in its own process)."""
    src, block, joiners, per_cell, n_perm, n_boot = job
    df, pure = tables(block, joiners, SOURCES[src])
    result = analyse(df, per_cell, n_perm, n_boot, joiners_only=pure)
    means = add_primary(df).groupby("cell")[["Y3", "Y4", "callout_rate"]].mean().round(4).to_dict(orient="index")
    return src, result, means


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    p = argparse.ArgumentParser()
    p.add_argument("block", type=Path)
    p.add_argument("--joiners-only", type=Path)
    p.add_argument("--per-cell", type=int, default=6)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=list(SOURCES),
                   help="run only these sources and keep the others already saved in --out")
    p.add_argument("--out", type=Path, default=ROOT / "notes/label_robustness.json")
    a = p.parse_args()
    jobs = [(src, a.block, a.joiners_only, a.per_cell, a.n_perm, a.n_boot) for src in a.sources]
    with ProcessPoolExecutor(len(jobs)) as pool:
        done = list(pool.map(one_source, jobs))
    old = json.loads(a.out.read_text()) if a.out.exists() else {}
    result = {"by_source": dict(old.get("by_source", {})), "cell_means": dict(old.get("cell_means", {}))}
    for src, res, means in done:
        result["by_source"][src] = res["confirmatory"]
        result["cell_means"][src] = means
    a.out.write_text(json.dumps(result, indent=1, default=float))  # saved before the comparison
    rows = {s: {"confirmatory": c} for s, c in result["by_source"].items()}
    result["robustness"] = verdict(rows)
    a.out.write_text(json.dumps(result, indent=1, default=float))
    for key, v in result["robustness"].items():
        cells = "  ".join(f"{s} {v['estimate'][s]:+.3f} (holm {v['p_holm'][s]:.3f})" for s in v["estimate"])
        print(f"{key:26s} {cells}  robust={v['robust']}  names_agrees={v.get('names_agrees')}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
