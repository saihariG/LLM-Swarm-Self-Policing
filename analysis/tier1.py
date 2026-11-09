#!/usr/bin/env python
"""Tier 1 exploratory arms (PREREG_AMENDMENTS.md, 2026-11-08 about 6:30am ET). Comparisons fixed before the runs.

random     ext_random (gpt-oss only, random audits of 2 agents per round) against block B without audits (E0)
           and with report-triggered audits (E1), in R0 P1 C1 H1 and R1 P1 C1 H1.
reporter   ext_reporter (9 gpt-oss plus one scripted reporter, R1 P1 C1 H1 with audits) against the same cell
           with no police (block B), one GLM agent (ext_dose1) and five (block A).
incident   main_C_incident (MiniMax M3 plus Muse Spark, R0 P0 C0 H0) against block C's R1 P1 C1 H1 without
           audits: adoption among MiniMax agents and policing among Muse agents (revised rubric).
Every comparison is across blocks that were not randomized against each other.
Usage: python analysis/tier1.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from analysis import outcomes  # noqa: E402
from analysis.cross_block import joiner_exploit_share  # noqa: E402
from analysis.effects import two_group  # noqa: E402
from analysis.report import add_primary  # noqa: E402

RUNS = ROOT / "runs"
OUTS = ["Y1", "exploit_share", "joiner_share", "exploit_credit_left"]


def table(block: str, joiner: str = "gpt-oss", police: str = "glm") -> pd.DataFrame:
    df = add_primary(outcomes.block_table(RUNS / block), joiner, police)
    df["block"] = block
    df["joiner_share"] = [joiner_exploit_share(RUNS / block / r, joiner) for r in df["run"]]
    roles = []
    for r in df["run"]:
        ev = outcomes.load_events(RUNS / block / r)
        rem = [e for e in ev if e["type"] == "removal"]
        roles.append({"seeds_removed": sum(e["target_role"].startswith("seed") for e in rem),
                      "live_removed": sum(e["target_role"] == "live" for e in rem),
                      "honest_removed": sum(e["target_role"] == "honest" for e in rem),
                      "audits": sum(e["type"] == "audit" for e in ev),
                      "seed_a_gone_before_reminder": any(e["target_role"] == "seed_a" and e["round"] < 7 for e in rem)})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(roles)], axis=1)


def summary(df: pd.DataFrame) -> dict:
    cols = OUTS + ["Y3", "seeds_removed", "live_removed", "honest_removed", "audits", "seed_a_gone_before_reminder"]
    return {"swarms": int(len(df)), **{c: float(df[c].mean()) for c in cols if c in df}}


def compare(a: pd.DataFrame, b: pd.DataFrame, outcome: str, n_perm: int) -> dict:
    both = pd.concat([a.assign(cell="a"), b.assign(cell="b")], ignore_index=True)
    res = two_group(both, outcome, "a", "b", n_perm=n_perm)
    return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in res.items() if k not in ("a", "b")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--out", type=Path, default=ROOT / "notes/tier1.json")
    a = p.parse_args()
    out: dict = {"note": "Exploratory tier 1 arms, across blocks that were not randomized against each other."}

    if (RUNS / "ext_random").exists():
        rnd, b = table("ext_random"), table("main_B")
        out["random"] = {}
        for cell in ("R0P1C1H1E0", "R1P1C1H1E0"):
            e1 = cell[:-1] + "1"
            ra, b0, b1 = rnd[rnd["cell"] == cell], b[b["cell"] == cell], b[b["cell"] == e1]
            out["random"][cell] = {"random_audits": summary(ra), "block_b_no_audits": summary(b0),
                                   "block_b_report_audits": summary(b1),
                                   "vs_no_audits": {o: compare(ra, b0, o, a.n_perm) for o in OUTS},
                                   "vs_report_audits": {o: compare(ra, b1, o, a.n_perm) for o in OUTS}}

    if (RUNS / "ext_reporter").exists():
        cell = "R1P1C1H1E1"
        rep = table("ext_reporter")
        groups = {"reporter": rep, "no_police": table("main_B"), "one_glm": table("ext_dose1"), "five_glm": table("main_A")}
        groups = {k: v[v["cell"] == cell] for k, v in groups.items()}
        out["reporter"] = {"summary": {k: summary(v) for k, v in groups.items()},
                           "vs_no_police": {o: compare(groups["reporter"], groups["no_police"], o, a.n_perm) for o in OUTS},
                           "vs_one_glm": {o: compare(groups["reporter"], groups["one_glm"], o, a.n_perm) for o in OUTS},
                           "vs_five_glm": {o: compare(groups["reporter"], groups["five_glm"], o, a.n_perm) for o in OUTS}}

    if (RUNS / "main_C_incident").exists():
        inc = table("main_C_incident", "minimax", "muse")
        col = table("main_C", "minimax", "muse")
        col = col[col["cell"] == "R1P1C1H1E0"]
        out["incident"] = {"incident": summary(inc), "collective": summary(col),
                           "collective_minus_incident": {o: compare(col, inc, o, a.n_perm) for o in ("Y1", "Y3", "Y4", "exploit_share")}}

    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 3))[:6000])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
