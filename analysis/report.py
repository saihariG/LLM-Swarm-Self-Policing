#!/usr/bin/env python
"""The pre-registered analysis for one block. Usage: python analysis/report.py runs/<block> --per-cell 6"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.effects import balanced, bootstrap_ci, holm, interaction, main_effect, permutation_p, two_group  # noqa: E402
from analysis.outcomes import block_table  # noqa: E402

HF, DM = "R0P0C0H0E0", "R1P1C1H1E0"
ENF_CELLS = ["R0P1C1H1E0", "R1P1C1H1E0", "R0P1C1H1E1", "R1P1C1H1E1"]


def factor_row(df: pd.DataFrame, outcome: str, factor: str, n_perm: int, n_boot: int) -> dict:
    lo, hi = bootstrap_ci(df, lambda d: main_effect(d, outcome, factor), ["R", "P", "C", "H"], n_boot=n_boot)
    return {"term": factor, "outcome": outcome, "estimate": main_effect(df, outcome, factor), "lo": lo, "hi": hi,
            "p": permutation_p(df, outcome, factor, n_perm=n_perm), "n": len(df)}


def enforcement_effect(df: pd.DataFrame, outcome: str, n_perm: int, n_boot: int, seed: int = 9) -> dict:
    """E1 minus E0 at P1 C1 H1, averaged over R, with E shuffled within each R arm."""
    rng = np.random.default_rng(seed)

    def stat(d: pd.DataFrame) -> float:
        parts = [g[g["E"] == 1][outcome].mean() - g[g["E"] == 0][outcome].mean() for _, g in d.groupby("R")]
        return float(np.mean(parts))

    observed = stat(df)
    hits = 0
    for _ in range(n_perm):
        work = df.copy()
        work["E"] = work.groupby("R")["E"].transform(lambda s: rng.permutation(s.to_numpy()))
        hits += abs(stat(work)) >= abs(observed) - 1e-12
    lo, hi = bootstrap_ci(df, stat, ["R", "E"], n_boot=n_boot)
    return {"term": "E", "outcome": outcome, "estimate": observed, "lo": lo, "hi": hi, "p": (hits + 1) / (n_perm + 1), "n": len(df)}


def add_primary(df: pd.DataFrame, joiner: str = "gpt-oss", police: str = "glm") -> pd.DataFrame:
    """Y1 is adoption among the joiner family, Y3 is policing among the police family.

    In a single-model block both fall back to the whole swarm.
    """
    df = df.copy()
    df["Y1"], df["Y3"], df["Y4"] = df["adoption"], df["police_rate"], df["report_rate"]
    for i in (0, 1):
        name = f"m{i}_name"
        if name not in df:
            continue
        is_joiner = df[name].fillna("").str.contains(joiner, case=False)
        is_police = df[name].fillna("").str.contains(police, case=False)
        df.loc[is_joiner, "Y1"] = df.loc[is_joiner, f"m{i}_adoption"]
        df.loc[is_police, "Y3"] = df.loc[is_police, f"m{i}_police"]
        df.loc[is_police, "Y4"] = df.loc[is_police, f"m{i}_report"]
    return df


def analyse(df: pd.DataFrame, per_cell: int, n_perm: int = 10000, n_boot: int = 5000,
            joiners_only: pd.DataFrame | None = None, joiner: str = "gpt-oss", police: str = "glm") -> dict:
    df = add_primary(df, joiner, police)
    out: dict = {"n_swarms": len(df), "cell_counts": df["cell"].value_counts().sort_index().to_dict()}
    rows: dict[str, dict] = {}
    bal = balanced(df, per_cell)
    if bal.groupby(["R", "P", "C", "H"]).ngroups == 16:
        rows["H1 P on policing"] = factor_row(bal, "Y3", "P", n_perm, n_boot)
        rows["H2a R on adoption"] = factor_row(bal, "Y1", "R", n_perm, n_boot)
        rows["H2b R on policing"] = factor_row(bal, "Y3", "R", n_perm, n_boot)
        rows["H3a H on policing"] = factor_row(bal, "Y3", "H", n_perm, n_boot)
        rows["H3b C on policing"] = factor_row(bal, "Y3", "C", n_perm, n_boot)
        out["secondary_adoption"] = {f: factor_row(bal, "Y1", f, n_perm, n_boot) for f in ("P", "C", "H")}
    enf = df[df["cell"].isin(ENF_CELLS)]
    if enf["cell"].nunique() == 4:
        rows["H4 E on exploit share"] = enforcement_effect(enf, "exploit_share", n_perm, n_boot)
        rows["H5 R x E on exploit share"] = interaction(enf, "exploit_share", "R", "E", n_perm=n_perm)
    if {HF, DM} <= set(df["cell"]):
        rows["H6a corners on adoption"] = {"term": "DM minus HF", **two_group(df, "Y1", DM, HF, n_perm=n_perm)}
        rows["H6b corners on policing"] = {"term": "DM minus HF", **two_group(df, "Y3", DM, HF, n_perm=n_perm)}
    if joiners_only is not None:
        pure = joiners_only[joiners_only["cell"].isin(ENF_CELLS)].assign(M=0)
        both = pd.concat([enf.assign(M=1), pure], ignore_index=True)
        if both.groupby(["E", "M"]).ngroups == 4:
            rows["H7 E x police on exploit share"] = interaction(both, "exploit_share", "E", "M", n_perm=n_perm, covariates=("R",))
    adjusted = holm({k: v["p"] for k, v in rows.items()})
    for key, row in rows.items():
        row["p_holm"] = adjusted[key]
    out["confirmatory"] = rows
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("block", type=Path)
    p.add_argument("--per-cell", type=int, default=6)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--joiners-only", type=Path, help="block of joiner-only swarms, for H7")
    p.add_argument("--joiner", default="gpt-oss", help="name fragment of the joiner model")
    p.add_argument("--police", default="glm", help="name fragment of the police model")
    a = p.parse_args()
    df = block_table(a.block)
    df.to_csv(a.block / "outcomes.csv", index=False)
    pure = block_table(a.joiners_only) if a.joiners_only else None
    result = analyse(df, a.per_cell, a.n_perm, a.n_boot, joiners_only=pure, joiner=a.joiner, police=a.police)
    (a.block / "confirmatory.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"swarms {result['n_swarms']}")
    for key, row in result["confirmatory"].items():
        est = row.get("estimate", row.get("diff"))
        print(f"{key:28s} {est:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}]  p={row['p']:.4f}  holm={row['p_holm']:.4f}")


if __name__ == "__main__":
    main()
