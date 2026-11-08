"""Pre-registered estimators: effects with stratified bootstrap and permutation tests, Holm across a family."""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTORS = ["R", "P", "C", "H"]


def cell_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(["all"] * len(df), index=df.index)
    return df[cols].astype(str).agg("".join, axis=1)


def balanced(df: pd.DataFrame, per_cell: int) -> pd.DataFrame:
    """Factorial subset: E0 swarms with replicate below per_cell, fixed by the schedule."""
    sub = df[(df["E"] == 0) & (df["replicate"] < per_cell)]
    return sub


def main_effect(df: pd.DataFrame, outcome: str, factor: str) -> float:
    """Level 1 minus level 0, weighting every combination of the other factors equally."""
    others = [f for f in FACTORS if f != factor]
    diffs = []
    for _, g in df.groupby(others):
        a, b = g[g[factor] == 1][outcome], g[g[factor] == 0][outcome]
        if len(a) and len(b):
            diffs.append(a.mean() - b.mean())
    return float(np.mean(diffs)) if diffs else float("nan")


def permutation_p(df: pd.DataFrame, outcome: str, factor: str, n_perm: int = 5000, seed: int = 1) -> float:
    """Shuffle the factor within strata of the other factors, as the randomization did."""
    rng = np.random.default_rng(seed)
    others = [f for f in FACTORS if f != factor]
    work = df[[outcome, factor] + others].copy()
    observed = abs(main_effect(work, outcome, factor))
    groups = [g.index.to_numpy() for _, g in work.groupby(others)]
    values = work[factor].to_numpy().copy()
    pos = {idx: i for i, idx in enumerate(work.index)}
    locs = [np.array([pos[i] for i in g]) for g in groups]
    y = work[outcome].to_numpy()
    hits = 0
    for _ in range(n_perm):
        perm = values.copy()
        total, count = 0.0, 0
        for loc in locs:
            shuffled = rng.permutation(values[loc])
            perm[loc] = shuffled
            ones, zeros = y[loc][shuffled == 1], y[loc][shuffled == 0]
            if len(ones) and len(zeros):
                total += ones.mean() - zeros.mean()
                count += 1
        if count and abs(total / count) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def bootstrap_ci(df: pd.DataFrame, stat, strata: list[str], n_boot: int = 2000, seed: int = 2,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval, resampling swarms within strata."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(strata)]
    values = []
    for _ in range(n_boot):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        values.append(stat(pd.concat(parts)))
    lo, hi = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def two_group(df: pd.DataFrame, outcome: str, a: str, b: str, n_perm: int = 10000, seed: int = 3) -> dict:
    """Difference in means between two cells with a permutation test. a minus b."""
    x = df[df["cell"] == a][outcome].to_numpy(dtype=float)
    y = df[df["cell"] == b][outcome].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    both = np.concatenate([x, y])
    observed = x.mean() - y.mean()
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(both)
        if abs(perm[: len(x)].mean() - perm[len(x):].mean()) >= abs(observed) - 1e-12:
            hits += 1
    boots = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"a": a, "b": b, "outcome": outcome, "mean_a": x.mean(), "mean_b": y.mean(), "n_a": len(x),
            "n_b": len(y), "diff": observed, "lo": lo, "hi": hi, "p": (hits + 1) / (n_perm + 1)}


def interaction(df: pd.DataFrame, outcome: str, f1: str = "R", f2: str = "E", n_perm: int = 5000, seed: int = 4,
                covariates: tuple = ()) -> dict:
    """2x2 interaction with a Freedman-Lane test: permute residuals of the main-effects model."""
    rng = np.random.default_rng(seed)
    y = df[outcome].to_numpy(dtype=float)
    a = df[f1].to_numpy(dtype=float)
    b = df[f2].to_numpy(dtype=float)
    extra = [df[c].to_numpy(dtype=float) for c in covariates]
    reduced = np.column_stack([np.ones(len(y)), a, b, *extra])
    full = np.column_stack([reduced, a * b])

    def coef(target: np.ndarray) -> float:
        return float(np.linalg.lstsq(full, target, rcond=None)[0][-1])

    observed = coef(y)
    fitted = reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
    resid = y - fitted
    hits = sum(abs(coef(fitted + rng.permutation(resid))) >= abs(observed) - 1e-12 for _ in range(n_perm))
    lo, hi = bootstrap_ci(df, lambda d: float(np.linalg.lstsq(
        np.column_stack([np.ones(len(d)), d[f1], d[f2], *[d[c] for c in covariates], d[f1] * d[f2]]).astype(float),
        d[outcome].to_numpy(dtype=float), rcond=None)[0][-1]), [f1, f2])
    return {"outcome": outcome, "term": f"{f1}x{f2}", "estimate": observed, "lo": lo, "hi": hi,
            "p": (hits + 1) / (n_perm + 1), "n": len(y)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted
