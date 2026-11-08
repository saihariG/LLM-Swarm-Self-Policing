#!/usr/bin/env python
"""Paper figures from a block's logs. Usage: python analysis/figures.py runs/<block> --per-cell 6 --out paper/figures"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.effects import balanced, bootstrap_ci, main_effect  # noqa: E402
from analysis.outcomes import block_table  # noqa: E402
from analysis.report import add_primary  # noqa: E402

HF, DM = "R0P0C0H0E0", "R1P1C1H1E0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FACTOR_NAMES = {"R": "Rivalry", "P": "Public work", "C": "Sanctioned channel", "H": "Route to humans"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def mean_ci(values: np.ndarray, seed: int = 21, n_boot: int = 4000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def corners(df: pd.DataFrame, out: Path) -> None:
    """Dot and interval: the two corners on the pre-registered outcomes, each in its own model family."""
    df = add_primary(df)
    rows = [("Y1", "gpt-oss: joined"), ("Y3", "GLM: policed"), ("Y4", "GLM: reported")]
    fig, ax = plt.subplots(figsize=(3.9, 1.55))
    for offset, cell, color, name in ((0.14, HF, BLUE, "Incident corner"), (-0.14, DM, ORANGE, "Collective corner")):
        sub = df[df["cell"] == cell]
        for i, (col, _) in enumerate(rows):
            m, lo, hi = mean_ci(sub[col].to_numpy(dtype=float))
            y = len(rows) - 1 - i + offset
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.scatter([m], [y], s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=name if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows][::-1])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Share of the family's 5 agents (mean over swarms, 95% interval)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, 1.3), ncol=2, handletextpad=0.2)
    fig.savefig(out / "corners.pdf", bbox_inches="tight")
    plt.close(fig)


def factors(confirmatory: dict, out: Path) -> None:
    """Small multiples of the main effects, read from report.py's output so they match the table."""
    conf, adoption = confirmatory["confirmatory"], confirmatory["secondary_adoption"]
    rows = {"Adoption (gpt-oss)": {"R": conf["H2a R on adoption"], **adoption},
            "Policing (GLM)": {"R": conf["H2b R on policing"], "P": conf["H1 P on policing"],
                               "C": conf["H3b C on policing"], "H": conf["H3a H on policing"]}}
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.0), sharey=True, sharex=True)
    names = list(FACTOR_NAMES)
    for ax, (title, effects) in zip(axes, rows.items()):
        for i, f in enumerate(names):
            e = effects[f]
            y = len(names) - 1 - i
            ax.plot([e["lo"], e["hi"]], [y, y], color=BLUE, lw=2, solid_capstyle="round")
            ax.scatter([e["estimate"]], [y], s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Main effect on the share, 95% interval")
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([FACTOR_NAMES[f] for f in names][::-1])
    fig.savefig(out / "factors.pdf", bbox_inches="tight")
    plt.close(fig)


def enforcement(panels: list[tuple[str, pd.DataFrame]], out: Path) -> None:
    """Rivalry by enforcement on exploit share, one panel per population."""
    cells = {(0, 0): "R0P1C1H1E0", (1, 0): "R1P1C1H1E0", (0, 1): "R0P1C1H1E1", (1, 1): "R1P1C1H1E1"}
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 1.9), sharey=True, squeeze=False)
    for ax, (title, df) in zip(axes[0], panels):
        for e, color, name, dx in ((0, BLUE, "No audits", -0.03), (1, ORANGE, "Audits", 0.03)):
            means = []
            for r in (0, 1):
                sub = df[df["cell"] == cells[(r, e)]]["exploit_share"].to_numpy(dtype=float)
                m, lo, hi = mean_ci(sub)
                means.append(m)
                ax.plot([r + dx, r + dx], [lo, hi], color=color, lw=2, solid_capstyle="round")
                ax.scatter([r + dx], [m], s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.plot([dx, 1 + dx], means, color=color, lw=2, label=name)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No rivalry", "Rivalry"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Exploit share")
    axes[0][0].legend(loc="upper center", bbox_to_anchor=(0.55 * len(panels), 1.38), ncol=2)
    fig.savefig(out / "enforcement.pdf", bbox_inches="tight")
    plt.close(fig)


def time_course(block: Path, df: pd.DataFrame, out: Path) -> None:
    """Cumulative share of live agents who cheated, by round, in the two corners."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for cell, color, name in ((HF, BLUE, "Incident corner"), (DM, ORANGE, "Collective corner")):
        curves = []
        for run in df[df["cell"] == cell]["run"]:
            events = [json.loads(x) for x in (block / run / "events.jsonl").read_text().splitlines()]
            roles = events[0]["roles"]
            rounds = events[0]["config"]["rounds"]
            labels = {e["id"]: e["label"] for e in events if e["type"] == "verdict"}
            first: dict[str, int] = {}
            for e in events:
                if e["type"] == "submission" and e["accepted"] and roles[e["agent"]] == "live" and labels.get(e["id"]) == "exploit":
                    first.setdefault(e["agent"], e["round"])
            live = sum(1 for r in roles.values() if r == "live")
            curves.append([sum(1 for v in first.values() if v <= t) / live for t in range(1, rounds + 1)])
        arr = np.array(curves)
        x = np.arange(1, arr.shape[1] + 1)
        ax.plot(x, arr.mean(axis=0), color=color, lw=2, label=name)
    ax.set_xlabel("Round")
    ax.set_xticks([1, 5, 10, 15])
    ax.set_ylabel("Live agents who have cheated")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    fig.savefig(out / "time_course.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("block", type=Path)
    p.add_argument("--per-cell", type=int, default=6)
    p.add_argument("--joiners-only", type=Path, help="block without police, drawn as a second enforcement panel")
    p.add_argument("--out", type=Path, default=ROOT / "paper" / "figures")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = block_table(a.block)
    if {HF, DM} <= set(df["cell"]):
        corners(df, a.out)
        time_course(a.block, df, a.out)
    conf = a.block / "confirmatory.json"
    if conf.exists():
        factors(json.loads(conf.read_text()), a.out)
    enf = {"R0P1C1H1E0", "R1P1C1H1E0", "R0P1C1H1E1", "R1P1C1H1E1"}
    if enf <= set(df["cell"]):
        panels = [("Block A, with GLM agents", df)]
        if a.joiners_only:
            pure = block_table(a.joiners_only)
            if enf <= set(pure["cell"]):
                panels.append(("Block B, gpt-oss only", pure))
        enforcement(panels, a.out)
    print("figures in", a.out)


if __name__ == "__main__":
    main()
