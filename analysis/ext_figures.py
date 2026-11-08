#!/usr/bin/env python
"""Figures for the exploratory arms, from notes/extensions_report.json.

Usage: python analysis/ext_figures.py [--report notes/extensions_report.json] [--out paper/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.figures import BLUE, MUTED, ORANGE, SURFACE  # noqa: E402

DOSE_PANELS = [("joiner_share", "Exploit share of gpt-oss work"), ("reporters", "Reporters per swarm")]
ARM_NAMES = {"none": "No enforcement", "audit": "Organizer audit", "vote": "Peer vote"}
DIFF_ROWS = [("vote", "none"), ("audit", "none"), ("vote", "audit")]


def dose(rows: list[dict], out: Path) -> None:
    """Small multiples over police agents per swarm. Each point comes from a different block."""
    fig, axes = plt.subplots(1, len(DOSE_PANELS), figsize=(6.2, 2.1), sharex=True)
    for ax, (metric, title) in zip(axes, DOSE_PANELS):
        xs = [r["police"] for r in rows]
        ax.plot(xs, [r[metric] for r in rows], color=BLUE, lw=2)
        for r in rows:
            ax.plot([r["police"]] * 2, [r[f"{metric}_lo"], r[f"{metric}_hi"]], color=BLUE, lw=2, solid_capstyle="round")
            ax.scatter([r["police"]], [r[metric]], s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks(xs)
        ax.set_xlabel("GLM agents among the 10 live agents")
        ax.grid(axis="x", visible=False)
    fig.savefig(out / "dose.pdf", bbox_inches="tight")
    plt.close(fig)


def vote(differences: list[dict], metric: str, out: Path) -> None:
    """Dot and interval on the differences between arms, for one metric."""
    rows = [d for pair in DIFF_ROWS for d in differences
            if d["metric"] == metric and (d["arm"], d["versus"]) == pair]
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    for i, d in enumerate(rows):
        y = len(rows) - 1 - i
        color = ORANGE if d["arm"] == "vote" else BLUE
        ax.plot([d["lo"], d["hi"]], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.scatter([d["difference"]], [y], s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{ARM_NAMES[d['arm']]} minus {ARM_NAMES[d['versus']].lower()}" for d in rows][::-1])
    ax.set_xlabel("Difference, with 95% interval")
    ax.grid(axis="y", visible=False)
    fig.savefig(out / f"vote_{metric}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, default=ROOT / "notes" / "extensions_report.json")
    p.add_argument("--out", type=Path, default=ROOT / "paper" / "figures")
    p.add_argument("--vote-metric", default="credit_left")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = json.loads(a.report.read_text())
    dose(report["dose"], a.out)
    vote(report["vote"]["differences"], a.vote_metric, a.out)
    print("figures in", a.out)


if __name__ == "__main__":
    main()
