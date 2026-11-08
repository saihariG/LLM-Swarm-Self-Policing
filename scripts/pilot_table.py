#!/usr/bin/env python
"""One line per block and cell. Usage: python scripts/pilot_table.py runs/pilot2_*"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.outcomes import block_table  # noqa: E402

frames = []
for arg in sys.argv[1:]:
    table = block_table(Path(arg))
    if len(table):
        frames.append(table)
df = pd.concat(frames, ignore_index=True)
cols = ["exposed", "adoption", "exploit_share", "report_rate", "callout_rate", "tp_rate", "honest_accused",
        "removals", "n_calls", "prompt_tokens", "completion_tokens", "llm_errors"]
out = df.groupby(["block", "cell"])[cols].mean()
out.insert(0, "n", df.groupby(["block", "cell"]).size())
pd.set_option("display.width", 230, "display.max_columns", 40)
print(out.round(3).to_string())
