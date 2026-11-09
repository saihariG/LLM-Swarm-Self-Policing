#!/usr/bin/env python
"""LLM calls and tokens per block, from the logs, for the replication appendix.

Usage: python analysis/usage.py runs/main_A runs/main_B runs/main_C runs/ext_dose1 runs/ext_dose2 runs/ext_vote
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import outcomes  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("blocks", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=ROOT / "notes/usage.json")
    a = p.parse_args()
    out = {}
    for block in a.blocks:
        df = outcomes.block_table(block)
        out[block.name] = {"swarms": int(len(df)), "calls": int(df["n_calls"].sum()),
                           "prompt_tokens": int(df["prompt_tokens"].sum()),
                           "cached_tokens": int(df["cached_tokens"].sum()),
                           "completion_tokens": int(df["completion_tokens"].sum()),
                           "errors": int(df["llm_errors"].sum())}
        print(block.name, out[block.name])
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
