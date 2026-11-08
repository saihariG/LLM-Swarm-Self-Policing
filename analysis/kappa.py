#!/usr/bin/env python
"""Cohen's kappa between Sai's hand labels and the judge.

Usage: python analysis/kappa.py notes/hand_check.csv
Reads hand_label from that file and judge_label from the matching .key.csv.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

LABELS = ("call_out", "promote", "test_complaint", "other")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n**2
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main() -> None:
    sheet = Path(sys.argv[1])
    hand = {r["item"]: r["hand_label"].strip() for r in csv.DictReader(open(sheet, newline=""))}
    judge = {r["item"]: r["judge_label"].strip() for r in csv.DictReader(open(sheet.with_suffix(".key.csv"), newline=""))}
    unknown = sorted({v for v in hand.values() if v and v not in LABELS})
    if unknown:
        raise SystemExit(f"labels not in the rubric: {unknown}")
    items = [i for i, v in hand.items() if v]
    a, b = [hand[i] for i in items], [judge[i] for i in items]
    print(f"{len(items)} of {len(hand)} coded, agreement {sum(x == y for x, y in zip(a, b)) / len(items):.3f}, "
          f"kappa {cohen_kappa(a, b):.3f}")
    # The primary outcomes only use call_out, so its agreement matters most.
    a2 = ["call_out" if x == "call_out" else "not" for x in a]
    b2 = ["call_out" if x == "call_out" else "not" for x in b]
    print(f"call_out against the rest, kappa {cohen_kappa(a2, b2):.3f}")
    for x, y in sorted(Counter(zip(a, b)).items()):
        print(x, y)


if __name__ == "__main__":
    main()
