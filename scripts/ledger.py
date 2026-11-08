#!/usr/bin/env python
"""Build notes/verification_ledger.md: each cited number with the PDF page it was found on.

A claim is listed only if its exact search string is found in the PDF's text.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sources"

# (source file, what we may cite, exact string to find)
CLAIMS = [
    ("metr.pdf", "533 agents active on the board, over 90% joined the attack", "Of the 533 agents active on the message board"),
    ("metr.pdf", "Figure 11: over 90% of board agents took part in the attack", "Over 90% of agents on the message board participated"),
    ("metr.pdf", "Sweep found 3-6 examples of agents considering alerting humans", "only found 3-6 actual examples"),
    ("metr.pdf", "No agent actually pursued alerting humans", "actually pursue alerting humans at all"),
    ("metr.pdf", "Agents saw no easily available route to humans", "easily-available route to alerting humans"),
    ("metr.pdf", "Some agents escalated to the board as an authority", "escalated to the message board"),
    ("metr.pdf", "Roughly 7% of transcripts successfully spoofed", "Roughly 7% of the transcripts"),
    ("metr.pdf", "Analysis agent adopted the transcript agent's perspective", "uncritically adopt the perspective"),
    ("paglieri.pdf", "Cohorts: exploiters 9%, converts 5%, whistleblowers 24%, unaware 62%", "exploiters (9%), converts (5%), whistleblowers (24%)"),
    ("paglieri.pdf", "Exploit contagion and whistleblowing reproduced across runs", "reliably reproduced"),
    ("paglieri.pdf", "Ostrom: ease of monitoring as the critical factor", "ease of monitoring is"),
    ("paglieri.pdf", "Organizer feedback channel was unmonitored during the run", "operated unmonitored in real"),
    ("paglieri.pdf", "Agents lacked tools to sanction", "institutional affordances"),
    ("institutional_2601.11369.pdf", "90 runs per condition, six model configurations", "N = 90 runs/condition"),
    ("institutional_2601.11369.pdf", "Mean collusion tier 3.1 to 1.8, d = 1.28", "mean tier falls from 3.1 to 1.8"),
    ("institutional_2601.11369.pdf", "Severe collusion 50% to 5.6%", "from 50% to 5.6%"),
    ("institutional_2601.11369.pdf", "Oracle is a programmatic detector", "programmatic detector"),
    ("persuasion_2607.08066.pdf", "40 tasks", "40 tasks"),
    ("persuasion_2607.08066.pdf", "CoT access raises approval of harmful actions by 9.5% on average", "9.5%"),
    ("persuasion_2607.08066.pdf", "Cross-family fact-checker cuts approval up to 45%, same-model 6%", "by up to 45%"),
    ("persuasion_2607.08066.pdf", "Limitation: hardcoded actions, not tool calls", "hardcoded agent actions"),
    ("persuasion_2607.08066.pdf", "Multi-agent coordination named as future work", "multi-agent"),
    ("infoaccess_2601.21112.pdf", "Less-is-more effect", "less-is-more effect"),
    ("infoaccess_2601.21112.pdf", "Filtering helped in seven of eight monitor-environment pairs", "out of eight"),
    ("scheme_2605.29178.pdf", "Undetected sabotage under 3% with full-visibility monitor", "stays at <3%"),
    ("scheme_2605.29178.pdf", "Even with fabricated cover stories", "fabricate cover stories"),
    ("scheme_2605.29178.pdf", "Comms-only TPR falls to 84% with fabricated justifications", "Comms TPR falls to 84%"),
    ("tamper.pdf", "Peer Logs: Opus-5 60%, Muse Spark and Opus 5.5 0%", "Opus-5 at 60%"),
    ("tamper.pdf", "Peer Workspaces: all but Muse Spark tampered in at least 90% of trials", "at least 90% of trials"),
    ("troy.pdf", "Forged authorization: Sol executes 2/7 under strict+open", "executed in 2/7"),
    ("metr.pdf", "Roughly 1,200 agents meant to be isolated", "Roughly 1200 agents meant to be isolated"),
    ("metr.pdf", "Over 70,000 messages and files", "70,000 messages and files"),
    ("metr.pdf", "700 went on to take part in the attack", "700 went on"),
    ("paglieri.pdf", "A cohort adopted the exploit in response to competitive pressure", "in response to competitive pressure"),
    ("paglieri.pdf", "100 autonomous LLM agents", "100 autonomous LLM agents"),
    ("paglieri.pdf", "Predicted tools: vote, reject, ban or expel", "temporarily ban or expel"),
    ("ye_steinhardt_2607.09766.pdf", "Populations are 50% good and 50% bad", "50% good and 50% bad"),
    ("ye_steinhardt_2607.09766.pdf", "Ground-truth norm verifier is gpt-5-mini", "with gpt-5-mini"),
    ("ye_steinhardt_2607.09766.pdf", "Bad agents file false reports against competitors", "false reports to eliminate competitors"),
    ("ye_steinhardt_2607.09766.pdf", "15 rounds in main experiments", "T = 15 rounds"),
    ("ye_steinhardt_2607.09766.pdf", "Checked uses a lightweight LM verifier", "lightweight LM verifier"),
    ("ssrn-5349484.pdf", "Communication plus sanctions reach 100% efficiency by round 20", "100% by Round 20"),
    ("ssrn-5349484.pdf", "Sanctions alone reach 95.2% at high endowment", "95.2% in high endowment"),
    ("ssrn-5349484.pdf", "Apparent retaliation by sanctioned defectors", "apparent retaliation"),
]


def pages(pdf: Path) -> list[str]:
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    return [re.sub(r"\s+", " ", p) for p in text.split("\f")]


def main() -> None:
    cache: dict[str, list[str]] = {}
    rows, missing = [], []
    for name, claim, needle in CLAIMS:
        if name not in cache:
            cache[name] = pages(SRC / name)
        hits = [i + 1 for i, page in enumerate(cache[name]) if needle.lower() in page.lower()]
        if hits:
            rows.append(f"| {name} | {claim} | p. {hits[0]} | `{needle}` |")
        else:
            missing.append(f"- {name}: {needle}")
    out = ["# Verification ledger", "",
           "Generated by `scripts/ledger.py`. Each row's search string was found in the PDF's own text on that page.", "",
           "| Source | Claim | Page | Found text |", "|---|---|---|---|", *rows, ""]
    if missing:
        out += ["## Not found, do not cite until checked", *missing, ""]
    out += ["## Not in this ledger",
            "- Check Point: web page only. Reread before citing.",
            "- Whistle While You Hack: GitHub README, copy in `sources/`.",
            "- Bell and Madisetti, Truong et al., Hu et al., Muralimohan et al.: abstracts only.",
            "- Hammond et al. (2502.14143), Agrawal et al. (2511.17085), Rehm (2609.22600), X / POLIS (2608.09828), Makins et al. (2607.07368): abstracts only, read on the arXiv listing on 2026-11-08."]
    (ROOT / "notes" / "verification_ledger.md").write_text("\n".join(out) + "\n")
    print(len(rows), "verified,", len(missing), "missing")
    print("\n".join(missing))


if __name__ == "__main__":
    main()
