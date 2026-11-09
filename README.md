# What makes an LLM Swarm Police itself?

Code, logs and analysis for a randomized study of when swarms of LLM agents police their own members. Built for the Apart Swarm Alignment Hackathon, Nov 6-8 2026.

The study holds the population of agents fixed and randomizes the institutions around a swarm: rivalry, public work, the legitimacy of the channel, a route to humans and enforcement. Two scripted agents cheat on a flawed scorer in every swarm, so every swarm gets the same stimulus. A strict verifier gives ground truth for cheating without any LLM judge.

## Layout
- `swarmlab/`: tasks, sandbox, flawed scorer, strict verifier, swarm world, scripted confederates, LLM backends, round loop
- `scripts/run_block.py`: runs a block of swarms from a fixed random schedule
- `analysis/`: per-swarm outcomes, pre-registered estimators, message judge, robustness checks, monitor arm
- `notes/`: results files that every number in the paper comes from, plus the hand-check sheet
- `tests/`: instrument and pipeline tests
- `runs/`: real logs. Full model transcripts are left out because of their size. Every message and submission is in each run's `events.jsonl`.
- `paper/`: LaTeX source, bibliography and figures

## Run
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python openai httpx numpy pandas scipy matplotlib pytest python-dotenv modal
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_block.py --block mock_smoke --provider mock --cells hf dm --n 2
```

The sandbox uses macOS `sandbox-exec`, so running candidate code needs a Mac with Homebrew Python, or a change to `swarmlab/sandbox.py`.

## Reproduce the paper's numbers
These steps only read the logs in `runs/` and make no API calls. The judge labels are already in each run as `judged_v1.jsonl` and `judged_v2.jsonl`, and `judged.jsonl` holds the active version (`scripts/use_rubric.py`).

```bash
P=.venv/bin/python
$P analysis/report.py runs/main_A --per-cell 6 --joiners-only runs/main_B
$P analysis/label_robustness.py runs/main_A --joiners-only runs/main_B
$P analysis/public_callouts.py runs/main_A
$P analysis/policing_channels.py runs/main_A
$P analysis/label_validity.py runs/main_A runs/main_B runs/main_C runs/ext_dose1 runs/ext_dose2 runs/ext_vote
$P analysis/cross_block.py runs/main_A runs/main_B runs/main_C
$P analysis/mechanisms.py runs/main_A runs/main_B
$P analysis/llm_coders.py notes/hand_check_2.csv notes/hand_check_2_claude_opus.csv notes/hand_check_2_claude_sonnet.csv
$P analysis/usage.py runs/main_A runs/main_B runs/main_C runs/ext_dose1 runs/ext_dose2 runs/ext_vote
$P analysis/tier1.py
$P analysis/rivalry_attempts.py runs/main_A
$P analysis/removal_timing.py
$P analysis/secondary.py
$P analysis/extensions.py
$P analysis/sensitivity.py runs/main_A --joiners-only runs/main_B
$P analysis/monitor_report.py runs/main_A > notes/monitor_report_main_A.txt
$P analysis/monitor_report.py runs/main_B > notes/monitor_report_main_B.txt
$P analysis/figures.py runs/main_A --per-cell 6 --joiners-only runs/main_B
$P analysis/ext_figures.py
```

`report.py` writes `runs/main_A/confirmatory.json` (Table 2), and the others write to `notes/`. The tests for `label_robustness.py`, `policing_channels.py` and `cross_block.py` are in `tests/test_labels.py` and `tests/test_followups.py`.

## Tier 1 arms (exploratory)
The first two need the gpt-oss server on Modal, the third needs Fireworks and Meta keys.

```bash
P=.venv/bin/python
$P scripts/run_block.py --block ext_random --provider modal --model openai/gpt-oss-120b --cells R0P1C1H1E0 R1P1C1H1E0 --n 12 --random-audits 2 --master-seed 20261113 --concurrency 12
$P scripts/run_block.py --block ext_reporter --provider modal --model openai/gpt-oss-120b --cells R1P1C1H1E1 --n 12 --scripted-reporter --master-seed 20261114 --concurrency 12
$P scripts/run_block.py --block main_C_incident --provider fireworks --model accounts/fireworks/models/minimax-m3 --provider2 meta --model2 muse-spark-1.3 --n-model2 5 --cells R0P0C0H0E0 --n 10 --concurrency 10 --master-seed 20261115 --max-usd 30
```
