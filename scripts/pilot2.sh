#!/bin/zsh
# Second pilot: push delivery of board entries, corners, enforcement and controls.
cd "$(dirname "$0")/.."
M=${1:-openai/gpt-oss-120b}
TAG=${2:-pilot2}
PROVIDER=${3:-modal}
PY=.venv/bin/python
$PY scripts/run_block.py --block ${TAG}_main --provider $PROVIDER --model $M --cells hf dm R1P1C1H1E1 --n 3 --concurrency 9 --master-seed 11 &
$PY scripts/run_block.py --block ${TAG}_report --provider $PROVIDER --model $M --cells dm --n 1 --instruct-report --master-seed 12 &
$PY scripts/run_block.py --block ${TAG}_norule --provider $PROVIDER --model $M --cells dm --n 1 --no-rule --master-seed 13 &
$PY scripts/run_block.py --block ${TAG}_noseeds --provider $PROVIDER --model $M --cells hf dm --n 1 --no-seeds --concurrency 2 --master-seed 14 &
wait
echo pilot-finished
