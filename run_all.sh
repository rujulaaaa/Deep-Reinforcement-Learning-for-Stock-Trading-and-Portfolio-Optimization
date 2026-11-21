#!/usr/bin/env bash
# Full pipeline: data → tuning (val) → 30-seed evaluation (test) → baselines → report.
# Pass --quick for a few-minute smoke run (results go to results/quick/).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}

$PY scripts/download_data.py      "$@"
$PY scripts/tune.py --algo dqn    "$@"
$PY scripts/tune.py --algo ddpg   "$@"
$PY scripts/evaluate.py           "$@"
$PY scripts/run_baselines.py      "$@"
$PY scripts/report.py             "$@"
