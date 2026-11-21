#!/usr/bin/env bash
# Live terminal demo of the project, 9 sections, about 3–5 minutes.
# Uses saved checkpoints and results only: nothing is trained, downloaded or written.
#
#   ./demo.sh                     start at section 1, press Enter to advance
#   ./demo.sh 5                   start at section 5
#   AUTO=1 PAUSE=8 ./demo.sh      advance automatically every 8 s
#   OPEN_IMAGES=0 ./demo.sh       don't open result images
#   STOCK=TCS.NS ./demo.sh 5      replay a different stock in section 5
set -uo pipefail
cd "$(dirname "$0")"

START=${1:-1}
AUTO=${AUTO:-0}
PAUSE=${PAUSE:-6}
OPEN_IMAGES=${OPEN_IMAGES:-1}
STOCK=${STOCK:-RELIANCE.NS}

# --- environment: keep output clean and the repo untouched ----------------
unset PYTHONPATH                      # system/ROS site-packages would shadow the venv
export PYTHONWARNINGS=ignore          # no library warnings on screen
export PYTHONDONTWRITEBYTECODE=1      # no new __pycache__ files
export TQDM_DISABLE=1 OMP_NUM_THREADS=4

fail() { printf '\033[1;91m%s\033[0m\n' "$1"; exit 1; }
[ -x .venv/bin/python ] || fail "No .venv found: see README 'Quick start'."
[ -f data/raw/nifty50_ohlcv.csv ] || fail "Missing data/raw/nifty50_ohlcv.csv: run scripts/download_data.py."
[ -f results/final/summary_trading.csv ] || fail "Missing results/final: run ./run_all.sh first."
ls results/runs/dqn/seed*.pt >/dev/null 2>&1 || fail "Missing results/runs checkpoints: run ./run_all.sh."
.venv/bin/python -c "import rich" 2>/dev/null || fail "Missing 'rich': .venv/bin/pip install rich"
# shellcheck disable=SC1091
source .venv/bin/activate
if [ "$OPEN_IMAGES" = 1 ] && ! command -v eog >/dev/null; then OPEN_IMAGES=0; fi

# --- helpers --------------------------------------------------------------
TITLE='\033[1;30;106m'   # bold black on bright cyan
PROMPT='\033[1;92m'      # bright green
CMD='\033[1;97m'         # bright white
KEY='\033[1;93m'         # bright yellow
RST='\033[0m'
IMG_PIDS=()

section() {  # section <n> <title>
  clear
  printf "${TITLE} %-94s ${RST}\n\n" "$1/9  $2"
}

run() {      # print the exact command like a shell prompt, then run it
  printf "${PROMPT}\$${RST} ${CMD}%s${RST}\n" "$1"
  eval "$1"
}

open_img() {
  [ "$OPEN_IMAGES" = 1 ] || return 0
  printf "${PROMPT}\$${RST} ${CMD}eog -n %s &${RST}\n" "$1"
  eog -n "$1" >/dev/null 2>&1 &
  IMG_PIDS+=($!)
}

close_imgs() {
  if [ ${#IMG_PIDS[@]} -gt 0 ]; then
    kill "${IMG_PIDS[@]}" 2>/dev/null
    wait "${IMG_PIDS[@]}" 2>/dev/null
  fi
  IMG_PIDS=()
}
trap close_imgs EXIT

pause() {
  if [ "$AUTO" = 1 ]; then
    sleep "$PAUSE"
  else
    printf "\n${KEY}▶ Enter${RST}"
    read -r _
  fi
  close_imgs
}

# --- sections -------------------------------------------------------------
s1() { section 1 "Deep RL for NIFTY50 trading: overview"
       run "python .demo/intro.py"; pause; }

s2() { section 2 "Data pipeline: NIFTY50 → state space"
       run "python .demo/data_summary.py"; pause; }

s3() { section 3 "Correctness checks: costs, no look-ahead, agents"
       run "python -m pytest -q -p no:cacheprovider -p no:warnings tests/"; pause; }

s4() { section 4 "Hyper-parameter tuning (saved sweep)"
       run "python .demo/tuning.py"
       open_img results/figures/hyperparameter_sweep.png; pause; }

s5() { section 5 "Live inference: DQN trading one stock (test 2023–26)"
       run "python .demo/infer_dqn_stock.py $STOCK"; pause; }

s6() { section 6 "Live inference: same DQN policy on all 20 stocks"
       run "python .demo/infer_dqn_all.py"; pause; }

s7() { section 7 "Live inference: DDPG portfolio allocation (test 2023–26)"
       run "python .demo/infer_ddpg.py"; pause; }

s8() { section 8 "Results: 30 seeds per method vs baselines (saved)"
       run "python .demo/results.py"
       open_img results/figures/equity_portfolio.png
       open_img results/figures/test_sharpe_distribution.png; pause; }

s9() { section 9 "Summary"
       run "python .demo/summary.py"; echo; }

for n in 1 2 3 4 5 6 7 8 9; do
  [ "$n" -ge "$START" ] && "s$n"
done
