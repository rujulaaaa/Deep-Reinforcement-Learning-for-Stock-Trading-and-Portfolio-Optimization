"""Guided terminal demo of the project (≈2–3 minutes, no long training).

    python scripts/demo.py            # full demo
    python scripts/demo.py --fast     # no pauses / animation delays
    python scripts/demo.py --no-train # skip the live mini-training section

Sections
  1. Data pipeline          — NIFTY50 dataset, features, chronological split
  2. Live training          — a short DQN run printing validation Sharpe
  3. DQN replay (test)      — the trained agent trading one stock, month by month
  4. DDPG replay (test)     — the trained agent managing the 20-stock portfolio
  5. Results vs baselines   — 30-seed test results against ARIMA / LSTM / Buy&Hold
  6. Figures                — opens the generated plots
Sections 3–5 use the checkpoints and CSVs written by ./run_all.sh.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from finsearch.agents import DDPGAgent, DQNAgent
from finsearch.config import ROOT, load_config
from finsearch.data import FEATURES, build_dataset, describe
from finsearch.envs import PortfolioEnv, TradingEnv
from finsearch.metrics import performance
from finsearch.runner import train_agent

RES = ROOT / os.environ.get("DEMO_RESULTS", "results")
B, DIM, RST = "\033[1m", "\033[2m", "\033[0m"
BLUE, GREEN, RED, YEL, CYAN = "\033[34m", "\033[32m", "\033[31m", "\033[33m", "\033[36m"
SPARK = "▁▂▃▄▅▆▇█"
ARGS = None


# ---------------------------------------------------------------- helpers
def pause(sec: float) -> None:
    if not ARGS.fast:
        time.sleep(sec)


def header(n: int, title: str, subtitle: str = "") -> None:
    w = min(shutil.get_terminal_size().columns, 90)
    print(f"\n{BLUE}{'━' * w}{RST}\n{B}{BLUE} {n}. {title}{RST}")
    if subtitle:
        print(f"{DIM}    {subtitle}{RST}")
    print(f"{BLUE}{'━' * w}{RST}")
    pause(1.0)


def typed(text: str, delay: float = 0.012) -> None:
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if not ARGS.fast:
            time.sleep(delay)
    print()


def sparkline(values, width: int = 50) -> str:
    v = np.asarray(values, dtype=float)
    idx = np.linspace(0, len(v) - 1, min(width, len(v))).astype(int)
    v = v[idx]
    lo, hi = v.min(), v.max()
    return "".join(SPARK[int((x - lo) / (hi - lo + 1e-12) * (len(SPARK) - 1))] for x in v)


def pct(x: float) -> str:
    c = GREEN if x >= 0 else RED
    return f"{c}{x:+.1%}{RST}"


def bar(frac: float, width: int = 30, color: str = CYAN) -> str:
    n = int(round(frac * width))
    return f"{color}{'█' * n}{RST}{DIM}{'░' * (width - n)}{RST}"


def best_checkpoint(algo: str):
    """Seed with the highest VALIDATION Sharpe (test is not used to pick it)."""
    runs = RES / "final" / f"{algo}_runs.csv"
    if not runs.exists():
        return None, None
    df = pd.read_csv(runs)
    seed = int(df.loc[df.val_sharpe.idxmax(), "seed"])
    path = RES / "runs" / algo / f"seed{seed}.pt"
    return (path, seed) if path.exists() else (None, None)


def need_results(what: str) -> None:
    print(f"{YEL}  ⚠ {what} not found — run ./run_all.sh first (or ./run_all.sh --quick).{RST}")


# ---------------------------------------------------------------- sections
def section_data(cfg):
    header(1, "Data pipeline — NIFTY50 → state space",
           "Yahoo Finance OHLCV · 11 causal technical features · chronological split")
    splits = build_dataset(cfg)
    d = splits["train"]
    print(f"  Universe : {len(d.tickers)} NIFTY50 stocks + ^NSEI index")
    print(f"  {DIM}{', '.join(t.replace('.NS', '') for t in d.tickers)}{RST}")
    print(f"  Features : {', '.join(FEATURES)}\n")
    print(describe(splits).to_string())
    total = sum(len(s.dates) - s.trade_start for s in splits.values())
    print(f"\n  {B}{total:,} trading days × {d.n_assets} stocks = {total * d.n_assets:,} stock-days "
          f"({total * d.n_assets * 5:,} OHLCV points){RST}")
    print(f"  {DIM}Normalisation fitted on TRAIN only; test (2023→) is never seen during training/tuning.{RST}")
    pause(2.5)
    return splits


def section_train(cfg, splits):
    header(2, "Live training — DQN on the trading environment",
           "Short 15k-step run (the real runs use 100k steps × 30 seeds); best-val checkpoint kept")
    c = load_config(overrides=["dqn.total_steps=15000", "train.eval_every=2500",
                               "dqn.eps_decay_steps=8000", "dqn.learning_starts=1000"])
    torch.set_num_threads(4)
    t0 = time.time()
    _, hist = train_agent("dqn", splits, c, seed=7, verbose=True)
    print(f"  {DIM}done in {time.time() - t0:.0f}s · ε decayed {c['dqn']['eps_start']} → "
          f"{hist.explore.iloc[-1]:.2f} · best val Sharpe {hist.val_sharpe.max():+.3f}{RST}")
    pause(2.0)


def section_dqn(splits_default):
    path, seed = best_checkpoint("dqn")
    header(3, "DQN agent trading on the unseen TEST period (2023 → 2026)",
           f"checkpoint: results/runs/dqn/seed{seed}.pt (seed chosen by validation Sharpe)")
    if path is None:
        return need_results("DQN checkpoints")
    ck = torch.load(path, weights_only=False)
    cfg = ck["config"]
    splits = build_dataset(cfg)
    test = splits["test"]
    env0 = TradingEnv(test, cfg, asset=0)
    agent = DQNAgent(env0.obs_dim, env0.n_actions, cfg["dqn"])
    agent.load_state_dict(ck["state_dict"])
    positions = np.asarray(cfg["trading"]["positions"])

    # pick the stock to replay
    tick = ARGS.stock if ARGS.stock in test.tickers else "RELIANCE.NS"
    i = test.tickers.index(tick)
    env = TradingEnv(test, cfg, asset=i)
    obs, done = env.reset(), False
    start_px = test.close[env.t, i]
    print(f"  Replaying {B}{tick}{RST} with ₹{cfg['env']['initial_cash']:,.0f} · actions = target exposure "
          f"{[float(x) for x in positions]} · cost {cfg['env']['cost']:.1%}/turnover\n")
    print(f"  {B}{'Date':<12}{'Price':>10}  {'Position':<22}{'Agent':>9}{'Buy&Hold':>10}{RST}")
    k = 0
    while not done:
        a = agent.act(obs)
        obs, _, done, _ = env.step(a)
        k += 1
        if k % 21 == 0 or done:  # roughly monthly
            px = test.close[env.t, i]
            print(f"  {str(env.dates[-1].date()):<12}{px:>10,.1f}  {bar(env.weights[1], 14)} "
                  f"{env.weights[1]:>4.0%}  {pct(env.value / env.values[0] - 1):>18}"
                  f"{pct(px / start_px - 1):>19}")
            pause(0.08)
    m = performance(env.values, env.turnover)
    bh = performance(test.close[env.first_t:, i])
    print(f"\n  {B}{tick}{RST}  agent: Sharpe {m['sharpe']:+.2f}, return {pct(m['cum_return'])}, "
          f"max DD {m['max_drawdown']:.1%}   |   buy&hold: Sharpe {bh['sharpe']:+.2f}, "
          f"return {pct(bh['cum_return'])}, max DD {bh['max_drawdown']:.1%}")
    print(f"  equity {CYAN}{sparkline(env.values)}{RST}")

    # all 20 stocks at once
    print(f"\n  {B}Same policy across all {test.n_assets} stocks (test):{RST}")
    rows = []
    for j, t in enumerate(test.tickers):
        e = TradingEnv(test, cfg, asset=j)
        o, dn = e.reset(), False
        while not dn:
            o, _, dn, _ = e.step(agent.act(o))
        rows.append((t.replace(".NS", ""), performance(e.values)["sharpe"],
                     performance(test.close[e.first_t:, j])["sharpe"]))
    beat = sum(r[1] > r[2] for r in rows)
    for name, s_a, s_b in sorted(rows, key=lambda r: -r[1])[:8]:
        print(f"    {name:<12} agent Sharpe {s_a:+.2f}   buy&hold {s_b:+.2f}")
    print(f"    {DIM}… agent has the higher Sharpe on {beat}/{len(rows)} stocks{RST}")
    pause(2.0)


def section_ddpg():
    path, seed = best_checkpoint("ddpg")
    header(4, "DDPG agent allocating a 20-stock portfolio on the TEST period",
           f"checkpoint: results/runs/ddpg/seed{seed}.pt · continuous weights over cash + 20 stocks")
    if path is None:
        return need_results("DDPG checkpoints")
    ck = torch.load(path, weights_only=False)
    cfg = ck["config"]
    test = build_dataset(cfg)["test"]
    env = PortfolioEnv(test, cfg)
    agent = DDPGAgent(env.obs_dim, env.action_dim, cfg["ddpg"])
    agent.load_state_dict(ck["state_dict"])
    names = ["CASH"] + [t.replace(".NS", "") for t in test.tickers]

    obs, done, k = env.reset(), False, 0
    print(f"  {B}{'Date':<12}{'Value (₹)':>14}{'Return':>16}   {'Cash':>5}   Top-3 holdings{RST}")
    while not done:
        obs, _, done, _ = env.step(agent.act(obs))
        k += 1
        if k % 42 == 0 or done:  # roughly every two months
            w = env.weights
            top = np.argsort(-w[1:])[:3] + 1
            print(f"  {str(env.dates[-1].date()):<12}{env.value:>14,.0f}{pct(env.value / env.values[0] - 1):>25}"
                  f"   {w[0]:>5.0%}   " + ", ".join(f"{names[j]} {w[j]:.0%}" for j in top))
            pause(0.12)
    m = performance(env.values, env.turnover)
    print(f"\n  {B}Final allocation{RST}")
    for j in np.argsort(-env.weights)[:8]:
        print(f"    {names[j]:<12}{bar(env.weights[j] / env.weights.max(), 30)} {env.weights[j]:.1%}")
    print(f"\n  Sharpe {m['sharpe']:+.2f} · return {pct(m['cum_return'])} · max DD {m['max_drawdown']:.1%} · "
          f"avg daily turnover {m['avg_daily_turnover']:.2%}")
    print(f"  equity {CYAN}{sparkline(env.values)}{RST}")
    pause(2.0)


def section_results():
    header(5, "Results vs baselines — 30 seeds per method, held-out TEST split",
           "Same environment, same 0.1% costs for every method · mean ± std across runs")
    found = False
    for task, algo in [("trading", "DQN"), ("portfolio", "DDPG")]:
        p = RES / "final" / f"summary_{task}.csv"
        if not p.exists():
            continue
        found = True
        s = pd.read_csv(p, index_col=0)
        print(f"\n  {B}{task.capitalize()} task ({algo}){RST}")
        print(f"  {B}{'Method':<11}{'Runs':>5}{'Sharpe':>17}{'Ann. return':>20}{'Max DD':>10}{RST}")
        for m, r in s.iterrows():
            hl = B + CYAN if m == algo else ""
            sd = "" if np.isnan(r.sharpe_std) else f" ± {r.sharpe_std:.2f}"
            rsd = "" if np.isnan(r.ann_return_std) else f" ± {r.ann_return_std:.1%}"
            print(f"  {hl}{m:<11}{int(r.runs):>5}{r.sharpe_mean:>+10.2f}{sd:<7}"
                  f"{r.ann_return_mean:>+12.1%}{rsd:<8}{r.max_drawdown_mean:>10.1%}{RST}")
            pause(0.15)
        imp = RES / "final" / f"improvement_{task}.csv"
        if imp.exists():
            d = pd.read_csv(imp)
            d = d[(d.metric == "sharpe") & d.baseline.isin(["ARIMA", "LSTM"])]
            for _, r in d.iterrows():
                rel = "n/a" if np.isnan(r.rel_improvement_pct) else f"{r.rel_improvement_pct:+.1f}%"
                print(f"    {algo} vs {r.baseline:<6} Sharpe: {rel:>8}  "
                      f"({r.seeds_beating_baseline_pct:.0f}% of seeds beat it)")
    if not found:
        need_results("results/final/summary_*.csv")
    pause(2.0)


def section_figures():
    header(6, "Figures", "results/figures/")
    figs = sorted((RES / "figures").glob("*.png"))
    if not figs:
        return need_results("figures")
    for f in figs:
        print(f"  • {f.relative_to(ROOT)}")
    if not ARGS.no_open and shutil.which("xdg-open"):
        for f in figs:
            subprocess.Popen(["xdg-open", str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pause(0.5)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fast", action="store_true", help="no pauses")
    p.add_argument("--no-train", action="store_true", help="skip the live mini-training")
    p.add_argument("--no-open", action="store_true", help="do not open figure windows")
    p.add_argument("--stock", default="RELIANCE.NS", help="stock replayed in section 3")
    ARGS = p.parse_args()

    print(f"\n{B}{CYAN}  Deep RL to Optimise Stock Trading Strategy — NIFTY50{RST}")
    typed(f"{DIM}  FinSearch · Finance Club, IIT Bombay · DQN (trading) & DDPG (portfolio) vs ARIMA / LSTM{RST}")
    pause(1.0)
    cfg = load_config()
    splits = section_data(cfg)
    if not ARGS.no_train:
        section_train(cfg, splits)
    section_dqn(splits)
    section_ddpg()
    section_results()
    section_figures()
    print(f"\n{B}{GREEN}  ✔ Demo complete.{RST}  Code: finsearch/ · configs/ · scripts/ · README.md\n")
