"""Baselines on both tasks, evaluated in the same environments as the agents.

    python scripts/run_baselines.py

ARIMA     per-stock ARMA on returns (order by train AIC)      — deterministic, 1 run
LSTM      pooled LSTM next-day return forecaster              — experiment.n_seeds runs
Buy&Hold  hold every stock (trading) / equal-weight drift (portfolio)
NIFTY 50  the index itself (reference only, no costs)

Forecasts become positions: trading → long if forecast > threshold else flat;
portfolio → equal-weight the top-k stocks with positive forecasts, rest cash.
"""

import pandas as pd
from common import base_parser, get_config, results_dir, run_pool, worker_splits

from finsearch.baselines import (arima_forecasts, backtest_forecasts, buy_and_hold,
                                 index_benchmark, lstm_forecasts)
from finsearch.data import build_dataset

TASKS = {"trading": "DQN task", "portfolio": "DDPG task"}


def evaluate_forecasts(name, fc, splits, cfg, seed):
    rows, curves = [], {}
    for task in TASKS:
        row = {"method": name, "task": task, "seed": seed}
        for split in ("val", "test"):
            res = backtest_forecasts(fc, splits[split], cfg, task)
            row.update({f"{split}_{k}": v for k, v in res["metrics"].items()})
            if split == "test":
                curves[task] = res["curve"]
        rows.append(row)
    return rows, curves


def lstm_job(args):
    cfg, seed = args
    splits = worker_splits(cfg)
    return seed, evaluate_forecasts("LSTM", lstm_forecasts(splits, cfg, seed), splits, cfg, seed)


if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    cfg = get_config(args)
    splits = build_dataset(cfg)
    out = results_dir(cfg, "final", quick=args.quick)
    rows, curves = [], {t: {} for t in TASKS}

    print("ARIMA: selecting orders by AIC on train ...")
    fc, orders = arima_forecasts(splits, cfg["baselines"]["arima_orders"])
    pd.Series({k: str(v) for k, v in orders.items()}, name="order").to_csv(out / "arima_orders.csv")
    r, c = evaluate_forecasts("ARIMA", fc, splits, cfg, 0)
    rows += r
    for t in TASKS:
        curves[t]["ARIMA"] = c[t]

    print(f"LSTM: {cfg['experiment']['n_seeds']} seeds ...")
    lstm_curves = {t: {} for t in TASKS}
    for seed, (r, c) in run_pool(lstm_job, [(cfg, s) for s in range(cfg["experiment"]["n_seeds"])],
                                 cfg, "lstm"):
        rows += r
        for t in TASKS:
            lstm_curves[t][seed] = c[t]
    for t in TASKS:
        curves[t]["LSTM"] = pd.concat(lstm_curves[t], axis=1).mean(axis=1)

    for task in TASKS:
        row = {"method": "Buy&Hold", "task": task, "seed": 0}
        for split in ("val", "test"):
            res = buy_and_hold(splits[split], cfg, task)
            row.update({f"{split}_{k}": v for k, v in res["metrics"].items()})
            if split == "test":
                curves[task]["Buy&Hold"] = res["curve"]
        rows.append(row)
        row = {"method": "NIFTY 50", "task": task, "seed": 0}
        for split in ("val", "test"):
            res = index_benchmark(splits[split], cfg)
            row.update({f"{split}_{k}": v for k, v in res["metrics"].items()})
            if split == "test":
                curves[task]["NIFTY 50"] = res["curve"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "baselines_runs.csv", index=False)
    for t in TASKS:
        pd.DataFrame(curves[t]).to_csv(out / f"baselines_{t}_test_curves.csv")
    cols = ["test_sharpe", "test_ann_return", "test_max_drawdown"]
    print(df.groupby(["task", "method"])[cols].mean().round(3).to_string())
