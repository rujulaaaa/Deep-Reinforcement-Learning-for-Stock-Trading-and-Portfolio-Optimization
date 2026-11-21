"""Passive benchmarks: buy-and-hold every stock, and the NIFTY 50 index itself."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import MarketData
from ..metrics import performance
from ..runner import run_portfolio, run_trading


def buy_and_hold(data: MarketData, cfg: dict, task: str) -> dict:
    if task == "trading":
        return run_trading(data, cfg, lambda obs, t: np.ones(data.n_assets))
    n = data.n_assets + 1
    first = {"done": False}

    def policy(obs, t):  # equal weights once, then let them drift (zero turnover)
        if not first["done"]:
            first["done"] = True
            return np.r_[0.0, np.full(n - 1, 1.0 / (n - 1))]
        return obs[-n:]

    return run_portfolio(data, cfg, policy)


def index_benchmark(data: MarketData, cfg: dict) -> dict:
    s = max(data.trade_start - 1, 0)
    px = data.index_close[s:]
    curve = pd.Series(px / px[0] * cfg["env"]["initial_cash"], index=data.dates[s:])
    return {"metrics": performance(curve.values), "curve": curve}
