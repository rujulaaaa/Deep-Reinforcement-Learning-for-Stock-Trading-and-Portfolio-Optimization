"""Performance metrics computed from an equity curve (daily values)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance(values, turnover: float | None = None) -> dict:
    v = np.asarray(values, dtype=np.float64)
    r = v[1:] / v[:-1] - 1.0
    n = len(r)
    total = v[-1] / v[0] - 1.0
    ann_ret = (v[-1] / v[0]) ** (TRADING_DAYS / max(n, 1)) - 1.0
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS) if n > 1 else 0.0
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS) if n > 1 and r.std() > 0 else 0.0
    downside = r[r < 0]
    dd_std = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    sortino = r.mean() / dd_std * np.sqrt(TRADING_DAYS) if dd_std > 0 else 0.0
    peak = np.maximum.accumulate(v)
    mdd = float(((peak - v) / peak).max())
    out = {
        "cum_return": total,
        "ann_return": ann_ret,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": ann_ret / mdd if mdd > 0 else 0.0,
        "final_value": v[-1],
    }
    if turnover is not None:
        out["avg_daily_turnover"] = turnover / max(n, 1)
    return {k: float(x) for k, x in out.items()}


def summarise(df: pd.DataFrame, by: str = "method",
              cols=("sharpe", "ann_return", "cum_return", "max_drawdown", "sortino", "calmar")) -> pd.DataFrame:
    """mean ± std (and run count) of each metric per method."""
    g = df.groupby(by)[list(cols)]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    out["runs"] = df.groupby(by).size()
    return out
