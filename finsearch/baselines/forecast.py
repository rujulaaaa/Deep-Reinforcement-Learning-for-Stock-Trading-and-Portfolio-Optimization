"""
Forecast-driven baselines: ARIMA and LSTM predict each stock's next-day
return; the forecasts are turned into positions by simple, standard rules
and pushed through the *same* environments (costs, timing) as the RL agents.

A forecast frame is a DataFrame (dates × tickers) whose row d holds the
forecast of the return on the trading day *after* d, made at the close of d.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data import MarketData
from ..runner import run_portfolio, run_trading, set_seed


def full_returns(splits: dict[str, MarketData]) -> pd.DataFrame:
    """Stitch train/val/test into one non-overlapping return series."""
    parts = []
    for name in ("train", "val", "test"):
        d = splits[name]
        s = d.trade_start
        parts.append(pd.DataFrame(d.returns[s:], index=d.dates[s:], columns=d.tickers))
    return pd.concat(parts)


# ----------------------------------------------------------------------
# ARIMA
# ----------------------------------------------------------------------
def arima_forecasts(splits: dict, orders: list, verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """Per-stock ARMA(p,0,q) on daily returns (×100).

    The order is chosen by AIC on TRAIN only and the parameters are fitted on
    TRAIN only; ``results.apply`` then runs the Kalman filter over the whole
    series with those frozen parameters, giving genuine one-step-ahead
    forecasts for val/test without refitting on future data.
    """
    from statsmodels.tsa.arima.model import ARIMA

    rets = full_returns(splits) * 100.0
    train_end = splits["train"].dates[-1]
    preds, chosen = {}, {}
    for tkr in rets.columns:
        y_tr = rets.loc[:train_end, tkr].values[1:]  # drop the r_0 = 0 placeholder
        best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in orders:
                try:
                    res = ARIMA(y_tr, order=tuple(order), trend="c").fit()
                except Exception:  # noqa: BLE001 — skip orders that fail to converge
                    continue
                if best is None or res.aic < best[0]:
                    best = (res.aic, tuple(order), res)
            full = best[2].apply(rets[tkr].values)
        one_step = full.predict()                # one_step[i] = E[y_i | y_<i]
        preds[tkr] = np.append(one_step[1:], np.nan) / 100.0   # row t → forecast of t+1
        chosen[tkr] = best[1]
        if verbose:
            print(f"  ARIMA {tkr:<14} order={best[1]}  AIC={best[0]:.1f}", flush=True)
    return pd.DataFrame(preds, index=rets.index), chosen


# ----------------------------------------------------------------------
# LSTM
# ----------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


def _windows(d: MarketData, window: int, only_trading: bool):
    """(samples, window, F) inputs and next-day return targets for every stock."""
    lo = max(window - 1, d.trade_start - 1) if only_trading else window - 1
    ts = np.arange(lo, len(d) - 1)
    X = np.stack([d.features[t - window + 1:t + 1] for t in ts])          # (S, W, N, F)
    X = X.transpose(0, 2, 1, 3).reshape(-1, window, d.n_features)          # (S·N, W, F)
    y = d.returns[ts + 1].reshape(-1) * 100.0
    return X.astype(np.float32), y.astype(np.float32), ts


def lstm_forecasts(splits: dict, cfg: dict, seed: int = 0, verbose: bool = False) -> pd.DataFrame:
    """Pooled LSTM (one model for all stocks), early-stopped on validation MSE."""
    torch.set_num_threads(1)
    set_seed(seed)
    lc, window = cfg["baselines"]["lstm"], int(cfg["env"]["window"])
    Xtr, ytr, _ = _windows(splits["train"], window, False)
    Xva, yva, _ = _windows(splits["val"], window, True)
    model = LSTMForecaster(Xtr.shape[2], lc["hidden"], lc["layers"], lc["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=lc["lr"])
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    gen = torch.Generator().manual_seed(seed)

    best, best_state, bad = np.inf, None, 0
    for epoch in range(lc["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=gen)
        for i in range(0, len(perm), lc["batch_size"]):
            idx = perm[i:i + lc["batch_size"]]
            loss = nn.functional.mse_loss(model(Xtr_t[idx]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(Xva_t), yva_t).item()
        if verbose:
            print(f"  LSTM seed={seed} epoch {epoch + 1:>2}  val MSE {val_loss:.4f}", flush=True)
        if val_loss < best - 1e-5:
            best, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= lc["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()

    frames = []
    for name in ("train", "val", "test"):
        d = splits[name]
        X, _, ts = _windows(d, window, name != "train")
        with torch.no_grad():
            p = model(torch.from_numpy(X)).numpy().reshape(len(ts), d.n_assets) / 100.0
        frames.append(pd.DataFrame(p, index=d.dates[ts], columns=d.tickers))
    return pd.concat(frames)


# ----------------------------------------------------------------------
# Forecast → positions → backtest
# ----------------------------------------------------------------------
def backtest_forecasts(forecasts: pd.DataFrame, data: MarketData, cfg: dict, task: str) -> dict:
    thr = float(cfg["baselines"]["threshold"])
    fc = forecasts.reindex(data.dates).fillna(0.0).values           # aligned to split rows

    if task == "trading":
        return run_trading(data, cfg, lambda obs, t: (fc[t] > thr).astype(float))

    k = int(cfg["baselines"]["top_k"])

    def policy(obs, t):
        w = np.zeros(data.n_assets + 1)
        picks = [i for i in np.argsort(-fc[t])[:k] if fc[t, i] > thr]
        w[[i + 1 for i in picks]] = 1.0 / k
        w[0] = 1.0 - w.sum()                                         # unfilled slots stay in cash
        return w

    return run_portfolio(data, cfg, policy)
