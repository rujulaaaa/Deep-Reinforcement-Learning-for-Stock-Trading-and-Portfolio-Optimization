"""
NIFTY50 data pipeline: download → clean → engineer features → split → normalise.

Everything downstream consumes a :class:`MarketData` object holding aligned
arrays over a common trading calendar:

    close    (T, N)      adjusted close prices
    returns  (T, N)      simple returns r_t = P_t / P_{t-1} - 1  (r_0 = 0)
    features (T, N, F)   technical features known at the close of day t

Leakage rules followed here
---------------------------
* Every feature at row t uses only prices/volumes up to and including day t.
* Normalisation statistics (mean / std) are fitted on the TRAIN rows only.
* The split is chronological; val/test slices carry ``lookback`` rows of
  history in front of them purely as observation context — no trading,
  reward or fitting happens on those rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve

FEATURES = [
    "ret_1", "ret_5", "ret_20",     # momentum at three horizons (log returns)
    "vol_20",                       # realised volatility
    "rsi_14",                       # relative strength index (centred)
    "macd_hist",                    # MACD histogram / price
    "bb_pctb",                      # Bollinger %B
    "sma_20_gap", "sma_50_gap",     # distance to moving averages
    "atr_14",                       # average true range / price
    "vol_z",                        # volume z-score
]
WARMUP = 60  # rows dropped so every rolling indicator is fully formed


# ----------------------------------------------------------------------
# Download / cache
# ----------------------------------------------------------------------
def download(cfg: dict, force: bool = False) -> pd.DataFrame:
    """Download OHLCV for all tickers + index, cached as one CSV."""
    dc = cfg["data"]
    raw_dir = resolve(dc["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "nifty50_ohlcv.csv"
    if path.exists() and not force:
        return pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)

    import yfinance as yf

    symbols = list(dc["tickers"]) + [dc["index"]]
    df = yf.download(symbols, start=dc["start"], end=dc["end"], auto_adjust=True,
                     progress=False, group_by="ticker", threads=True)
    missing = [s for s in symbols if s not in df.columns.get_level_values(0)
               or df[s]["Close"].dropna().empty]
    if missing:
        raise RuntimeError(f"No data downloaded for {missing}; Yahoo may be rate-limiting, retry later.")
    df.to_csv(path)
    return df


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def stock_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Technical features for one stock; every value at row t uses data ≤ t."""
    c, h, l, v = ohlcv["Close"], ohlcv["High"], ohlcv["Low"], ohlcv["Volume"]
    logc = np.log(c)
    f = pd.DataFrame(index=ohlcv.index)
    f["ret_1"] = logc.diff(1)
    f["ret_5"] = logc.diff(5)
    f["ret_20"] = logc.diff(20)
    f["vol_20"] = f["ret_1"].rolling(20).std()
    f["rsi_14"] = _rsi(c) / 100.0 - 0.5
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (sma20 - 2 * std20)) / (4 * std20)
    f["sma_20_gap"] = c / sma20 - 1
    f["sma_50_gap"] = c / c.rolling(50).mean() - 1
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean() / c
    lv = np.log1p(v)
    f["vol_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    return f[FEATURES]


# ----------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------
@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    tickers: list
    close: np.ndarray          # (T, N)
    returns: np.ndarray        # (T, N)
    features: np.ndarray       # (T, N, F) — normalised
    index_close: np.ndarray    # (T,)
    trade_start: int = 0       # first row on which trading is allowed
    meta: dict = field(default_factory=dict)

    @property
    def n_assets(self) -> int:
        return self.close.shape[1]

    @property
    def n_features(self) -> int:
        return self.features.shape[2]

    def __len__(self) -> int:
        return len(self.dates)

    def n_points(self) -> int:
        """Number of raw OHLCV observations parsed (rows × assets × 5 fields)."""
        return int(self.close.size * 5)

    def subset(self, tickers: list) -> "MarketData":
        idx = [self.tickers.index(t) for t in tickers]
        return MarketData(self.dates, list(tickers), self.close[:, idx], self.returns[:, idx],
                          self.features[:, idx], self.index_close, self.trade_start, self.meta)


def build_dataset(cfg: dict, force_download: bool = False) -> dict[str, MarketData]:
    """Return ``{"train", "val", "test"}`` MarketData splits (features normalised)."""
    dc = cfg["data"]
    raw = download(cfg, force=force_download)
    tickers = list(dc["tickers"])

    # Common trading calendar: NSE dates where the index traded; forward-fill
    # isolated gaps (circuit halts / missing prints), then drop the warm-up.
    cal = raw[dc["index"]]["Close"].dropna().index
    closes, feats = [], []
    for t in tickers:
        o = raw[t].reindex(cal).ffill()
        closes.append(o["Close"])
        feats.append(stock_features(o))
    close = pd.concat(closes, axis=1, keys=tickers)
    feat = np.stack([f.values for f in feats], axis=1)            # (T, N, F)
    index_close = raw[dc["index"]]["Close"].reindex(cal).ffill()

    keep = np.arange(len(cal)) >= WARMUP
    keep &= ~np.isnan(feat).any(axis=(1, 2)) & ~close.isna().any(axis=1).values
    dates = cal[keep]
    close_v = close.values[keep]
    feat = feat[keep]
    index_v = index_close.values[keep]
    returns = np.vstack([np.zeros((1, len(tickers))), close_v[1:] / close_v[:-1] - 1])

    # Chronological split boundaries
    train_end = np.searchsorted(dates, pd.Timestamp(dc["train_end"]), side="right")
    val_end = np.searchsorted(dates, pd.Timestamp(dc["val_end"]), side="right")

    # Normalise with TRAIN statistics only (pooled across stocks per feature)
    mu = feat[:train_end].reshape(-1, feat.shape[2]).mean(0)
    sd = feat[:train_end].reshape(-1, feat.shape[2]).std(0) + 1e-8
    feat = np.clip((feat - mu) / sd, -dc["clip"], dc["clip"]).astype(np.float32)

    lookback = max(cfg["env"]["window"], cfg["portfolio"]["window"], 60)

    def make(lo: int, hi: int) -> MarketData:
        ctx = max(0, lo - lookback)
        return MarketData(dates[ctx:hi], tickers, close_v[ctx:hi], returns[ctx:hi],
                          feat[ctx:hi], index_v[ctx:hi], trade_start=lo - ctx,
                          meta={"feature_mean": mu, "feature_std": sd})

    return {"train": make(0, train_end), "val": make(train_end, val_end),
            "test": make(val_end, len(dates))}


def describe(splits: dict[str, MarketData]) -> pd.DataFrame:
    rows = []
    for name, d in splits.items():
        tr = d.dates[d.trade_start:]
        rows.append({"split": name, "start": tr[0].date(), "end": tr[-1].date(),
                     "trading_days": len(tr), "assets": d.n_assets,
                     "ohlcv_points": len(tr) * d.n_assets * 5})
    return pd.DataFrame(rows).set_index("split")
