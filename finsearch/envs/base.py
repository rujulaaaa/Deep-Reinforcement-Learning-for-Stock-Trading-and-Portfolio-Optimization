"""
Shared portfolio accounting used by *every* strategy — RL agents and
baselines alike — so that all methods face identical costs and timing.

Timing convention
-----------------
At the close of day t the strategy observes features up to t and chooses
target weights w_t (index 0 = cash).  It pays ``cost × turnover`` on the
risky legs, then earns the close-to-close return of day t+1:

    V_{t+1} = V_t · (1 − c·Σ_i |w_t,i − w̃_t,i|) · (1 + Σ_i w_t,i · r_{t+1,i})

where w̃_t are yesterday's weights after drifting with prices.  Cash earns 0.
"""

from __future__ import annotations

import numpy as np

from ..data import MarketData
from .rewards import RewardShaper


def rebalance_step(prev_w: np.ndarray, target_w: np.ndarray, r_next: np.ndarray, cost: float):
    """Apply one rebalance + one day of returns. Returns (growth, drifted_w, turnover)."""
    turnover = float(np.abs(target_w[1:] - prev_w[1:]).sum())
    gross = 1.0 + float(target_w[1:] @ r_next)
    growth = (1.0 - cost * turnover) * gross
    drifted = target_w * np.concatenate(([1.0], 1.0 + r_next)) / max(gross, 1e-12)
    return growth, drifted, turnover


class BaseMarketEnv:
    """Minimal gym-style environment over a :class:`MarketData` split.

    Subclasses define the observation and how actions map to weights.
    ``random_start=True`` samples ``episode_length``-day segments (training);
    otherwise an episode covers the whole tradable part of the split.
    """

    def __init__(self, data: MarketData, cfg: dict, window: int, random_start: bool = False,
                 seed: int | None = None):
        self.data = data
        self.cfg = cfg
        self.window = window
        self.cost = float(cfg["env"]["cost"])
        self.initial_cash = float(cfg["env"]["initial_cash"])
        self.episode_length = int(cfg["env"]["episode_length"])
        self.random_start = random_start
        self.rng = np.random.default_rng(seed)
        self.shaper = RewardShaper(cfg["env"])
        # First decision row: needs `window` rows of history; on val/test we
        # decide at the last pre-split close so returns cover the full split.
        self.first_t = max(window - 1, data.trade_start - 1)
        self.last_t = len(data) - 2  # last decision row (earns return at T-1)

    # -- to be specified by subclasses --------------------------------
    n_weights: int

    def _observation(self) -> np.ndarray:
        raise NotImplementedError

    # -----------------------------------------------------------------
    def reset(self, start: int | None = None) -> np.ndarray:
        if start is None and self.random_start:
            hi = max(self.first_t, self.last_t - self.episode_length)
            start = int(self.rng.integers(self.first_t, hi + 1))
        self.t = self.first_t if start is None else start
        # end_t = last decision row of the episode
        self.end_t = (min(self.last_t, self.t + self.episode_length - 1) if self.random_start
                      else self.last_t)
        self.weights = np.zeros(self.n_weights)
        self.weights[0] = 1.0
        self.value = self.initial_cash
        self.values = [self.value]
        self.dates = [self.data.dates[self.t]]
        self.turnover = 0.0
        self.shaper.reset()
        return self._observation()

    def _asset_returns(self, t: int) -> np.ndarray:
        raise NotImplementedError

    def step_weights(self, target_w: np.ndarray):
        target_w = np.asarray(target_w, dtype=np.float64)
        growth, self.weights, turnover = rebalance_step(
            self.weights, target_w, self._asset_returns(self.t + 1), self.cost)
        log_ret = float(np.log(max(growth, 1e-12)))
        reward = self.shaper(log_ret)
        self.value *= growth
        self.turnover += turnover
        self.t += 1
        self.values.append(self.value)
        self.dates.append(self.data.dates[self.t])
        done = self.t > self.end_t
        info = {"log_return": log_ret, "value": self.value, "turnover": turnover}
        return self._observation(), reward, done, info

    def equity_curve(self):
        import pandas as pd
        return pd.Series(self.values, index=pd.DatetimeIndex(self.dates), name="value")
