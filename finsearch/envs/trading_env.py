"""
Single-stock trading environment (discrete actions) — used by DQN.

State   : last `window` days of the stock's F normalised features (flattened)
          + current (drifted) exposure                → dim = window·F + 1
Action  : index into ``trading.positions``, e.g. {0, 0.5, 1} = flat / half / fully long
Reward  : shaped net log-return (see rewards.py)

During training each episode samples a random stock from the universe and a
random one-year segment, so a single policy learns from all 20 stocks
(≈ 20× more experience than a per-stock agent, and less over-fitting).
"""

from __future__ import annotations

import numpy as np

from ..data import MarketData
from .base import BaseMarketEnv


class TradingEnv(BaseMarketEnv):
    n_weights = 2  # [cash, stock]

    def __init__(self, data: MarketData, cfg: dict, asset: int | None = None,
                 random_start: bool = False, seed: int | None = None):
        super().__init__(data, cfg, window=int(cfg["env"]["window"]),
                         random_start=random_start, seed=seed)
        self.positions = np.asarray(cfg["trading"]["positions"], dtype=np.float64)
        self.fixed_asset = asset
        self.asset = asset if asset is not None else 0

    @property
    def n_actions(self) -> int:
        return len(self.positions)

    @property
    def obs_dim(self) -> int:
        return self.window * self.data.n_features + 1

    def reset(self, start: int | None = None, asset: int | None = None) -> np.ndarray:
        if asset is not None:
            self.asset = asset
        elif self.fixed_asset is None:
            self.asset = int(self.rng.integers(self.data.n_assets))
        return super().reset(start)

    def _asset_returns(self, t: int) -> np.ndarray:
        return self.data.returns[t, self.asset:self.asset + 1]

    def _observation(self) -> np.ndarray:
        f = self.data.features[self.t - self.window + 1:self.t + 1, self.asset, :]
        return np.concatenate([f.ravel(), [self.weights[1]]]).astype(np.float32)

    def step(self, action: int):
        x = self.positions[int(action)]
        return self.step_weights(np.array([1.0 - x, x]))

    def step_exposure(self, x: float):
        """Used by baselines: trade to an arbitrary exposure in [0, 1]."""
        return self.step_weights(np.array([1.0 - x, x]))
