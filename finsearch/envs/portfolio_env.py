"""
Multi-stock portfolio environment (continuous actions) — used by DDPG.

State   : last `portfolio.window` days of features for all N stocks (flattened)
          + current portfolio weights (cash + N)        → dim = window·N·F + N + 1
Action  : target weights on the simplex (cash + N stocks), long-only, fully invested
Reward  : shaped net log-return of the whole portfolio
"""

from __future__ import annotations

import numpy as np

from ..data import MarketData
from .base import BaseMarketEnv


class PortfolioEnv(BaseMarketEnv):
    def __init__(self, data: MarketData, cfg: dict, random_start: bool = False,
                 seed: int | None = None):
        self.n_weights = data.n_assets + 1
        super().__init__(data, cfg, window=int(cfg["portfolio"]["window"]),
                         random_start=random_start, seed=seed)

    @property
    def action_dim(self) -> int:
        return self.n_weights

    @property
    def obs_dim(self) -> int:
        return self.window * self.data.n_assets * self.data.n_features + self.n_weights

    def _asset_returns(self, t: int) -> np.ndarray:
        return self.data.returns[t]

    def _observation(self) -> np.ndarray:
        f = self.data.features[self.t - self.window + 1:self.t + 1]
        return np.concatenate([f.ravel(), self.weights]).astype(np.float32)

    def step(self, weights: np.ndarray):
        w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.eye(self.n_weights)[0]
        return self.step_weights(w)
