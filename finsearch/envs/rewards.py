"""
Reward shaping.

The raw per-step signal is the portfolio log-return net of transaction costs,
``r_t = log(V_{t+1} / V_t)``.  Pure log-return rewards make an agent
risk-neutral, so we offer shaped alternatives that are tuned as
hyper-parameters (``env.reward``):

log_return     s·r_t
risk_adjusted  s·r_t − λ·(s·r_t)²            per-step mean–variance utility
drawdown       s·r_t − λ_dd·s·max(0, ΔDD_t)  penalise *new* drawdown only
dsr            differential Sharpe ratio (Moody & Saffell, 2001) — the
               first-order change of an EMA Sharpe ratio caused by r_t
"""

from __future__ import annotations

import numpy as np


class RewardShaper:
    KINDS = ("log_return", "risk_adjusted", "drawdown", "dsr")

    def __init__(self, env_cfg: dict):
        self.kind = env_cfg.get("reward", "log_return")
        if self.kind not in self.KINDS:
            raise ValueError(f"unknown reward '{self.kind}', choose from {self.KINDS}")
        self.scale = float(env_cfg.get("reward_scale", 100.0))
        self.lam = float(env_cfg.get("risk_lambda", 0.1))
        self.dd_lam = float(env_cfg.get("dd_lambda", 0.5))
        self.eta = float(env_cfg.get("dsr_eta", 0.01))
        self.reset()

    def reset(self) -> None:
        self.peak = 1.0
        self.value = 1.0
        self.dd = 0.0
        self.A = 0.0   # EMA of returns         (DSR)
        self.B = 0.0   # EMA of squared returns (DSR)

    def __call__(self, log_ret: float) -> float:
        self.value *= np.exp(log_ret)
        self.peak = max(self.peak, self.value)
        s = self.scale

        if self.kind == "log_return":
            return s * log_ret
        if self.kind == "risk_adjusted":
            return s * log_ret - self.lam * (s * log_ret) ** 2
        if self.kind == "drawdown":
            dd = 1.0 - self.value / self.peak
            penalty = max(0.0, dd - self.dd)
            self.dd = dd
            return s * log_ret - self.dd_lam * s * penalty
        # Differential Sharpe ratio
        dA, dB = log_ret - self.A, log_ret ** 2 - self.B
        var = self.B - self.A ** 2
        d = 0.0 if var <= 1e-12 else (self.B * dA - 0.5 * self.A * dB) / var ** 1.5
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(np.clip(d, -10.0, 10.0))
