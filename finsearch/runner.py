"""
Training / evaluation loops shared by all scripts.

Two tasks:
  "trading"   — DQN, one policy trading each of the 20 stocks independently
                (equal capital per stock); headline curve = their combination
  "portfolio" — DDPG allocating one pot of capital across the 20 stocks + cash

Model selection: during training the policy is evaluated on the VALIDATION
split every ``train.eval_every`` steps and the best-Sharpe checkpoint is kept.
The test split is only ever used by ``evaluate`` after training finishes.
"""

from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import torch

from .agents import DDPGAgent, DQNAgent
from .data import MarketData
from .envs import PortfolioEnv, TradingEnv
from .metrics import performance

TASK_OF = {"dqn": "trading", "ddpg": "portfolio"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# Evaluation of arbitrary policies on a split
# ----------------------------------------------------------------------
def run_trading(data: MarketData, cfg: dict, policy) -> dict:
    """Trade every stock with ``policy(obs_batch, t) -> exposures (N,)``.

    Stocks are stepped in lock-step so the policy sees a batch of N
    observations per day (one forward pass instead of N).
    """
    envs = [TradingEnv(data, cfg, asset=i) for i in range(data.n_assets)]
    obs = np.stack([e.reset() for e in envs])
    done = False
    while not done:
        x = policy(obs, envs[0].t)
        out = [e.step_exposure(float(xi)) for e, xi in zip(envs, x)]
        obs = np.stack([o[0] for o in out])
        done = out[0][2]
    curves = pd.concat([e.equity_curve() for e in envs], axis=1, keys=data.tickers)
    combined = (curves / curves.iloc[0]).mean(axis=1) * cfg["env"]["initial_cash"]
    per_asset = pd.DataFrame({t: performance(curves[t].values, e.turnover)
                              for t, e in zip(data.tickers, envs)}).T
    turnover = sum(e.turnover for e in envs) / len(envs)
    return {"metrics": performance(combined.values, turnover), "curve": combined,
            "per_asset": per_asset}


def run_portfolio(data: MarketData, cfg: dict, policy) -> dict:
    """Allocate with ``policy(obs, t) -> weights (N+1,)``."""
    env = PortfolioEnv(data, cfg)
    obs, done = env.reset(), False
    while not done:
        obs, _, done, _ = env.step(policy(obs, env.t))
    curve = env.equity_curve()
    return {"metrics": performance(curve.values, env.turnover), "curve": curve}


def evaluate_agent(agent, task: str, data: MarketData, cfg: dict) -> dict:
    if task == "trading":
        positions = np.asarray(cfg["trading"]["positions"])

        @torch.no_grad()
        def policy(obs, t):
            q = agent.q(torch.as_tensor(obs, device=agent.device))
            return positions[q.argmax(dim=1).cpu().numpy()]

        return run_trading(data, cfg, policy)
    return run_portfolio(data, cfg, lambda obs, t: agent.act(obs, explore=False))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def build_agent(algo: str, splits: dict, cfg: dict, seed: int):
    task = TASK_OF[algo]
    if task == "trading":
        env = TradingEnv(splits["train"], cfg, random_start=True, seed=seed)
        agent = DQNAgent(env.obs_dim, env.n_actions, cfg["dqn"], cfg["train"]["device"], seed)
    else:
        env = PortfolioEnv(splits["train"], cfg, random_start=True, seed=seed)
        agent = DDPGAgent(env.obs_dim, env.action_dim, cfg["ddpg"], cfg["train"]["device"], seed)
    return env, agent


def train_agent(algo: str, splits: dict, cfg: dict, seed: int = 0, verbose: bool = False):
    """Train one agent; returns (agent loaded with best-val weights, history DataFrame)."""
    set_seed(seed)
    task = TASK_OF[algo]
    env, agent = build_agent(algo, splits, cfg, seed)
    total = int(cfg[algo]["total_steps"])
    eval_every = int(cfg["train"]["eval_every"])

    best_sharpe, best_state = -np.inf, agent.state_dict()
    history, ep_reward, ep_rewards, losses = [], 0.0, [], []
    obs = env.reset()
    t0 = time.time()
    for step in range(1, total + 1):
        action = agent.act(obs, explore=True)
        next_obs, reward, done, _ = env.step(action)
        # Episode ends are time-limit truncations, not true terminals → keep bootstrapping.
        stats = agent.observe(obs, action, reward, next_obs, terminal=False)
        if stats:
            losses.append(stats)
        ep_reward += reward
        obs = next_obs
        if done:
            ep_rewards.append(ep_reward)
            ep_reward, obs = 0.0, env.reset()

        if step % eval_every == 0 or step == total:
            val = evaluate_agent(agent, task, splits["val"], cfg)["metrics"]
            row = {"step": step, "val_sharpe": val["sharpe"], "val_ann_return": val["ann_return"],
                   "val_max_drawdown": val["max_drawdown"],
                   "train_ep_reward": float(np.mean(ep_rewards[-20:])) if ep_rewards else np.nan,
                   "explore": agent.epsilon() if algo == "dqn" else agent.sigma(),
                   "elapsed_s": time.time() - t0}
            if losses:
                row.update({k: float(np.mean([l[k] for l in losses[-1000:]])) for k in losses[-1].keys()})
            history.append(row)
            if val["sharpe"] > best_sharpe:
                best_sharpe, best_state = val["sharpe"], agent.state_dict()
            if verbose:
                print(f"[{algo} seed={seed}] step {step:>7}/{total}  val Sharpe {val['sharpe']:+.3f}  "
                      f"ann.ret {val['ann_return']:+.2%}  best {best_sharpe:+.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

    agent.load_state_dict(best_state)
    return agent, pd.DataFrame(history)


def run_experiment(algo: str, cfg: dict, splits: dict, seed: int, eval_splits=("val", "test"),
                   verbose: bool = False) -> dict:
    """Train + evaluate. Returns metrics per split, equity curves, history and weights."""
    torch.set_num_threads(1)
    agent, history = train_agent(algo, splits, cfg, seed, verbose)
    out = {"algo": algo, "seed": seed, "history": history, "state_dict": agent.state_dict()}
    for split in eval_splits:
        res = evaluate_agent(agent, TASK_OF[algo], splits[split], cfg)
        out[split] = res["metrics"]
        out[f"{split}_curve"] = res["curve"]
    return out
