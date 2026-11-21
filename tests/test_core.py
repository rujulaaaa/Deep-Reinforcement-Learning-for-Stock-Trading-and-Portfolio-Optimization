"""Unit tests for the pieces whose bugs would silently inflate results:
accounting, look-ahead leakage, reward shaping, metrics and the agents' plumbing.
They run on synthetic data — no download needed.
"""

import numpy as np
import pandas as pd
import pytest

from finsearch.agents import DDPGAgent, DQNAgent
from finsearch.config import load_config
from finsearch.data import MarketData, stock_features
from finsearch.envs import PortfolioEnv, RewardShaper, TradingEnv, rebalance_step
from finsearch.metrics import performance


@pytest.fixture
def cfg():
    return load_config(overrides=["env.reward=log_return", "env.reward_scale=1.0"])


def synthetic(T=300, N=3, F=11, seed=0, trade_start=20):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, (T, N))
    rets[0] = 0
    close = 100 * np.cumprod(1 + rets, axis=0)
    return MarketData(pd.bdate_range("2020-01-01", periods=T), [f"S{i}" for i in range(N)], close,
                      rets, rng.normal(size=(T, N, F)).astype(np.float32), close.mean(1), trade_start)


# ---------------------------------------------------------------- accounting
def test_rebalance_step_costs_and_drift():
    prev = np.array([1.0, 0.0, 0.0])
    tgt = np.array([0.0, 0.5, 0.5])
    g, drifted, turnover = rebalance_step(prev, tgt, np.array([0.10, -0.10]), cost=0.001)
    assert turnover == pytest.approx(1.0)
    assert g == pytest.approx((1 - 0.001) * 1.0)
    assert drifted.sum() == pytest.approx(1.0)
    assert drifted[1] == pytest.approx(0.55)


def test_buy_and_hold_matches_price_path(cfg):
    d = synthetic()
    env = TradingEnv(d, cfg, asset=1)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step_exposure(1.0)
    curve = env.equity_curve().values
    px = d.close[env.first_t:, 1]
    expected = (1 - cfg["env"]["cost"]) * px[-1] / px[0]
    assert curve[-1] / curve[0] == pytest.approx(expected, rel=1e-9)


def test_eval_episode_covers_exactly_the_split(cfg):
    d = synthetic(trade_start=50)
    env = TradingEnv(d, cfg, asset=0)
    env.reset()
    assert env.dates[0] == d.dates[49]           # decide at last pre-split close
    done = False
    while not done:
        _, _, done, _ = env.step(0)
    assert env.dates[-1] == d.dates[-1]
    assert len(env.values) - 1 == len(d) - 50   # one return per split day


def test_reward_equals_net_log_return(cfg):
    d = synthetic()
    env = PortfolioEnv(d, cfg)
    env.reset()
    w = np.array([0.2, 0.3, 0.3, 0.2])
    v0 = env.value
    _, r, _, info = env.step(w)
    assert r == pytest.approx(np.log(env.value / v0))
    assert r == pytest.approx(info["log_return"])


# ---------------------------------------------------------------- leakage
def test_observation_has_no_lookahead(cfg):
    """Changing data *after* day t must not change the observation at day t."""
    d1 = synthetic()
    d2 = synthetic()
    t = 100
    d2.features[t + 1:] += 99.0
    e1, e2 = TradingEnv(d1, cfg, asset=0), TradingEnv(d2, cfg, asset=0)
    np.testing.assert_array_equal(e1.reset(start=t), e2.reset(start=t))


def test_features_are_causal():
    rng = np.random.default_rng(1)
    n = 200
    c = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                       "Volume": rng.integers(1e5, 1e6, n).astype(float)})
    f1 = stock_features(df)
    df2 = df.copy()
    df2.iloc[150:] *= 3.0
    f2 = stock_features(df2)
    pd.testing.assert_frame_equal(f1.iloc[:150], f2.iloc[:150])


# ---------------------------------------------------------------- rewards / metrics
@pytest.mark.parametrize("kind", RewardShaper.KINDS)
def test_reward_shapers_finite(kind, cfg):
    cfg["env"]["reward"] = kind
    s = RewardShaper(cfg["env"])
    rs = [s(x) for x in np.random.default_rng(0).normal(0, 0.01, 500)]
    assert np.all(np.isfinite(rs))


def test_risk_adjusted_penalises_variance(cfg):
    cfg["env"].update(reward="risk_adjusted", reward_scale=100.0, risk_lambda=0.1)
    s = RewardShaper(cfg["env"])
    assert s(0.02) < 100 * 0.02


def test_metrics_basic():
    v = np.array([100, 110, 99, 120.0])
    m = performance(v)
    assert m["cum_return"] == pytest.approx(0.2)
    assert m["max_drawdown"] == pytest.approx(0.1)


# ---------------------------------------------------------------- agents
def test_dqn_learns_and_acts(cfg):
    d = synthetic()
    env = TradingEnv(d, cfg, random_start=True, seed=0)
    c = dict(cfg["dqn"], learning_starts=64, batch_size=32)
    agent = DQNAgent(env.obs_dim, env.n_actions, c)
    obs = env.reset()
    for _ in range(200):
        a = agent.act(obs, explore=True)
        nxt, r, done, _ = env.step(a)
        agent.observe(obs, a, r, nxt, False)
        obs = env.reset() if done else nxt
    assert agent.updates > 0
    assert 0 <= agent.act(obs) < env.n_actions


def test_ddpg_actions_on_simplex(cfg):
    d = synthetic()
    env = PortfolioEnv(d, cfg, random_start=True, seed=0)
    c = dict(cfg["ddpg"], learning_starts=64, batch_size=32, train_freq=1)
    agent = DDPGAgent(env.obs_dim, env.action_dim, c)
    obs = env.reset()
    for _ in range(150):
        a = agent.act(obs, explore=True)
        assert a.min() >= 0 and a.sum() == pytest.approx(1.0, abs=1e-5)
        nxt, r, done, _ = env.step(a)
        agent.observe(obs, a, r, nxt, False)
        obs = env.reset() if done else nxt
    assert agent.steps == 150
