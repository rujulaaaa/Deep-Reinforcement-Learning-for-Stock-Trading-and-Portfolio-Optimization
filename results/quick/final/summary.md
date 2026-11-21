# Results summary

All numbers are on the held-out **test split** unless stated. Mean ± std across runs; deterministic methods have one run. Costs of 0.10% per unit turnover are charged to every method.

## Single-stock trading (DQN, 20 stocks, equal capital)

| Method | Runs | Sharpe | Ann. return | Cum. return | Max DD | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| DQN | 2 | -0.245 ± 0.143 | -2.02% ± 1.05% | -7.34% ± 3.73% | 18.28% | -0.240 |
| ARIMA | 1 | +0.010 | -0.39% | -1.44% | 20.58% | +0.010 |
| LSTM | 2 | +0.621 ± 0.044 | +6.25% ± 0.47% | +25.49% ± 2.09% | 14.01% | +0.623 |
| Buy&Hold | 1 | +0.725 | +8.26% | +34.61% | 15.21% | +0.716 |
| NIFTY 50 | 1 | +0.573 | +6.66% | +27.33% | 15.77% | +0.558 |

**DQN vs baselines (test)** — relative change of the DQN mean, and the share of the 2 DQN seeds that beat the baseline mean:

| Baseline | Metric | DQN mean | Baseline mean | Abs. Δ | Relative Δ | Seeds beating |
|---|---|---:|---:|---:|---:|---:|
| ARIMA | sharpe | -0.245 | +0.010 | -0.255 | n/a | 0% |
| ARIMA | ann_return | -2.02% | -0.39% | -1.64% | n/a | 0% |
| ARIMA | cum_return | -7.34% | -1.44% | -5.90% | -411.3% | 0% |
| LSTM | sharpe | -0.245 | +0.621 | -0.865 | -139.4% | 0% |
| LSTM | ann_return | -2.02% | +6.25% | -8.27% | -132.4% | 0% |
| LSTM | cum_return | -7.34% | +25.49% | -32.83% | -128.8% | 0% |
| Buy&Hold | sharpe | -0.245 | +0.725 | -0.969 | -133.7% | 0% |
| Buy&Hold | ann_return | -2.02% | +8.26% | -10.28% | -124.5% | 0% |
| Buy&Hold | cum_return | -7.34% | +34.61% | -41.95% | -121.2% | 0% |

## Portfolio allocation (DDPG, 20 stocks + cash)

| Method | Runs | Sharpe | Ann. return | Cum. return | Max DD | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| DDPG | 2 | +0.665 ± 0.015 | +8.31% ± 0.44% | +34.86% ± 2.05% | 19.18% | +0.669 |
| ARIMA | 1 | -1.439 | -19.38% | -55.38% | 56.36% | -1.335 |
| LSTM | 2 | -0.610 ± 0.021 | -9.95% ± 0.10% | -32.47% ± 0.27% | 38.58% | -0.618 |
| Buy&Hold | 1 | +0.725 | +8.26% | +34.61% | 15.21% | +0.716 |
| NIFTY 50 | 1 | +0.573 | +6.66% | +27.33% | 15.77% | +0.558 |

**DDPG vs baselines (test)** — relative change of the DDPG mean, and the share of the 2 DDPG seeds that beat the baseline mean:

| Baseline | Metric | DDPG mean | Baseline mean | Abs. Δ | Relative Δ | Seeds beating |
|---|---|---:|---:|---:|---:|---:|
| ARIMA | sharpe | +0.665 | -1.439 | +2.104 | +146.2% | 100% |
| ARIMA | ann_return | +8.31% | -19.38% | +27.69% | +142.9% | 100% |
| ARIMA | cum_return | +34.86% | -55.38% | +90.24% | +162.9% | 100% |
| LSTM | sharpe | +0.665 | -0.610 | +1.275 | +209.0% | 100% |
| LSTM | ann_return | +8.31% | -9.95% | +18.26% | +183.5% | 100% |
| LSTM | cum_return | +34.86% | -32.47% | +67.33% | +207.4% | 100% |
| Buy&Hold | sharpe | +0.665 | +0.725 | -0.060 | -8.2% | 0% |
| Buy&Hold | ann_return | +8.31% | +8.26% | +0.05% | +0.6% | 50% |
| Buy&Hold | cum_return | +34.86% | +34.61% | +0.25% | +0.7% | 50% |

Total RL test runs: **4**.

DQN tuning: 3 configurations evaluated on the validation split; best = config #2 (val Sharpe +0.383).

DDPG tuning: 3 configurations evaluated on the validation split; best = config #2 (val Sharpe +1.416).

Figures: `results/figures/`.