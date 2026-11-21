# Results summary

All numbers are on the held-out **test split** unless stated. Mean ± std across runs; deterministic methods have one run. Costs of 0.10% per unit turnover are charged to every method.

## Single-stock trading (DQN, 20 stocks, equal capital)

| Method | Runs | Sharpe | Ann. return | Cum. return | Max DD | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| DQN | 30 | -0.064 ± 0.347 | -0.36% ± 2.41% | -1.05% ± 9.11% | 16.13% | -0.062 |
| ARIMA | 1 | +0.010 | -0.39% | -1.44% | 20.58% | +0.010 |
| LSTM | 30 | +0.626 ± 0.146 | +6.67% ± 1.86% | +27.54% ± 8.04% | 15.41% | +0.621 |
| Buy&Hold | 1 | +0.725 | +8.26% | +34.61% | 15.21% | +0.716 |
| NIFTY 50 | 1 | +0.573 | +6.66% | +27.33% | 15.77% | +0.558 |

**DQN vs baselines (test)** — relative change of the DQN mean, and the share of the 30 DQN seeds that beat the baseline mean:

| Baseline | Metric | DQN mean | Baseline mean | Abs. Δ | Relative Δ | Seeds beating |
|---|---|---:|---:|---:|---:|---:|
| ARIMA | sharpe | -0.064 | +0.010 | -0.074 | n/a | 43% |
| ARIMA | ann_return | -0.36% | -0.39% | +0.02% | n/a | 47% |
| ARIMA | cum_return | -1.05% | -1.44% | +0.38% | +26.5% | 47% |
| LSTM | sharpe | -0.064 | +0.626 | -0.690 | -110.2% | 0% |
| LSTM | ann_return | -0.36% | +6.67% | -7.03% | -105.4% | 0% |
| LSTM | cum_return | -1.05% | +27.54% | -28.60% | -103.8% | 0% |
| Buy&Hold | sharpe | -0.064 | +0.725 | -0.788 | -108.8% | 0% |
| Buy&Hold | ann_return | -0.36% | +8.26% | -8.62% | -104.4% | 0% |
| Buy&Hold | cum_return | -1.05% | +34.61% | -35.67% | -103.0% | 0% |

## Portfolio allocation (DDPG, 20 stocks + cash)

| Method | Runs | Sharpe | Ann. return | Cum. return | Max DD | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| DDPG | 30 | +0.148 ± 0.468 | +1.93% ± 8.45% | +11.12% ± 34.52% | 30.60% | +0.156 |
| ARIMA | 1 | -1.439 | -19.38% | -55.38% | 56.36% | -1.335 |
| LSTM | 30 | -0.836 ± 0.260 | -12.65% ± 3.35% | -39.30% ± 9.14% | 45.11% | -0.829 |
| Buy&Hold | 1 | +0.725 | +8.26% | +34.61% | 15.21% | +0.716 |
| NIFTY 50 | 1 | +0.573 | +6.66% | +27.33% | 15.77% | +0.558 |

**DDPG vs baselines (test)** — relative change of the DDPG mean, and the share of the 30 DDPG seeds that beat the baseline mean:

| Baseline | Metric | DDPG mean | Baseline mean | Abs. Δ | Relative Δ | Seeds beating |
|---|---|---:|---:|---:|---:|---:|
| ARIMA | sharpe | +0.148 | -1.439 | +1.586 | +110.3% | 100% |
| ARIMA | ann_return | +1.93% | -19.38% | +21.31% | +110.0% | 97% |
| ARIMA | cum_return | +11.12% | -55.38% | +66.50% | +120.1% | 97% |
| LSTM | sharpe | +0.148 | -0.836 | +0.984 | +117.7% | 97% |
| LSTM | ann_return | +1.93% | -12.65% | +14.58% | +115.3% | 97% |
| LSTM | cum_return | +11.12% | -39.30% | +50.42% | +128.3% | 97% |
| Buy&Hold | sharpe | +0.148 | +0.725 | -0.577 | -79.6% | 7% |
| Buy&Hold | ann_return | +1.93% | +8.26% | -6.32% | -76.6% | 10% |
| Buy&Hold | cum_return | +11.12% | +34.61% | -23.49% | -67.9% | 10% |

Total RL test runs: **60**.

DQN tuning: 16 configurations evaluated on the validation split; best = config #13 (val Sharpe +0.700).

DDPG tuning: 16 configurations evaluated on the validation split; best = config #5 (val Sharpe +1.049).

Figures: `results/figures/`.