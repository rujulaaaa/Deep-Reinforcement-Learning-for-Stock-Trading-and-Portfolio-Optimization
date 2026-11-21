"""Aggregate results into tables + figures.

    python scripts/report.py

Reads results/final/*.csv (+ results/tuning/*) and writes
    results/final/summary.md          headline tables, improvement vs baselines
    results/final/summary_<task>.csv  mean/std of every metric per method
    results/figures/*.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import base_parser, get_config, results_dir  # noqa: E402

from finsearch.metrics import summarise  # noqa: E402

# Fixed categorical order (colour follows the method, never its rank)
COLORS = {"RL": "#2a78d6", "ARIMA": "#eb6834", "LSTM": "#1baf7a", "Buy&Hold": "#eda100",
          "NIFTY 50": "#e87ba4"}
TASKS = {"trading": ("DQN", "Single-stock trading (DQN, 20 stocks, equal capital)"),
         "portfolio": ("DDPG", "Portfolio allocation (DDPG, 20 stocks + cash)")}
METRICS = ["sharpe", "ann_return", "cum_return", "max_drawdown", "sortino", "calmar"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def color(method: str) -> str:
    return COLORS.get(method, COLORS["RL"])


def load(out):
    base = pd.read_csv(out / "baselines_runs.csv")
    agents = {}
    for task, (algo, _) in TASKS.items():
        p = out / f"{algo.lower()}_runs.csv"
        if p.exists():
            agents[task] = pd.read_csv(p).assign(task=task)
    return base, agents


def improvement(rl: pd.Series, base: pd.Series, metric: str) -> dict:
    """Relative improvement of the RL mean over a baseline mean, plus seed win-rate.
    Relative change is undefined (reported as NaN) when the baseline is ~0."""
    b = base.mean()
    tiny = 0.05 if metric == "sharpe" else 0.005
    return {"rl_mean": rl.mean(), "baseline_mean": b, "abs_diff": rl.mean() - b,
            "rel_improvement_pct": 100 * (rl.mean() - b) / abs(b) if abs(b) > tiny else np.nan,
            "seeds_beating_baseline_pct": 100 * (rl > b).mean()}


def fmt(m, s=None, pct=False):
    if pct:
        return f"{m:+.2%}" if s is None or np.isnan(s) else f"{m:+.2%} ± {s:.2%}"
    return f"{m:+.3f}" if s is None or np.isnan(s) else f"{m:+.3f} ± {s:.3f}"


def table_md(summary: pd.DataFrame) -> str:
    lines = ["| Method | Runs | Sharpe | Ann. return | Cum. return | Max DD | Sortino |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, r in summary.iterrows():
        lines.append(
            f"| {method} | {int(r['runs'])} | {fmt(r['sharpe_mean'], r['sharpe_std'])} | "
            f"{fmt(r['ann_return_mean'], r['ann_return_std'], True)} | "
            f"{fmt(r['cum_return_mean'], r['cum_return_std'], True)} | "
            f"{r['max_drawdown_mean']:.2%} | {fmt(r['sortino_mean'])} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def fig_equity(out, figs, task, algo, title):
    rl = pd.read_csv(out / f"{algo.lower()}_test_curves.csv", index_col=0, parse_dates=True)
    base = pd.read_csv(out / f"baselines_{task}_test_curves.csv", index_col=0, parse_dates=True)
    rl, base = rl / rl.iloc[0], base / base.iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(rl.index, rl.quantile(0.25, axis=1), rl.quantile(0.75, axis=1),
                    color=color("RL"), alpha=0.18, linewidth=0, label=f"{algo} inter-quartile range")
    ax.plot(rl.index, rl.median(axis=1), color=color("RL"), label=f"{algo} median of {rl.shape[1]} seeds")
    for m in ["ARIMA", "LSTM", "Buy&Hold", "NIFTY 50"]:
        if m in base:
            ax.plot(base.index, base[m], color=color(m), lw=1.6,
                    ls="--" if m == "NIFTY 50" else "-", label=m + (" (mean of seeds)" if m == "LSTM" else ""))
    for line in ax.get_lines():  # direct end labels
        ax.annotate(line.get_label().split(" ")[0], (line.get_xdata()[-1], line.get_ydata()[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_title(f"Test-period growth of ₹1 — {title}", loc="left")
    ax.set_ylabel("Portfolio value (× initial)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / f"equity_{task}.png", dpi=150)
    plt.close(fig)


def fig_sharpe_box(figs, data: dict):
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 4), squeeze=False)
    for ax, (task, df) in zip(axes[0], data.items()):
        order = [m for m in [TASKS[task][0], "ARIMA", "LSTM", "Buy&Hold", "NIFTY 50"] if m in set(df.method)]
        vals = [df.loc[df.method == m, "test_sharpe"].values for m in order]
        bp = ax.boxplot(vals, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "lw": 1.5})
        for patch, m in zip(bp["boxes"], order):
            patch.set(facecolor=color(m), alpha=0.35, edgecolor=color(m))
        for i, (v, m) in enumerate(zip(vals, order), 1):
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
            ax.scatter(np.full(len(v), i) + jitter, v, s=14, color=color(m), zorder=3,
                       edgecolor="#fcfcfb", linewidth=0.6)
        ax.set_xticks(range(1, len(order) + 1), order)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"Test Sharpe per run — {task}", loc="left")
        ax.set_ylabel("Annualised Sharpe ratio")
    fig.tight_layout()
    fig.savefig(figs / "test_sharpe_distribution.png", dpi=150)
    plt.close(fig)


def fig_learning(out, figs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, algo in zip(axes, ["dqn", "ddpg"]):
        p = out / f"{algo}_histories.csv"
        if not p.exists():
            ax.set_visible(False)
            continue
        h = pd.read_csv(p).groupby("step")["val_sharpe"]
        m, s = h.mean(), h.std()
        ax.fill_between(m.index, m - s, m + s, color=color("RL"), alpha=0.18, linewidth=0)
        ax.plot(m.index, m.values, color=color("RL"), marker="o", ms=4)
        ax.set_title(f"{algo.upper()} — validation Sharpe during training (mean ± 1 sd)", loc="left", fontsize=10)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Val. Sharpe")
    fig.tight_layout()
    fig.savefig(figs / "learning_curves.png", dpi=150)
    plt.close(fig)


def fig_sweep(tuning, figs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, algo in zip(axes, ["dqn", "ddpg"]):
        p = tuning / f"{algo}_sweep.csv"
        if not p.exists():
            ax.set_visible(False)
            continue
        s = pd.read_csv(p).sort_values("val_sharpe_mean", ascending=False)
        x = np.arange(len(s))
        ax.bar(x, s.val_sharpe_mean, yerr=s.val_sharpe_std.fillna(0), color=color("RL"),
               alpha=0.85, width=0.7, error_kw={"ecolor": MUTED, "lw": 1})
        ax.bar(x[:1], s.val_sharpe_mean.iloc[:1], color="#1c5cab", width=0.7)
        ax.set_xticks(x, s.config_id.astype(str), fontsize=8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{algo.upper()} sweep — {len(s)} configs ranked by val. Sharpe", loc="left", fontsize=10)
        ax.set_xlabel("Config id (darkest = selected)")
        ax.set_ylabel("Val. Sharpe (mean ± sd)")
    fig.tight_layout()
    fig.savefig(figs / "hyperparameter_sweep.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    cfg = get_config(args)
    out = results_dir(cfg, "final", quick=args.quick)
    figs = results_dir(cfg, "figures", quick=args.quick)
    tuning = results_dir(cfg, "tuning", quick=args.quick)
    base, agents = load(out)

    md = ["# Results summary", "",
          "All numbers are on the held-out **test split** unless stated. Mean ± std across runs; "
          "deterministic methods have one run. Costs of "
          f"{cfg['env']['cost']:.2%} per unit turnover are charged to every method.", ""]
    per_task = {}
    for task, (algo, title) in TASKS.items():
        b = base[base.task == task]
        df = pd.concat([agents[task], b]) if task in agents else b
        per_task[task] = df
        test = df.rename(columns=lambda c: c[5:] if c.startswith("test_") else c)
        summary = summarise(test, cols=METRICS)
        order = [m for m in [algo, "ARIMA", "LSTM", "Buy&Hold", "NIFTY 50"] if m in summary.index]
        summary = summary.loc[order]
        summary.to_csv(out / f"summary_{task}.csv")
        md += [f"## {title}", "", table_md(summary), ""]

        if task in agents:
            rl = agents[task]
            imp_rows = []
            for bname in ["ARIMA", "LSTM", "Buy&Hold"]:
                bb = b[b.method == bname]
                if bb.empty:
                    continue
                for metric in ["sharpe", "ann_return", "cum_return"]:
                    imp = improvement(rl[f"test_{metric}"], bb[f"test_{metric}"], metric)
                    imp_rows.append({"baseline": bname, "metric": metric, **imp})
            imp = pd.DataFrame(imp_rows)
            imp.to_csv(out / f"improvement_{task}.csv", index=False)
            md += [f"**{algo} vs baselines (test)** — relative change of the {algo} mean, and the share "
                   f"of the {len(rl)} {algo} seeds that beat the baseline mean:", "",
                   "| Baseline | Metric | " + algo + " mean | Baseline mean | Abs. Δ | Relative Δ | Seeds beating |",
                   "|---|---|---:|---:|---:|---:|---:|"]
            for _, r in imp.iterrows():
                pct = r.metric != "sharpe"
                md.append(f"| {r.baseline} | {r.metric} | {fmt(r.rl_mean, pct=pct)} | "
                          f"{fmt(r.baseline_mean, pct=pct)} | {fmt(r.abs_diff, pct=pct)} | "
                          f"{'n/a' if np.isnan(r.rel_improvement_pct) else f'{r.rel_improvement_pct:+.1f}%'} | "
                          f"{r.seeds_beating_baseline_pct:.0f}% |")
            md.append("")
            fig_equity(out, figs, task, algo, title)

    total_runs = sum(len(a) for a in agents.values())
    md += [f"Total RL test runs: **{total_runs}**.", ""]
    for algo in ["dqn", "ddpg"]:
        p = tuning / f"{algo}_sweep.csv"
        if p.exists():
            s = pd.read_csv(p)
            md += [f"{algo.upper()} tuning: {len(s)} configurations evaluated on the validation split; "
                   f"best = config #{int(s.iloc[0].config_id)} (val Sharpe {s.iloc[0].val_sharpe_mean:+.3f}).", ""]
    md += ["Figures: `results/figures/`."]
    (out / "summary.md").write_text("\n".join(md))

    fig_sharpe_box(figs, per_task)
    fig_learning(out, figs)
    fig_sweep(tuning, figs)
    print("\n".join(md))
