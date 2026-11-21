"""Random-search hyper-parameter tuning on the VALIDATION split.

    python scripts/tune.py --algo dqn
    python scripts/tune.py --algo ddpg --n-configs 20

Writes results/tuning/<algo>_sweep.csv (every config, mean ± std val Sharpe)
and results/tuning/best_<algo>.yaml (the winner, used by evaluate.py).
"""

import numpy as np
import pandas as pd
import yaml
from common import base_parser, best_params_path, get_config, results_dir, run_pool, worker_splits

from finsearch.config import ROOT, apply_params
from finsearch.runner import run_experiment


def sample_configs(space: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs, seen = [{}], {"{}"}
    while len(configs) < n:
        c = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = repr(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            configs.append(c)
    return configs


def job(args):
    cfg, cid, params, seed = args
    r = run_experiment(cfg["_algo"], apply_params(cfg, params), worker_splits(cfg), seed, ("val",))
    return {"config_id": cid, "seed": seed, **{f"val_{k}": v for k, v in r["val"].items()}}


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--algo", choices=["dqn", "ddpg"], required=True)
    p.add_argument("--n-configs", type=int, default=None)
    args = p.parse_args()
    cfg = get_config(args)
    sweep = yaml.safe_load((ROOT / "configs" / "sweep.yaml").read_text())
    n = args.n_configs or (3 if args.quick else sweep["n_configs"])
    configs = sample_configs(sweep[args.algo], n, sweep["seed"])
    seeds = range(int(cfg["experiment"]["tune_seeds"]))
    cfg["_algo"] = args.algo
    jobs = [(cfg, i, c, 1000 + s) for i, c in enumerate(configs) for s in seeds]
    print(f"Tuning {args.algo}: {len(configs)} configs x {len(seeds)} seeds = {len(jobs)} runs")

    rows = list(run_pool(job, jobs, cfg, f"tune-{args.algo}"))
    runs = pd.DataFrame(rows)
    agg = runs.groupby("config_id").agg(
        val_sharpe_mean=("val_sharpe", "mean"), val_sharpe_std=("val_sharpe", "std"),
        val_ann_return_mean=("val_ann_return", "mean"),
        val_max_drawdown_mean=("val_max_drawdown", "mean"))
    agg["params"] = [yaml.safe_dump(configs[i], default_flow_style=True).strip() for i in agg.index]
    agg = agg.sort_values("val_sharpe_mean", ascending=False)

    out = results_dir(cfg, "tuning", quick=args.quick)
    runs.to_csv(out / f"{args.algo}_sweep_runs.csv", index=False)
    agg.to_csv(out / f"{args.algo}_sweep.csv")
    best = int(agg.index[0])
    best_params_path(cfg, args.algo, args.quick).write_text(yaml.safe_dump({
        "algo": args.algo, "config_id": best,
        "val_sharpe_mean": float(agg.loc[best, "val_sharpe_mean"]),
        "params": configs[best]}, sort_keys=False))
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(agg.head(10).to_string())
    print(f"\nbest config #{best} -> {best_params_path(cfg, args.algo, args.quick)}")
