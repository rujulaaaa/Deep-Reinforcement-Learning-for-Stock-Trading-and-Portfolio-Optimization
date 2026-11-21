"""Final evaluation: train each agent with its tuned hyper-parameters over many
seeds (experiment.n_seeds, default 30) and record val + TEST metrics.

    python scripts/evaluate.py --algo dqn ddpg

Outputs (results/final/):
    <algo>_runs.csv          one row per seed: val_* and test_* metrics
    <algo>_test_curves.csv   test equity curve of every seed
    <algo>_histories.csv     validation learning curves of every seed
"""

import pandas as pd
import torch
from common import (base_parser, curves_frame, get_config, load_best, results_dir, run_pool,
                    save_json, worker_splits)

from finsearch.runner import run_experiment


def job(args):
    algo, cfg, seed, ckpt_dir = args
    r = run_experiment(algo, cfg, worker_splits(cfg), seed, ("val", "test"))
    torch.save({"state_dict": r.pop("state_dict"), "config": cfg}, f"{ckpt_dir}/seed{seed}.pt")
    return r


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--algo", nargs="+", choices=["dqn", "ddpg"], default=["dqn", "ddpg"])
    args = p.parse_args()
    base = get_config(args)
    out = results_dir(base, "final", quick=args.quick)

    for algo in args.algo:
        cfg, params = load_best(base, algo, args.quick)
        print(f"\n== {algo.upper()} | tuned params: {params or '(defaults — run tune.py first)'}")
        ckpt = results_dir(cfg, "runs", algo, quick=args.quick)
        seeds = range(int(cfg["experiment"]["n_seeds"]))
        results = list(run_pool(job, [(algo, cfg, s, str(ckpt)) for s in seeds], cfg, algo))

        rows = [{"method": algo.upper(), "seed": r["seed"],
                 **{f"val_{k}": v for k, v in r["val"].items()},
                 **{f"test_{k}": v for k, v in r["test"].items()}} for r in results]
        runs = pd.DataFrame(rows).sort_values("seed")
        runs.to_csv(out / f"{algo}_runs.csv", index=False)
        curves_frame(results, "test_curve").to_csv(out / f"{algo}_test_curves.csv")
        pd.concat({r["seed"]: r["history"] for r in results}, names=["seed", "i"]).to_csv(
            out / f"{algo}_histories.csv")
        save_json({"algo": algo, "params": params, "seeds": list(seeds)}, out / f"{algo}_meta.json")
        print(runs[["test_sharpe", "test_ann_return", "test_max_drawdown"]].describe().round(3))
