"""Train a single agent (one seed) and report validation + test metrics.

    python scripts/train.py --algo dqn --seed 0
    python scripts/train.py --algo ddpg --seed 1 --set ddpg.actor_lr=3e-4 env.reward=dsr
    python scripts/train.py --algo dqn --use-best      # start from tuned hyper-parameters
"""

import torch
from common import base_parser, get_config, load_best, results_dir

from finsearch.data import build_dataset
from finsearch.runner import run_experiment

if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--algo", choices=["dqn", "ddpg"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-best", action="store_true", help="apply results/tuning/best_<algo>.yaml")
    args = p.parse_args()
    cfg = get_config(args)
    if args.use_best:
        cfg, params = load_best(cfg, args.algo, args.quick)
        print("tuned params:", params or "(none found — using defaults)")

    torch.set_num_threads(4)
    res = run_experiment(args.algo, cfg, build_dataset(cfg), args.seed, verbose=True)
    out = results_dir(cfg, "runs", args.algo, quick=args.quick)
    torch.save({"state_dict": res["state_dict"], "config": cfg}, out / f"seed{args.seed}.pt")
    res["history"].to_csv(out / f"seed{args.seed}_history.csv", index=False)
    for split in ("val", "test"):
        m = res[split]
        print(f"{split:>4}: Sharpe {m['sharpe']:+.3f} | ann. return {m['ann_return']:+.2%} | "
              f"max DD {m['max_drawdown']:.2%} | turnover/day {m['avg_daily_turnover']:.3f}")
    print(f"checkpoint -> {out / f'seed{args.seed}.pt'}")
