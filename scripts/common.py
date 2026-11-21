"""Helpers shared by the experiment scripts: CLI args, process pool, I/O."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from finsearch.config import ROOT, apply_params, load_config, resolve
from finsearch.data import build_dataset

_SPLITS = None


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                   help="override config values, e.g. --set dqn.lr=1e-4 env.cost=0.002")
    p.add_argument("--workers", type=int, default=None, help="parallel processes")
    p.add_argument("--quick", action="store_true",
                   help="tiny budgets / few seeds — checks the pipeline end-to-end in minutes")
    return p


QUICK = ["dqn.total_steps=4000", "ddpg.total_steps=4000", "train.eval_every=2000",
         "dqn.learning_starts=500", "ddpg.learning_starts=500", "experiment.n_seeds=2",
         "experiment.tune_seeds=1", "baselines.lstm.epochs=2"]


def get_config(args) -> dict:
    cfg = load_config(args.config, (QUICK if args.quick else []) + list(args.set))
    if args.workers:
        cfg["experiment"]["workers"] = args.workers
    return cfg


def results_dir(cfg: dict, *parts: str, quick: bool = False) -> Path:
    base = resolve(cfg["experiment"]["results_dir"])
    if quick:
        base = base / "quick"
    d = base.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def best_params_path(cfg: dict, algo: str, quick: bool = False) -> Path:
    return results_dir(cfg, "tuning", quick=quick) / f"best_{algo}.yaml"


def load_best(cfg: dict, algo: str, quick: bool = False) -> tuple[dict, dict]:
    """Apply tuned params (if tune.py has been run) on top of ``cfg``."""
    p = best_params_path(cfg, algo, quick)
    if not p.exists():
        return cfg, {}
    params = yaml.safe_load(p.read_text())["params"]
    return apply_params(cfg, params), params


# ----------------------------------------------------------------------
# Process pool: each worker builds the dataset once (from the CSV cache)
# ----------------------------------------------------------------------
def _init_worker(cfg_data: dict) -> None:
    global _SPLITS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _SPLITS = build_dataset(cfg_data)


def worker_splits(cfg: dict) -> dict:
    """Splits for this worker; rebuilt if the job changes the observation window
    (the lookback context depends on it, the features do not)."""
    global _SPLITS
    need = max(cfg["env"]["window"], cfg["portfolio"]["window"], 60)
    if _SPLITS is None or _SPLITS["val"].trade_start < need:
        _SPLITS = build_dataset(cfg)
    return _SPLITS


def run_pool(fn, jobs: list, cfg: dict, desc: str = ""):
    """Run ``fn(job)`` for each job in parallel; yields results as they finish."""
    workers = min(int(cfg["experiment"]["workers"]), len(jobs)) or 1
    build_dataset(cfg)  # make sure the CSV cache exists before forking
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg,)) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            print(f"  [{desc} {i}/{len(jobs)}] done", flush=True)
            yield res


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def curves_frame(results: list, key: str) -> pd.DataFrame:
    return pd.concat({r["seed"]: r[key] for r in results}, axis=1).sort_index(axis=1)


__all__ = ["ROOT", "base_parser", "get_config", "results_dir", "load_best", "best_params_path",
           "run_pool", "worker_splits", "save_json", "curves_frame"]
