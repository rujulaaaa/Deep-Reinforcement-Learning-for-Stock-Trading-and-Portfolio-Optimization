"""YAML config loading with dotted-key CLI overrides (``--set dqn.lr=1e-4``)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, _, value = item.partition("=")
        set_by_path(cfg, key, yaml.safe_load(value))
    return cfg


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def apply_params(cfg: dict, params: dict) -> dict:
    """Return a copy of ``cfg`` with ``{"dqn.lr": 1e-4, ...}`` applied."""
    out = copy.deepcopy(cfg)
    for k, v in params.items():
        set_by_path(out, k, v)
    return out


def resolve(path: str | Path) -> Path:
    """Paths in the config are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
