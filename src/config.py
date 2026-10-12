"""Config loading: config.yaml < config.local.yaml < $TOMATO_CONFIG < env path overrides."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _deep_update(base: dict, new: dict | None) -> dict:
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    for extra in (ROOT / "config.local.yaml", os.environ.get("TOMATO_CONFIG")):
        if extra and Path(extra).exists():
            with open(extra) as f:
                _deep_update(cfg, yaml.safe_load(f))

    env_map = {
        "TOMATO_DATA_ROOT": "data_root",
        "TOMATO_WORK_DIR": "work_dir",
        "TOMATO_RESULTS_DIR": "results_dir",
        "TOMATO_MANIFEST": "manifest",
    }
    for env, key in env_map.items():
        if os.environ.get(env):
            cfg["paths"][key] = os.environ[env]

    for key, val in list(cfg["paths"].items()):
        p = Path(val).expanduser()
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p

    for key in ("work_dir", "results_dir"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    cfg["paths"]["manifest"].parent.mkdir(parents=True, exist_ok=True)
    return cfg
