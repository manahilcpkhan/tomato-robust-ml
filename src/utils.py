from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


class Timer:
    """with Timer('label'): ...  -> prints elapsed wall time."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{self.label}] ...", flush=True)
        return self

    def __exit__(self, *exc):
        self.dt = time.time() - self.t0
        print(f"[{self.label}] done in {self.dt:.1f}s", flush=True)
