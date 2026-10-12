"""Stage 3 (Member B): extract hand-crafted features for train/val/test and cache them.

    python scripts/03_extract_features.py --benchmark 100   # time 100 random images, save nothing
    python scripts/03_extract_features.py                    # full extraction (resumes: skips finished splits)
    python scripts/03_extract_features.py --force

Output: work/features/clean_{train,val,test}.npy (rows in manifest order) + feature_names.json
This is CPU-bound: a GPU does not help. Use as many cores as the machine has.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import SPLITS, cache_file, load_cache, load_manifest, split_indices  # noqa: E402
from src.features import extract_matrix, feature_dir, feature_schema, worker_cfg  # noqa: E402
from src.utils import save_json, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=int, default=0, help="time N random images and exit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    df = load_manifest(cfg)
    load_cache(cfg, df)                      # validates cache <-> manifest
    idx = split_indices(df)
    sub = worker_cfg(cfg)
    names = feature_schema(sub)
    print(f"{len(names)} features: " + ", ".join(
        f"{g}={sum(n.startswith(g + '_') for n in names)}" for g in ("color", "texture", "shape")))

    if args.benchmark:
        rng = np.random.default_rng(cfg["seed"])
        pick = rng.choice(len(df), size=min(args.benchmark, len(df)), replace=False)
        t0 = time.time()
        X = extract_matrix(cache_file(cfg), pick, sub, cfg["n_jobs"])
        dt = time.time() - t0
        print(f"{len(pick)} images in {dt:.1f}s -> {1000 * dt / len(pick):.0f} ms/img (wall). "
              f"Estimated full dataset: {dt / len(pick) * len(df) / 60:.1f} min. NaNs: {int(np.isnan(X).sum())}")
        return

    out_dir = feature_dir(cfg)
    save_json(names, out_dir / "feature_names.json")
    for split in SPLITS:
        out = out_dir / f"clean_{split}.npy"
        if out.exists() and not args.force:
            print(f"{split}: exists, skipping ({out.name})")
            continue
        t0 = time.time()
        X = extract_matrix(cache_file(cfg), idx[split], sub, cfg["n_jobs"])
        np.save(out, X)
        print(f"{split}: {X.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
