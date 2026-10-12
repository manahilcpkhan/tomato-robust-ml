"""Stage 6 (Member C, uses A's perturb.py and B's features.py): robustness under controlled perturbations.

Models are trained ONCE on clean train+val. Each perturbation is applied to the TEST images only, features are
re-extracted from the perturbed images (segmentation is re-run too, as a deployed pipeline would), and every
model / feature set is re-evaluated.

Outputs: results/robustness.csv, results/robustness_conditions.csv, work/features/perturbed/{kind}_{level}.npy
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import cache_file, load_cache, load_manifest, split_indices  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.features import (FEATURE_SETS, columns_for_set, extract_matrix, load_clean_features,  # noqa: E402
                          perturbed_dir, worker_cfg)
from src.models import make_model  # noqa: E402
from src.perturb import all_conditions, condition_name  # noqa: E402
from src.utils import Timer, load_json, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-extract perturbed features even if cached")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    res = cfg["paths"]["results_dir"]
    classes = cfg["dataset"]["classes"]
    best = load_json(res / "best_params.json")

    df = load_manifest(cfg)
    load_cache(cfg, df)
    idx = split_indices(df)
    y = df["class_id"].values
    X, names = load_clean_features(cfg)
    Xtv, ytv = np.vstack([X["train"], X["val"]]), np.concatenate([y[idx["train"]], y[idx["val"]]])
    yte = y[idx["test"]]
    leaf_col = names.index("shape_leaf_area_frac")

    # models: RF on every feature set; SVM and LR on all features
    plan = [("rf", s) for s in FEATURE_SETS] + [("svm", "all"), ("lr", "all")]
    fitted = {}
    with Timer("fit models on clean train+val"):
        for mname, sname in plan:
            cols = columns_for_set(names, sname)
            fitted[(mname, sname)] = (make_model(mname, best[mname], cfg).fit(Xtv[:, cols], ytv), cols)

    sub = worker_cfg(cfg)
    pdir = perturbed_dir(cfg)
    conds = [("clean", "none")] + all_conditions(cfg)
    rows, cond_rows = [], []
    for kind, level in conds:
        name = "clean" if kind == "clean" else condition_name(kind, level)
        if kind == "clean":
            Xt = X["test"]
        else:
            f = pdir / f"{name}.npy"
            if f.exists() and not args.force:
                Xt = np.load(f)
            else:
                t0 = time.time()
                Xt = extract_matrix(cache_file(cfg), idx["test"], sub, cfg["n_jobs"], cond=(kind, level))
                np.save(f, Xt)
                print(f"[{name}] extracted {Xt.shape} in {time.time() - t0:.0f}s", flush=True)
        cond_rows.append({"condition": name, "kind": kind, "level": level,
                          "mean_leaf_area_frac": float(Xt[:, leaf_col].mean())})
        for (mname, sname), (model, cols) in fitted.items():
            m = compute_metrics(yte, model.predict(Xt[:, cols]), classes)
            rows.append({"condition": name, "kind": kind, "level": level, "model": mname,
                         "feature_set": sname, **m})

    pd.DataFrame(rows).to_csv(res / "robustness.csv", index=False)
    pd.DataFrame(cond_rows).to_csv(res / "robustness_conditions.csv", index=False)
    r = pd.DataFrame(rows)
    sel = r[(r.model == "rf") & (r.feature_set == "all")][["condition", "accuracy", "macro_f1"]]
    print(sel.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
