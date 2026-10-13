"""Stage 11: repeat the grouped split with several seeds so every comparison has an error bar.

    python scripts/11_repeated_splits.py [--seeds 5]

For each seed the manifest's source-photo groups are re-split 70/15/15 per class (same method as stage 1). Models use
the hyper-parameters tuned in stage 4 (stated simplification: no re-tuning per seed). Fit on train+val, score on test.
Feature sets: all / colour / texture / shape / colour+texture, for RF, SVM, LR.
Needs clean features for ALL images, so it reads the three cached split matrices and re-assembles them in manifest order.
Outputs (results/): repeated_splits.csv (one row per seed x model x feature set), repeated_splits_summary.csv (mean, std)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import group_split, load_manifest, split_indices  # noqa: E402
from src.features import columns_for_set, load_clean_features  # noqa: E402
from src.models import MODEL_NAMES, make_model  # noqa: E402
from src.utils import Timer, load_json  # noqa: E402

SETS = ("all", "color", "texture", "shape", "color+texture")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    res = cfg["paths"]["results_dir"]
    df = load_manifest(cfg)
    idx = split_indices(df)
    Xs, names = load_clean_features(cfg)
    X = np.zeros((len(df), len(names)), dtype=np.float64)
    for s in ("train", "val", "test"):
        X[idx[s]] = Xs[s]
    y = df["class_id"].values
    best = load_json(res / "best_params.json")
    fracs = cfg["dataset"]["group_split"]
    sets = {s: columns_for_set(names, s) for s in SETS if s != "all"}
    sets["all"] = np.arange(len(names))
    rows = []
    for k in range(args.seeds):
        seed = cfg["seed"] + k
        split = group_split(df, fracs, seed).values
        tv, te = split != "test", split == "test"
        with Timer(f"seed {seed}: train+val {int(tv.sum())}, test {int(te.sum())}"):
            for sname, cols in sets.items():
                for name in MODEL_NAMES:
                    m = make_model(name, best[name], cfg).fit(X[tv][:, cols], y[tv])
                    p = m.predict(X[te][:, cols])
                    rows.append({"seed": seed, "feature_set": sname, "model": name,
                                 "test_accuracy": accuracy_score(y[te], p),
                                 "test_macro_f1": f1_score(y[te], p, average="macro")})
    out = pd.DataFrame(rows)
    out.to_csv(res / "repeated_splits.csv", index=False)
    summ = (out.groupby(["model", "feature_set"])[["test_accuracy", "test_macro_f1"]]
               .agg(["mean", "std"]).round(4))
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(res / "repeated_splits_summary.csv", index=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
