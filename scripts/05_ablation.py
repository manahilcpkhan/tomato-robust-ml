"""Stage 5 (Member B): feature-group ablation (color / texture / shape and combinations) x {RF, SVM, LR}.

Hyper-parameters are those tuned on ALL features in stage 4 (stated as a limitation in the report);
every model is fitted on train+val and evaluated once on the clean test set.
Output: results/ablation.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_manifest, split_indices  # noqa: E402
from src.evaluate import bootstrap_ci, compute_metrics  # noqa: E402
from src.features import FEATURE_SETS, columns_for_set, load_clean_features  # noqa: E402
from src.models import MODEL_NAMES, make_model  # noqa: E402
from src.utils import Timer, load_json, set_seed  # noqa: E402


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    res = cfg["paths"]["results_dir"]
    classes = cfg["dataset"]["classes"]
    best = load_json(res / "best_params.json")

    df = load_manifest(cfg)
    idx = split_indices(df)
    y = df["class_id"].values
    X, names = load_clean_features(cfg)
    Xtv, ytv = np.vstack([X["train"], X["val"]]), np.concatenate([y[idx["train"]], y[idx["val"]]])
    yte = y[idx["test"]]

    rows = []
    for set_name in FEATURE_SETS:
        cols = columns_for_set(names, set_name)
        for mname in MODEL_NAMES:
            with Timer(f"{mname} / {set_name} ({len(cols)} feats)"):
                model = make_model(mname, best[mname], cfg).fit(Xtv[:, cols], ytv)
                pred = model.predict(X["test"][:, cols])
            m = compute_metrics(yte, pred, classes)
            lo, hi = bootstrap_ci(yte, pred)
            rows.append({"feature_set": set_name, "n_features": len(cols), "model": mname,
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "macro_f1_ci_lo": lo, "macro_f1_ci_hi": hi,
                         **{k: v for k, v in m.items() if k.startswith("f1_")}})
    out = pd.DataFrame(rows)
    out.to_csv(res / "ablation.csv", index=False)
    print(out.pivot(index="feature_set", columns="model", values="macro_f1").round(4))


if __name__ == "__main__":
    main()
