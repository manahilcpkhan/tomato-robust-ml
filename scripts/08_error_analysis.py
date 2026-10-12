"""Stage 8 (Member C): where and why does the model fail?

Outputs (results/): error_pairs.csv, misclassified_examples.png, confusion_rf_clean_vs_worst.png,
  prediction_shift_worst.csv, per_class_f1_drop_worst.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_cache, load_manifest, split_indices  # noqa: E402
from src.evaluate import compute_metrics, confusion, plot_confusion, pretty  # noqa: E402
from src.features import load_clean_features, perturbed_dir  # noqa: E402
from src.utils import set_seed  # noqa: E402


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    res = cfg["paths"]["results_dir"]
    classes = cfg["dataset"]["classes"]
    k = len(classes)

    df = load_manifest(cfg)
    imgs = load_cache(cfg, df)
    idx = split_indices(df)
    y = df["class_id"].values
    yte = y[idx["test"]]
    X, names = load_clean_features(cfg)
    rf = joblib.load(cfg["paths"]["work_dir"] / "models" / "rf_all.joblib")
    pred = rf.predict(X["test"])

    # ---- most confused class pairs on clean test
    cm = confusion(yte, pred, k)
    rows = [{"true": classes[i], "predicted": classes[j], "count": int(cm[i, j]),
             "rate_of_true_class": cm[i, j] / max(cm[i].sum(), 1)}
            for i in range(k) for j in range(k) if i != j and cm[i, j] > 0]
    pairs = pd.DataFrame(rows, columns=["true", "predicted", "count", "rate_of_true_class"]
                         ).sort_values("count", ascending=False)
    pairs.to_csv(res / "error_pairs.csv", index=False)
    print(pairs.head(8).to_string(index=False))

    # ---- misclassified example grid (prioritise the most confused pairs)
    wrong = np.where(pred != yte)[0]
    order, seen = [], set()
    for _, r in pairs.head(4).iterrows():
        t, p = classes.index(r["true"]), classes.index(r["predicted"])
        for w in wrong:
            if yte[w] == t and pred[w] == p and w not in seen and sum(1 for o in order if (yte[o], pred[o]) == (t, p)) < 3:
                order.append(w)
                seen.add(w)
    n = min(12, len(order))
    if n:
        fig, axes = plt.subplots(2, 6, figsize=(13, 5))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, w in zip(axes.ravel(), order[:n]):
            ax.imshow(imgs[idx["test"][w]])
            ax.set_title(f"T: {pretty(classes[yte[w]])[:16]}\nP: {pretty(classes[pred[w]])[:16]}", fontsize=7)
        fig.tight_layout()
        fig.savefig(res / "misclassified_examples.png", dpi=110)
        plt.close(fig)

    # ---- worst perturbation for RF/all: confusion + prediction shift
    rob = pd.read_csv(res / "robustness.csv")
    r = rob[(rob.model == "rf") & (rob.feature_set == "all") & (rob.condition != "clean")]
    worst = r.sort_values("macro_f1").iloc[0]["condition"]
    print(f"Worst condition for RF/all: {worst}")
    Xp = np.load(perturbed_dir(cfg) / f"{worst}.npy")
    predp = rf.predict(Xp)
    plot_confusion(confusion(yte, predp, k), classes, res / "confusion_rf_worst_condition.png",
                   f"RF (all features) under {worst}")

    shift = pd.DataFrame({"class": classes,
                          "share_pred_clean": np.bincount(pred, minlength=k) / len(pred),
                          f"share_pred_{worst}": np.bincount(predp, minlength=k) / len(predp)})
    shift.to_csv(res / "prediction_shift_worst.csv", index=False)
    mc, mp = compute_metrics(yte, pred, classes), compute_metrics(yte, predp, classes)
    pd.DataFrame({"class": classes, "f1_clean": [mc[f"f1_{c}"] for c in classes],
                  f"f1_{worst}": [mp[f"f1_{c}"] for c in classes],
                  "drop": [mc[f"f1_{c}"] - mp[f"f1_{c}"] for c in classes]}
                 ).to_csv(res / "per_class_f1_drop_worst.csv", index=False)
    print(shift.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
