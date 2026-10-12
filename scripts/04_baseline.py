"""Stage 4 (Member B): tune on validation, fit on train+val, report on test. RF (main), SVM, LR (brief check).

Outputs (results/): baseline_tuning_{rf,svm,lr}.csv, best_params.json, baseline_test_metrics.csv,
  baseline_per_class_test.csv, confusion_{model}_test.png, learning_curve_rf.csv/.png
Models:  work/models/{rf,svm,lr}_all.joblib
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_manifest, split_indices  # noqa: E402
from src.evaluate import bootstrap_ci, compute_metrics, confusion, per_class_table, plot_confusion, pretty  # noqa: E402
from src.features import load_clean_features  # noqa: E402
from src.models import MODEL_NAMES, make_model, tune  # noqa: E402
from src.utils import Timer, save_json, set_seed  # noqa: E402


def learning_curve(cfg, best_rf, Xtr, ytr, Xva, yva, res):
    """Train/val macro-F1 vs training-set size: a direct look at the bias-variance trade-off."""
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for frac in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        sel = np.concatenate([rng.permutation(np.where(ytr == c)[0])[: max(2, int(frac * (ytr == c).sum()))]
                              for c in np.unique(ytr)])
        m = make_model("rf", best_rf, cfg).fit(Xtr[sel], ytr[sel])
        rows.append({"train_fraction": frac, "n_train": len(sel),
                     "train_macro_f1": f1_score(ytr[sel], m.predict(Xtr[sel]), average="macro"),
                     "val_macro_f1": f1_score(yva, m.predict(Xva), average="macro")})
    lc = pd.DataFrame(rows)
    lc.to_csv(res / "learning_curve_rf.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(lc.n_train, lc.train_macro_f1, "o-", label="train")
    ax.plot(lc.n_train, lc.val_macro_f1, "s-", label="validation")
    ax.set_xlabel("training images")
    ax.set_ylabel("macro-F1")
    ax.set_title("Random Forest learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(res / "learning_curve_rf.png", dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    res = cfg["paths"]["results_dir"]
    models_dir = cfg["paths"]["work_dir"] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    classes = cfg["dataset"]["classes"]

    df = load_manifest(cfg)
    idx = split_indices(df)
    y = df["class_id"].values
    X, names = load_clean_features(cfg)
    ytr, yva, yte = y[idx["train"]], y[idx["val"]], y[idx["test"]]
    Xtv, ytv = np.vstack([X["train"], X["val"]]), np.concatenate([ytr, yva])

    leak = None
    flag_file = res / "test_leakage_flags.csv"
    if flag_file.exists():
        fl = pd.read_csv(flag_file).set_index("path")["leaky"]
        leak = fl.reindex(df["path"].values[idx["test"]]).fillna(False).values.astype(bool)
        print(f"Leakage flags found: {int(leak.sum())}/{len(leak)} test images have a near-duplicate in train/val")

    best, rows, per_class = {}, [], []
    for name in MODEL_NAMES:
        with Timer(f"tune {name} on validation"):
            best[name], table = tune(name, X["train"], ytr, X["val"], yva, cfg)
        table.to_csv(res / f"baseline_tuning_{name}.csv", index=False)
        print(f"  best {name}: {best[name]}  val macro-F1={table.val_macro_f1.max():.4f}")

        model = make_model(name, best[name], cfg).fit(Xtv, ytv)      # refit on train+val
        pred = model.predict(X["test"])
        m = compute_metrics(yte, pred, classes)
        lo, hi = bootstrap_ci(yte, pred)
        row = {"model": name, "best_params": json.dumps(best[name]),
               "val_macro_f1_at_tuning": table.val_macro_f1.max(),
               "trainval_accuracy": model.score(Xtv, ytv),
               "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"],
               "test_macro_f1_ci_lo": lo, "test_macro_f1_ci_hi": hi}
        if leak is not None and leak.any() and (~leak).any():
            m2 = compute_metrics(yte[~leak], pred[~leak], classes)
            row.update({"n_test_noleak": int((~leak).sum()), "test_accuracy_noleak": m2["accuracy"],
                        "test_macro_f1_noleak": m2["macro_f1"]})
        rows.append(row)
        pc = per_class_table(yte, pred, classes)
        pc.insert(0, "model", name)
        per_class.append(pc)
        plot_confusion(confusion(yte, pred, len(classes)), classes, res / f"confusion_{name}_test.png",
                       f"{name.upper()} (all features) - test")
        joblib.dump(model, models_dir / f"{name}_all.joblib")
        print(f"  {name}: test acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} [{lo:.3f}, {hi:.3f}]")

    pd.DataFrame(rows).to_csv(res / "baseline_test_metrics.csv", index=False)
    pd.concat(per_class).to_csv(res / "baseline_per_class_test.csv", index=False)
    save_json(best, res / "best_params.json")
    with Timer("RF learning curve"):
        learning_curve(cfg, best["rf"], X["train"], ytr, X["val"], yva, res)


if __name__ == "__main__":
    main()
