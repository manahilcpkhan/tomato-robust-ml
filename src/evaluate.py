"""Metrics and plots.  (Owner: Member C)"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support)


def pretty(c: str) -> str:
    return c.replace("_", " ").title()


def compute_metrics(y_true, y_pred, classes) -> dict:
    labels = list(range(len(classes)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": float(f.mean()),
           "macro_precision": float(p.mean()), "macro_recall": float(r.mean())}
    for i, c in enumerate(classes):
        out[f"f1_{c}"] = float(f[i])
    return out


def per_class_table(y_true, y_pred, classes) -> pd.DataFrame:
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s})


def bootstrap_ci(y_true, y_pred, n: int = 500, seed: int = 0, alpha: float = 0.05):
    """95% bootstrap CI of macro-F1 (resampling test images)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y_true), len(y_true))
        vals.append(f1_score(y_true[ix], y_pred[ix], average="macro"))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def confusion(y_true, y_pred, k: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(k)))


def plot_confusion(cm, classes, path, title="", normalize=True):
    cm = np.asarray(cm, float)
    show = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(classes)), [pretty(c) for c in classes], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), [pretty(c) for c in classes], fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = f"{show[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if show[i, j] > 0.5 * show.max() else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
