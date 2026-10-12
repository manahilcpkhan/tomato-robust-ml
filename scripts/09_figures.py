"""Stage 9 (Member C): report-ready figures and the robustness-drop table, built only from results/*.csv|json.

Outputs: robustness_drop.csv, fig_ablation.png, fig_robustness_curves.png, fig_robustness_drop_heatmap.png,
         fig_group_importance.png, fig_importance_stability.png
Missing inputs are skipped, so this can be re-run at any time to see progress.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.utils import load_json  # noqa: E402


def fig_ablation(res):
    f = res / "ablation.csv"
    if not f.exists():
        return
    a = pd.read_csv(f)
    sets = list(dict.fromkeys(a.feature_set))
    models = list(dict.fromkeys(a.model))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, m in enumerate(models):
        sub = a[a.model == m].set_index("feature_set").loc[sets]
        err = np.vstack([sub.macro_f1 - sub.macro_f1_ci_lo, sub.macro_f1_ci_hi - sub.macro_f1])
        ax.bar(np.arange(len(sets)) + j * w, sub.macro_f1, w, yerr=err, capsize=2, label=m.upper())
    ax.set_xticks(np.arange(len(sets)) + w * (len(models) - 1) / 2, sets, rotation=20)
    ax.set_ylabel("test macro-F1 (95% bootstrap CI)")
    ax.set_title("Feature-group ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(res / "fig_ablation.png", dpi=150)
    plt.close(fig)


def fig_robustness(res):
    f = res / "robustness.csv"
    if not f.exists():
        return
    r = pd.read_csv(f)
    rf = r[r.model == "rf"].copy()
    clean = rf[rf.condition == "clean"].set_index("feature_set")["macro_f1"]
    pert = rf[rf.condition != "clean"].copy()
    pert["macro_f1_drop"] = pert["feature_set"].map(clean) - pert["macro_f1"]
    clean_acc = rf[rf.condition == "clean"].set_index("feature_set")["accuracy"]
    pert["accuracy_drop"] = pert["feature_set"].map(clean_acc) - pert["accuracy"]
    full = r[r.condition != "clean"].copy()
    base = r[r.condition == "clean"].set_index(["model", "feature_set"])
    full["macro_f1_drop"] = [base.loc[(m, s), "macro_f1"] - v for m, s, v in zip(full.model, full.feature_set, full.macro_f1)]
    full["accuracy_drop"] = [base.loc[(m, s), "accuracy"] - v for m, s, v in zip(full.model, full.feature_set, full.accuracy)]
    full[["condition", "kind", "level", "model", "feature_set", "accuracy", "macro_f1",
          "accuracy_drop", "macro_f1_drop"]].to_csv(res / "robustness_drop.csv", index=False)

    kinds = list(dict.fromkeys(pert.kind))
    sets = ["all", "color", "texture", "shape"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        sub = pert[pert.kind == kind]
        levels = list(dict.fromkeys(sub.level))
        for s in sets:
            ss = sub[sub.feature_set == s].set_index("level").loc[levels]
            ax.plot(range(len(levels)), ss.macro_f1, "o-", label=s)
            ax.axhline(clean[s], color="grey", lw=0.4, ls=":")
        ax.set_xticks(range(len(levels)), levels, rotation=20)
        ax.set_title(kind)
    axes[0].set_ylabel("test macro-F1 (RF)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(res / "fig_robustness_curves.png", dpi=150)
    plt.close(fig)

    pv = pert.pivot(index="condition", columns="feature_set", values="macro_f1_drop")
    pv = pv[[s for s in ["all", "color", "texture", "shape", "color+texture", "color+shape", "texture+shape"] if s in pv]]
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(pv) + 1.5))
    im = ax.imshow(pv.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(pv.shape[1]), pv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(pv.shape[0]), pv.index, fontsize=8)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title("Macro-F1 drop vs clean (RF)")
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout()
    fig.savefig(res / "fig_robustness_drop_heatmap.png", dpi=150)
    plt.close(fig)


def fig_importance(res):
    f = res / "importance_perturbed_group.csv"
    if f.exists():
        g = pd.read_csv(f, index_col=0)
        fig, ax = plt.subplots(figsize=(4.8, 0.3 * len(g) + 1.4))
        im = ax.imshow(g.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(g.shape[1]), g.columns)
        ax.set_yticks(range(g.shape[0]), g.index, fontsize=8)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="w")
        ax.set_title("Grouped permutation importance (true-class-prob drop)")
        fig.colorbar(im, fraction=0.05)
        fig.tight_layout()
        fig.savefig(res / "fig_group_importance.png", dpi=150)
        plt.close(fig)

    s, p = res / "importance_stability_cv.json", res / "importance_stability_perturbed.csv"
    if s.exists() and p.exists():
        cv, pt = load_json(s), pd.read_csv(p)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        names = ["impurity", "permutation"]
        axes[0].bar(names, [cv[n]["spearman_mean"] for n in names],
                    yerr=[cv[n]["spearman_std"] for n in names], capsize=3)
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Across CV folds: mean pairwise Spearman")
        axes[1].barh(pt["condition"], pt["spearman"])
        axes[1].set_xlim(0, 1)
        axes[1].set_title("Perturbed vs clean: Spearman of permutation importance", fontsize=9)
        axes[1].tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(res / "fig_importance_stability.png", dpi=150)
        plt.close(fig)


def main():
    res = load_config()["paths"]["results_dir"]
    fig_ablation(res)
    fig_robustness(res)
    fig_importance(res)
    print("figures written to", res)


if __name__ == "__main__":
    main()
