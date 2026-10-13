"""Stage 2 (Member A): EDA, shortcut checks, segmentation check, duplicate/leakage check, perturbation gallery.

Outputs (results/):
  eda_split_counts.csv, eda_image_sizes.csv, eda_samples.png, eda_class_stats.csv, eda_brightness_by_class.png,
  eda_segmentation.png, eda_segmentation_stats.json, duplicate_pairs.csv, duplicates_summary.json,
  duplicate_examples.png, test_leakage_flags.csv, perturbation_gallery.png

Read these before modelling: they feed the "dataset" and "challenges" sections of the report.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import SPLITS, load_cache, load_manifest, split_indices  # noqa: E402
from src.duplicates import find_near_duplicates  # noqa: E402
from src.evaluate import pretty  # noqa: E402
from src.features import worker_cfg  # noqa: E402
from src.perturb import all_conditions, condition_name, perturb  # noqa: E402
from src.segment import segment_leaf  # noqa: E402
from src.utils import Timer, save_json  # noqa: E402


def step_counts_and_sizes(cfg, df, res):
    pd.crosstab(df["label"], df["split"])[list(SPLITS)].to_csv(res / "eda_split_counts.csv")
    root = cfg["paths"]["data_root"]

    def size(p):
        with Image.open(root / p) as im:
            return im.size

    with ThreadPoolExecutor(8) as ex:
        sizes = list(ex.map(size, df["path"]))
    s = pd.DataFrame(sizes, columns=["width", "height"])
    s["orientation"] = np.where(s.width >= s.height, "landscape", "portrait")
    s["path"] = df["path"].values
    s.to_csv(res / "eda_image_sizes.csv", index=False)
    print(s.describe().loc[["min", "50%", "max"]])
    print(s["orientation"].value_counts().to_dict())


def step_samples(cfg, df, imgs, res, n_per=6):
    classes = cfg["dataset"]["classes"]
    rng = np.random.default_rng(cfg["seed"])
    fig, axes = plt.subplots(len(classes), n_per, figsize=(2 * n_per, 2 * len(classes)))
    for r, c in enumerate(classes):
        idx = np.where((df["label"] == c) & (df["split"] == "train"))[0]
        pick = rng.choice(idx, size=min(n_per, len(idx)), replace=False)
        for k in range(n_per):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(pick):
                ax.imshow(imgs[pick[k]])
            if k == 0:
                ax.set_title(pretty(c), fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(res / "eda_samples.png", dpi=110)
    plt.close(fig)


def step_class_stats(cfg, df, imgs, res):
    """Per-class brightness/saturation: large between-class differences = possible illumination shortcut."""
    rows = []
    for i in range(len(df)):
        hsv = cv2.cvtColor(np.asarray(imgs[i]), cv2.COLOR_RGB2HSV)
        rows.append((hsv[..., 2].mean(), hsv[..., 1].mean()))
    st = pd.DataFrame(rows, columns=["mean_V", "mean_S"])
    st["label"], st["split"] = df["label"].values, df["split"].values
    summary = st.groupby("label")[["mean_V", "mean_S"]].agg(["mean", "std"])
    summary.to_csv(res / "eda_class_stats.csv")
    print(summary.round(1))
    classes = cfg["dataset"]["classes"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col in zip(axes, ("mean_V", "mean_S")):
        ax.boxplot([st.loc[st.label == c, col] for c in classes])
        ax.set_xticks(range(1, len(classes) + 1), [pretty(c) for c in classes])
        ax.set_title(f"{col} per class")
        ax.tick_params(axis="x", rotation=40, labelsize=7)
    fig.tight_layout()
    fig.savefig(res / "eda_brightness_by_class.png", dpi=130)
    plt.close(fig)


def step_segmentation(cfg, df, imgs, res, n_stats=1000, n_show=12):
    rng = np.random.default_rng(cfg["seed"])
    sample = rng.choice(len(df), size=min(n_stats, len(df)), replace=False)
    fr = np.array([segment_leaf(np.asarray(imgs[i]), cfg["segmentation"]).mean() for i in sample])
    fallback = float((fr >= 0.999).mean())
    stats = {"method": cfg["segmentation"]["method"], "n_checked": len(sample),
             "mask_frac_mean": float(fr.mean()), "mask_frac_median": float(np.median(fr)),
             "frac_mask_above_0.95": float((fr > 0.95).mean()), "frac_fallback_whole_image": fallback}
    save_json(stats, res / "eda_segmentation_stats.json")
    print(stats)
    fig, axes = plt.subplots(3, n_show // 3 * 2, figsize=(2.2 * n_show // 3 * 2, 6.6))
    axes = axes.ravel()
    for k, i in enumerate(sample[:n_show]):
        img = np.asarray(imgs[i])
        m = segment_leaf(img, cfg["segmentation"]).astype(np.uint8)
        over = img.copy()
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (255, 0, 0), 2)
        axes[2 * k].imshow(img)
        axes[2 * k + 1].imshow(over)
        axes[2 * k].set_title(df["label"][i][:12], fontsize=7)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Segmentation check: left original, right mask outline (red). If outlines are poor, try method: grabcut",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(res / "eda_segmentation.png", dpi=110)
    plt.close(fig)


def step_duplicates(cfg, df, imgs, res):
    d = cfg["duplicates"]
    with Timer("near-duplicate search"):
        pairs = find_near_duplicates(imgs, d["hash_size"], d["hamming_threshold"], min_corr=d["min_correlation"])
    if len(pairs):
        for side in ("i", "j"):
            pairs[f"path_{side}"] = df["path"].values[pairs[side]]
            pairs[f"split_{side}"] = df["split"].values[pairs[side]]
            pairs[f"label_{side}"] = df["label"].values[pairs[side]]
    pairs.to_csv(res / "duplicate_pairs.csv", index=False)

    idx = split_indices(df)
    test_leaky = np.zeros(len(df), bool)
    val_leaky = np.zeros(len(df), bool)
    if len(pairs):
        for _, r in pairs.iterrows():
            si, sj = r.split_i, r.split_j
            if si == "test" and sj in ("train", "val"):
                test_leaky[r.i] = True
            if sj == "test" and si in ("train", "val"):
                test_leaky[r.j] = True
            if si == "val" and sj == "train":
                val_leaky[r.i] = True
            if sj == "val" and si == "train":
                val_leaky[r.j] = True
    # source-photo leakage (from filenames): test image whose source photo also appears in train/val
    trval_groups = set(df.loc[idx["train"], "group"]) | set(df.loc[idx["val"], "group"])
    group_leaky = df["group"].isin(trval_groups).values & (df["split"].values == "test")
    flags = pd.DataFrame({"path": df["path"].values[idx["test"]],
                          "leaky_hash": test_leaky[idx["test"]],
                          "leaky_group": group_leaky[idx["test"]]})
    flags["leaky"] = flags["leaky_hash"] | flags["leaky_group"]
    flags.to_csv(res / "test_leakage_flags.csv", index=False)

    summary = {"n_test_images_with_source_photo_in_trainval": int(group_leaky.sum()),
               "frac_test_group_leaky": float(group_leaky[idx["test"]].mean()),
               "frac_test_leaky_any": float(flags["leaky"].mean()),
               "n_pairs_total": int(len(pairs)),
               "n_pairs_cross_split": int((pairs.split_i != pairs.split_j).sum()) if len(pairs) else 0,
               "n_pairs_label_conflict": int((pairs.label_i != pairs.label_j).sum()) if len(pairs) else 0,
               "n_test_images_with_duplicate_in_trainval": int(test_leaky.sum()),
               "frac_test_leaky": float(test_leaky[idx["test"]].mean()),
               "n_val_images_with_duplicate_in_train": int(val_leaky.sum()),
               "threshold_bits": d["hamming_threshold"], "hash_bits": d["hash_size"] ** 2 - 1,
               "hamming_histogram": (pairs["hamming"].value_counts().sort_index().to_dict() if len(pairs) else {})}
    save_json(summary, res / "duplicates_summary.json")
    print(summary)

    cross = pairs[pairs.split_i != pairs.split_j].head(6) if len(pairs) else pairs
    if len(cross):
        fig, axes = plt.subplots(len(cross), 2, figsize=(4.4, 2.2 * len(cross)))
        axes = np.atleast_2d(axes)
        for r, (_, p) in enumerate(cross.iterrows()):
            for c, k in enumerate(("i", "j")):
                axes[r, c].imshow(imgs[int(p[k])])
                axes[r, c].set_title(f"{p[f'split_{k}']} / {p[f'label_{k}'][:14]}", fontsize=7)
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(res / "duplicate_examples.png", dpi=110)
        plt.close(fig)


def step_gallery(cfg, df, imgs, res):
    sub = worker_cfg(cfg)
    i = int(np.where((df["split"] == "test") & (df["label"] == "healthy"))[0][0]) if (
        (df["split"] == "test") & (df["label"] == "healthy")).any() else int(np.where(df["split"] == "test")[0][0])
    conds = all_conditions(cfg)
    ncol = 5
    nrow = int(np.ceil((len(conds) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow))
    axes = axes.ravel()
    axes[0].imshow(imgs[i])
    axes[0].set_title("clean", fontsize=8)
    for k, (kind, level) in enumerate(conds, start=1):
        axes[k].imshow(perturb(np.asarray(imgs[i]), kind, level, i, sub))
        axes[k].set_title(condition_name(kind, level), fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(res / "perturbation_gallery.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dups", action="store_true")
    ap.add_argument("--skip-sizes", action="store_true", help="skip reading original image headers (slow on Drive)")
    args = ap.parse_args()

    cfg = load_config()
    res = cfg["paths"]["results_dir"]
    df = load_manifest(cfg)
    imgs = load_cache(cfg, df)

    if not args.skip_sizes:
        with Timer("counts + original sizes"):
            step_counts_and_sizes(cfg, df, res)
    with Timer("sample grid"):
        step_samples(cfg, df, imgs, res)
    with Timer("class brightness stats"):
        step_class_stats(cfg, df, imgs, res)
    with Timer("segmentation check"):
        step_segmentation(cfg, df, imgs, res)
    with Timer("perturbation gallery"):
        step_gallery(cfg, df, imgs, res)
    if not args.skip_dups:
        step_duplicates(cfg, df, imgs, res)


if __name__ == "__main__":
    main()
