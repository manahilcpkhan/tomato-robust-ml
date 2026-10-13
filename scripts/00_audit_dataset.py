"""Stage 0 (Member A): audit the dataset BEFORE building anything. Reads folder/file names only (fast).

    python scripts/00_audit_dataset.py

Answers: are class names recognised? how many images per split/class? how many are augmented copies?
does any SOURCE PHOTO appear in more than one split (=> inflated test scores)?
Writes results/dataset_audit.json. Run it for every candidate data_root (e.g. 'Dataset (raw)' and
'Augmented Dataset') and keep the output for the report.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import audit_dataset, refine_groups, scan_images  # noqa: E402
from src.utils import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true", help="also cluster visually near-duplicate images (slower)")
    args = ap.parse_args()
    cfg = load_config()
    df = scan_images(cfg)
    if args.hash:
        df = refine_groups(cfg, df)
    print(f"data_root: {cfg['paths']['data_root']}")
    print(f"{len(df)} images, {df['label'].nunique()} classes: {sorted(df['label'].unique())}")
    counts = df["label"].value_counts()
    print("\nimages per class" + (" (after merging near-duplicates: unique photos per class below)" if args.hash else ""))
    print(pd.DataFrame({"images": counts, "unique_photos": df.groupby("label")["group"].nunique()})
          .reindex(counts.index).to_string())
    print(f"imbalance (largest/smallest class): {counts.max() / counts.min():.1f}x")

    if df["official_split"].notna().all():
        print("\n--- using the dataset's own train/val/test folders ---")
        rep = audit_dataset(df, "official_split")
        print(pd.crosstab(df["label"], df["official_split"]).reindex(columns=["train", "val", "test"]))
    else:
        print("\nNo train/val/test folders found under this data_root -> official split unavailable "
              "(use split_strategy: group).")
        rep = audit_dataset(df.assign(official_split=None), "official_split")

    print(f"\nsource photos (groups)        : {rep['n_source_groups']}  (~{rep['mean_images_per_group']:.1f} images each)")
    print(f"images flagged as augmented   : {rep['frac_images_marked_augmented']:.1%}")
    print(f"source photos in >1 split     : {rep['n_groups_spanning_multiple_splits']}")
    if "frac_test_images_with_source_in_trainval" in rep:
        f = rep["frac_test_images_with_source_in_trainval"]
        print(f"test images whose source photo is also in train/val: {f:.1%}")
        if f > 0.01:
            print("\n>>> LEAKAGE: set dataset.split_strategy: group in config.yaml (re-splits by source photo).")
        else:
            print("\nNo source-photo leakage between official splits. (Test may still contain several copies of one "
                  "photo; that makes results correlated, not leaked.)")
    save_json(rep, cfg["paths"]["results_dir"] / "dataset_audit.json")

    ex = df[df["is_aug"]].head(5)["path"].tolist()
    if ex:
        print("\nexample augmented file names:", ex[:3])
        print(f"grouping regex: {cfg['dataset']['group_regex']!r}  -> check these groups look right:")
        print(df[df["is_aug"]][["path", "group"]].head(4).to_string(index=False))


if __name__ == "__main__":
    main()
