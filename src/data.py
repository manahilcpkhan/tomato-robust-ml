"""Dataset indexing, source-photo grouping, audit and the 256x256 image cache.  (Owner: Member A)

manifest.csv  : path (relative to data_root), label, split, class_id, group, is_aug  -> one row per image
                `group` = the source photo an image was derived from (augmented copies share it).
images_cache  : uint8 RGB array (N, S, S, 3) in manifest order, memory-mapped by every later stage
"""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .duplicates import find_near_duplicates
from .utils import load_json, save_json

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- scanning
def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _match_class(parts, cfg) -> str | None:
    hits = set()
    for part in parts:
        toks = _tokens(part)
        for cls, keys in cfg["dataset"]["class_keywords"].items():
            if any(t == k or t.startswith(k) for t in toks for k in keys):
                hits.add(cls)
    if len(hits) > 1:
        raise ValueError(f"Ambiguous class for folders {parts}: {sorted(hits)}")
    return next(iter(hits)) if hits else None


def _match_split(parts, cfg) -> str | None:
    for part in parts:
        toks = _tokens(part)
        for split, keys in cfg["dataset"]["split_names"].items():
            if any(t in keys for t in toks):
                return split
    return None


def source_group(stem: str, label: str, cfg) -> tuple[str, bool]:
    """(group id, is_augmented). Group = class + filename text before the augmentation marker."""
    rx = cfg["dataset"].get("group_regex") or ""
    if rx:
        m = re.search(rx, stem, flags=re.IGNORECASE)
        if m:
            base = stem[: m.start()].rstrip("_- ") or stem
            return f"{label}/{base}", True
    return f"{label}/{stem}", False


def scan_images(cfg) -> pd.DataFrame:
    """Find every labelled image under data_root. Columns: path, label, official_split, group, is_aug."""
    root = cfg["paths"]["data_root"]
    if not root.exists():
        raise FileNotFoundError(
            f"data_root does not exist: {root}\n"
            "Set paths.data_root in config.local.yaml or the TOMATO_DATA_ROOT env variable."
        )
    exts = {e.lower() for e in cfg["dataset"]["image_extensions"]}
    rows, unmatched, skipped = [], {}, {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            if p.suffix.lower() not in (".md", ".txt", ".pdf", ".db", ".csv", ".json", ".zip", ""):
                skipped[p.suffix.lower()] = skipped.get(p.suffix.lower(), 0) + 1
            continue
        rel = p.relative_to(root)
        parts = rel.parts[:-1]
        label = _match_class(parts, cfg)
        if label is None:
            key = "/".join(parts) if parts else "."
            unmatched[key] = unmatched.get(key, 0) + 1
            continue
        group, is_aug = source_group(p.stem, label, cfg)
        rows.append({"path": rel.as_posix(), "label": label, "official_split": _match_split(parts, cfg),
                     "group": group, "is_aug": is_aug})
    if not rows:
        raise RuntimeError(
            f"No labelled images found under {root}.\n"
            f"Folders seen without a class match: {list(unmatched)[:15]}\n"
            "Fix dataset.class_keywords in config.yaml so each class folder name matches a keyword."
        )
    if unmatched:
        print(f"WARNING: {sum(unmatched.values())} images in folders with no class match: {list(unmatched)[:10]}")
    if skipped:
        print(f"WARNING: files with other extensions were ignored: {skipped} "
              "(add the extension to dataset.image_extensions if they are images)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- visual grouping
def _load_small(path, size: int = 64) -> np.ndarray:
    img = _read_rgb(path)
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def refine_groups(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Merge images into one group when they are near-duplicates (dHash, flip-aware) of the same class.

    Catches resized / recompressed / flipped / slightly re-lit copies of one photo, which file names cannot reveal.
    It does NOT catch a different photo of the same leaf taken from another angle.
    """
    root = cfg["paths"]["data_root"]
    n = len(df)
    d = cfg["duplicates"]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first = {}
    for i, g in enumerate(df["group"].values):
        if g in first:
            union(i, first[g])
        else:
            first[g] = i

    print(f"Hashing {n} images to cluster near-duplicates (reads every file once) ...")
    paths = [root / p for p in df["path"]]
    with ThreadPoolExecutor(max_workers=8) as ex:
        small = np.stack(list(ex.map(_load_small, paths)))
    pairs = find_near_duplicates(small, d["hash_size"], d["hamming_threshold"], min_corr=d["min_correlation"])
    labels = df["label"].values
    cross, cross_idx = 0, set()
    for i, j in zip(pairs["i"].values, pairs["j"].values):
        if labels[i] == labels[j]:
            union(int(i), int(j))
        else:
            cross += 1
            cross_idx.update((int(i), int(j)))
    roots = np.array([find(i) for i in range(n)])
    new_group = df["group"].values[roots]
    out = df.copy()
    before, after = out["group"].nunique(), len(set(new_group))
    out["group"] = new_group

    n_dropped = 0
    if d.get("drop_cross_class", False) and cross_idx:
        # the same photo is filed under two diseases -> its label is unknowable. Drop every image that belongs to
        # the same-class group of an affected image (so no copy of an ambiguous photo survives).
        bad_groups = {new_group[i] for i in cross_idx}
        drop = out["group"].isin(bad_groups).values
        n_dropped = int(drop.sum())
        rep = out[drop][["path", "label", "group"]].sort_values("group")
        rep.to_csv(cfg["paths"]["results_dir"] / "dropped_cross_class.csv", index=False)
        print(f"Dropping {n_dropped} images that are (copies of) a photo filed under two different diseases "
              f"-> results/dropped_cross_class.csv")
        out = out[~drop].reset_index(drop=True)
        new_group = out["group"].values
    sizes = pd.Series(new_group).value_counts()
    print(f"near-duplicate pairs: {len(pairs)} ({cross} across different classes = label noise). "
          f"groups {before} -> {after}; largest group: {int(sizes.max())} images")
    if sizes.max() > max(20, 0.02 * n):
        print(f"WARNING: one group has {int(sizes.max())} images ({sizes.max() / n:.1%} of the data). Near-duplicate "
              "matching is probably chaining unrelated leaves: lower duplicates.hamming_threshold or raise "
              "duplicates.min_correlation in config.yaml, then rebuild.")
    save_json({"n_images": n, "n_pairs": int(len(pairs)), "n_cross_class_pairs": int(cross),
               "groups_before": int(before), "groups_after": int(after),
               "n_images_dropped_cross_class": n_dropped,
               "largest_group": int(sizes.max()), "n_groups_with_copies": int((sizes > 1).sum())},
              cfg["paths"]["results_dir"] / "grouping_summary.json")
    return out


# --------------------------------------------------------------------------- audit
def audit_dataset(df: pd.DataFrame, split_col: str) -> dict:
    """Class balance, augmentation share and source-photo leakage between splits."""
    d = df[df[split_col].notna()]
    g_splits = d.groupby("group")[split_col].nunique()
    multi = g_splits[g_splits > 1].index
    out = {
        "n_images": int(len(df)),
        "n_source_groups": int(df["group"].nunique()),
        "mean_images_per_group": float(len(df) / df["group"].nunique()),
        "frac_images_marked_augmented": float(df["is_aug"].mean()),
        "n_groups_spanning_multiple_splits": int(len(multi)),
        "frac_images_in_cross_split_groups": float(d["group"].isin(multi).mean()),
        "counts": pd.crosstab(d["label"], d[split_col]).to_dict(),
    }
    tr = set(d.loc[d[split_col].isin(["train", "val"]), "group"])
    te = d[d[split_col] == "test"]
    if len(te):
        out["frac_test_images_with_source_in_trainval"] = float(te["group"].isin(tr).mean())
        out["n_test_groups"] = int(te["group"].nunique())
    return out


# --------------------------------------------------------------------------- splitting
def group_split(df: pd.DataFrame, fracs: dict, seed: int) -> pd.Series:
    """Per-class split by whole groups, so no source photo is shared between splits."""
    rng = np.random.default_rng(seed)
    split = pd.Series(index=df.index, dtype=object)
    for label, sub in df.groupby("label"):
        sizes = sub.groupby("group").size()
        groups = sizes.index.to_numpy().copy()
        rng.shuffle(groups)
        total = float(sizes.sum())
        t_tr, t_va = fracs["train"] * total, (fracs["train"] + fracs["val"]) * total
        assign, cum = {}, 0
        for g in groups:
            assign[g] = "train" if cum < t_tr else ("val" if cum < t_va else "test")
            cum += sizes[g]
        split[sub.index] = sub["group"].map(assign)
        got = split[sub.index].value_counts()
        if any(got.get(s, 0) == 0 for s in SPLITS):
            print(f"WARNING: class {label} has too few source groups for all three splits: {got.to_dict()}")
    return split


def build_manifest(cfg) -> pd.DataFrame:
    df = scan_images(cfg)
    if cfg["dataset"].get("group_by_hash"):
        df = refine_groups(cfg, df)
    strategy = cfg["dataset"]["split_strategy"]
    have_official = df["official_split"].notna().all()
    if strategy == "official":
        if not have_official:
            raise RuntimeError(
                "split_strategy is 'official' but some/all images are not under train/val/test folders. "
                "Point data_root at the folder that contains train/val/test, or set split_strategy: group."
            )
        df["split"] = df["official_split"]
    elif strategy == "group":
        df["split"] = group_split(df, cfg["dataset"]["group_split"], cfg["seed"])
    else:
        raise ValueError(f"unknown split_strategy: {strategy}")

    audit = audit_dataset(df.assign(split=df["split"]), "split")
    if audit["n_groups_spanning_multiple_splits"]:
        print(f"WARNING: {audit['n_groups_spanning_multiple_splits']} source photos appear in more than one split "
              f"({audit['frac_images_in_cross_split_groups']:.1%} of images). Test scores will be inflated. "
              "Use dataset.split_strategy: group.")

    classes = cfg["dataset"]["classes"]
    absent = set(classes) - set(df["label"])
    if absent:
        print(f"WARNING: classes with no images: {sorted(absent)}")
    df["class_id"] = df["label"].map({c: i for i, c in enumerate(classes)})
    order = {s: i for i, s in enumerate(SPLITS)}
    df = (df.assign(_o=df["split"].map(order)).sort_values(["_o", "class_id", "path"])
            .drop(columns=["_o", "official_split"]).reset_index(drop=True))
    df = df[["path", "label", "split", "class_id", "group", "is_aug"]]

    df.to_csv(cfg["paths"]["manifest"], index=False)
    print(f"Wrote {cfg['paths']['manifest']}  ({len(df)} images, {df['group'].nunique()} source photos)")
    print(pd.crosstab(df["label"], df["split"])[list(SPLITS)])
    return df


def load_manifest(cfg) -> pd.DataFrame:
    p = cfg["paths"]["manifest"]
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run scripts/01_prepare_data.py first.")
    return pd.read_csv(p)


def split_indices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {s: np.where(df["split"].values == s)[0] for s in SPLITS}


# --------------------------------------------------------------------------- image cache
def cache_file(cfg):
    return cfg["paths"]["work_dir"] / "images_cache.npy"


def _meta_file(cfg):
    return cfg["paths"]["work_dir"] / "images_cache.meta.json"


def _paths_hash(df: pd.DataFrame, size: int) -> str:
    return hashlib.md5(("|".join(df["path"]) + f"#{size}").encode()).hexdigest()


def _read_rgb(path) -> np.ndarray | None:
    # imdecode(fromfile) also works with unicode paths on Windows
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_image_cache(cfg, df: pd.DataFrame, force: bool = False, n_threads: int = 8) -> None:
    size = cfg["preprocess"]["image_size"]
    out, meta = cache_file(cfg), _meta_file(cfg)
    h = _paths_hash(df, size)
    if out.exists() and meta.exists() and not force and load_json(meta).get("hash") == h:
        print(f"Image cache up to date: {out}")
        return

    root = cfg["paths"]["data_root"]
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(df), size, size, 3))
    paths = df["path"].tolist()

    def work(i):
        img = _read_rgb(root / paths[i])
        if img is None:
            return i, False
        mm[i] = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return i, True

    print(f"Building image cache ({len(df)} images -> {size}x{size}) ...")
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        results = list(ex.map(work, range(len(df))))
    mm.flush()
    bad = [paths[i] for i, ok in results if not ok]
    if bad:
        raise RuntimeError(f"{len(bad)} unreadable images, e.g. {bad[:5]}")
    save_json({"hash": h, "n": len(df), "size": size}, meta)
    print(f"Saved {out}  ({out.stat().st_size / 1e9:.2f} GB)")


def load_cache(cfg, df: pd.DataFrame | None = None) -> np.ndarray:
    f = cache_file(cfg)
    if not f.exists():
        raise FileNotFoundError(f"{f} missing. Run scripts/01_prepare_data.py first.")
    if df is not None and _meta_file(cfg).exists():
        if load_json(_meta_file(cfg)).get("hash") != _paths_hash(df, cfg["preprocess"]["image_size"]):
            raise RuntimeError("Image cache does not match manifest.csv. Re-run scripts/01_prepare_data.py --force")
    return np.load(f, mmap_mode="r")
