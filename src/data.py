"""Dataset indexing and the 256x256 image cache.  (Owner: Member A)

manifest.csv  : path (relative to data_root), label, split, class_id   -> one row per image, fixed order
images_cache  : uint8 RGB array (N, S, S, 3) in manifest order, memory-mapped by every later stage
"""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import load_json, save_json

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- manifest
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


def build_manifest(cfg) -> pd.DataFrame:
    root = cfg["paths"]["data_root"]
    if not root.exists():
        raise FileNotFoundError(
            f"data_root does not exist: {root}\n"
            "Set paths.data_root in config.local.yaml or the TOMATO_DATA_ROOT env variable."
        )
    exts = {e.lower() for e in cfg["dataset"]["image_extensions"]}
    rows, unmatched = [], {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(root)
        parts = rel.parts[:-1]
        label = _match_class(parts, cfg)
        if label is None:
            key = "/".join(parts) if parts else "."
            unmatched[key] = unmatched.get(key, 0) + 1
            continue
        rows.append({"path": rel.as_posix(), "label": label, "split": _match_split(parts, cfg)})

    if not rows:
        raise RuntimeError(
            f"No labelled images found under {root}.\n"
            f"Folders seen without a class match: {list(unmatched)[:15]}\n"
            "Fix dataset.class_keywords in config.yaml so each class folder name matches a keyword."
        )
    if unmatched:
        print(f"WARNING: {sum(unmatched.values())} images in folders with no class match: {list(unmatched)[:10]}")

    df = pd.DataFrame(rows)
    n_missing = int(df["split"].isna().sum())
    if n_missing == len(df):
        print("No train/val/test folders detected -> creating a stratified fallback split "
              f"{cfg['dataset']['fallback_split']} (seed={cfg['seed']}).")
        fr = cfg["dataset"]["fallback_split"]
        tr, rest = train_test_split(df.index, test_size=1 - fr["train"], stratify=df["label"], random_state=cfg["seed"])
        va, te = train_test_split(
            rest, test_size=fr["test"] / (fr["val"] + fr["test"]),
            stratify=df.loc[rest, "label"], random_state=cfg["seed"],
        )
        df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"
    elif n_missing:
        raise RuntimeError(f"{n_missing} images have no split folder while others do; check split_names in config.yaml")

    classes = cfg["dataset"]["classes"]
    absent = set(classes) - set(df["label"])
    if absent:
        print(f"WARNING: classes with no images: {sorted(absent)}")
    df["class_id"] = df["label"].map({c: i for i, c in enumerate(classes)})
    order = {s: i for i, s in enumerate(SPLITS)}
    df = (df.assign(_o=df["split"].map(order)).sort_values(["_o", "class_id", "path"])
            .drop(columns="_o").reset_index(drop=True))

    df.to_csv(cfg["paths"]["manifest"], index=False)
    print(f"Wrote {cfg['paths']['manifest']}  ({len(df)} images)")
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
