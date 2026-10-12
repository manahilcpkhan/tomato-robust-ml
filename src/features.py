"""Hand-crafted features: color / texture / shape.  (Owner: Member B)

Every feature name starts with its group prefix: color_*, texture_*, shape_*.
All statistics are computed on the leaf mask only (see segment.py), not on the background.
Features are extracted in parallel from the cached 256x256 images and stored as ONE matrix per split
(work/features/clean_{split}.npy) plus feature_names.json; feature sets are column slices by prefix.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from .data import SPLITS
from .perturb import perturb
from .segment import segment_leaf
from .utils import load_json, save_json

EPS = 1e-8
GROUPS = ("color", "texture", "shape")
FEATURE_SETS = {
    "color": ["color"],
    "texture": ["texture"],
    "shape": ["shape"],
    "color+texture": ["color", "texture"],
    "color+shape": ["color", "shape"],
    "texture+shape": ["texture", "shape"],
    "all": ["color", "texture", "shape"],
}


def worker_cfg(cfg) -> dict:
    """Small picklable subset of the config sent to joblib workers."""
    return {"seed": cfg["seed"], "segmentation": cfg["segmentation"], "features": cfg["features"]}


# --------------------------------------------------------------------------- color
def _stats(x: np.ndarray):
    m, s = float(x.mean()), float(x.std())
    return m, s, float(((x - m) ** 3).mean() / (s ** 3 + EPS))


def color_features(img, mask, fc, out):
    spaces = {
        "rgb": (img, "RGB"),
        "hsv": (cv2.cvtColor(img, cv2.COLOR_RGB2HSV), "HSV"),
        "lab": (cv2.cvtColor(img, cv2.COLOR_RGB2LAB), "LAB"),
    }
    hsv = None
    for sp, (arr, chans) in spaces.items():
        px = arr[mask].astype(np.float32)
        if sp == "hsv":
            hsv = px
        for c, ch in enumerate(chans):
            for stat, v in zip(("mean", "std", "skew"), _stats(px[:, c])):
                out[f"color_{sp}_{ch}_{stat}"] = v
    n = len(hsv)
    hh = np.histogram(hsv[:, 0], bins=fc["hue_bins"], range=(0, 180))[0] / n
    sh = np.histogram(hsv[:, 1], bins=fc["sat_bins"], range=(0, 256))[0] / n
    for k, v in enumerate(hh):
        out[f"color_hue_hist_{k}"] = v
    for k, v in enumerate(sh):
        out[f"color_sat_hist_{k}"] = v


# --------------------------------------------------------------------------- texture
def texture_features(img, mask, inner, ft, out):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    n_in = max(int(inner.sum()), 1)
    for P, R in ft["lbp"]:
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist = np.bincount(lbp[inner].astype(int), minlength=P + 2)[: P + 2] / n_in
        for k, v in enumerate(hist):
            out[f"texture_lbp_P{P}R{R}_{k}"] = v

    levels = ft["glcm_levels"]
    g = np.where(mask, gray, int(gray[mask].mean()))          # background -> neutral leaf-mean grey
    q = np.minimum((g.astype(np.float32) / 256.0 * levels).astype(np.uint8), levels - 1)
    glcm = graycomatrix(q, distances=ft["glcm_distances"], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels, symmetric=True, normed=True)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        v = graycoprops(glcm, prop)                           # (n_dist, n_angles)
        for di, d in enumerate(ft["glcm_distances"]):
            out[f"texture_glcm_{prop}_d{d}"] = float(v[di].mean())


# --------------------------------------------------------------------------- shape / edge / lesions
def shape_features(img, mask, inner, ft, out):
    h, w = mask.shape
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = float(mask.sum()) + EPS
    out["shape_leaf_area_frac"] = area / (h * w)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        harea = cv2.contourArea(cv2.convexHull(c)) + EPS
        x, y, bw, bh = cv2.boundingRect(c)
        out["shape_circularity"] = 4 * np.pi * area / (per ** 2 + EPS)
        out["shape_solidity"] = area / harea
        out["shape_extent"] = area / (bw * bh + EPS)
    else:
        out["shape_circularity"] = out["shape_solidity"] = out["shape_extent"] = 0.0

    M = cv2.moments(m8, binaryImage=True)
    m00 = M["m00"] + EPS
    mu20, mu02, mu11 = M["mu20"] / m00, M["mu02"] / m00, M["mu11"] / m00
    tmp = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    l1, l2 = (mu20 + mu02) / 2 + tmp, (mu20 + mu02) / 2 - tmp
    out["shape_eccentricity"] = float(np.sqrt(max(0.0, 1 - l2 / (l1 + EPS))))
    hu = cv2.HuMoments(M).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    for k, v in enumerate(hu):
        out[f"shape_hu{k + 1}"] = float(v)

    # edges / gradients inside the leaf
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    out["shape_edge_density"] = float(edges[inner].mean() / 255.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)[inner]
    out["shape_grad_mean"] = float(mag.mean())
    out["shape_grad_std"] = float(mag.std())
    out["shape_laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_32F)[inner].var())

    # lesion morphology: connected "non-green" regions inside the leaf
    f = img.astype(np.float32)
    s = f.sum(2) + 1.0
    exg = 2 * f[..., 1] / s - f[..., 0] / s - f[..., 2] / s
    les = ((exg < ft["lesion_exg_thresh"]) & inner).astype(np.uint8)
    les = cv2.morphologyEx(les, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(les, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([], int)
    areas = areas[areas >= ft["min_lesion_px"]]
    leaf = float(inner.sum()) + EPS
    out["shape_lesion_frac"] = float(les.sum()) / leaf
    out["shape_lesion_count_log"] = float(np.log1p(len(areas)))
    out["shape_lesion_largest_frac"] = float(areas.max() / leaf) if len(areas) else 0.0
    out["shape_lesion_mean_frac"] = float(areas.mean() / leaf) if len(areas) else 0.0


# --------------------------------------------------------------------------- one image
def extract_named(img: np.ndarray, sub: dict) -> OrderedDict:
    mask = segment_leaf(img, sub["segmentation"])
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if inner.sum() < 100:
        inner = mask
    out: OrderedDict = OrderedDict()
    color_features(img, mask, sub["features"], out)
    texture_features(img, mask, inner, sub["features"], out)
    shape_features(img, mask, inner, sub["features"], out)
    return out


def feature_schema(sub: dict) -> list[str]:
    rng = np.random.default_rng(0)
    dummy = cv2.resize(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8), (256, 256), interpolation=cv2.INTER_CUBIC)
    return list(extract_named(dummy, sub).keys())


# --------------------------------------------------------------------------- many images
def _worker(cache_path: str, idx_chunk, sub: dict, cond):
    arr = np.load(cache_path, mmap_mode="r")
    rows = []
    for i in idx_chunk:
        img = np.ascontiguousarray(arr[i])
        if cond is not None:
            img = perturb(img, cond[0], cond[1], int(i), sub)
        rows.append(np.fromiter(extract_named(img, sub).values(), dtype=np.float32))
    return np.stack(rows)


def extract_matrix(cache_path, idx, sub: dict, n_jobs: int = -1, cond=None) -> np.ndarray:
    """Feature matrix (len(idx), n_features) for cache rows `idx`; `cond=(kind, level)` perturbs first."""
    idx = np.asarray(idx)
    n_workers = os.cpu_count() or 1 if n_jobs in (-1, None) else n_jobs
    chunks = [c for c in np.array_split(idx, max(1, min(len(idx), n_workers * 8))) if len(c)]
    parts = Parallel(n_jobs=n_jobs)(delayed(_worker)(str(cache_path), c, sub, cond) for c in chunks)
    X = np.concatenate(parts)
    bad = ~np.isfinite(X)
    if bad.any():
        print(f"  note: replaced {int(bad.sum())} non-finite feature values with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# --------------------------------------------------------------------------- storage helpers
def feature_dir(cfg):
    d = cfg["paths"]["work_dir"] / "features"
    d.mkdir(parents=True, exist_ok=True)
    return d


def perturbed_dir(cfg):
    d = feature_dir(cfg) / "perturbed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_clean_features(cfg):
    d = feature_dir(cfg)
    if not (d / "feature_names.json").exists():
        raise FileNotFoundError("Clean features missing. Run scripts/03_extract_features.py first.")
    names = load_json(d / "feature_names.json")
    X = {s: np.load(d / f"clean_{s}.npy") for s in SPLITS}
    return X, names


def columns_for_set(names: list[str], set_name: str) -> np.ndarray:
    groups = FEATURE_SETS[set_name]
    return np.array([i for i, n in enumerate(names) if n.split("_")[0] in groups])


def group_columns(names: list[str]) -> dict[str, np.ndarray]:
    return {g: np.array([i for i, n in enumerate(names) if n.split("_")[0] == g]) for g in GROUPS}
