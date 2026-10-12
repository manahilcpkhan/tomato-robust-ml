"""Near-duplicate detection with a difference hash (dHash), flip-aware.  (Owner: Member A)

Why: if near-identical photos sit in both train and test, test scores are inflated.
Catches resized/recompressed copies and h/v flips. It will NOT catch crops or arbitrary rotations.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

FLIPS = (None, "h", "v", "hv")


def _dhash(img_rgb: np.ndarray, hash_size: int, flip) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if flip == "h":
        g = g[:, ::-1]
    elif flip == "v":
        g = g[::-1]
    elif flip == "hv":
        g = g[::-1, ::-1]
    small = cv2.resize(np.ascontiguousarray(g), (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).ravel()


def find_near_duplicates(imgs, hash_size: int = 16, thr: int = 10, chunk: int = 1024) -> pd.DataFrame:
    """Return DataFrame(i, j, hamming, variant) of unordered near-duplicate pairs (i < j)."""
    n = len(imgs)
    H = {f: np.zeros((n, hash_size * hash_size), np.float32) for f in FLIPS}
    for i in range(n):
        img = np.asarray(imgs[i])
        for f in FLIPS:
            H[f][i] = _dhash(img, hash_size, f)

    base = H[None]
    sb = base.sum(1)
    found = []
    for f in FLIPS:
        other = H[f]
        so = other.sum(1)
        for s in range(0, n, chunk):
            ham = sb[s:s + chunk, None] + so[None, :] - 2.0 * (base[s:s + chunk] @ other.T)
            ii, jj = np.where(ham <= thr)
            ii = ii + s
            keep = ii != jj
            for a, b, hv in zip(ii[keep], jj[keep], ham[ii[keep] - s, jj[keep]]):
                found.append((min(a, b), max(a, b), int(hv), f or "identity"))
    if not found:
        return pd.DataFrame(columns=["i", "j", "hamming", "variant"])
    df = pd.DataFrame(found, columns=["i", "j", "hamming", "variant"])
    return df.sort_values("hamming").drop_duplicates(["i", "j"]).reset_index(drop=True)
