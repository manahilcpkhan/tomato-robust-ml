"""Near-duplicate detection with a DCT perceptual hash (pHash), flip-aware.  (Owner: Member A)

Why: if near-identical photos sit in both train and test, test scores are inflated.
Catches resized / recompressed / slightly re-lit copies and h/v flips (pHash keeps only coarse low-frequency
structure, so JPEG noise and brightness scaling barely move it). It will NOT catch crops, rotations, or a
different photo of the same leaf.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

FLIPS = (None, "h", "v", "hv")


def _dhash(img_rgb: np.ndarray, hash_size: int, flip) -> np.ndarray:
    """pHash bits: low-frequency (hash_size x hash_size) DCT block of a 32x32 grey image, DC dropped, vs median.
    (Function name kept for compatibility.)  Returns hash_size**2 - 1 bits."""
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if flip == "h":
        g = g[:, ::-1]
    elif flip == "v":
        g = g[::-1]
    elif flip == "hv":
        g = g[::-1, ::-1]
    small = cv2.resize(np.ascontiguousarray(g), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    block = cv2.dct(small)[:hash_size, :hash_size].ravel()[1:]
    return block > np.median(block)


def _small_gray(img_rgb: np.ndarray) -> np.ndarray:
    """32x32 COLOUR thumbnail (name kept for compatibility): more discriminative than grey."""
    return cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def _flip(a: np.ndarray, f) -> np.ndarray:
    return a if f is None else (a[:, ::-1] if f == "h" else (a[::-1] if f == "v" else a[::-1, ::-1]))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def find_near_duplicates(imgs, hash_size: int = 8, thr: int = 10, chunk: int = 1024,
                         min_corr: float = 0.95) -> pd.DataFrame:
    """DataFrame(i, j, hamming, corr, variant) of unordered near-duplicate pairs (i < j).

    Stage 1: pHash Hamming distance <= thr (cheap, all pairs, flip-aware).
    Stage 2: the candidate must also have normalised correlation >= min_corr between 32x32 colour thumbnails
    (after the same flip). This removes the false matches a coarse hash makes between different leaves
    that merely look alike (centred leaf, similar background)."""
    n = len(imgs)
    H = {f: np.zeros((n, hash_size * hash_size - 1), np.float32) for f in FLIPS}
    S = np.zeros((n, 32, 32, 3), np.float32)
    for i in range(n):
        img = np.asarray(imgs[i])
        S[i] = _small_gray(img)
        for f in FLIPS:
            H[f][i] = _dhash(img, hash_size, f)

    base = H[None]
    sb = base.sum(1)
    cand = {}
    for f in FLIPS:
        other = H[f]
        so = other.sum(1)
        for s in range(0, n, chunk):
            ham = sb[s:s + chunk, None] + so[None, :] - 2.0 * (base[s:s + chunk] @ other.T)
            ii, jj = np.where(ham <= thr)
            ii = ii + s
            for a, b in zip(ii, jj):
                if a == b:
                    continue
                key = (int(min(a, b)), int(max(a, b)))
                h = int(round(ham[a - s, b]))
                if key not in cand or h < cand[key][0]:
                    cand[key] = (h, f, int(a), int(b))
    found = []
    for (i, j), (h, f, a, b) in cand.items():
        c = _ncc(S[a], _flip(S[b], f))
        if c >= min_corr:
            found.append((i, j, h, round(c, 4), f or "identity"))
    if not found:
        return pd.DataFrame(columns=["i", "j", "hamming", "corr", "variant"])
    return pd.DataFrame(found, columns=["i", "j", "hamming", "corr", "variant"]).sort_values(
        ["hamming", "corr"], ascending=[True, False]).reset_index(drop=True)
