"""Leaf segmentation.  (Owner: Member A)

Deliberately simple and unsupervised (hue/saturation gate + morphology): the dataset has NO leaf masks, so this is an approximation.
Its quality directly limits (i) the shape features and (ii) the realism of the background perturbation.
Always inspect results/eda_segmentation.png and report failure rates in the Challenges section.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def _clean(m: np.ndarray, close_px: int = 5) -> np.ndarray:
    h, w = m.shape
    m8 = m.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kc)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.where(areas >= max(0.2 * areas.max(), 0.01 * h * w))[0]
    return ndi.binary_fill_holes(np.isin(lab, keep))


def _grabcut(img_rgb: np.ndarray, init: np.ndarray, iters: int) -> np.ndarray:
    h, w = init.shape
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[init] = cv2.GC_PR_FGD
    b = max(2, int(0.03 * min(h, w)))
    border = np.zeros((h, w), bool)
    border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
    gc[border & ~init] = cv2.GC_BGD
    cy, cx, r = h // 2, w // 2, int(0.1 * min(h, w))
    centre = np.zeros((h, w), bool)
    centre[cy - r:cy + r, cx - r:cx + r] = True
    gc[centre & init] = cv2.GC_FGD
    if not (gc == cv2.GC_BGD).any() or not (gc == cv2.GC_FGD).any():
        return init
    try:
        bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
        cv2.grabCut(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return init
    return (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)


def segment_leaf(img_rgb: np.ndarray, seg: dict) -> np.ndarray:
    """Boolean (H, W) mask of the leaf. Falls back to the whole image when segmentation fails."""
    h, w = img_rgb.shape[:2]
    method = seg["method"]
    if method == "none":
        return np.ones((h, w), bool)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    # leaf tissue = yellow-green-green hues (OpenCV hue 0-179). Brown soil/lesions fall outside this range;
    # lesions fully enclosed by healthy tissue are recovered by hole filling, lesions on the leaf edge are lost.
    fg = (hsv[..., 1] >= seg["sat_min"]) & (hsv[..., 2] >= seg["val_min"]) \
        & (hue >= seg["hue_min"]) & (hue <= seg["hue_max"])
    fg = _clean(fg, seg["close_px"])
    if method == "grabcut" and fg.any():
        fg = _clean(_grabcut(img_rgb, fg, seg["grabcut_iters"]), seg["close_px"])
    if fg.mean() < seg["min_leaf_frac"]:
        return np.ones((h, w), bool)
    return fg
