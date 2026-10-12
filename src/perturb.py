"""Controlled test-time perturbations.  (Owner: Member A)

perturb(img, kind, level, idx, sub) -> uint8 RGB image, deterministic given (seed, image idx, kind, level).
kinds: brightness | contrast | shadow | background
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np

from .segment import segment_leaf


def condition_name(kind: str, level) -> str:
    return f"{kind}_{level}"


def _rng(seed: int, idx: int, kind: str, level) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(idx), zlib.crc32(condition_name(kind, level).encode())])


def _shadow_map(shape, strength: float, rng) -> np.ndarray:
    h, w = shape
    theta = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = ((xx - w / 2) * np.cos(theta) + (yy - h / 2) * np.sin(theta)) / (max(h, w) / 2)
    t = 1.0 / (1.0 + np.exp(-(proj - rng.uniform(-0.3, 0.3)) / 0.15))
    return 1.0 - strength * t


def _lowfreq(h, w, scale, rng) -> np.ndarray:
    small = rng.random((scale, scale, 3), dtype=np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _synthetic_background(kind: str, shape, rng) -> np.ndarray:
    h, w = shape
    if kind == "gray":
        return np.full((h, w, 3), 128, np.float32)
    if kind == "noise":
        return _lowfreq(h, w, 6, rng) * 255.0
    base = {"soil": (110, 80, 55), "grass": (60, 115, 40)}[kind]
    bg = np.array(base, np.float32) + (_lowfreq(h, w, 10, rng) - 0.5) * 60
    bg += (rng.random((h, w, 1), dtype=np.float32) - 0.5) * 30
    return bg


def _replace_background(img: np.ndarray, kind: str, rng, sub: dict) -> np.ndarray:
    mask = segment_leaf(img, sub["segmentation"])
    if mask.all():                      # segmentation failed -> nothing to replace
        return img.astype(np.float32)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    alpha = cv2.GaussianBlur(mask, (0, 0), 2)[..., None]
    bg = _synthetic_background(kind, img.shape[:2], rng)
    return alpha * img.astype(np.float32) + (1 - alpha) * bg


def perturb(img: np.ndarray, kind: str, level, idx: int, sub: dict) -> np.ndarray:
    rng = _rng(sub["seed"], idx, kind, level)
    x = img.astype(np.float32)
    if kind == "brightness":
        x = x * float(level)
    elif kind == "contrast":
        m = x.mean()
        x = (x - m) * float(level) + m
    elif kind == "shadow":
        x = x * _shadow_map(x.shape[:2], float(level), rng)[..., None]
    elif kind == "background":
        x = _replace_background(img, str(level), rng, sub)
    else:
        raise ValueError(f"unknown perturbation kind: {kind}")
    return np.clip(x, 0, 255).astype(np.uint8)


def all_conditions(cfg) -> list[tuple[str, object]]:
    return [(kind, level) for kind, levels in cfg["perturbations"].items() for level in levels]
