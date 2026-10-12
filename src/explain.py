"""Feature-importance computation and stability statistics.  (Owner: Member C)

Why grouped permutation importance: hand-crafted features are strongly correlated (e.g. RGB/HSV/LAB means),
so per-feature importance is split arbitrarily among correlated features and is expected to be unstable.
Permuting a whole group (color / texture / shape) jointly is a more reliable question to ask.
"""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import f1_score


def _score(model, X, y, metric: str) -> float:
    if metric == "macro_f1":
        return f1_score(y, model.predict(X), average="macro")
    # default: mean probability assigned to the true class. Continuous, so far fewer ties than F1/accuracy,
    # which matters because rank correlations are meaningless on vectors full of exact zeros.
    P = model.predict_proba(X)
    return float(P[np.arange(len(y)), np.searchsorted(model.classes_, y)].mean())


def permutation_importance(model, X, y, col_groups, n_repeats: int, rng, metric: str = "true_class_prob") -> np.ndarray:
    """Mean score drop when each column-group is permuted jointly (rows shuffled together)."""
    base = _score(model, X, y, metric)
    out = np.zeros((len(col_groups), n_repeats))
    for gi, cols in enumerate(col_groups):
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            out[gi, r] = base - _score(model, Xp, y, metric)
    return out.mean(1)


def topk_set(v: np.ndarray, k: int) -> set:
    return set(np.argsort(-v)[:k].tolist())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def rank_stability(M: np.ndarray, topk=(10, 20)) -> dict:
    """M: (n_runs, n_features) importances. Pairwise agreement between runs."""
    sp, kt = [], []
    jac = {k: [] for k in topk}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a, b in combinations(range(len(M)), 2):
            sp.append(spearmanr(M[a], M[b])[0])
            kt.append(kendalltau(M[a], M[b])[0])
            for k in topk:
                jac[k].append(jaccard(topk_set(M[a], k), topk_set(M[b], k)))
    out = {"spearman_mean": float(np.nanmean(sp)), "spearman_std": float(np.nanstd(sp)),
           "kendall_mean": float(np.nanmean(kt)), "kendall_std": float(np.nanstd(kt)), "n_pairs": len(sp)}
    for k in topk:
        out[f"jaccard_top{k}_mean"] = float(np.mean(jac[k]))
        out[f"jaccard_top{k}_std"] = float(np.std(jac[k]))
    return out


def agreement(u: np.ndarray, v: np.ndarray, topk=(10, 20)) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {"spearman": float(spearmanr(u, v)[0]), "kendall": float(kendalltau(u, v)[0])}
    for k in topk:
        out[f"jaccard_top{k}"] = jaccard(topk_set(u, k), topk_set(v, k))
    return out
