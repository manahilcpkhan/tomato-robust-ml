"""Classical models + validation-based tuning.  (Owner: Member B)

Protocol: hyper-parameters are chosen on the VALIDATION split with the model fitted on TRAIN only.
Final models are refitted on TRAIN+VAL with the chosen parameters and touch the TEST split only once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ("rf", "svm", "lr")


def make_model(name: str, params: dict, cfg: dict):
    seed = cfg["seed"]
    cw = cfg["models"].get("class_weight")
    if name == "rf":
        p = {"n_estimators": cfg["models"]["rf"]["n_estimators"], **params}
        return RandomForestClassifier(random_state=seed, n_jobs=cfg["n_jobs"], class_weight=cw, **p)
    if name == "svm":      # RBF-kernel SVM; scaling is essential for margin-based methods
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=seed, class_weight=cw, **params))
    if name == "lr":       # L2-regularised multinomial logistic regression
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, random_state=seed, class_weight=cw, **params))
    raise ValueError(name)


def tune(name: str, Xtr, ytr, Xva, yva, cfg: dict):
    """Grid search on the validation split. Returns (best_params, table of all configurations)."""
    grid = list(ParameterGrid(cfg["models"][name]["grid"]))
    rows = []
    for p in grid:
        pred = make_model(name, p, cfg).fit(Xtr, ytr).predict(Xva)
        rows.append({**{k: str(v) for k, v in p.items()},
                     "val_accuracy": accuracy_score(yva, pred),
                     "val_macro_f1": f1_score(yva, pred, average="macro")})
    table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    best_i = int(np.argmax([r["val_macro_f1"] for r in rows]))
    return grid[best_i], table
