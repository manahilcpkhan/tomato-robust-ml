"""Stage 7 (Member C): is explainability stable?  (importance = drop in mean true-class probability)
 Feature-importance agreement across CV folds and perturbations.

A) Stratified K-fold on train+val: RF fitted per fold; impurity importance + permutation importance (held-out fold).
   Pairwise Spearman / Kendall / top-k Jaccard between folds.
B) Fixed RF (stage 4) on clean vs each perturbed TEST set: rank agreement of permutation importance with clean.
Both per-feature and per feature GROUP (color / texture / shape permuted jointly).

Needs stage 6 (perturbed features). Outputs: results/importance_*.csv, importance_stability_*.json/csv, cv_scores.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_manifest, split_indices  # noqa: E402
from src.explain import agreement, permutation_importance, rank_stability  # noqa: E402
from src.features import GROUPS, group_columns, load_clean_features, perturbed_dir  # noqa: E402
from src.models import make_model  # noqa: E402
from src.perturb import all_conditions, condition_name  # noqa: E402
from src.utils import Timer, load_json, save_json, set_seed  # noqa: E402


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    res = cfg["paths"]["results_dir"]
    ex = cfg["explain"]
    best = load_json(res / "best_params.json")

    df = load_manifest(cfg)
    idx = split_indices(df)
    y = df["class_id"].values
    X, names = load_clean_features(cfg)
    Xtv, ytv = np.vstack([X["train"], X["val"]]), np.concatenate([y[idx["train"]], y[idx["val"]]])
    yte = y[idx["test"]]
    nf = len(names)
    singles = [np.array([i]) for i in range(nf)]
    gcols = group_columns(names)
    groups = [gcols[g] for g in GROUPS]
    rng = np.random.default_rng(cfg["seed"])

    # ---------------- A) across CV folds
    skf = StratifiedKFold(ex["cv_folds"], shuffle=True, random_state=cfg["seed"])
    imp_imp, imp_perm, grp_perm, cv_f1 = [], [], [], []
    for f, (tr, va) in enumerate(skf.split(Xtv, ytv)):
        with Timer(f"CV fold {f + 1}/{ex['cv_folds']}"):
            m = make_model("rf", best["rf"], cfg).fit(Xtv[tr], ytv[tr])
            cv_f1.append(f1_score(ytv[va], m.predict(Xtv[va]), average="macro"))
            imp_imp.append(m.feature_importances_)
            imp_perm.append(permutation_importance(m, Xtv[va], ytv[va], singles, ex["perm_repeats_cv"], rng))
            grp_perm.append(permutation_importance(m, Xtv[va], ytv[va], groups, ex["perm_repeats_cv"], rng))
    imp_imp, imp_perm, grp_perm = map(np.array, (imp_imp, imp_perm, grp_perm))
    pd.DataFrame({"fold": range(len(cv_f1)), "macro_f1": cv_f1}).to_csv(res / "cv_scores.csv", index=False)
    pd.DataFrame(imp_imp, columns=names).to_csv(res / "importance_cv_impurity.csv", index=False)
    pd.DataFrame(imp_perm, columns=names).to_csv(res / "importance_cv_permutation.csv", index=False)
    pd.DataFrame(grp_perm, columns=GROUPS).to_csv(res / "importance_cv_group_permutation.csv", index=False)
    save_json({"cv_macro_f1_mean": float(np.mean(cv_f1)), "cv_macro_f1_std": float(np.std(cv_f1)),
               "impurity": rank_stability(imp_imp, ex["topk"]),
               "permutation": rank_stability(imp_perm, ex["topk"]),
               "group_permutation": rank_stability(grp_perm, (1, 2))},
              res / "importance_stability_cv.json")
    top = pd.DataFrame({"feature": names, "impurity_mean": imp_imp.mean(0), "impurity_std": imp_imp.std(0),
                        "permutation_mean": imp_perm.mean(0), "permutation_std": imp_perm.std(0)})
    top["group"] = top["feature"].str.split("_").str[0]
    top.sort_values("permutation_mean", ascending=False).to_csv(res / "importance_top_features.csv", index=False)

    # ---------------- B) across perturbations (fixed model from stage 4)
    model = joblib.load(cfg["paths"]["work_dir"] / "models" / "rf_all.joblib")
    pdir = perturbed_dir(cfg)
    clean_feat = permutation_importance(model, X["test"], yte, singles, ex["perm_repeats_perturbed"], rng)
    clean_grp = permutation_importance(model, X["test"], yte, groups, ex["perm_repeats_perturbed"], rng)
    feat_rows, grp_rows, stab_rows = {"clean": clean_feat}, {"clean": clean_grp}, []
    for kind, level in all_conditions(cfg):
        cname = condition_name(kind, level)
        f = pdir / f"{cname}.npy"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - run scripts/06_robustness.py first")
        Xp = np.load(f)
        with Timer(f"permutation importance on {cname}"):
            feat_rows[cname] = permutation_importance(model, Xp, yte, singles, ex["perm_repeats_perturbed"], rng)
            grp_rows[cname] = permutation_importance(model, Xp, yte, groups, ex["perm_repeats_perturbed"], rng)
        stab_rows.append({"condition": cname, **agreement(clean_feat, feat_rows[cname], ex["topk"]),
                          "macro_f1_after": f1_score(yte, model.predict(Xp), average="macro")})
    pd.DataFrame(feat_rows).T.set_axis(names, axis=1).to_csv(res / "importance_perturbed_permutation.csv")
    pd.DataFrame(grp_rows).T.set_axis(list(GROUPS), axis=1).to_csv(res / "importance_perturbed_group.csv")
    pd.DataFrame(stab_rows).to_csv(res / "importance_stability_perturbed.csv", index=False)
    print(pd.DataFrame(stab_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
