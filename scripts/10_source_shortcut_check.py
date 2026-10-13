"""Stage 10: can the class be predicted from IMAGE METADATA ALONE (size, aspect ratio, file size)?

    python scripts/10_source_shortcut_check.py

If yes, the dataset has a source/provenance shortcut (e.g. one class mostly from stock photos, another from phone
photos), and part of any classifier's accuracy may come from the source, not the disease.
Reads image headers only (no decoding). Uses the manifest's grouped split (train+val -> test) and a grouped 5-fold CV.
Outputs (results/): source_shortcut_metadata.csv, source_shortcut_by_class.csv, source_shortcut_result.json
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_manifest  # noqa: E402
from src.utils import save_json  # noqa: E402


def main():
    cfg = load_config()
    res, root = cfg["paths"]["results_dir"], cfg["paths"]["data_root"]
    df = load_manifest(cfg)

    def meta(p):
        with Image.open(root / p) as im:
            w, h = im.size
        return w, h, (root / p).stat().st_size

    with ThreadPoolExecutor(max_workers=8) as ex:
        m = list(ex.map(meta, df["path"]))
    md = pd.DataFrame(m, columns=["width", "height", "file_bytes"])
    md["megapixels"] = md.width * md.height / 1e6
    md["aspect"] = md.width / md.height
    md["log_pixels"] = np.log10(md.width * md.height)
    md["bytes_per_pixel"] = md.file_bytes / (md.width * md.height)
    feats = ["width", "height", "aspect", "log_pixels", "file_bytes", "bytes_per_pixel"]
    out = pd.concat([df[["path", "label", "split", "group"]], md], axis=1)
    out.to_csv(res / "source_shortcut_metadata.csv", index=False)
    (out.groupby("label")[["megapixels", "aspect", "bytes_per_pixel"]].median()
        .join(out.groupby("label")["megapixels"].agg(p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9)))
        .round(3).to_csv(res / "source_shortcut_by_class.csv"))

    X, y, g = md[feats].values, df["class_id"].values, df["group"].values
    seed = cfg["seed"]
    rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                        random_state=seed, n_jobs=cfg["n_jobs"])
    tv, te = (df["split"] != "test").values, (df["split"] == "test").values
    pred = rf().fit(X[tv], y[tv]).predict(X[te])
    dummy = DummyClassifier(strategy="prior").fit(X[tv], y[tv]).predict(X[te])
    cvp = cross_val_predict(rf(), X, y, groups=g,
                            cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed))
    n_cls = len(np.unique(y))
    r = {"features": feats, "n_classes": n_cls, "chance_macro_f1_approx": 1 / n_cls,
         "test_accuracy_metadata_only": accuracy_score(y[te], pred),
         "test_macro_f1_metadata_only": f1_score(y[te], pred, average="macro"),
         "test_accuracy_majority_baseline": accuracy_score(y[te], dummy),
         "cv_macro_f1_metadata_only": f1_score(y, cvp, average="macro"),
         "cv_accuracy_metadata_only": accuracy_score(y, cvp)}
    save_json(r, res / "source_shortcut_result.json")
    print(pd.read_csv(res / "source_shortcut_by_class.csv").to_string(index=False))
    print(f"\nmetadata-only test macro-F1 = {r['test_macro_f1_metadata_only']:.3f}, "
          f"CV macro-F1 = {r['cv_macro_f1_metadata_only']:.3f}  (chance ~ {1 / n_cls:.3f})")
    print("A value well above chance means image size/compression alone reveals the class (source shortcut).")


if __name__ == "__main__":
    main()
