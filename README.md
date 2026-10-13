# Robust & Explainable Classical ML for Tomato Disease Recognition (CS-871)

Hand-crafted features (color / texture / shape) + Random Forest (SVM, LR as a brief check), evaluated for
**robustness** (controlled test-time perturbations) and **explainability stability** (feature-importance agreement).

## Do we need a GPU?
**No.** Everything is classical ML on CPU (OpenCV / scikit-image / scikit-learn). The slow part is feature
extraction (GLCM/LBP/segmentation per image), which is parallelised over CPU cores and cached. A laptop, a free
or a Colab CPU runtime all work. A GPU would only matter for the optional CNN reference baseline.

## Layout
```
config.yaml            all parameters (seed, paths, features, grids, perturbation levels)
src/                   all logic (importable, no notebook state)
  data.py              manifest + 256x256 image cache          [A]
  segment.py           leaf segmentation (hue gate + morphology) [A]
  perturb.py           brightness / contrast / shadow / background [A]
  duplicates.py        flip-aware near-duplicate (leakage) check [A]
  features.py          color, texture (LBP, GLCM), shape/edge/lesion features [B]
  models.py            RF / SVM / LR + validation tuning        [B]
  evaluate.py          metrics, bootstrap CI, confusion plots    [C]
  explain.py           permutation importance + stability stats  [C]
scripts/               one runnable script per stage (below)
notebooks/00_colab_runner.ipynb   thin launcher for Colab
data/manifest.csv      path,label,split,class_id (commit this once)
results/               small csv/json/png outputs (commit)
work/                  image cache, features, models (git-ignored; large)
```

## Dataset notes (Pakistan field tomato leaves, 6 classes)
We use **`Dataset (raw)`**: original photos, no pre-made split. (`Augmented Dataset` was generated from it, so its test
set contains augmented siblings of photos that may also be in train, and its brightness/flip/rotation changes would
confound the perturbation study.) Folder names are matched by keyword; the dataset's own spelling quirks are handled.
- **Our own split:** 70/15/15 per class, by **group**, with `config.yaml` `split_strategy: group`, seed 42. `data/manifest.csv`
  is committed once so all three of us use the identical split.
- **Groups:** file names are random, so images are grouped by visual similarity (`group_by_hash`): a 63-bit DCT
  perceptual hash (flip-aware) proposes candidate pairs, and a 32x32 colour-thumbnail correlation (>= 0.98) confirms them.
  Resized / recompressed / flipped / re-lit copies of one photo therefore never straddle train/test.
  It does NOT catch a different photo of the same leaf from another angle: state this in the Limitations.
- **Check after stage 01:** `results/grouping_summary.json` (pairs found, groups before/after, largest group).
  A suspiciously large group means unrelated leaves were chained together: raise `duplicates.min_correlation`.
  `results/duplicate_examples.png` (stage 02) shows example pairs to eyeball.
- Cross-class near-duplicate pairs are reported as possible label noise (worth a sentence in Challenges).
- CV in stage 7 is grouped as well, so copies never straddle folds.
- If you ever want to use `Augmented Dataset`: set `split_strategy: official` and run stage 0 to quantify leakage first.

## Pipeline (run in order; each stage is resumable)
| # | Script | Owner | Produces | Report section |
|---|---|---|---|---|
| 0 | `00_audit_dataset.py` | A | class/split counts, augmented share, **source-photo leakage** | Dataset, Challenges |
| 1 | `01_prepare_data.py` | A | `manifest.csv`, image cache | Dataset |
| 2 | `02_eda_duplicates.py` | A | class/size stats, brightness-by-class, segmentation check, **near-duplicate / leakage report**, perturbation gallery | Dataset, Challenges |
| 3 | `03_extract_features.py` | B | cached feature matrices (105 features) | Methodology |
| 4 | `04_baseline.py` | B | tuning tables, test metrics + 95% CI, confusion matrices, learning curve, leak-free test metrics | Preliminary results |
| 5 | `05_ablation.py` | B | feature-group ablation x {RF,SVM,LR} | Results |
| 6 | `06_robustness.py` | C | accuracy / macro-F1 per perturbation, per feature set | Results |
| 7 | `07_importance_stability.py` | C | CV-fold and perturbation rank stability (Spearman, Kendall, top-k Jaccard) | Results |
| 8 | `08_error_analysis.py` | C | most-confused pairs, misclassified examples, prediction shift under worst perturbation | Error analysis |
| 9 | `09_figures.py` | C | report-ready figures + `robustness_drop.csv` | all |

## Run it
**Local**
```bash
pip install -r requirements.txt
# point to the dataset (or put it in data/raw/):
export TOMATO_DATA_ROOT=/path/to/tomato_dataset      # Windows PowerShell: $env:TOMATO_DATA_ROOT="C:\..."
python scripts/00_audit_dataset.py
python scripts/01_prepare_data.py
python scripts/02_eda_duplicates.py
python scripts/03_extract_features.py --benchmark 100   # time it first
python scripts/03_extract_features.py
python scripts/04_baseline.py && python scripts/05_ablation.py
python scripts/06_robustness.py && python scripts/07_importance_stability.py
python scripts/08_error_analysis.py && python scripts/09_figures.py
```
**Colab:** open `notebooks/00_colab_runner.ipynb`, edit the CONFIG cell, run cells top to bottom.

Personal paths never go in `config.yaml`: use `config.local.yaml` (git-ignored) or the `TOMATO_*` env variables.

## Design decisions to cite in the report
- **Protocol:** hyper-parameters tuned on *validation* (fit on train); final models refit on train+val; the test split is used only for reported numbers. Stage 5/6 reuse the parameters tuned on all features (a stated simplification).
- **Leakage check:** stage 2 hashes every image (also h/v flips) and flags test images that have a near-duplicate in train/val; stage 4 reports metrics on the leak-free subset too.
- **Perturbations are test-time only.** Features *and segmentation* are recomputed on the perturbed image, so the numbers measure the whole pipeline.
- **Features** are computed on the leaf mask only. Color: mean/std/skew in RGB, HSV, LAB + hue/sat histograms (47). Texture: uniform LBP (two scales) + GLCM props (38). Shape: leaf outline, Hu moments, edge/gradient, non-green lesion morphology (20).
- **Explainability:** per-feature importance is expected to be unstable because features are correlated, so stage 7 also permutes whole groups (color/texture/shape) jointly. Importance is the drop in mean true-class probability (continuous, fewer ties than F1).

## Known limitations (put these in the Challenges / Limitations sections)
- No leaf masks exist; segmentation is unsupervised (hue gate). It separates a leaf from **soil**, but **fails on green backgrounds** (grass, other plants), where it falls back to the whole image. `results/eda_segmentation_stats.json` reports the failure rate. `method: grabcut` is a slower alternative: compare both.
- Brown lesions on the leaf *edge* are excluded by the hue gate; enclosed lesions are recovered by hole-filling.
- Background perturbation replaces whatever the mask calls background, so its realism depends on mask quality (a thin halo can remain).
- Images are resized to 256x256 without preserving aspect ratio (consistent across all splits).
- Brightness/contrast are global, shadows are a single soft diagonal; real fields mix these.

## Collaboration workflow
- `main` is protected; work on `a/...`, `b/...`, `c/...` branches and open a PR; a teammate reviews and merges (at least twice a week).
- Only edit files you own (table above); changes to `config.yaml` or an interface go through a PR everyone sees.
- Commit: code, `data/manifest.csv` (once), small files in `results/`. Never commit `work/`, `data/raw/`, `*.npy`, `*.joblib`.
- Notebook runs: `git pull --rebase` first; push results to your branch, not `main`.
- Colab wipes its disk: the runner backs `work/` up to Drive and restores it.

## Extending (only after the core is done)
Optional CNN reference baseline (e.g. MobileNetV2 fine-tune) would be the one place a GPU helps; add it as `src/cnn.py` + `scripts/10_cnn_baseline.py`.
