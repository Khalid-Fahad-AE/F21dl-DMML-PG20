# Tomato leaf disease detection: F21DL group coursework 2026/27

Classifying 10 tomato leaf conditions (9 diseases + healthy) from the **original PlantVillage** images,
from classic models on hand-crafted features up to CNNs and transfer learning. Every model is then
tested on **PlantDoc** field photos, to measure how far lab results carry over to the real world.

| Requirement | Where |
|---|---|
| R1 questions & objectives | below, and report §1 |
| R2 preprocessing, EDA, clustering | `src/data.py`, `src/eda.py`, `src/features.py`, `src/clustering.py` |
| Generative task | `src/generative.py` (class-conditional GMM vs Naive Bayes) |
| R3 ≥3 classic models + DT discussion | `src/classic.py`, `src/dt_explain.py`, shared metrics in `src/evaluate.py` |
| R4 neural networks | `src/mlp.py` (MLP), `src/cnn.py` + `src/gradcam.py` (CNN), `src/transfer.py` (transfer learning) |
| Real-world robustness | `src/plantdoc.py` (+ the PlantDoc step inside `cnn.py` / `transfer.py`) |

## Research questions
1. How well do classic ML models on hand-crafted colour/texture/shape features separate the 10 classes, and which features matter?
2. Does unsupervised structure in the images (k-means, GMM) follow disease labels, or something else?
3. Does a generative model (one GMM per class) beat Naive Bayes, i.e. does modelling feature correlations pay off?
4. How much do MLPs, a CNN trained from scratch, and pretrained CNNs gain over the classic models, and at what cost?
5. Do the models look at lesions, and do they still work on field photos (PlantDoc)?

## Data
* **PlantVillage**, original colour images (github.com/spMohanty/PlantVillage-Dataset, `raw/color/Tomato___*`):
  18,160 images, 256×256 RGB. We deliberately avoid the Kaggle "augmented" copy, whose rotated/flipped
  duplicates leak across splits.
* **De-duplication** (`data.py`): 14 exact duplicates (MD5); near-duplicates found with a 64-bit dHash
  (≤4 bits) and *confirmed* by a pixel check (mean abs difference ≤8 on 32×32 greyscale). The pixel check
  matters: dHash alone flagged 936 pairs, 333 of them across different diseases. Visual inspection showed these
  were different leaves on similar backdrops. After confirmation: 28 true pairs, **27 images removed, 18,133 kept**,
  0 label conflicts. All removed files are listed in `results/tables/removed_duplicates.csv`.
* **Leaf grouping**: PlantVillage file names contain a leaf ID; 291 leaves were photographed more than once
  (e.g. `GHLB2 Leaf 131` and `GHLB2 Leaf 131.1`). All photos of one leaf go into the same split.
* **Split**: stratified *and grouped* 70/15/15, seed 42 → `data/splits.csv` (train 12,696 / val 2,718 / test 2,719).
  **Everyone uses this file.** Test is used once per final model; all tuning uses CV on train or the val split.
* **PlantDoc** (github.com/pratikkayal/PlantDoc-Dataset): 746 tomato field photos in 9 of our classes
  (no Target spot), used **only** as an outside test set.

## Results (PlantVillage test split, 2,719 images)

| Model | Workstream | Accuracy | Macro-P | Macro-R | **Macro-F1** | PlantDoc macro-F1 |
|---|---|---|---|---|---|---|
| Small CNN from scratch, no augmentation (128 px) | E | 0.996 | 0.995 | 0.995 | **0.995** | 0.142 |
| Small CNN from scratch, geometric aug. | E | 0.994 | 0.993 | 0.993 | **0.993** | 0.134 |
| EfficientNet-B0, fine-tuned (224 px) | F | 0.993 | 0.992 | 0.991 | **0.991** | 0.176 |
| Small CNN from scratch, full aug. | E | 0.992 | 0.989 | 0.992 | **0.990** | 0.175 |
| ResNet50, fine-tuned | F | 0.990 | 0.988 | 0.988 | **0.988** | 0.206 |
| MobileNetV2, fine-tuned | F | 0.986 | 0.983 | 0.985 | **0.984** | 0.150 |
| EfficientNet-B0 / MobileNetV2 / ResNet50, frozen backbone | F | 0.885 / 0.874 / 0.861 | | | 0.862 / 0.840 / 0.831 | |
| Random Forest (300 trees) | C | 0.922 | 0.917 | 0.894 | **0.904** | 0.060 |
| MLP on features (1024-512, BN, dropout 0.3) | D | 0.918 | 0.902 | 0.898 | **0.899** | 0.071 |
| Logistic Regression (PCA features) | C | 0.886 | 0.866 | 0.855 | 0.859 | 0.080 |
| MLP on 64×64 pixels | D | 0.885 | 0.861 | 0.848 | 0.853 | 0.062 |
| Class-conditional GMM (generative) | B | 0.835 | 0.811 | 0.789 | 0.799 | 0.064 |
| k-NN (k=3, cosine) | C | 0.847 | 0.813 | 0.792 | 0.799 | 0.076 |
| Decision Tree (depth 20) | C | 0.812 | 0.777 | 0.771 | 0.773 | 0.094 |
| Gaussian Naive Bayes | B/C | 0.698 | 0.659 | 0.708 | 0.672 | 0.075 |
| Decision Tree, depth 5 / depth 3 (interpretable) | C | 0.656 / 0.535 | | | 0.505 / 0.319 | |

Full table: `results/tables/summary_test.md`; per-class F1: `results/tables/per_class_f1.csv`;
confusion matrices: `results/figures/cm_*.png`; headline figure: `results/figures/model_comparison.png`.

### Hypotheses
* **H1: transfer learning beats every classic model.** Supported: fine-tuned backbones reach macro-F1
  0.984–0.991 vs 0.904 for the random forest. But a small CNN trained from scratch does just as well (0.995),
  so ImageNet pretraining is not needed on 12,700 uniform lab images.
* **H2: every model loses substantial accuracy on field photos.** Supported for all 14 models: macro-F1 drops
  by 0.60–0.86 from PlantVillage to PlantDoc (`results/tables/summary_plantdoc.md`).
* **H3: augmentation narrows that loss.** Partly supported: full augmentation (colour jitter + zoom) raises the
  small CNN's PlantDoc macro-F1 from 0.14 to 0.18, a small gain on only 746 photos, so a trend rather than a
  proven effect. Geometric augmentation alone (0.13) has no effect, because rotations and flips do not change
  backdrop or lighting.

### Key findings
* **Field photos break every feature-based model.** On PlantDoc all models fall to macro-F1 0.06–0.09
  (accuracy 15–17%, below the 20% majority-class rate). Predictions collapse onto *Late blight* and *YLCV*:
  the features encode photo conditions (backdrop, lighting) as much as lesions. This is the "controlled
  conditions" risk from our pitch, now measured.
* **CNNs do better in the field, but are still far from usable.** PlantDoc macro-F1 is 0.13–0.21 (accuracy
  22–33%) for the six neural image models, 2–3× the feature-based models. ResNet50 is best (0.206) at ~131 ms
  per image on a CPU, vs ~28 ms for MobileNetV2 (`results/tables/transfer_cost.csv`). Healthy field leaves are
  almost never recognised (Healthy F1 ≤ 0.06).
* **Fine-tuning matters for transfer learning**: unfreezing the last blocks adds 0.13–0.16 macro-F1 over a
  frozen backbone (0.83–0.86 → 0.98–0.99).
* **Grad-CAM: the CNNs look at the leaf, but not only at the leaf.** On correct predictions the leaf-focus ratio
  (share of attention on the leaf ÷ share of the image that is leaf) is 1.48 for the plain CNN, 1.67 with full
  augmentation and 1.60 for MobileNetV2; wrong predictions focus less (1.25–1.51). Still, 37–43% of the attention
  on correct predictions falls off the leaf (`results/tables/gradcam_summary_*.csv`).
* **Backdrop bias is visible in the EDA**: hue histograms differ by class in the 0.6–0.95 (purple/grey backdrop)
  range, not only in leaf colours. t-SNE shows several separate "Healthy" islands and an isolated Late-blight group,
  consistent with photo sessions.
* **Clustering finds structure, but not disease structure**: k-means / GMM with k=10 reach ARI ≤ 0.23 and
  NMI ≤ 0.31 against the true labels (silhouette ≤ 0.07). Colour features alone cluster closest to the labels
  (ARI 0.22), shape (HOG) the least (0.08).
* **Generative**: one GMM per class (PCA-200, 8 diagonal components) gives macro-F1 0.799 vs 0.621 for Naive Bayes on
  the same input. Most of the gain comes from capturing correlations and multi-modal classes. Full covariance with
  many components overfits badly at 200 dims (val F1 0.09).
* **Decision tree**: colour histograms carry 59% of the importance, LBP texture 20%; the root split is a texture
  feature (share of flat LBP patterns). A depth-3 tree gives readable rules (e.g. *busy, edge-rich texture
  (few flat LBP patterns, >18% edge patterns) and >2.4% strongly saturated pixels → YLCV*, a leaf that is 95% pure
  over 1,837 training images, which matches YLCV's crinkled, yellowed leaves) but only macro-F1 0.32, so a farmer-facing rule card
  would need to be paired with a stronger model.
* **MLP**: depth, width, dropout and batch norm move val macro-F1 only within 0.88–0.91; early stopping reaches the
  same score in ~⅓ of the epochs; without dropout the network memorises the training set by epoch 3. Hand-crafted
  features beat raw pixels (0.899 vs 0.853) with 11× fewer parameters (1.1M vs 12.9M), which is the motivation for convolutions.
* **Hardest class**: Early blight (best F1 0.73), confused with Late blight and YLCV.
* **Feature selection** (`src/feature_selection.py`, validation split, random forest): the 64 colour-histogram
  features alone score macro-F1 0.936, *higher* than all 2,624 features (0.887). Removing colour costs 0.119;
  removing HOG shape slightly *improves* the score. A mutual-information filter peaks at the top 500 features (0.914).
  Colour drives the classifier, consistent with the background/session bias above.
* **Reliability** (`src/seeds.py`): over 5 seeds, random forest 0.898 ± 0.003 and MLP 0.898 ± 0.004 test macro-F1.
  Paired bootstrap 95% CI for RF − MLP: [−0.014, +0.016], so the two are tied; both beat logistic regression
  (0.859, CI 0.844–0.873).

## How to reproduce

```bash
# Python >= 3.11
pip install -r requirements.txt

# 1. data (only the tomato folders, ~314 MB)
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git ../pv
cd ../pv && git sparse-checkout init --no-cone && echo '/raw/color/Tomato___*/' > .git/info/sparse-checkout && git checkout master && cd -
# symlink targets are resolved from inside data/, so they need ../../ to reach the repo's sibling folders
ln -s ../../pv/raw/color data/plantvillage_color
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git ../plantdoc
cd ../plantdoc && git sparse-checkout init --no-cone && printf '/train/Tomato*\n/test/Tomato*\n' > .git/info/sparse-checkout && git checkout master && cd -
ln -s ../../plantdoc data/plantdoc
ls data/plantvillage_color data/plantdoc > /dev/null && echo "data links OK"

# 2. pipeline (CPU; times measured on a 2-core machine)
python src/data.py            # A  splits.csv (already committed; re-running reproduces it exactly)  ~1 min
python src/eda.py             # A  EDA figures                                                      <1 min
python src/features.py        # B  features.npz                                                     ~2 min
python src/clustering.py      # B  k-means / GMM                                                    <1 min
python src/generative.py      # B  class-conditional GMM vs NB                                      ~1 min
python src/classic.py         # C  DT, NB, KNN, LR, RF with 5-fold CV                               ~45 min
python src/dt_explain.py      # C  interpretable shallow trees                                      <1 min
python src/mlp.py             # D  MLP experiments                                                  ~20 min
python src/plantdoc.py        # F  field-photo test of the models above                              ~1 min
python src/feature_selection.py  # B/C block ablation + mutual-information filter                   ~12 min
python src/seeds.py           # C/D seed variance + bootstrap confidence intervals                  ~12 min
# 3. GPU part (E, F): notebooks/run_EF_on_colab.ipynb, or:
python src/cnn.py --aug all --size 128
python src/gradcam.py --model CNN_scratch_full_128
python src/transfer.py --arch all
# 4. tables + headline figure
python src/summarise.py
python tests/test_mlp_grad.py  # gradient check for the NumPy MLP
python tests/test_splits.py    # split integrity: sizes, no leaf leakage, no duplicates, stratification
```

## Repository layout
```
data/splits.csv                the shared split (committed)
data/plantdoc_tomato.csv       PlantDoc index used for the field test
src/                           one script per workstream (see table at the top)
notebooks/run_EF_on_colab.ipynb   GPU runs for the CNN, Grad-CAM, transfer learning
results/tables/                every number in the report (CSV / JSON / markdown)
results/figures/               every figure in the report
tests/test_mlp_grad.py         finite-difference gradient check
tests/test_splits.py           integrity checks on the shared split (standard library only)
```

## Notes
* All randomness is seeded with 42 (`data.SEED`).
* Metrics: macro-F1 is the selection metric everywhere because classes are imbalanced 14:1 (YLCV 29.5% vs
  Mosaic virus 2.1%); accuracy alone would reward predicting YLCV.
* The MLP is written in NumPy (scikit-learn's MLP has no dropout or batch norm); `tests/test_mlp_grad.py`
  checks its gradients against finite differences (max relative error ~1e-8).

## References
* Hughes, D. P. & Salathé, M. (2015). An open access repository of images on plant health… arXiv:1511.08060.
* Mohanty, S. P., Hughes, D. P. & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science* 7:1419.
* Singh, D. et al. (2020). PlantDoc: a dataset for visual plant disease detection. *CoDS-COMAD 2020*.
* Selvaraju, R. R. et al. (2017). Grad-CAM. *ICCV 2017*.
* Sandler et al. (2018) MobileNetV2; Tan & Le (2019) EfficientNet; He et al. (2016) ResNet.
