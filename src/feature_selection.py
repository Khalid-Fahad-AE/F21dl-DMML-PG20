"""Workstream B/C - feature selection and evaluation (requirement R2).

Two experiments with the best random forest from classic.py (300 trees, max_features='sqrt',
min_samples_leaf=3, class_weight='balanced_subsample'), scored on the VALIDATION split so the test
split stays untouched:

  1. Leave-one-block-out ablation: drop each feature block (hsv / hog / lbp / pix) in turn.
     The drop in macro-F1 shows how much each kind of information is worth.
  2. Filter selection with mutual information: rank all 2,624 features by mutual information with
     the label (computed on TRAIN only), keep the top k, retrain.

    python src/feature_selection.py
Writes results/tables/feature_ablation.csv, feature_mi_topk.csv, mi_top_features.csv
and results/figures/feature_selection.png
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score

from classic import feature_names
from data import ROOT, SEED, set_seed
from features import load_features

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"


def rf():
    return RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=3,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)


def score(cols, d):
    t0 = time.time()
    m = rf().fit(d["Xtr"][:, cols], d["ytr"])
    return f1_score(d["yva"], m.predict(d["Xva"][:, cols]), average="macro"), time.time() - t0


def main():
    set_seed()
    d = load_features(pca=False)
    n = d["Xtr"].shape[1]
    names = feature_names(d)
    blocks = {b: np.arange(s, e) for b, (s, e) in zip(d["block_names"], d["block_slices"])}

    # 1. leave-one-block-out
    full, t = score(np.arange(n), d)
    rows = [{"features": "all 4 blocks", "n_features": n, "val_macro_f1": full, "drop_vs_all": 0.0, "fit_s": t}]
    for b, idx in blocks.items():
        keep = np.setdiff1d(np.arange(n), idx)
        f, t = score(keep, d)
        rows.append({"features": f"without {b}", "n_features": len(keep), "val_macro_f1": f,
                     "drop_vs_all": full - f, "fit_s": t})
        print(rows[-1])
    for b, idx in blocks.items():  # each block on its own, for reference
        f, t = score(idx, d)
        rows.append({"features": f"{b} only", "n_features": len(idx), "val_macro_f1": f,
                     "drop_vs_all": full - f, "fit_s": t})
        print(rows[-1])
    abl = pd.DataFrame(rows)
    abl.round(4).to_csv(TAB / "feature_ablation.csv", index=False)

    # 2. mutual-information filter (train only)
    mi = mutual_info_classif(d["Xtr"], d["ytr"], random_state=SEED)
    order = np.argsort(-mi)
    pd.DataFrame({"feature": np.array(names)[order[:25]], "mutual_information": mi[order[:25]]}).round(4).to_csv(
        TAB / "mi_top_features.csv", index=False)
    block_of = np.empty(n, dtype=object)
    for b, idx in blocks.items():
        block_of[idx] = b
    rows = []
    for k in [25, 50, 100, 250, 500, 1000, n]:
        cols = order[:k]
        f, t = score(cols, d)
        mix = pd.Series(block_of[cols]).value_counts().to_dict()
        rows.append({"top_k": k, "val_macro_f1": f, "fit_s": t, **{f"n_{b}": mix.get(b, 0) for b in blocks}})
        print(rows[-1])
    topk = pd.DataFrame(rows)
    topk.round(4).to_csv(TAB / "feature_mi_topk.csv", index=False)

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    a = abl[abl.features.str.startswith("without")]
    axes[0].barh(a.features, a.drop_vs_all, color="#3b6ea8")
    axes[0].set_xlabel("drop in validation macro-F1 when removed")
    axes[0].set_title(f"Leave-one-block-out (all blocks: {full:.3f})", fontsize=9)
    axes[0].invert_yaxis()
    axes[1].plot(topk.top_k, topk.val_macro_f1, marker="o", color="#3b6ea8")
    axes[1].axhline(full, ls=":", color="grey")
    axes[1].set_xscale("log"); axes[1].set_xlabel("top-k features by mutual information")
    axes[1].set_ylabel("validation macro-F1"); axes[1].set_title("Mutual-information filter", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "feature_selection.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
