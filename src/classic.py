"""Workstream C (R3) - classic models with stratified 5-fold CV on the TRAIN split.

Models: Decision Tree, Gaussian Naive Bayes, k-NN, Logistic Regression, Random Forest (ensemble).
Input features: the shared ``features.npz``.
  - tree models and NB use the standardised named features (X) so splits are interpretable;
  - KNN and Logistic Regression use the PCA projection (X_pca): distances / linear weights
    behave better after decorrelation and the fit is ~10x faster.
Selection metric: macro-F1 (classes are imbalanced 14:1, so accuracy alone would favour YLCV).
The best configuration is refitted on the full train split, checked on val, and scored once on test.

    python src/classic.py            # all models
    python src/classic.py dt rf      # a subset
"""
import json
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from data import ROOT, SEED, SHORT, set_seed
from evaluate import evaluate, timer
from features import load_features

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models"

SPECS = {
    "dt": ("DecisionTree", "raw", DecisionTreeClassifier(random_state=SEED),
           {"criterion": ["gini", "entropy"], "max_depth": [8, 12, 20, None], "min_samples_leaf": [1, 10]}),
    "nb": ("GaussianNB", "raw", GaussianNB(), {"var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1, 1.0]}),
    "knn": ("KNN", "pca", KNeighborsClassifier(algorithm="brute", n_jobs=-1),
            {"n_neighbors": [1, 3, 5, 9, 15], "weights": ["uniform", "distance"], "metric": ["euclidean", "cosine"]}),
    "lr": ("LogisticRegression", "pca", LogisticRegression(max_iter=3000),
           {"C": [0.001, 0.01, 0.1, 1.0], "class_weight": [None, "balanced"]}),
    "rf": ("RandomForest", "raw", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
           {"max_features": ["sqrt"], "min_samples_leaf": [1, 3], "class_weight": [None, "balanced_subsample"]}),
}


def feature_names(raw):
    names = []
    for b, (s, e) in zip(raw["block_names"], raw["block_slices"]):
        n = e - s
        if b == "hsv":
            names += [f"hue_bin{i}" for i in range(32)] + [f"sat_bin{i}" for i in range(16)] + \
                     [f"val_bin{i}" for i in range(16)]
        elif b == "hog":
            names += [f"hog_{i}" for i in range(n)]
        elif b == "lbp":
            names += [f"lbp8_{i}" for i in range(10)] + [f"lbp16_{i}" for i in range(18)]
        elif b == "pix":
            names += [f"pix_r{(i // 3) // 16}_c{(i // 3) % 16}_{'RGB'[i % 3]}" for i in range(n)]
    return names


def block_importance(imp, raw):
    return {b: float(imp[s:e].sum()) for b, (s, e) in zip(raw["block_names"], raw["block_slices"])}


def run(key, data):
    name, feat, est, grid = SPECS[key]
    d = data[feat]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=1 if key in ("rf", "knn") else -1,
                      refit=True, return_train_score=True, verbose=1)
    with timer() as t:
        gs.fit(d["Xtr"], d["ytr"])
    search_time = t()
    cvres = pd.DataFrame(gs.cv_results_)[["params", "mean_train_score", "mean_test_score", "std_test_score",
                                          "mean_fit_time"]].sort_values("mean_test_score", ascending=False)
    cvres.to_csv(TAB / f"cv_{name}.csv", index=False)
    best = gs.best_estimator_
    # refit time of the chosen config alone
    with timer() as t:
        best.fit(d["Xtr"], d["ytr"])
    fit_time = t()
    val_f1 = f1_score(d["yva"], best.predict(d["Xva"]), average="macro")
    with timer() as ti:
        pred = best.predict(d["Xte"])
    res = evaluate(name, d["yte"], pred, train_time=fit_time, infer_time=ti(), group="C classic",
                   notes=f"{feat} features; best {json.dumps(gs.best_params_)}; cv macroF1={gs.best_score_:.4f}; "
                         f"val macroF1={val_f1:.4f}")
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / f"{name}.pkl", "wb") as f:
        pickle.dump(best, f)
    print(f"{name}: best={gs.best_params_} cv={gs.best_score_:.4f} val={val_f1:.4f} search {search_time:.0f}s")
    return best, res


def explain_tree(dt, raw):
    """Decision-tree interpretation for the report: top splits and importances."""
    names = feature_names(raw)
    (TAB / "dt_top_rules.txt").write_text(export_text(dt, feature_names=names, max_depth=4, decimals=2))
    imp = pd.Series(dt.feature_importances_, index=names).sort_values(ascending=False)
    imp.head(25).round(4).to_csv(TAB / "dt_top_features.csv", header=["importance"])
    json.dump(block_importance(dt.feature_importances_, raw), open(TAB / "dt_block_importance.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(dt, max_depth=2, feature_names=names, class_names=SHORT, filled=True, impurity=False,
              proportion=True, fontsize=7, ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "dt_top_levels.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    imp.head(15)[::-1].plot.barh(ax=ax, color="#3b6ea8")
    ax.set_title("Decision tree: top-15 feature importances", fontsize=9); ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "dt_feature_importance.png", dpi=200); plt.close(fig)
    print("Top DT features:\n", imp.head(10))


def main(keys):
    set_seed()
    data = {"raw": load_features(pca=False), "pca": load_features(pca=True)}
    for k in keys:
        best, _ = run(k, data)
        if k == "dt":
            explain_tree(best, data["raw"])
        if k == "rf":
            json.dump(block_importance(best.feature_importances_, data["raw"]),
                      open(TAB / "rf_block_importance.json", "w"), indent=1)
            imp = pd.Series(best.feature_importances_, index=feature_names(data["raw"]))
            imp.sort_values(ascending=False).head(25).round(5).to_csv(TAB / "rf_top_features.csv", header=["importance"])


if __name__ == "__main__":
    main(sys.argv[1:] or list(SPECS))
