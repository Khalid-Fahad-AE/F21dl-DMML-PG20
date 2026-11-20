"""Workstream C - Decision-tree discussion for a farmer or a phone app.

The tuned tree (depth 20) is accurate but unreadable, so we also fit a SHALLOW tree (depth 3,
8 leaves) on the same features and report what it costs in accuracy. Its rules are what a
farmer-facing app could display ("high share of yellow pixels and little fine texture -> YLCV").
Thresholds are converted back from standardised units to raw histogram fractions so they are readable.

    python src/dt_explain.py
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from classic import feature_names
from data import DATA_DIR, ROOT, SEED, SHORT, set_seed
from evaluate import evaluate
from features import load_features

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"


def main():
    set_seed()
    d = load_features(pca=False)
    names = feature_names(d)
    scaler = pickle.load(open(DATA_DIR / "feature_transform.pkl", "rb"))["scaler"]
    for depth in (3, 5):
        dt = DecisionTreeClassifier(max_depth=depth, criterion="entropy", random_state=SEED).fit(d["Xtr"], d["ytr"])
        evaluate(f"DecisionTree_depth{depth}", d["yte"], dt.predict(d["Xte"]), group="C classic",
                 notes=f"interpretable tree, {dt.get_n_leaves()} leaves")
        if depth == 3:
            # undo standardisation on thresholds -> raw feature values
            tree = dt.tree_
            raw_thr = tree.threshold.copy()
            internal = tree.feature >= 0
            raw_thr[internal] = tree.threshold[internal] * scaler.scale_[tree.feature[internal]] + \
                scaler.mean_[tree.feature[internal]]
            lines = []
            for node in np.where(internal)[0]:
                lines.append({"node": int(node), "feature": names[tree.feature[node]],
                              "threshold_standardised": round(float(tree.threshold[node]), 3),
                              "threshold_raw": round(float(raw_thr[node]), 4), "samples": int(tree.n_node_samples[node])})
            pd.DataFrame(lines).to_csv(TAB / "dt_depth3_splits.csv", index=False)
            leaves = []
            for node in np.where(~internal)[0]:
                v = tree.value[node][0]
                leaves.append({"leaf": int(node), "samples": int(tree.n_node_samples[node]),
                               "majority": SHORT[int(v.argmax())], "purity": round(float(v.max() / v.sum()), 3)})
            pd.DataFrame(leaves).to_csv(TAB / "dt_depth3_leaves.csv", index=False)
            (TAB / "dt_depth3_rules.txt").write_text(export_text(dt, feature_names=names, decimals=3))
            fig, ax = plt.subplots(figsize=(13, 5))
            plot_tree(dt, feature_names=names, class_names=SHORT, filled=True, impurity=False, proportion=True,
                      fontsize=6, ax=ax)
            fig.tight_layout(); fig.savefig(FIG / "dt_depth3.png", dpi=170); plt.close(fig)
            print(pd.DataFrame(lines)); print(pd.DataFrame(leaves))


if __name__ == "__main__":
    main()
