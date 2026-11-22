"""How reliable are the differences between the top models?

1. Seed variance: retrain the random forest and the MLP with 5 different random seeds (same split,
   same hyper-parameters) and report test macro-F1 as mean +- standard deviation.
2. Bootstrap: resample the 2,719 test images 2,000 times to get a 95% confidence interval for each
   model's macro-F1, and a paired interval for the difference RF - MLP.
   (Logistic regression with lbfgs is deterministic, so it is only bootstrapped.)

    python src/seeds.py
Writes results/tables/seed_variance.csv and results/tables/bootstrap_ci.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from data import ROOT, set_seed
from features import load_features
from mlp import NumpyMLP

TAB = ROOT / "results" / "tables"
SEEDS = [0, 1, 2, 3, 4]


def bootstrap(y, preds: dict, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in preds}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        for k, p in preds.items():
            out[k].append(f1_score(y[idx], p[idx], average="macro"))
    return {k: np.array(v) for k, v in out.items()}


def main():
    set_seed()
    raw, pca = load_features(pca=False), load_features(pca=True)
    rows, keep = [], {}
    for s in SEEDS:
        rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=3,
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=s)
        p = rf.fit(raw["Xtr"], raw["ytr"]).predict(raw["Xte"])
        rows.append({"model": "RandomForest", "seed": s, "test_macro_f1": f1_score(raw["yte"], p, average="macro")})
        keep.setdefault("RandomForest", p)
        m = NumpyMLP(pca["Xtr"].shape[1], 10, hidden=(1024, 512), dropout=0.3, batchnorm=True, lr=1e-3,
                     patience=8, max_epochs=60, seed=s).fit(pca["Xtr"], pca["ytr"], pca["Xva"], pca["yva"])
        p = m.predict(pca["Xte"])
        rows.append({"model": "MLP_features", "seed": s, "test_macro_f1": f1_score(pca["yte"], p, average="macro")})
        keep.setdefault("MLP_features", p)
        print(rows[-2], rows[-1])
    sv = pd.DataFrame(rows)
    summ = sv.groupby("model").test_macro_f1.agg(["mean", "std", "min", "max"]).round(4)
    sv.round(4).to_csv(TAB / "seed_variance.csv", index=False)
    print(summ)

    lr = LogisticRegression(C=0.01, max_iter=3000).fit(pca["Xtr"], pca["ytr"])
    keep["LogisticRegression"] = lr.predict(pca["Xte"])
    y = raw["yte"]
    bs = bootstrap(y, keep)
    rows = [{"model": k, "macro_f1": f1_score(y, p, average="macro"),
             "ci95_low": np.percentile(bs[k], 2.5), "ci95_high": np.percentile(bs[k], 97.5)} for k, p in keep.items()]
    diff = bs["RandomForest"] - bs["MLP_features"]
    rows.append({"model": "difference RF - MLP (paired)", "macro_f1": f1_score(y, keep["RandomForest"], average="macro")
                 - f1_score(y, keep["MLP_features"], average="macro"),
                 "ci95_low": np.percentile(diff, 2.5), "ci95_high": np.percentile(diff, 97.5)})
    ci = pd.DataFrame(rows).round(4)
    ci.to_csv(TAB / "bootstrap_ci.csv", index=False)
    print(ci.to_string(index=False))


if __name__ == "__main__":
    import seeds  # noqa: F401  (keeps pickling/imports consistent with the other scripts)
    seeds.main()
