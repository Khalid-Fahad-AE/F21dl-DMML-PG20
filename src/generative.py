"""Workstream B - generative classification.

Class-conditional GMM classifier: one Gaussian mixture p(x | c) per class, fitted with EM on that
class's training images, combined with the class prior p(c) through Bayes' rule:
    y_hat = argmax_c  log p(x | c) + log p(c)
Compared with Gaussian Naive Bayes (the special case: one component, diagonal covariance,
features independent given the class).

Hyper-parameters (number of PCA dims, components per class, covariance type) are chosen on the
VALIDATION split; the final model is refitted on train+val and scored once on test.

    python src/generative.py
"""
import itertools

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB

from data import ROOT, set_seed
from evaluate import evaluate, timer
from features import load_features

TAB = ROOT / "results" / "tables"


class ClassConditionalGMM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, covariance_type="full", reg_covar=1e-3, random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.log_prior_ = np.log(np.bincount(y)[self.classes_] / len(y))
        self.gmms_ = [GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                      reg_covar=self.reg_covar, n_init=2, max_iter=300,
                                      random_state=self.random_state).fit(X[y == c]) for c in self.classes_]
        return self

    def log_joint(self, X):
        return np.column_stack([g.score_samples(X) for g in self.gmms_]) + self.log_prior_

    def predict_proba(self, X):
        lj = self.log_joint(X)
        return np.exp(lj - logsumexp(lj, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[self.log_joint(X).argmax(1)]

    def sample(self, n_per_class=5):
        """Draw synthetic feature vectors per class (the 'generative' direction)."""
        return {c: g.sample(n_per_class)[0] for c, g in zip(self.classes_, self.gmms_)}


def main():
    set_seed()
    d = load_features(pca=True)
    rows = []
    # ---- Gaussian NB baselines (raw standardised features and PCA)
    raw = load_features(pca=False)
    for feat_name, D in [("raw", raw), ("PCA50", None)]:
        Xtr = D["Xtr"] if D is not None else d["Xtr"][:, :50]
        Xva = D["Xva"] if D is not None else d["Xva"][:, :50]
        nb = GaussianNB().fit(Xtr, d["ytr"])
        rows.append({"model": "GaussianNB", "features": feat_name, "K": 1, "cov": "diag",
                     "val_macro_f1": f1_score(d["yva"], nb.predict(Xva), average="macro")})
        print(rows[-1])
    # ---- class-conditional GMM grid on validation
    for dims, K, cov in itertools.product([20, 50, 100, 200], [1, 2, 4, 8], ["diag", "full"]):
        m = ClassConditionalGMM(n_components=K, covariance_type=cov).fit(d["Xtr"][:, :dims], d["ytr"])
        f = f1_score(d["yva"], m.predict(d["Xva"][:, :dims]), average="macro")
        rows.append({"model": "ClassGMM", "features": f"PCA{dims}", "K": K, "cov": cov, "val_macro_f1": f})
        print(rows[-1])
    grid = pd.DataFrame(rows); grid.round(4).to_csv(TAB / "generative_val_grid.csv", index=False)

    # ---- final models: refit on train+val, test once
    trva = np.isin(d["split"], ["train", "val"])
    best = grid[grid.model == "ClassGMM"].sort_values("val_macro_f1", ascending=False).iloc[0]
    dims = int(best.features.replace("PCA", ""))
    Xtv, ytv = d["Xall"][trva][:, :dims], d["y"][trva]
    with timer() as t:
        gmm = ClassConditionalGMM(n_components=int(best.K), covariance_type=best["cov"]).fit(Xtv, ytv)
    with timer() as ti:
        pred = gmm.predict(d["Xte"][:, :dims])
    import pickle
    (ROOT / "results" / "models").mkdir(parents=True, exist_ok=True)
    pickle.dump({"model": gmm, "dims": dims}, open(ROOT / "results" / "models" / "ClassGMM.pkl", "wb"))
    evaluate("ClassGMM", d["yte"], pred, train_time=t(), infer_time=ti(), group="B generative",
             notes=f"PCA{dims}, K={int(best.K)}, {best['cov']} cov")

    best_nb = grid[grid.model == "GaussianNB"].sort_values("val_macro_f1", ascending=False).iloc[0]
    if best_nb.features == "raw":
        Xtv_nb, Xte_nb = raw["Xall"][trva], raw["Xte"]
    else:
        Xtv_nb, Xte_nb = d["Xall"][trva][:, :50], d["Xte"][:, :50]
    with timer() as t:
        nb = GaussianNB().fit(Xtv_nb, ytv)
    with timer() as ti:
        pred = nb.predict(Xte_nb)
    evaluate("GaussianNB_generative", d["yte"], pred, train_time=t(), infer_time=ti(), group="B generative",
             notes=f"{best_nb.features} features")
    # NB on the same PCA dims as the GMM, for a like-for-like comparison
    nb2 = GaussianNB().fit(Xtv, ytv)
    evaluate("GaussianNB_sameDims", d["yte"], nb2.predict(d["Xte"][:, :dims]), group="B generative",
             notes=f"PCA{dims}, same input as ClassGMM")


if __name__ == "__main__":
    import generative  # so pickled classes are stored as generative.<Class>, not __main__.<Class>
    generative.main()
