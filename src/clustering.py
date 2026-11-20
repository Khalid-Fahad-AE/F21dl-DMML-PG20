"""Workstream B - unsupervised analysis: k-means and Gaussian Mixture Model (EM).

Both are fitted on the TRAIN split, on the first 50 PCA components of the shared features.
Internal metric: silhouette. External metrics against the true disease labels: ARI and NMI.
Also: k sweep (k = 2..20), cluster-vs-class contingency heatmaps, PCA and t-SNE maps.

    python src/clustering.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

from data import ROOT, SHORT, set_seed
from features import load_features

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
N_DIM = 50
SIL_SAMPLE = 4000


def metrics(X, y, labels, rng):
    idx = rng.choice(len(X), min(SIL_SAMPLE, len(X)), replace=False)
    return {"silhouette": silhouette_score(X[idx], labels[idx]),
            "ARI": adjusted_rand_score(y, labels), "NMI": normalized_mutual_info_score(y, labels)}


def fit_models(X, k, seed=42):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    gmm = GaussianMixture(n_components=k, covariance_type="diag", n_init=3, max_iter=300, random_state=seed).fit(X)
    gmmf = GaussianMixture(n_components=k, covariance_type="full", n_init=1, max_iter=300, reg_covar=1e-3,
                           random_state=seed).fit(X)
    return {"k-means": km, "GMM (diag)": gmm, "GMM (full)": gmmf}


def contingency_plot(y, labels, title, out):
    ct = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series([SHORT[i] for i in y], name="class"))
    ct = ct[SHORT]
    ct = ct.loc[ct.values.argmax(1).argsort(kind="stable")]  # order clusters by dominant class
    frac = ct / ct.sum(axis=0)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.imshow(frac.values, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(SHORT))); ax.set_xticklabels(SHORT, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(ct))); ax.set_yticklabels([f"c{c}" for c in ct.index], fontsize=7)
    ax.set_title(title, fontsize=9); ax.set_ylabel("cluster")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return ct


def main():
    set_seed()
    rng = np.random.default_rng(42)
    d = load_features(pca=True)
    Xtr, ytr = d["Xtr"][:, :N_DIM], d["ytr"]

    # ---- k sweep
    rows = []
    for k in [2, 4, 6, 8, 10, 12, 15, 20]:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xtr)
        gm = GaussianMixture(n_components=k, covariance_type="diag", n_init=1, random_state=42).fit(Xtr)
        rows.append({"k": k, "algo": "k-means", "inertia": km.inertia_, **metrics(Xtr, ytr, km.labels_, rng)})
        rows.append({"k": k, "algo": "GMM (diag)", "bic": gm.bic(Xtr), **metrics(Xtr, ytr, gm.predict(Xtr), rng)})
        print(rows[-2]); print(rows[-1])
    sweep = pd.DataFrame(rows); sweep.round(4).to_csv(TAB / "clustering_k_sweep.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, m in zip(axes, ["silhouette", "ARI", "NMI"]):
        for algo, g in sweep.groupby("algo"):
            ax.plot(g.k, g[m], marker="o", ms=3, label=algo)
        ax.axvline(10, color="grey", ls=":", lw=1); ax.set_title(m, fontsize=9); ax.set_xlabel("k")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "clustering_k_sweep.png", dpi=200); plt.close(fig)

    # ---- k = 10 (number of classes): train fit, report on train and test
    models = fit_models(Xtr, 10)
    res = []
    Xte, yte = d["Xte"][:, :N_DIM], d["yte"]
    for name, m in models.items():
        lab_tr = m.labels_ if hasattr(m, "labels_") else m.predict(Xtr)
        lab_te = m.predict(Xte)
        res.append({"algo": name, "split": "train", **metrics(Xtr, ytr, lab_tr, rng)})
        res.append({"algo": name, "split": "test", **metrics(Xte, yte, lab_te, rng)})
        contingency_plot(ytr, lab_tr, f"{name}, k=10: share of each class per cluster (train)",
                         FIG / f"clustering_contingency_{name.split()[0].lower().replace('-', '')}"
                               f"{'_full' if 'full' in name else ''}.png")
    # same metrics on the separate feature blocks (which features carry class structure?)
    from sklearn.decomposition import PCA
    raw = load_features(pca=False)
    for b, (s, e) in zip(raw["block_names"], raw["block_slices"]):
        Z = PCA(n_components=min(30, e - s), random_state=42).fit_transform(raw["Xtr"][:, s:e])
        km = KMeans(n_clusters=10, n_init=5, random_state=42).fit(Z)
        res.append({"algo": f"k-means on {b} only", "split": "train", **metrics(Z, ytr, km.labels_, rng)})
    out = pd.DataFrame(res); out.round(4).to_csv(TAB / "clustering_k10.csv", index=False)
    print(out.round(3).to_string(index=False))

    # ---- 2-D maps: PCA and t-SNE, coloured by true class and by k-means cluster
    idx = rng.choice(len(Xtr), 3000, replace=False)
    emb_pca = Xtr[idx, :2]
    emb_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(Xtr[idx])
    km_lab = models["k-means"].labels_[idx]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, emb, col, t in [(axes[0], emb_pca, ytr[idx], "PCA (PC1 vs PC2), true class"),
                            (axes[1], emb_tsne, ytr[idx], "t-SNE, true class"),
                            (axes[2], emb_tsne, km_lab, "t-SNE, k-means cluster (k=10)")]:
        cm_ = cmap if "true" in t else plt.get_cmap("tab20b")  # clusters get their own palette
        ax.scatter(emb[:, 0], emb[:, 1], c=[cm_(c * 2 if cm_ is not cmap else c) for c in col], s=3, alpha=0.7,
                   linewidths=0)
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(i), label=n, ms=4) for i, n in enumerate(SHORT)]
    axes[1].legend(handles=handles, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5)
    fig.tight_layout(); fig.savefig(FIG / "clustering_pca_tsne.png", dpi=200); plt.close(fig)

    ev = d["pca_explained"]
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.plot(np.arange(1, len(ev) + 1), np.cumsum(ev)); ax.axvline(N_DIM, ls=":", c="grey")
    ax.set_xlabel("components"); ax.set_ylabel("cumulative explained variance"); ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "pca_explained_variance.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
