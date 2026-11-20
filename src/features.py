"""Workstream B - shared hand-crafted feature file ``data/features.npz``.

Feature blocks (all computed per image, in splits.csv order):
  hsv   : HSV colour histograms (H 32 bins, S 16, V 16) on 64x64 RGB         -> 64
  hog   : HOG on 128x128 greyscale (9 orientations, 16px cells, 2x2 blocks)  -> 1764
  lbp   : uniform LBP histograms, (P=8,R=1) and (P=16,R=2), 128x128 grey     -> 28
  pix   : down-sampled 16x16 RGB pixels                                       -> 768

The scaler and PCA are fitted on the TRAIN split only, then applied to val/test.
PCA keeps 95% of the variance.

npz keys
  X        standardised concatenated features (n, 2624)
  X_pca    PCA projection of X (n, k)
  y, split, path, block_names, block_slices
Usage:  python src/features.py
"""
import time

import numpy as np
from joblib import Parallel, delayed
from skimage.color import rgb2hsv
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data import DATA_DIR, load_images, load_splits, set_seed

FEATURES_NPZ = DATA_DIR / "features.npz"
TRANSFORM_PKL = DATA_DIR / "feature_transform.pkl"


def hsv_hist(rgb):
    hsv = rgb2hsv(rgb.astype(np.float32) / 255.0)
    parts = [np.histogram(hsv[..., c], bins=b, range=(0, 1))[0] for c, b in zip(range(3), (32, 16, 16))]
    v = np.concatenate(parts).astype(np.float32)
    return v / v.sum() * 3


def hog_feat(gray):
    return hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def lbp_feat(gray):
    out = []
    for P, R in ((8, 1), (16, 2)):
        codes = local_binary_pattern(gray, P, R, method="uniform")
        h = np.histogram(codes, bins=P + 2, range=(0, P + 2))[0].astype(np.float32)
        out.append(h / h.sum())
    return np.concatenate(out)


def pix_feat(rgb):
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel() / 255.0


def one(rgb, gray):
    return hsv_hist(rgb), hog_feat(gray), lbp_feat(gray), pix_feat(rgb)


def extract(rgb64, gray128, n_jobs=-1):
    res = Parallel(n_jobs=n_jobs, batch_size=256)(delayed(one)(r, g) for r, g in zip(rgb64, gray128))
    blocks = {name: np.stack([r[k] for r in res]) for k, name in enumerate(["hsv", "hog", "lbp", "pix"])}
    return blocks


def main():
    set_seed()
    df = load_splits()
    t0 = time.time()
    blocks = extract(load_images(64), load_images(128, gray=True))
    print(f"Extracted features in {time.time() - t0:.0f}s:", {k: v.shape[1] for k, v in blocks.items()})
    names = list(blocks)
    raw = np.hstack([blocks[n] for n in names])
    slices, s = [], 0
    for n in names:
        slices.append((s, s + blocks[n].shape[1])); s += blocks[n].shape[1]
    tr = (df.split == "train").to_numpy()
    scaler = StandardScaler().fit(raw[tr])
    X = scaler.transform(raw).astype(np.float32)
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42).fit(X[tr])
    X_pca = pca.transform(X).astype(np.float32)
    print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} components (95% variance)")
    import pickle
    with open(TRANSFORM_PKL, "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca, "block_names": names, "block_slices": slices}, f)
    np.savez_compressed(
        FEATURES_NPZ, X=X, X_pca=X_pca, y=df.label.to_numpy(), split=df.split.to_numpy().astype(str),
        path=df.path.to_numpy().astype(str), block_names=np.array(names), block_slices=np.array(slices),
        pca_explained=pca.explained_variance_ratio_,
    )
    print("Wrote", FEATURES_NPZ)


def transform_new(rgb64, gray128):
    """Features for images outside PlantVillage, using the train-fitted scaler and PCA."""
    import pickle
    t = pickle.load(open(TRANSFORM_PKL, "rb"))
    blocks = extract(rgb64, gray128)
    raw = np.hstack([blocks[n] for n in t["block_names"]])
    X = t["scaler"].transform(raw).astype(np.float32)
    return X, t["pca"].transform(X).astype(np.float32)


def load_features(pca=False):
    """Returns dict with X (or X_pca), y, split and helpers Xtr, ytr, Xva, yva, Xte, yte."""
    d = dict(np.load(FEATURES_NPZ, allow_pickle=False))
    X = d["X_pca"] if pca else d["X"]
    sp = d["split"]
    for s, k in (("train", "tr"), ("val", "va"), ("test", "te")):
        d["X" + k], d["y" + k] = X[sp == s], d["y"][sp == s]
    d["Xall"] = X
    return d


if __name__ == "__main__":
    main()
