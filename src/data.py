"""Workstream A - data pipeline for the PlantVillage tomato subset.

Builds the single shared split file ``data/splits.csv`` that every other script uses.

Steps
1. Index the ORIGINAL PlantVillage colour images for the 10 tomato classes
   (github.com/spMohanty/PlantVillage-Dataset, folder raw/color/Tomato___*).
2. Remove duplicates:
   - exact duplicates (same MD5 of the file bytes);
   - near-duplicates (64-bit difference hash, Hamming distance <= NEAR_DUP_BITS).
   A near-duplicate pair that carries two different labels is dropped entirely
   (label noise); otherwise one representative is kept.
3. Group repeat photos of the same physical leaf. PlantVillage file names carry a
   leaf ID (e.g. "GHLB2 Leaf 131" and "GHLB2 Leaf 131.1"); every photo of one leaf
   (and every member of a near-duplicate cluster) is kept in the same split so the
   test set never contains a leaf the model has already seen.
4. Stratified, grouped 70/15/15 split with seed 42 -> data/splits.csv

Usage
    python src/data.py                       # build splits.csv
    python src/data.py --root /path/to/raw/color

Other scripts import ``load_splits`` and ``load_images`` from here.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_IMG_ROOT = DATA_DIR / "plantvillage_color"
SPLITS_CSV = DATA_DIR / "splits.csv"
CACHE_DIR = DATA_DIR / "cache"
NEAR_DUP_BITS = 4  # Hamming threshold on the 64-bit dHash (candidate stage)
NEAR_DUP_MAD = 8.0  # confirm: mean abs grey-level difference on 32x32 thumbnails

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
SHORT = [
    "Bacterial spot", "Early blight", "Late blight", "Leaf mold", "Septoria",
    "Spider mites", "Target spot", "YLCV", "Mosaic virus", "Healthy",
]


def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------------------------------------------------------- indexing
def index_images(img_root: Path) -> pd.DataFrame:
    rows = []
    for label, cls in enumerate(CLASSES):
        folder = img_root / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append({"path": f"{cls}/{f.name}", "label": label, "class_name": SHORT[label]})
    return pd.DataFrame(rows)


def leaf_id(path: str) -> str:
    """'Tomato___X/uuid___GHLB2 Leaf 131.1.JPG' -> 'Tomato___X|GHLB2 Leaf 131'."""
    cls, name = path.split("/", 1)
    stem = name.split("___", 1)[-1].rsplit(".", 1)[0]
    stem = re.sub(r"\.\d+$", "", stem)  # drop repeat-photo suffix .1, .2 ...
    return f"{cls}|{stem}"


# ----------------------------------------------------------------------------- hashing
def dhash(img: Image.Image, size: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def hash_images(df: pd.DataFrame, img_root: Path) -> pd.DataFrame:
    md5s, dh, w, h = [], [], [], []
    for p in df["path"]:
        fp = img_root / p
        md5s.append(hashlib.md5(fp.read_bytes()).hexdigest())
        with Image.open(fp) as im:
            w.append(im.width)
            h.append(im.height)
            dh.append(dhash(im))
    df = df.copy()
    df["md5"], df["dhash"], df["width"], df["height"] = md5s, np.array(dh, dtype=np.uint64), w, h
    return df


def _popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).reshape(x.shape)


def near_duplicate_pairs(hashes: np.ndarray, max_bits: int, chunk: int = 512) -> list[tuple[int, int]]:
    """All pairs (i<j) whose dHash differs in at most ``max_bits`` bits."""
    pairs = []
    n = len(hashes)
    for s in range(0, n, chunk):
        a = hashes[s:s + chunk, None] ^ hashes[None, :]
        d = np.unpackbits(a.view(np.uint8).reshape(a.shape[0], n, 8), axis=2).sum(2)
        ii, jj = np.nonzero(d <= max_bits)
        ii = ii + s
        keep = ii < jj
        pairs.extend(zip(ii[keep].tolist(), jj[keep].tolist()))
    return pairs


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _thumb(fp: Path) -> np.ndarray:
    with Image.open(fp) as im:
        return np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)


def confirm_pairs(df: pd.DataFrame, pairs, img_root: Path, max_mad: float = NEAR_DUP_MAD):
    """dHash alone collides for centred leaves on similar backgrounds, so each candidate
    pair is confirmed by a pixel check on 32x32 greyscale thumbnails."""
    cache, out = {}, []
    for i, j in pairs:
        for k in (i, j):
            if k not in cache:
                cache[k] = _thumb(img_root / df.at[k, "path"])
        if np.abs(cache[i] - cache[j]).mean() <= max_mad:
            out.append((i, j))
    return out


def deduplicate(df: pd.DataFrame, img_root: Path, max_bits: int = NEAR_DUP_BITS) -> tuple[pd.DataFrame, dict]:
    """Collapse exact and near duplicates; drop clusters whose members disagree on label."""
    df = df.reset_index(drop=True)
    uf = _UnionFind(len(df))
    for _, idx in df.groupby("md5").indices.items():
        for k in idx[1:]:
            uf.union(idx[0], k)
    n_exact = int(len(df) - df["md5"].nunique())
    candidates = near_duplicate_pairs(df["dhash"].to_numpy(np.uint64), max_bits)
    pairs = confirm_pairs(df, candidates, img_root)
    for i, j in pairs:
        uf.union(i, j)
    df["dup_cluster"] = [uf.find(i) for i in range(len(df))]
    sizes = df.groupby("dup_cluster")["label"].agg(["size", "nunique"])
    conflicting = sizes.index[sizes["nunique"] > 1]
    conflict_rows = df["dup_cluster"].isin(conflicting)
    clean = df[~conflict_rows]
    kept = clean.sort_values("path").drop_duplicates("dup_cluster", keep="first")
    stats = {
        "images_indexed": len(df),
        "exact_duplicate_images": n_exact,
        "dhash_candidate_pairs": len(candidates),
        "confirmed_near_duplicate_pairs": len(pairs),
        "images_in_label_conflict_clusters": int(conflict_rows.sum()),
        "images_removed_total": len(df) - len(kept),
        "images_kept": len(kept),
    }
    return kept.reset_index(drop=True), stats, df


# ----------------------------------------------------------------------------- splitting
def grouped_stratified_split(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """70/15/15 split that is stratified by class and keeps each leaf group together."""
    groups = df["group"].to_numpy()
    y = df["label"].to_numpy()
    split = np.empty(len(df), dtype=object)
    # 20 folds of 5% each: 14 -> train, 3 -> val, 3 -> test
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_of = np.empty(len(df), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(df)), y, groups)):
        fold_of[te] = k
    split[fold_of < 14] = "train"
    split[(fold_of >= 14) & (fold_of < 17)] = "val"
    split[fold_of >= 17] = "test"
    return pd.Series(split, index=df.index)


def build_splits(img_root: Path = DEFAULT_IMG_ROOT, out: Path = SPLITS_CSV) -> pd.DataFrame:
    set_seed()
    df = index_images(img_root)
    print(f"Indexed {len(df)} images in {len(CLASSES)} classes")
    df = hash_images(df, img_root)
    kept, stats, full = deduplicate(df, img_root)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    # group = leaf id, merged with near-duplicate clusters (via union-find on both)
    kept["leaf_id"] = kept["path"].map(leaf_id)
    uf = _UnionFind(len(kept))
    for _, idx in kept.groupby("leaf_id").indices.items():
        for k in idx[1:]:
            uf.union(idx[0], k)
    kept["group"] = [uf.find(i) for i in range(len(kept))]
    kept["split"] = grouped_stratified_split(kept)
    # sanity: no group crosses splits
    assert kept.groupby("group")["split"].nunique().max() == 1
    cols = ["path", "label", "class_name", "split", "leaf_id", "group", "md5", "dhash", "width", "height"]
    kept = kept[cols].sort_values(["split", "label", "path"]).reset_index(drop=True)
    kept["dhash"] = kept["dhash"].map(lambda v: f"{int(v):016x}")
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, index=False)
    pd.Series(stats).to_csv(ROOT / "results" / "tables" / "dedup_stats.csv", header=["value"])
    # record what was removed, for transparency
    removed = full.loc[~full["path"].isin(kept["path"]), ["path", "label", "md5", "dup_cluster"]]
    removed.to_csv(ROOT / "results" / "tables" / "removed_duplicates.csv", index=False)
    print(kept.groupby(["class_name", "split"]).size().unstack()[["train", "val", "test"]])
    print(f"Wrote {out}")
    return kept


# ----------------------------------------------------------------------------- loading helpers
def load_splits(path: Path = SPLITS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("splits.csv not found - run `python src/data.py` first")
    return pd.read_csv(path)


def load_images(size: int = 64, gray: bool = False, img_root: Path = DEFAULT_IMG_ROOT,
                df: pd.DataFrame | None = None) -> np.ndarray:
    """All images in splits.csv order, resized to size x size, uint8. Cached to data/cache."""
    df = load_splits() if df is None else df
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"img_{size}{'_gray' if gray else ''}.npy"
    if cache.exists():
        arr = np.load(cache)
        if len(arr) == len(df):
            return arr
    mode = "L" if gray else "RGB"
    arr = np.stack([
        np.asarray(Image.open(img_root / p).convert(mode).resize((size, size), Image.BILINEAR))
        for p in df["path"]
    ]).astype(np.uint8)
    np.save(cache, arr)
    return arr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_IMG_ROOT,
                    help="folder that contains the Tomato___* class folders (PlantVillage raw/color)")
    args = ap.parse_args()
    build_splits(args.root)
