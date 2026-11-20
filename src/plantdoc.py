"""Workstream F - real-world robustness test on PlantDoc field photos.

PlantDoc (Singh et al., 2020; github.com/pratikkayal/PlantDoc-Dataset) contains tomato leaves
photographed in the field: cluttered backgrounds, several leaves, shadows. We NEVER train on it;
all PlantDoc tomato images (its train and test folders) form an outside test set for models trained on
PlantVillage. PlantDoc has no Target-spot class, so macro metrics average over the 9 shared classes.

    python src/plantdoc.py        # index + evaluate the saved scikit-learn / NumPy models
The PyTorch scripts (cnn.py, transfer.py) import ``index_plantdoc`` and run the same test.
"""
import pickle
import time

import numpy as np
import pandas as pd
from PIL import Image

from data import DATA_DIR, ROOT, SHORT, set_seed
from evaluate import evaluate

PLANTDOC_ROOT = DATA_DIR / "plantdoc"
FOLDER_TO_CLASS = {
    "Tomato leaf bacterial spot": "Bacterial spot",
    "Tomato Early blight leaf": "Early blight",
    "Tomato leaf late blight": "Late blight",
    "Tomato mold leaf": "Leaf mold",
    "Tomato Septoria leaf spot": "Septoria",
    "Tomato two spotted spider mites leaf": "Spider mites",
    "Tomato leaf yellow virus": "YLCV",
    "Tomato leaf mosaic virus": "Mosaic virus",
    "Tomato leaf": "Healthy",
}
MODELS = ROOT / "results" / "models"


def index_plantdoc(root=PLANTDOC_ROOT) -> pd.DataFrame:
    rows = []
    for sub in ("train", "test"):
        for folder, cls in FOLDER_TO_CLASS.items():
            d = root / sub / folder
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        rows.append({"path": str(f.relative_to(root)), "label": SHORT.index(cls), "class_name": cls,
                                     "plantdoc_split": sub})
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "plantdoc_tomato.csv", index=False)
    return df


def load_square(path, size, gray=False):
    """Centre-crop to square then resize (PlantVillage images are already square)."""
    im = Image.open(path).convert("L" if gray else "RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((size, size), Image.BILINEAR))


def present_labels(y):
    return sorted(set(np.asarray(y).tolist()))


def _register_legacy_pickles():
    """Models pickled by an older run of mlp.py / generative.py refer to __main__.<Class>."""
    import __main__
    import generative
    import mlp
    __main__.NumpyMLP = mlp.NumpyMLP
    __main__.ClassConditionalGMM = generative.ClassConditionalGMM


def main():
    set_seed()
    _register_legacy_pickles()
    from features import transform_new
    df = index_plantdoc()
    print(df.class_name.value_counts())
    paths = [PLANTDOC_ROOT / p for p in df.path]
    rgb64 = np.stack([load_square(p, 64) for p in paths])
    gray128 = np.stack([load_square(p, 128, gray=True) for p in paths])
    X, X_pca = transform_new(rgb64, gray128)
    y = df.label.to_numpy()
    labels = present_labels(y)

    feats_for = {"DecisionTree": X, "GaussianNB": X, "RandomForest": X, "KNN": X_pca, "LogisticRegression": X_pca}
    for name, F in feats_for.items():
        fp = MODELS / f"{name}.pkl"
        if fp.exists():
            m = pickle.load(open(fp, "rb"))
            t0 = time.time(); pred = m.predict(F); ti = time.time() - t0
            evaluate(f"{name}@PlantDoc", y, pred, split="plantdoc", infer_time=ti, group="F field test", labels=labels)
    fp = MODELS / "ClassGMM.pkl"
    if fp.exists():
        g = pickle.load(open(fp, "rb"))
        evaluate("ClassGMM@PlantDoc", y, g["model"].predict(X_pca[:, :g["dims"]]), split="plantdoc",
                 group="F field test", labels=labels)
    fp = MODELS / "MLP_features.pkl"
    if fp.exists():
        m = pickle.load(open(fp, "rb"))["model"]
        evaluate("MLP_features@PlantDoc", y, m.predict(X_pca), split="plantdoc", group="F field test", labels=labels)
    fp = MODELS / "MLP_pixels.pkl"
    if fp.exists():
        b = pickle.load(open(fp, "rb"))
        Xp = (rgb64.reshape(len(df), -1).astype(np.float32) / 255.0 - b["mu"]) / b["sd"]
        evaluate("MLP_pixels@PlantDoc", y, b["model"].predict(Xp), split="plantdoc", group="F field test",
                 labels=labels)


if __name__ == "__main__":
    main()
