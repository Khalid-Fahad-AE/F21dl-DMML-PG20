"""Workstream A - exploratory data analysis. Writes figures to results/figures/ and tables to results/tables/.

    python src/eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from data import DEFAULT_IMG_ROOT, ROOT, SHORT, load_images, load_splits, set_seed

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"


def class_counts(df):
    t = df.groupby(["class_name", "split"]).size().unstack()[["train", "val", "test"]]
    t = t.loc[SHORT]
    t["total"] = t.sum(axis=1)
    t["share_%"] = (100 * t["total"] / t["total"].sum()).round(1)
    t.to_csv(TAB / "class_counts.csv")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    t[["train", "val", "test"]].plot.barh(stacked=True, ax=ax, color=["#3b6ea8", "#8fb3d9", "#d98f3b"], width=0.75)
    ax.invert_yaxis()
    ax.set_xlabel("images")
    ax.set_ylabel("")
    ax.set_title(f"Class distribution after de-duplication (n={len(df)}); imbalance ratio "
                 f"{t['total'].max() / t['total'].min():.1f}:1", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_class_counts.png", dpi=200)
    plt.close(fig)
    print(t)


def size_check(df):
    s = df.groupby(["width", "height"]).size()
    s.to_csv(TAB / "image_sizes.csv", header=["count"])
    print("Image sizes:\n", s)


def sample_grid(df, n=6):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(len(SHORT), n, figsize=(n * 1.1, len(SHORT) * 1.15))
    for r, name in enumerate(SHORT):
        paths = df.loc[(df.class_name == name) & (df.split == "train"), "path"].to_numpy()
        for c, p in enumerate(rng.choice(paths, n, replace=False)):
            ax = axes[r, c]
            ax.imshow(Image.open(DEFAULT_IMG_ROOT / p))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
    fig.subplots_adjust(wspace=0.03, hspace=0.05, left=0.17, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(FIG / "eda_sample_grid.png", dpi=150)
    plt.close(fig)


def colour_histograms(df, imgs):
    """Per-class mean hue / saturation / value histograms (train split only)."""
    from skimage.color import rgb2hsv
    tr = (df.split == "train").to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    cmap = plt.get_cmap("tab10")
    rows = []
    for k, name in enumerate(SHORT):
        idx = np.where(tr & (df.class_name == name).to_numpy())[0][:300]
        hsv = rgb2hsv(imgs[idx].astype(np.float32) / 255.0).reshape(-1, 3)
        for ch, ax in enumerate(axes):
            h, edges = np.histogram(hsv[:, ch], bins=32, range=(0, 1), density=True)
            ax.plot((edges[:-1] + edges[1:]) / 2, h, color=cmap(k), lw=1.2, label=name)
        rows.append({"class": name, "mean_hue": hsv[:, 0].mean(), "mean_sat": hsv[:, 1].mean(),
                     "mean_val": hsv[:, 2].mean()})
    for ax, t in zip(axes, ["Hue", "Saturation", "Value"]):
        ax.set_title(t, fontsize=9); ax.spines[["top", "right"]].set_visible(False); ax.set_yticks([])
    axes[2].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Per-class HSV colour distributions (train, 64x64)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "eda_hsv_histograms.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).round(3).to_csv(TAB / "class_mean_hsv.csv", index=False)


def mean_images(df, imgs):
    fig, axes = plt.subplots(1, len(SHORT), figsize=(len(SHORT) * 1.1, 1.4))
    for ax, name in zip(axes, SHORT):
        m = imgs[((df.class_name == name) & (df.split == "train")).to_numpy()].mean(0)
        ax.imshow(m.astype(np.uint8)); ax.set_title(name, fontsize=6); ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "eda_mean_images.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    set_seed()
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    df = load_splits()
    imgs = load_images(64)
    class_counts(df)
    size_check(df)
    sample_grid(df)
    colour_histograms(df, imgs)
    mean_images(df, imgs)
    print("EDA figures written to", FIG)
