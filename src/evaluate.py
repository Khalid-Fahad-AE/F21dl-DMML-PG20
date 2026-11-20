"""Shared evaluation code (owned by workstream C). Every model in the project reports through
``evaluate()`` so all tables in the report use the same metrics and format.

Metrics: accuracy, macro precision / recall / F1, weighted F1, per-class F1,
confusion matrix (counts + row-normalised figure), training time, inference time.
Results are upserted (by model name) into results/tables/metrics_all.csv and
results/tables/per_class_f1.csv; confusion matrices go to results/figures/cm_<name>.png.
Works with numpy arrays, so the PyTorch scripts (E, F) can call it too.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support)

from data import ROOT, SHORT

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
METRICS_CSV = TAB / "metrics_all.csv"
PER_CLASS_CSV = TAB / "per_class_f1.csv"


@contextmanager
def timer():
    """with timer() as t: ...;  t() -> elapsed seconds"""
    t0 = time.perf_counter()
    box = {}
    yield lambda: box.get("dt", time.perf_counter() - t0)
    box["dt"] = time.perf_counter() - t0


def _upsert(path: Path, row: pd.DataFrame, key="model"):
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old[key].isin(row[key])]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(path, index=False)


def plot_confusion(cm: np.ndarray, title: str, out: Path):
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(SHORT))); ax.set_xticklabels(SHORT, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(SHORT))); ax.set_yticklabels(SHORT, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def evaluate(name: str, y_true, y_pred, *, split: str = "test", train_time: float | None = None,
             infer_time: float | None = None, n_params: int | None = None, group: str = "",
             notes: str = "", save: bool = True, labels=None) -> dict:
    """``labels``: classes to macro-average over (default all 10). The PlantDoc field test has
    no Target-spot images, so it passes the 9 classes that are present."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(SHORT))) if labels is None else list(labels)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    res = {
        "model": name, "group": group, "split": split, "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p, "macro_recall": r, "macro_f1": f,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "train_time_s": train_time, "infer_ms_per_img": None if infer_time is None else 1000 * infer_time / len(y_true),
        "n_params": n_params, "notes": notes,
    }
    if save:
        TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
        _upsert(METRICS_CSV, pd.DataFrame([res]))
        pc = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        _upsert(PER_CLASS_CSV, pd.DataFrame([{"model": name, **{SHORT[l]: v for l, v in zip(labels, pc)}}]))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(SHORT))))
        np.save(TAB / f"cm_{name}.npy", cm)
        plot_confusion(cm, f"{name} ({split}) acc={res['accuracy']:.3f} macro-F1={f:.3f}", FIG / f"cm_{name}.png")
    print(f"[{name}] acc={res['accuracy']:.4f} macroP={p:.4f} macroR={r:.4f} macroF1={f:.4f}")
    return res


def summary_table(models=None) -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV)
    if models is not None:
        df = df[df.model.isin(models)]
    return df


if __name__ == "__main__":
    print(summary_table().round(4).to_string(index=False))
