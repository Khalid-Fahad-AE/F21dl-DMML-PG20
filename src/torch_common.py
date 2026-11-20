"""Shared PyTorch utilities for workstreams E (CNN) and F (transfer learning).

Uses the same splits.csv, the same evaluate.py and the same PlantDoc field test as the
scikit-learn models, so every number in the report is comparable.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data import DEFAULT_IMG_ROOT, ROOT, SEED, load_splits
from plantdoc import PLANTDOC_ROOT, index_plantdoc

MODELS = ROOT / "results" / "models"
TAB = ROOT / "results" / "tables"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform, square_crop=False):
        self.paths = [Path(root) / p for p in df.path]
        self.labels = df.label.to_numpy()
        self.transform = transform
        self.square_crop = square_crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        if self.square_crop:  # PlantDoc photos are not square
            w, h = im.size; s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return self.transform(im), int(self.labels[i])


def build_transforms(size: int, aug: str):
    """aug: 'none' | 'geo' (rotation + flips) | 'full' (geo + brightness/contrast + zoom), as in our pitch."""
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    test_tf = T.Compose([T.Resize((size, size)), *norm])
    if aug == "none":
        train_tf = test_tf
    elif aug == "geo":
        train_tf = T.Compose([T.Resize((size, size)), T.RandomRotation(25), T.RandomHorizontalFlip(),
                              T.RandomVerticalFlip(), *norm])
    elif aug == "full":
        train_tf = T.Compose([T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),  # zoom
                              T.RandomRotation(25), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                              T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), *norm])
    else:
        raise ValueError(aug)
    return train_tf, test_tf


def make_loaders(size, aug, batch_size=64, workers=4, img_root=DEFAULT_IMG_ROOT):
    df = load_splits()
    train_tf, test_tf = build_transforms(size, aug)
    g = torch.Generator(); g.manual_seed(SEED)
    mk = lambda s, tf, sh: DataLoader(ImageDataset(df[df.split == s], img_root, tf), batch_size=batch_size,
                                      shuffle=sh, num_workers=workers, pin_memory=True, generator=g if sh else None)
    pdoc = index_plantdoc()
    plantdoc = DataLoader(ImageDataset(pdoc, PLANTDOC_ROOT, test_tf, square_crop=True), batch_size=batch_size,
                          num_workers=workers) if len(pdoc) else None
    return mk("train", train_tf, True), mk("val", test_tf, False), mk("test", test_tf, False), plantdoc


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, ys = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x, y in loader:
        preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
        ys.append(y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(ys).numpy(), torch.cat(preds).numpy(), time.perf_counter() - t0


@torch.no_grad()
def latency_ms(model, size, device, n=50):
    """Single-image latency (batch 1), the number that matters for a phone app."""
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / n


def train(model, train_loader, val_loader, device, *, epochs=30, lr=1e-3, weight_decay=1e-4, patience=6,
          param_groups=None, log_name="run"):
    """AdamW + cosine schedule + early stopping on validation macro-F1. Restores the best weights."""
    model.to(device)
    opt = torch.optim.AdamW(param_groups or [p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, wait, history = -1.0, None, 0, []
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        # BatchNorm layers whose weights are frozen keep their ImageNet running statistics
        for mod in model.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in mod.parameters()):
                mod.eval()
        tl, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
        sched.step()
        # validation
        model.eval()
        vl, vys, vps = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vl += loss_fn(out, y).item() * len(y)
                vys.append(y.cpu()); vps.append(out.argmax(1).cpu())
        vy, vp = torch.cat(vys).numpy(), torch.cat(vps).numpy()
        rec = {"epoch": ep, "train_loss": tl / n, "train_acc": correct / n, "val_loss": vl / len(vy),
               "val_acc": float((vy == vp).mean()), "val_macro_f1": float(f1_score(vy, vp, average="macro")),
               "elapsed_s": time.time() - t_start}
        history.append(rec)
        print(f"[{log_name}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
              flush=True)
        if rec["val_macro_f1"] > best:
            best, wait = rec["val_macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if patience and wait >= patience:
                print(f"[{log_name}] early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(history, open(TAB / f"history_{log_name}.json", "w"), indent=1)
    return model, history, time.time() - t_start


def plot_history(histories: dict, out: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for label, h in histories.items():
        h = pd.DataFrame(h)
        l, = axes[0].plot(h.epoch, h.train_loss, lw=1, label=f"{label} train")
        axes[0].plot(h.epoch, h.val_loss, lw=1, ls="--", color=l.get_color(), label=f"{label} val")
        axes[1].plot(h.epoch, h.val_macro_f1, lw=1, color=l.get_color(), label=label)
    axes[0].set_title("loss (solid train, dashed val)", fontsize=9); axes[1].set_title("val macro-F1", fontsize=9)
    for ax in axes:
        ax.set_xlabel("epoch"); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=6, frameon=False)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
