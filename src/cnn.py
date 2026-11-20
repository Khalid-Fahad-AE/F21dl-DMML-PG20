"""Workstream E (R4) - small CNN trained from scratch, with the augmentation ablation from our pitch.

Architecture: 4 blocks of [Conv3x3-BN-ReLU, Conv3x3-BN-ReLU, MaxPool2], channels 32-64-128-256,
then global average pooling, dropout 0.3 and a 10-way linear layer (~1.2M parameters).
Augmentation ablation (train split only; val/test are never augmented):
    none  -> resize only
    geo   -> + rotation (+-25 deg), horizontal and vertical flips
    full  -> + brightness/contrast/saturation jitter and random zoom (RandomResizedCrop 0.7-1.0)
Each run: AdamW lr 1e-3, cosine schedule, up to 30 epochs, early stopping (patience 6) on val macro-F1.
Every run is scored on the PlantVillage test split AND on the PlantDoc field photos.

    python src/cnn.py --aug all --size 128          # the three runs (~10 min each on a Colab T4)
    python src/cnn.py --aug full --size 224
"""
import argparse

import torch
import torch.nn as nn

from data import ROOT
from evaluate import evaluate
from plantdoc import present_labels
from torch_common import (MODELS, count_params, get_device, latency_ms, make_loaders, plot_history, predict,
                          seed_everything, train)

FIG = ROOT / "results" / "figures"


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers.append(block(cin, w)); cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

    @property
    def gradcam_layer(self):
        return self.features[-1][5]  # ReLU output of the last conv in the last block (before pooling)


def run(aug, size, epochs, batch_size, workers):
    seed_everything()
    device = get_device()
    tr, va, te, pdoc = make_loaders(size, aug, batch_size, workers)
    model = SmallCNN()
    name = f"CNN_scratch_{aug}_{size}"
    model, hist, train_time = train(model, tr, va, device, epochs=epochs, lr=1e-3, patience=6, log_name=name)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / f"{name}.pt")
    y, p, t = predict(model, te, device)
    lat = latency_ms(model, size, torch.device("cpu")) if device.type != "cpu" else latency_ms(model, size, device)
    evaluate(name, y, p, train_time=train_time, infer_time=t, n_params=count_params(model), group="E CNN",
             notes=f"aug={aug}, {size}px, epochs run={len(hist)}, CPU latency {lat:.1f} ms/img (batch 1)")
    if pdoc is not None:
        y2, p2, _ = predict(model, pdoc, device)
        evaluate(f"{name}@PlantDoc", y2, p2, split="plantdoc", group="F field test", labels=present_labels(y2))
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="all", choices=["none", "geo", "full", "all"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    augs = ["none", "geo", "full"] if a.aug == "all" else [a.aug]
    hists = {aug: run(aug, a.size, a.epochs, a.batch_size, a.workers) for aug in augs}
    FIG.mkdir(parents=True, exist_ok=True)
    plot_history(hists, FIG / f"cnn_curves_{a.size}.png", f"Small CNN from scratch ({a.size}px): augmentation ablation")


if __name__ == "__main__":
    main()
