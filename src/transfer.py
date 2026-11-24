"""Workstream F (R4) - transfer learning with ImageNet-pretrained MobileNetV2, EfficientNet-B0, ResNet50.

Two stages per backbone, at 224x224 with the 'full' augmentation from our pitch
(rotation, flips, brightness/contrast, zoom):
  stage 1  frozen backbone, train only the new 10-way head      (lr 1e-3, <= 5 epochs)
  stage 2  partial unfreeze of the last blocks + head            (lr 1e-4, <= 12 epochs, early stopping)
           MobileNetV2: features[14:]   EfficientNet-B0: features[6:]   ResNet50: layer4
Reported per model: accuracy / macro-P/R/F1 (evaluate.py), total and trainable parameters,
test throughput, single-image CPU latency, and the PlantDoc field-photo test.

    python src/transfer.py --arch all           # ~45-60 min total on a Colab T4
    python src/transfer.py --arch mobilenet_v2
"""
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from data import ROOT
from evaluate import evaluate
from plantdoc import present_labels
from torch_common import (MODELS, TAB, count_params, get_device, latency_ms, make_loaders, plot_history, predict,
                          seed_everything, train)

FIG = ROOT / "results" / "figures"
SIZE = 224


def build_model(arch, pretrained=True, n_classes=10):
    """Returns (model, list of modules to unfreeze in stage 2)."""
    if arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, n_classes))
        unfreeze = [m.features[14:]]
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, n_classes))
        unfreeze = [m.features[6:]]
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n_classes))
        unfreeze = [m.layer4]
    else:
        raise ValueError(arch)
    return m, unfreeze


def head_of(m, arch):
    return m.fc if arch == "resnet50" else m.classifier


def set_trainable(m, arch, unfreeze_modules):
    for p in m.parameters():
        p.requires_grad = False
    for mod in [head_of(m, arch), *unfreeze_modules]:
        for p in mod.parameters():
            p.requires_grad = True


def run(arch, epochs1, epochs2, batch_size, workers):
    seed_everything()
    device = get_device()
    tr, va, te, pdoc = make_loaders(SIZE, "full", batch_size, workers)
    model, unfreeze = build_model(arch)
    name = f"TL_{arch}"

    # stage 1: frozen backbone (BatchNorm statistics of the frozen part are also kept fixed)
    set_trainable(model, arch, [])
    model, h1, t1 = train(model, tr, va, device, epochs=epochs1, lr=1e-3, patience=0, log_name=f"{name}_stage1")
    y, p, _ = predict(model, te, device)
    evaluate(f"{name}_frozen", y, p, train_time=t1, n_params=count_params(model), group="F transfer",
             notes=f"frozen backbone, trainable={count_params(model, True)}")

    # stage 2: partial unfreeze, lower learning rate
    set_trainable(model, arch, unfreeze)
    model, h2, t2 = train(model, tr, va, device, epochs=epochs2, lr=1e-4, patience=4, log_name=f"{name}_stage2")
    MODELS.mkdir(parents=True, exist_ok=True)  # results/models/ is not in git, so a fresh clone lacks it
    torch.save(model.state_dict(), MODELS / f"{name}.pt")
    y, p, t_inf = predict(model, te, device)
    lat_cpu = latency_ms(model.to("cpu"), SIZE, torch.device("cpu"))
    model.to(device)
    evaluate(name, y, p, train_time=t1 + t2, infer_time=t_inf, n_params=count_params(model), group="F transfer",
             notes=f"partial unfreeze, trainable={count_params(model, True)}, CPU latency {lat_cpu:.1f} ms/img")
    if pdoc is not None:
        y2, p2, _ = predict(model, pdoc, device)
        evaluate(f"{name}@PlantDoc", y2, p2, split="plantdoc", group="F field test", labels=present_labels(y2))
    h = [*h1, *[{**r, "epoch": r["epoch"] + len(h1)} for r in h2]]
    return h, {"model": name, "total_params": count_params(model), "trainable_stage2": count_params(model, True),
               "cpu_latency_ms": lat_cpu, "train_time_s": t1 + t2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=["mobilenet_v2", "efficientnet_b0", "resnet50", "all"])
    ap.add_argument("--epochs1", type=int, default=5)
    ap.add_argument("--epochs2", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    archs = ["mobilenet_v2", "efficientnet_b0", "resnet50"] if a.arch == "all" else [a.arch]
    hists, cost = {}, []
    for arch in archs:
        hists[arch], c = run(arch, a.epochs1, a.epochs2, a.batch_size, a.workers)
        cost.append(c)
    pd.DataFrame(cost).to_csv(TAB / "transfer_cost.csv", index=False)
    plot_history(hists, FIG / "transfer_curves.png", "Transfer learning (stage 1 frozen, then partial unfreeze)")


if __name__ == "__main__":
    main()
