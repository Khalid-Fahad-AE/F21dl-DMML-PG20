"""Workstream E - Grad-CAM: does the CNN look at the lesions or at the background?

Grad-CAM (Selvaraju et al., 2017): weight each channel of the last conv feature map by the
spatially averaged gradient of the predicted class score, sum, ReLU, upsample.

Besides the overlay grids (correct and wrong test predictions) we compute a number for the report:
  leaf-focus ratio = (share of Grad-CAM mass that falls on the leaf) / (share of the image that is leaf)
A ratio > 1 means attention concentrates on the leaf; ~1 means no preference; < 1 means the
model is using the background. The leaf mask is a simple HSV threshold (PlantVillage backgrounds
are grey/purple and low-saturation), refined with morphological opening/closing.

    python src/gradcam.py --model CNN_scratch_full_128            # small CNN checkpoint
    python src/gradcam.py --model TL_mobilenet_v2 --size 224        # transfer model checkpoint
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2hsv

from data import DEFAULT_IMG_ROOT, ROOT, SHORT, load_splits
from torch_common import IMAGENET_MEAN, IMAGENET_STD, MODELS, build_transforms, get_device

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=None):
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        pred = logits.argmax(1)
        tgt = pred if target is None else target
        self.model.zero_grad()
        logits.gather(1, tgt[:, None]).sum().backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.flatten(1).max(1)[0][:, None, None] + 1e-8)
        return cam.detach().cpu().numpy(), pred.cpu().numpy(), logits.softmax(1).detach().cpu().numpy()


def leaf_mask(rgb):
    hsv = rgb2hsv(rgb)
    h = hsv[..., 0]
    m = (hsv[..., 1] > 0.18) & (hsv[..., 2] > 0.12) & (h > 0.02) & (h < 0.5)  # green-yellow-brown hues
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_closing(m, iterations=3)
    lab, n = ndimage.label(m)
    if n > 1:  # keep the biggest component (the leaf)
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(m)


def load_model(name, size):
    if name.startswith("CNN_scratch"):
        from cnn import SmallCNN
        model = SmallCNN()
        model.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location="cpu"))
        layer = model.gradcam_layer
    else:
        from transfer import build_model
        arch = name.replace("TL_", "")
        model, _ = build_model(arch, pretrained=False)
        model.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location="cpu"))
        layer = {"mobilenet_v2": lambda m: m.features[-1], "efficientnet_b0": lambda m: m.features[-1],
                 "resnet50": lambda m: m.layer4[-1]}[arch](model)
    return model, layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CNN_scratch_full_128")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-stats", type=int, default=600, help="test images used for the leaf-focus statistic")
    a = ap.parse_args()
    device = get_device()
    model, layer = load_model(a.model, a.size)
    model.to(device)
    cam_fn = GradCAM(model, layer)
    _, tf = build_transforms(a.size, "none")
    df = load_splits()
    te = df[df.split == "test"].sample(frac=1, random_state=42).reset_index(drop=True)

    rows, examples = [], {"correct": [], "wrong": []}
    for s in range(0, len(te), 32):
        chunk = te.iloc[s:s + 32]
        ims = [Image.open(DEFAULT_IMG_ROOT / p).convert("RGB").resize((a.size, a.size)) for p in chunk.path]
        x = torch.stack([tf(im) for im in ims]).to(device)
        cams, pred, prob = cam_fn(x)
        for im, cam, pr, pb, (_, r) in zip(ims, cams, pred, prob, chunk.iterrows()):
            rgb = np.asarray(im) / 255.0
            m = leaf_mask(rgb)
            leaf_share = m.mean()
            cam_on_leaf = (cam * m).sum() / (cam.sum() + 1e-8)
            ok = int(pr) == int(r.label)
            rows.append({"path": r.path, "label": int(r.label), "pred": int(pr), "correct": ok,
                         "leaf_area_share": leaf_share, "cam_mass_on_leaf": cam_on_leaf,
                         "leaf_focus_ratio": cam_on_leaf / max(leaf_share, 1e-3)})
            key = "correct" if ok else "wrong"
            if len(examples[key]) < 8:
                examples[key].append((rgb, cam, m, SHORT[int(r.label)], SHORT[int(pr)], float(pb.max())))
        if len(rows) >= a.n_stats and len(examples["wrong"]) >= 8:
            break

    st = pd.DataFrame(rows)
    # the HSV mask is a heuristic: drop images where it is implausible (beige backdrops etc.)
    st["mask_ok"] = st.leaf_area_share.between(0.05, 0.70)
    st.to_csv(TAB / f"gradcam_stats_{a.model}.csv", index=False)
    print(f"mask plausible for {st.mask_ok.mean():.1%} of images")
    summ = st[st.mask_ok].groupby("correct")[["cam_mass_on_leaf", "leaf_area_share", "leaf_focus_ratio"]].mean()
    summ.round(3).to_csv(TAB / f"gradcam_summary_{a.model}.csv")
    print(summ)

    for key, ex in examples.items():
        if not ex:
            continue
        fig, axes = plt.subplots(2, len(ex), figsize=(1.6 * len(ex), 3.6))
        axes = np.atleast_2d(axes).reshape(2, -1)
        for j, (rgb, cam, m, t, p, pb) in enumerate(ex):
            axes[0, j].imshow(rgb); axes[0, j].contour(m, levels=[0.5], colors="white", linewidths=0.5)
            axes[0, j].set_title(f"true: {t}\npred: {p} ({pb:.2f})", fontsize=6)
            axes[1, j].imshow(rgb); axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
            for ax in axes[:, j]:
                ax.axis("off")
        fig.suptitle(f"Grad-CAM, {a.model}: {key} test predictions (white line = leaf mask)", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG / f"gradcam_{a.model}_{key}.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
