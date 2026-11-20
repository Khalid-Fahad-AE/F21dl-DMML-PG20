"""Workstream D (R4) - Multi-Layer Perceptron.

A small NumPy MLP (ReLU, optional batch norm, inverted dropout, Adam, L2 weight decay,
early stopping on validation macro-F1). Written from scratch because scikit-learn's
MLPClassifier has no dropout or batch norm, and those are part of our experiments.
A gradient check is in ``tests/test_mlp_grad.py``.

Inputs
  feat : PCA projection of the shared hand-crafted features (features.npz, X_pca)
  pix  : flattened 64x64x3 pixels, scaled to [0,1] and standardised with train mean/std

Experiments (one factor at a time from a base config), all scored on VALIDATION:
  depth, width, learning rate, dropout, batch norm, early stopping.
The best configuration per input type is then scored once on TEST via evaluate.py.

    python src/mlp.py              # feature experiments + pixel runs
    python src/mlp.py --quick      # fewer epochs, for a smoke test
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from data import ROOT, SEED, load_images, load_splits, set_seed
from evaluate import evaluate
from features import load_features

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"


# ============================================================================ model
class NumpyMLP:
    def __init__(self, n_in, n_out, hidden=(512, 256), dropout=0.3, batchnorm=True, lr=1e-3,
                 weight_decay=1e-4, batch_size=128, max_epochs=60, patience=8, seed=SEED):
        self.cfg = dict(hidden=list(hidden), dropout=dropout, batchnorm=batchnorm, lr=lr,
                        weight_decay=weight_decay, batch_size=batch_size, max_epochs=max_epochs, patience=patience)
        self.rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.params = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = {"W": (self.rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32),
                     "b": np.zeros(b, np.float32)}
            if batchnorm and i < len(hidden):
                layer.update(g=np.ones(b, np.float32), be=np.zeros(b, np.float32),
                             rm=np.zeros(b, np.float32), rv=np.ones(b, np.float32))
            self.params.append(layer)
        self.n_layers = len(self.params)
        self._adam = [{k: (np.zeros_like(v), np.zeros_like(v)) for k, v in p.items() if k not in ("rm", "rv")}
                      for p in self.params]
        self.t = 0
        self.history = []

    @property
    def n_params(self):
        return int(sum(v.size for p in self.params for k, v in p.items() if k not in ("rm", "rv")))

    # ---------------------------------------------------------------- forward / backward
    def forward(self, X, train=False):
        cache = []
        h = X
        for i, p in enumerate(self.params):
            z = h @ p["W"] + p["b"]
            c = {"h_in": h}
            if i < self.n_layers - 1:
                if "g" in p:
                    if train:
                        mu, var = z.mean(0), z.var(0)
                        p["rm"] = 0.9 * p["rm"] + 0.1 * mu
                        p["rv"] = 0.9 * p["rv"] + 0.1 * var
                    else:
                        mu, var = p["rm"], p["rv"]
                    inv = 1.0 / np.sqrt(var + 1e-5)
                    zh = (z - mu) * inv
                    c.update(zh=zh, inv=inv)
                    z = p["g"] * zh + p["be"]
                c["pre"] = z
                h = np.maximum(z, 0)
                if train and self.cfg["dropout"] > 0:
                    keep = 1 - self.cfg["dropout"]
                    m = (self.rng.random(h.shape) < keep).astype(np.float32) / keep
                    c["mask"] = m
                    h = h * m
            else:
                h = z
            cache.append(c)
        return h, cache

    @staticmethod
    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def loss_and_grads(self, X, y, train=True):
        logits, cache = self.forward(X, train=train)
        P = self.softmax(logits)
        n = len(y)
        loss = -np.log(P[np.arange(n), y] + 1e-12).mean()
        d = P.copy(); d[np.arange(n), y] -= 1; d /= n
        grads = [None] * self.n_layers
        for i in reversed(range(self.n_layers)):
            p, c = self.params[i], cache[i]
            g = {"W": c["h_in"].T @ d + self.cfg["weight_decay"] * p["W"], "b": d.sum(0)}
            dh = d @ p["W"].T
            if i > 0:
                pc, pp = cache[i - 1], self.params[i - 1]
                if "mask" in pc:
                    dh = dh * pc["mask"]
                dz = dh * (pc["pre"] > 0)
                if "g" in pp:
                    zh, inv = pc["zh"], pc["inv"]
                    grads_bn = {"g": (dz * zh).sum(0), "be": dz.sum(0)}
                    dzh = dz * pp["g"]
                    m = dz.shape[0]
                    dz = inv / m * (m * dzh - dzh.sum(0) - zh * (dzh * zh).sum(0))
                    grads[i - 1] = grads_bn  # filled with W,b on the next iteration
                d = dz
            grads[i] = {**(grads[i] or {}), **g}
        return loss, grads

    def _step(self, grads):
        self.t += 1
        b1, b2, lr = 0.9, 0.999, self.cfg["lr"]
        for p, g, st in zip(self.params, grads, self._adam):
            for k in g:
                m, v = st[k]
                m[:] = b1 * m + (1 - b1) * g[k]
                v[:] = b2 * v + (1 - b2) * g[k] ** 2
                mh, vh = m / (1 - b1 ** self.t), v / (1 - b2 ** self.t)
                p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)

    # ---------------------------------------------------------------- training
    def fit(self, X, y, Xva, yva, verbose=False):
        X, Xva = X.astype(np.float32), Xva.astype(np.float32)
        bs, best, best_state, wait = self.cfg["batch_size"], -1, None, 0
        t0 = time.time()
        for ep in range(self.cfg["max_epochs"]):
            idx = self.rng.permutation(len(X))
            tl = []
            for s in range(0, len(X) - bs + 1, bs):  # drop last partial batch (batch norm)
                b = idx[s:s + bs]
                loss, grads = self.loss_and_grads(X[b], y[b])
                self._step(grads)
                tl.append(loss)
            va_logits, _ = self.forward(Xva)
            va_loss = -np.log(self.softmax(va_logits)[np.arange(len(yva)), yva] + 1e-12).mean()
            va_pred = va_logits.argmax(1)
            sub = idx[:3000]
            tr_pred = self.forward(X[sub])[0].argmax(1)
            rec = {"epoch": ep + 1, "train_loss": float(np.mean(tl)), "val_loss": float(va_loss),
                   "train_acc": float((tr_pred == y[sub]).mean()), "val_acc": float((va_pred == yva).mean()),
                   "val_macro_f1": float(f1_score(yva, va_pred, average="macro"))}
            self.history.append(rec)
            if verbose:
                print(rec)
            if rec["val_macro_f1"] > best:
                best, best_state, wait = rec["val_macro_f1"], copy.deepcopy(self.params), 0
            else:
                wait += 1
                if self.cfg["patience"] and wait >= self.cfg["patience"]:
                    break
        self.train_time = time.time() - t0
        if self.cfg["patience"]:  # early stopping: restore the best epoch
            self.params = best_state
            self.best_val_f1 = best
        else:  # no early stopping: the model we keep is the last epoch, so report that epoch
            self.best_val_f1 = self.history[-1]["val_macro_f1"]
        return self

    def predict(self, X, batch=2048):
        return np.concatenate([self.forward(X[s:s + batch].astype(np.float32))[0].argmax(1)
                               for s in range(0, len(X), batch)])


# ============================================================================ experiments
BASE = dict(hidden=(512, 256), dropout=0.3, batchnorm=True, lr=1e-3, patience=8)
EXPERIMENTS = {
    "depth": [dict(hidden=(256,)), dict(hidden=(512, 256)), dict(hidden=(512, 256, 128)),
              dict(hidden=(512, 512, 256, 128))],
    "width": [dict(hidden=(64, 32)), dict(hidden=(256, 128)), dict(hidden=(1024, 512))],
    "learning_rate": [dict(lr=1e-4), dict(lr=1e-3), dict(lr=1e-2)],
    "dropout": [dict(dropout=0.0), dict(dropout=0.3), dict(dropout=0.5)],
    "batchnorm": [dict(batchnorm=False), dict(batchnorm=True)],
    "early_stopping": [dict(patience=0), dict(patience=8)],
}


def pixel_inputs(df):
    imgs = load_images(64).reshape(len(df), -1).astype(np.float32) / 255.0
    tr = (df.split == "train").to_numpy()
    mu, sd = imgs[tr].mean(0), imgs[tr].std(0) + 1e-6
    imgs = (imgs - mu) / sd
    y = df.label.to_numpy()
    out = {"mu": mu, "sd": sd}
    for s, k in (("train", "tr"), ("val", "va"), ("test", "te")):
        m = (df.split == s).to_numpy()
        out["X" + k], out["y" + k] = imgs[m], y[m]
    return out


def describe(cfg):
    return f"h={list(cfg['hidden'])} do={cfg['dropout']} bn={cfg['batchnorm']} lr={cfg['lr']} es={cfg['patience']}"


def plot_curves(histories, out, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for label, h in histories.items():
        h = pd.DataFrame(h)
        l, = axes[0].plot(h.epoch, h.train_loss, lw=1, label=f"{label} train")
        axes[0].plot(h.epoch, h.val_loss, lw=1, ls="--", color=l.get_color(), label=f"{label} val")
        axes[1].plot(h.epoch, h.train_acc, lw=1, color=l.get_color())
        axes[1].plot(h.epoch, h.val_acc, lw=1, ls="--", color=l.get_color())
    axes[0].set_title("cross-entropy loss (solid train, dashed val)", fontsize=9)
    axes[1].set_title("accuracy", fontsize=9)
    for ax in axes:
        ax.set_xlabel("epoch"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, None)
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def _save(name, model, **extra):
    import pickle
    out = ROOT / "results" / "models"
    out.mkdir(parents=True, exist_ok=True)
    model._adam = None  # optimiser state is not needed for inference
    pickle.dump({"model": model, **extra}, open(out / f"{name}.pkl", "wb"))


def main(quick=False):
    set_seed()
    df = load_splits()
    feats = load_features(pca=True)
    rows, hist = [], {}
    max_ep = 8 if quick else 60
    for exp, variants in EXPERIMENTS.items():
        for v in variants:
            cfg = {**BASE, **v, "max_epochs": max_ep}
            m = NumpyMLP(feats["Xtr"].shape[1], 10, **cfg).fit(feats["Xtr"], feats["ytr"], feats["Xva"], feats["yva"])
            h = pd.DataFrame(m.history)
            rows.append({"input": "feat", "experiment": exp, "config": describe(cfg), **{k: (list(x) if k == "hidden" else x) for k, x in cfg.items()},
                         "params": m.n_params, "epochs_run": len(h), "best_epoch": int(h.val_macro_f1.idxmax() + 1),
                         "val_macro_f1": m.best_val_f1, "final_train_acc": h.train_acc.iloc[-1],
                         "final_val_acc": h.val_acc.iloc[-1], "train_time_s": m.train_time})
            hist[(exp, describe(cfg))] = m.history
            print(exp, describe(cfg), f"val F1={m.best_val_f1:.4f} epochs={len(h)} {m.train_time:.0f}s")
    res = pd.DataFrame(rows)
    res.round(4).to_csv(TAB / "mlp_experiments.csv", index=False)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in hist.items()}, open(TAB / "mlp_histories.json", "w"))

    # learning-curve figures for the dropout / batch-norm / early-stopping ablations
    for exp in ["dropout", "batchnorm", "early_stopping", "learning_rate"]:
        labels = {"dropout": "do", "batchnorm": "bn", "early_stopping": "es", "learning_rate": "lr"}[exp]
        hs = {[t for t in k[1].split() if t.startswith(labels + "=")][0]: v for k, v in hist.items() if k[0] == exp}
        plot_curves(hs, FIG / f"mlp_curves_{exp}.png", f"MLP on features: {exp.replace('_', ' ')}")

    # ---- best feature config -> test
    best = res.sort_values("val_macro_f1", ascending=False).iloc[0]
    best_cfg = {k: best[k] for k in ["dropout", "batchnorm", "lr", "patience"]}
    best_cfg["hidden"] = tuple(best["hidden"])
    trva_cfg = {**best_cfg, "max_epochs": max_ep}
    m = NumpyMLP(feats["Xtr"].shape[1], 10, **trva_cfg).fit(feats["Xtr"], feats["ytr"], feats["Xva"], feats["yva"])
    t0 = time.time(); pred = m.predict(feats["Xte"]); ti = time.time() - t0
    evaluate("MLP_features", feats["yte"], pred, train_time=m.train_time, infer_time=ti, n_params=m.n_params,
             group="D MLP", notes=f"PCA features; {describe(trva_cfg)}; val macroF1={m.best_val_f1:.4f}")
    _save("MLP_features", m)
    plot_curves({"best": m.history}, FIG / "mlp_curves_best_features.png", f"MLP on features, best config: {describe(trva_cfg)}")

    # ---- flattened 64x64 pixels: base config and best feature config
    pix = pixel_inputs(df)
    pix_rows, pix_hist = [], {}
    for label, cfg in [("base", {**BASE}), ("wide-highdrop", {**BASE, "hidden": (1024, 256), "dropout": 0.5})]:
        cfg = {**cfg, "max_epochs": 6 if quick else 40, "patience": 6}
        mp = NumpyMLP(pix["Xtr"].shape[1], 10, **cfg).fit(pix["Xtr"], pix["ytr"], pix["Xva"], pix["yva"])
        pix_rows.append({"input": "pix", "config": describe(cfg), "params": mp.n_params, "val_macro_f1": mp.best_val_f1,
                         "epochs_run": len(mp.history), "train_time_s": mp.train_time, "model": mp, "cfg": cfg})
        pix_hist[label] = mp.history
        print("pixels", describe(cfg), f"val F1={mp.best_val_f1:.4f} {mp.train_time:.0f}s")
    plot_curves(pix_hist, FIG / "mlp_curves_pixels.png", "MLP on flattened 64x64 pixels")
    bp = max(pix_rows, key=lambda r: r["val_macro_f1"])
    t0 = time.time(); pred = bp["model"].predict(pix["Xte"]); ti = time.time() - t0
    evaluate("MLP_pixels", pix["yte"], pred, train_time=bp["train_time_s"], infer_time=ti, n_params=bp["params"],
             group="D MLP", notes=f"64x64x3 pixels; {bp['config']}; val macroF1={bp['val_macro_f1']:.4f}")
    _save("MLP_pixels", bp["model"], mu=pix["mu"], sd=pix["sd"])
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("model", "cfg")} for r in pix_rows]).round(4).to_csv(
        TAB / "mlp_pixel_runs.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    import mlp  # so pickled classes are stored as mlp.<Class>, not __main__.<Class>
    mlp.main(ap.parse_args().quick)
