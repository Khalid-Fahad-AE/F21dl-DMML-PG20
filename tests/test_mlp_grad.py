"""Finite-difference gradient check for the NumPy MLP (run: python tests/test_mlp_grad.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlp import NumpyMLP  # noqa: E402


def check(batchnorm):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 7)).astype(np.float64)
    y = rng.integers(0, 4, 16)
    m = NumpyMLP(7, 4, hidden=(6, 5), dropout=0.0, batchnorm=batchnorm, weight_decay=0.0)
    for p in m.params:
        for k in p:
            p[k] = p[k].astype(np.float64)
    _, grads = m.loss_and_grads(X, y)
    worst = 0
    for li, p in enumerate(m.params):
        for k in grads[li]:
            for idx in [tuple(rng.integers(0, s) for s in p[k].shape) for _ in range(5)]:
                old = p[k][idx]
                p[k][idx] = old + 1e-5; lp, _ = m.loss_and_grads(X, y)
                p[k][idx] = old - 1e-5; lm, _ = m.loss_and_grads(X, y)
                p[k][idx] = old
                num = (lp - lm) / 2e-5
                worst = max(worst, abs(num - grads[li][k][idx]) / max(1e-8, abs(num) + abs(grads[li][k][idx])))
    return worst


if __name__ == "__main__":
    for bn in (False, True):
        w = check(bn)
        print(f"batchnorm={bn}: max relative error {w:.2e}")
        assert w < 1e-4, "gradient check failed"
    print("gradient check passed")
