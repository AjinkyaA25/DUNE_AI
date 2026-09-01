"""
Value model: features -> P(perspective player wins the game).

`ValueModel` is a small NumPy MLP (one hidden layer, tanh) trained with Adam on
binary cross-entropy.  `hidden=0` degenerates to logistic regression.  No torch
dependency.  Save/load via .npz.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.ai.features import FEATURE_DIM


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class ValueModel:
    def __init__(self, dim: int = FEATURE_DIM, hidden: int = 64, seed: int = 0):
        self.dim = dim
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        if hidden > 0:
            self.W1 = rng.normal(0, 1.0 / np.sqrt(dim), (dim, hidden)).astype(np.float64)
            self.b1 = np.zeros(hidden)
            self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1)).astype(np.float64)
            self.b2 = np.zeros(1)
        else:
            self.W1 = np.zeros((dim, 1))
            self.b1 = np.zeros(1)
            self.W2 = None
            self.b2 = None
        self._adam = {}

    # -- inference --------------------------------------------------------

    def _forward(self, X: np.ndarray):
        if self.hidden > 0:
            h = np.tanh(X @ self.W1 + self.b1)
            logit = (h @ self.W2 + self.b2).ravel()
            return logit, h
        logit = (X @ self.W1 + self.b1).ravel()
        return logit, None

    def predict(self, feats: np.ndarray) -> float:
        x = feats.reshape(1, -1).astype(np.float64)
        logit, _ = self._forward(x)
        return float(_sigmoid(logit)[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        logit, _ = self._forward(X.astype(np.float64))
        return _sigmoid(logit)

    # -- training --------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            epochs: int = 40, batch_size: int = 512, lr: float = 3e-3,
            l2: float = 1e-5, val_frac: float = 0.1, seed: int = 0,
            verbose: bool = False) -> dict:
        X = X.astype(np.float64)
        y = y.astype(np.float64).ravel()
        w = (np.ones_like(y) if sample_weight is None
             else sample_weight.astype(np.float64).ravel())
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        X, y, w = X[idx], y[idx], w[idx]
        n_val = max(1, int(len(y) * val_frac))
        Xtr, ytr, wtr = X[n_val:], y[n_val:], w[n_val:]
        Xva, yva = X[:n_val], y[:n_val]

        params = ["W1", "b1"] + (["W2", "b2"] if self.hidden > 0 else [])
        for p in params:
            self._adam.setdefault(p, [np.zeros_like(getattr(self, p)),
                                     np.zeros_like(getattr(self, p)), 0])

        def step(gW, name, b1=0.9, b2=0.999, eps=1e-8):
            m, v, t = self._adam[name]
            t += 1
            m = b1 * m + (1 - b1) * gW
            v = b2 * v + (1 - b2) * (gW * gW)
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            setattr(self, name, getattr(self, name) - lr * mhat / (np.sqrt(vhat) + eps))
            self._adam[name] = [m, v, t]

        hist = {"val_logloss": []}
        for ep in range(epochs):
            order = rng.permutation(len(ytr))
            for s in range(0, len(order), batch_size):
                bi = order[s:s + batch_size]
                xb, yb, wb = Xtr[bi], ytr[bi], wtr[bi]
                logit, h = self._forward(xb)
                p = _sigmoid(logit)
                g = (p - yb) * wb / len(bi)                      # dL/dlogit
                if self.hidden > 0:
                    gW2 = h.T @ g[:, None] + l2 * self.W2
                    gb2 = np.array([g.sum()])
                    gh = (g[:, None] @ self.W2.T) * (1 - h ** 2)
                    gW1 = xb.T @ gh + l2 * self.W1
                    gb1 = gh.sum(axis=0)
                    step(gW1, "W1"); step(gb1, "b1"); step(gW2, "W2"); step(gb2, "b2")
                else:
                    gW1 = xb.T @ g[:, None] + l2 * self.W1
                    gb1 = np.array([g.sum()])
                    step(gW1, "W1"); step(gb1, "b1")
            pv = np.clip(self.predict_batch(Xva), 1e-6, 1 - 1e-6)
            ll = float(-(yva * np.log(pv) + (1 - yva) * np.log(1 - pv)).mean())
            hist["val_logloss"].append(ll)
            if verbose:
                print(f"  epoch {ep+1:2d}/{epochs}  val_logloss={ll:.4f}")
        return hist

    # -- persistence ----------------------------------------------------

    def save(self, path: str) -> None:
        d = {"dim": self.dim, "hidden": self.hidden, "W1": self.W1, "b1": self.b1}
        if self.hidden > 0:
            d["W2"] = self.W2
            d["b2"] = self.b2
        np.savez(path, **d)

    @classmethod
    def load(cls, path: str) -> "ValueModel":
        z = np.load(path, allow_pickle=False)
        m = cls(dim=int(z["dim"]), hidden=int(z["hidden"]))
        m.W1 = z["W1"]; m.b1 = z["b1"]
        if m.hidden > 0:
            m.W2 = z["W2"]; m.b2 = z["b2"]
        return m
