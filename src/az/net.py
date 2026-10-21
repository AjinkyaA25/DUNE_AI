"""NumPy inference for the AlphaZero net (workers never import torch).

  trunk : obs -> 256 -> 256 (ReLU)                       = h
  value : h -> 64 -> 4   expected result of each seat, ordered from the
          acting player (me, next, next+1, next+2)
  policy: per legal move u = ReLU(h P + dense D + sum E[idx] + pb) (128),
          logit = u . pw + pc
"""
from __future__ import annotations

import numpy as np

from src.az.encode import DENSE_DIM, HASH_DIM, STATE_DIM

H, HV, HP = 256, 64, 128
SHAPES = {"W1": (STATE_DIM, H), "b1": (H,), "W2": (H, H), "b2": (H,),
          "V1": (H, HV), "vb1": (HV,), "V2": (HV, 4), "vb2": (4,),
          "P": (H, HP), "D": (DENSE_DIM, HP), "E": (HASH_DIM + 1, HP), "pb": (HP,),
          "pw": (HP,), "pc": (1,)}


def init_params(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for k, s in SHAPES.items():
        if len(s) == 2:
            out[k] = (rng.normal(0, 1.0 / np.sqrt(s[0]), s)).astype(np.float32)
        else:
            out[k] = np.zeros(s, np.float32)
    out["E"][HASH_DIM] = 0.0
    out["pw"] = (rng.normal(0, 0.01, SHAPES["pw"])).astype(np.float32)
    out["vb2"][:] = 0.25
    return out


class AZNet:
    def __init__(self, params: dict):
        self.p = {k: np.asarray(v, np.float32) for k, v in params.items()}

    @classmethod
    def load(cls, path: str) -> "AZNet":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def trunk(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        h = np.maximum(x @ p["W1"] + p["b1"], 0.0)
        return np.maximum(h @ p["W2"] + p["b2"], 0.0)

    def value(self, h: np.ndarray) -> np.ndarray:
        p = self.p
        return np.maximum(h @ p["V1"] + p["vb1"], 0.0) @ p["V2"] + p["vb2"]

    def logits(self, h: np.ndarray, dense: np.ndarray, idx: np.ndarray) -> np.ndarray:
        p = self.p
        u = h @ p["P"] + dense @ p["D"] + p["E"][idx].sum(axis=1) + p["pb"]
        return np.maximum(u, 0.0) @ p["pw"] + p["pc"][0]
