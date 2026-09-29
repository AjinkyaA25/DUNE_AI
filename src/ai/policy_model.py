"""
Policy head: P(action | state) over the legal action set.

`PolicyModel` scores each candidate action with a small NumPy MLP
`f([state ; action]) -> logit` and softmaxes over the legal set. It is trained
by advantage-weighted regression (AWR): imitate the action that was taken,
weighted by how much that trajectory beat the value net's prediction. Actions
from fast wins get up-weighted; actions from losses get ~0 weight. So over many
games the policy learns the moves that actually correlate with winning even
when the (heuristic / value) generator only played them sometimes — including
compounding moves like Swordmaster that a 1-ply value-max can't see.

No torch. Save/load via .npz, same style as ValueModel.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.ai.features import FEATURE_DIM
from src.ai.action_features import ACTION_FEATURE_DIM

_NEG = -1e9


class PolicyModel:
    def __init__(self, state_dim: int = FEATURE_DIM,
                 action_dim: int = ACTION_FEATURE_DIM,
                 hidden: int = 64, seed: int = 0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dim = state_dim + action_dim
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0 / np.sqrt(self.dim),
                             (self.dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, np.float32)
        self._adam = {}

    # -- inference -------------------------------------------------------

    def grow(self, state_dim: int) -> None:
        """Accept `state_dim` state features (new ones appended to the state
        block) with zero weights, so predictions are unchanged."""
        extra = state_dim - self.state_dim
        if extra <= 0:
            return
        z = np.zeros((extra, self.W1.shape[1]), self.W1.dtype)
        self.W1 = np.vstack([self.W1[:self.state_dim], z, self.W1[self.state_dim:]])
        self.state_dim = state_dim
        self.dim = state_dim + self.action_dim
        self._adam = {}

    def _logits_flat(self, XA: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """XA: (M, state_dim + action_dim) -> (logits (M,), hidden (M,H))."""
        h = np.tanh(XA @ self.W1 + self.b1)
        logit = (h @ self.W2 + self.b2).ravel()
        return logit, h

    def action_logits(self, state_feats: np.ndarray,
                      action_feats: np.ndarray) -> np.ndarray:
        """state_feats: (state_dim,)  action_feats: (K, action_dim) -> (K,) logits."""
        K = action_feats.shape[0]
        state_feats = state_feats[:self.state_dim]   # older model: fewer feats
        xa = np.concatenate(
            [np.repeat(state_feats.reshape(1, -1), K, axis=0),
             action_feats], axis=1).astype(np.float64)
        logit, _ = self._logits_flat(xa)
        return logit

    def policy(self, state_feats: np.ndarray,
               action_feats: np.ndarray) -> np.ndarray:
        z = self.action_logits(state_feats, action_feats)
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    # -- training (advantage-weighted CE over the legal set) -------------

    def fit(self, S: np.ndarray, A: np.ndarray, M: np.ndarray, ci: np.ndarray,
            weight: np.ndarray, epochs: int = 20, batch_size: int = 1024,
            lr: float = 2e-3, l2: float = 1e-5, val_frac: float = 0.1,
            seed: int = 0, verbose: bool = False) -> dict:
        """
        S: (N, state_dim)              per-decision state features
        A: (N, Kmax, action_dim)       padded candidate-action features
        M: (N, Kmax)  {0,1}            legal mask
        ci: (N,) int                   index (into Kmax) of the action taken
        weight: (N,)                   AWR sample weight (>= 0)
        (a model built for fewer state features grows to fit S)
        """
        if S.shape[1] > self.state_dim:
            self.grow(S.shape[1])
        S = np.ascontiguousarray(S, np.float32)
        A = np.ascontiguousarray(A, np.float32)
        M = np.ascontiguousarray(M, np.float32)
        w = np.ascontiguousarray(weight, np.float32).ravel()
        ci = ci.astype(np.int64).ravel()
        N, Kmax, _ = A.shape
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)
        S, A, M, ci, w = S[perm], A[perm], M[perm], ci[perm], w[perm]
        nv = max(1, int(N * val_frac))
        tr = slice(nv, N); va = slice(0, nv)

        for pn in ("W1", "b1", "W2", "b2"):
            self._adam.setdefault(pn, [np.zeros_like(getattr(self, pn)),
                                       np.zeros_like(getattr(self, pn)), 0])

        def astep(g, name, b1=0.9, b2=0.999, eps=1e-8):
            m, v, t = self._adam[name]
            t += 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            setattr(self, name, getattr(self, name)
                    - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps))
            self._adam[name] = [m, v, t]

        def batch_loss_acc(idx):
            ll, correct, tot = 0.0, 0, 0
            for s in range(0, len(idx), batch_size):
                bi = idx[s:s + batch_size]
                lp, _, _ = self._forward_batch(S[bi], A[bi], M[bi])
                pick = lp[np.arange(len(bi)), ci[bi]]
                ll += -(pick).sum()
                correct += int((lp.argmax(axis=1) == ci[bi]).sum())
                tot += len(bi)
            return ll / max(1, tot), correct / max(1, tot)

        tr_idx = np.arange(N)[tr]
        va_idx = np.arange(N)[va]
        hist = {"val_logloss": [], "val_acc": []}
        for ep in range(epochs):
            order = rng.permutation(len(tr_idx))
            for s in range(0, len(order), batch_size):
                bi = tr_idx[order[s:s + batch_size]]
                self._train_batch(S[bi], A[bi], M[bi], ci[bi], w[bi], l2, astep)
            vll, vacc = batch_loss_acc(va_idx)
            hist["val_logloss"].append(vll)
            hist["val_acc"].append(vacc)
            if verbose:
                print(f"  epoch {ep+1:2d}/{epochs}  val_logloss={vll:.4f}  acc={vacc:.3f}")
        return hist

    def _forward_batch(self, Sb, Ab, Mb):
        """-> log_softmax (B,K), softmax (B,K), hidden h (B,K,H).

        The state half of the input is identical across the K candidates, so we
        never materialise the (B,K,state_dim) tensor: h = tanh(S@W1s + A@W1a).
        """
        sd = self.state_dim
        pre_s = Sb @ self.W1[:sd] + self.b1                          # (B,H)
        pre_a = Ab @ self.W1[sd:]                                    # (B,K,H)
        h = np.tanh(pre_s[:, None, :] + pre_a)                       # (B,K,H)
        logit = h @ self.W2[:, 0] + self.b2[0]                       # (B,K)
        logit = np.where(Mb > 0, logit, _NEG)
        z = logit - logit.max(axis=1, keepdims=True)
        e = np.exp(z) * (Mb > 0)
        esum = e.sum(axis=1, keepdims=True)
        return z - np.log(esum), e / esum, h

    def _train_batch(self, Sb, Ab, Mb, cib, wb, l2, astep):
        B, K, H = Ab.shape[0], Ab.shape[1], self.hidden
        sd = self.state_dim
        _, sm, h = self._forward_batch(Sb, Ab, Mb)
        onehot = np.zeros((B, K), np.float32)
        onehot[np.arange(B), cib] = 1.0
        g = ((wb[:, None] * (sm - onehot)) * (Mb > 0) / B).astype(np.float32)
        h2 = h.reshape(B * K, H)
        gflat = g.reshape(B * K, 1)
        gW2 = h2.T @ gflat + l2 * self.W2                           # (H,1)
        gb2 = np.array([g.sum()])
        gh = (g[..., None] * self.W2[:, 0]) * (1 - h ** 2)          # (B,K,H)
        ghflat = gh.reshape(B * K, H)
        gW1s = Sb.T @ gh.sum(axis=1) + l2 * self.W1[:sd]            # state shared over K
        gW1a = Ab.reshape(B * K, -1).T @ ghflat + l2 * self.W1[sd:]
        gW1 = np.vstack([gW1s, gW1a])
        gb1 = ghflat.sum(axis=0)
        astep(gW1, "W1"); astep(gb1, "b1"); astep(gW2, "W2"); astep(gb2, "b2")

    # -- persistence ---------------------------------------------------

    def save(self, path: str) -> None:
        np.savez(path, state_dim=self.state_dim, action_dim=self.action_dim,
                 hidden=self.hidden, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str) -> "PolicyModel":
        z = np.load(path, allow_pickle=False)
        m = cls(state_dim=int(z["state_dim"]), action_dim=int(z["action_dim"]),
                hidden=int(z["hidden"]))
        m.W1, m.b1, m.W2, m.b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        return m


def awr_weights(value_model, X: np.ndarray, ret: np.ndarray, base_w: np.ndarray,
                beta: float = 0.4, w_max: float = 10.0):
    """Advantage-weighted-regression sample weights: up-weight the action taken
    when that trajectory's return beat the value net's prediction for the state.
    `ret` = discounted win return (winner GAMMA^rounds-to-win, loser 0).
    Returns (weights, mean_advantage) for logging."""
    V = value_model.predict_batch(X.astype(np.float64)).ravel()
    adv = np.clip(ret.ravel() - V, -1.0, 1.0)
    w = np.minimum(w_max, np.exp(adv / beta)) * base_w.ravel()
    return w.astype(np.float64), float(adv.mean())
