"""Dense reward and the all-moves (ragged) batch of the PPO trainer."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.selfplay.ppo_train import advantages, rows, to_batch


def _traj(vps, k=3, A=4):
    rng = np.random.default_rng(len(vps))
    return [{"s": rng.normal(size=5).astype(np.float16), "D": rng.normal(size=(k, A)).astype(np.float16),
             "I": np.zeros((k, 5), np.int16), "h": rng.normal(size=k).astype(np.float32), "k": 0,
             "logp": 0.0, "vp": v} for v in vps]


def test_dense_reward_sums_to_the_game_result():
    t = _traj([[1, 1, 1, 1], [3, 1, 2, 1], [3, 4, 2, 1], [6, 4, 2, 1]])
    R = 1.0
    vals = np.zeros(len(t), np.float32)
    a0, _ = advantages([(t, R, None)], vals, lam=1.0, dense=0.0)
    a1, _ = advantages([(t, R, None)], vals, lam=1.0, dense=0.1)
    assert np.allclose(a0, R)                    # end-only: every decision gets the result
    assert np.isclose(a1[0], R)                  # shaping telescopes: the total is still the result
    # ... but the VP swing between decisions 0 and 1 (+2 me, +1/3 opponents' average) is credited there
    assert np.isclose(a1[0] - a1[1], 0.1 * (2 - 1 / 3))


def test_ragged_rows_pad_each_decision():
    t = _traj([[1, 1, 1, 1]] * 2, k=2) + _traj([[1, 1, 1, 1]], k=5)
    B, N = to_batch([(t, 1.0, [0, 0, 0, 0])], "cpu")
    S, D, I, H, M = rows(B, torch.arange(N))
    assert D.shape[:2] == (3, 5) and M.sum(1).tolist() == [2, 2, 5]
    for i, d in enumerate(t):
        k = len(d["h"])
        assert torch.allclose(D[i, :k], torch.from_numpy(d["D"].astype(np.float32)))
        assert torch.allclose(H[i, :k], torch.from_numpy(d["h"]))
