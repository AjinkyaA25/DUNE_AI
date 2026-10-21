"""A widened PPO net plays exactly like the smaller one it was built from."""
import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ai.ppo import PPONet
from src.selfplay.ppo_train import make_torch_net, widen


def _inputs(S, A, sizes, n=6, k=5):
    g = torch.Generator().manual_seed(0)
    s = torch.randn(n, S, generator=g)
    D = torch.randn(n, k, A, generator=g)
    I = torch.stack([torch.randint(0, m, (n, k), generator=g) for m in sizes], dim=-1)
    H = torch.randn(n, k, generator=g) * 3
    M = torch.ones(n, k, dtype=torch.bool)
    return s, D, I, H, M


def test_widen_keeps_play_and_value():
    S, A, sizes = 20, 10, [5, 7, 3]
    torch.manual_seed(1)
    small = make_torch_net(S, A, sizes, 1.0, (12, 8, 9, 4))
    for p in small.parameters():                # a2 starts at 0: give every layer real weights
        torch.nn.init.normal_(p, std=0.3)
    big = make_torch_net(S, A, sizes, 1.0, (30, 20, 25, 6))
    widen(big, small.state_dict())
    x = _inputs(S, A, sizes)
    with torch.no_grad():
        l0, v0, _ = small(*x)
        a0 = small.last_aux
        l1, v1, _ = big(*x)
        a1 = big.last_aux
    assert torch.allclose(l0, l1, atol=1e-5)
    assert torch.allclose(v0, v1, atol=1e-5)
    assert torch.allclose(a0, a1, atol=1e-5)
    # the numpy forward used in games reads the wider export the same way
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.npz")
        big.export(path)
        net = PPONet(path)
        s, D, I, H, _ = (t.numpy() for t in x)
        lg, v = net.logits(s[0], D[0], I[0], H[0])
    assert np.allclose(lg, l1[0].numpy(), atol=1e-4)
    assert abs(v - float(v1[0])) < 1e-4
