"""The AlphaZero net as a playing agent (net-guided ISMCTS, src/az/mcts.py).

Spec (src/ai/agents.make_agent): 'az' | 'az:S<sims>' | 'az:net=<path.npz>'
| 'az:S128:style=sword' (a persona from src/az/personas.py tilts the pick)
| 'az:S128:coach' / 'az:S128:coach=1.5' (the user's advice, src/az/coach.py).
Without net= it loads the newest models_az/net_<gen>.npz when created, so a
new game picks up the latest generation of a running training job.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Optional

import numpy as np

from src.az.mcts import evaluate, legal, search
from src.az.net import AZNet

NET_DIR = "models_az"


def latest_net(out_dir: str = NET_DIR) -> Optional[str]:
    nets = [(int(m.group(1)), p) for p in glob.glob(os.path.join(out_dir, "net_*.npz"))
            if (m := re.search(r"net_(\d+)\.npz$", p))]
    return max(nets)[1] if nets else None


class AZAgent:
    def __init__(self, net_path: Optional[str] = None, sims: int = 32,
                 seed: Optional[int] = None, style: Optional[str] = None, lean: float = 1.5,
                 coach: float = 0.0):
        self.style, self.lean, self.coach = style, lean, coach
        path = net_path or latest_net()
        if path is None:
            raise FileNotFoundError(f"no AlphaZero net in {NET_DIR}/ (run src.az.train)")
        self.net_path = path
        m = re.search(r"(\d+)\.npz$", path)
        self.gen = int(m.group(1)) if m else None
        self.net = AZNet.load(path)
        self.sims = sims
        self.rng = np.random.default_rng(seed)

    def _tilt(self, gs, pid, acts, p):
        """log-probabilities plus the persona's lean (no persona: unchanged order)."""
        if not self.style and not self.coach:
            return np.asarray(p)
        out = np.log(np.asarray(p, np.float64) + 1e-3)
        if self.style:
            from src.az.personas import bias
            out = out + self.lean * bias(gs, pid, acts, self.style)
        if self.coach:
            from src.az.coach import bias as coach_bias
            out = out + self.coach * coach_bias(gs, pid, acts)
        return out

    def select_action(self, gs, pid, valid_actions):
        acts = legal(gs, pid)
        self.last_az = None
        if len(acts) == 1:
            a = acts[0]
        elif self.sims <= 0:   # az:S0 - the network's first instinct, no search
            prior, v = evaluate(self.net, gs, pid, acts)
            a = acts[int(np.argmax(self._tilt(gs, pid, acts, prior)))]
        else:   # no exploration noise: play the most-visited move
            visits = search(self.net, gs, pid, acts, self.sims, self.rng, noise=0.0)
            a = acts[int(np.argmax(self._tilt(gs, pid, acts, visits / max(visits.sum(), 1))))]
            prior, v = evaluate(self.net, gs, pid, acts)
            self.last_az = {"acts": acts, "visits": visits, "prior": prior,
                            "value": float(v[pid])}     # for the UI's "what it considered"
        key = repr(a)
        return next((v for v in valid_actions if repr(v) == key), a)
