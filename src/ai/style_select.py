"""
Learned style selector: which playstyle wins from this situation.

Trained on the style-search agent's own measurements (data/style_league/
adaptive_*.jsonl): at the start of each round it played the game out with
every style and logged the mean result per style, alongside its situation
(src/selfplay/style_league.situation). A linear model per style predicts
that result from the situation, so the choice is instant - no playouts.

  python -m src.ai.style_select      # fit -> config/style_select.npz
Agent spec: styleselect  (heuristic moves with the predicted best style)
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "config", "style_select.npz")


def _rows(paths: List[str]):
    X, Y, names = [], [], None
    for p in paths:
        for line in open(p):
            g = json.loads(line)
            hist = g.get("adaptive_history") or []
            seat = g["style"].index("adaptive") if "adaptive" in g["style"] else None
            if seat is None:
                continue
            snaps = {s["round"]: s["x"][seat] for s in g["snaps"]}
            for h in hist:
                x = snaps.get(h["round"])
                if x is None:
                    continue
                names = names or sorted(h["means"])
                X.append(x)
                Y.append([h["means"][k] for k in names])
    return np.array(X), np.array(Y), names


def fit(paths: Optional[List[str]] = None, l2: float = 1.0) -> Dict:
    paths = paths or sorted(glob.glob(os.path.join(ROOT, "data", "style_league", "adaptive_*.jsonl")))
    X, Y, names = _rows(paths)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    nv = len(X) // 5
    va, tr = idx[:nv], idx[nv:]

    def ridge(A, B):
        return np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ B)

    W = ridge(Xb[tr], Y[tr])
    P = Xb[va] @ W
    # how often the predicted best style is the playouts' best (or within 0.05)
    best_true = Y[va].argmax(1)
    pick = P.argmax(1)
    exact = float((pick == best_true).mean())
    near = float((Y[va][np.arange(len(va)), pick] >= Y[va].max(1) - 0.05).mean())
    regret = float((Y[va].max(1) - Y[va][np.arange(len(va)), pick]).mean())
    base_pick = Y[tr].mean(0).argmax()       # always the on-average best style
    base_regret = float((Y[va].max(1) - Y[va][:, base_pick]).mean())
    W = ridge(Xb, Y)
    np.savez(MODEL, W=W, names=np.array(names))
    return {"rows": len(X), "styles": names, "pick_exact": exact, "pick_within_0.05": near,
            "regret": regret, "always_" + names[base_pick] + "_regret": base_regret}


class StyleSelectAgent:
    name = "styleselect"

    def __init__(self, seed: Optional[int] = None):
        from src.ai.agents import HeuristicAgent
        from src.ai.style_search import load_styles
        z = np.load(MODEL, allow_pickle=False)
        self.W, self.names = z["W"], [str(n) for n in z["names"]]
        styles = load_styles()
        self.play = {k: HeuristicAgent(seed=seed, tuning=styles[k]) for k in self.names}
        self.current, self.round = None, -1
        self.history = []

    def reset(self) -> None:
        self.current, self.round, self.history = None, -1, []

    def select_action(self, gs, pid, valid):
        from src.selfplay.style_league import situation
        if gs.round != self.round and gs.phase.value == "player_turns" \
                and gs.player_in_reveal_buy is None:
            self.round = gs.round
            pred = np.append(situation(gs, pid), 1.0) @ self.W
            self.current = self.names[int(np.argmax(pred))]
            self.history.append({"round": gs.round, "style": self.current,
                                 "means": {k: round(float(v), 3) for k, v in zip(self.names, pred)}})
        return self.play[self.current or self.names[0]].select_action(gs, pid, valid)


if __name__ == "__main__":
    print(fit())
