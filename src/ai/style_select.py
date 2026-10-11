"""
Learned style selector: which playstyle wins from this situation.

Trained on the style-search agent's own measurements (data/style_league/
adaptive_*.jsonl): at the start of each round it played the game out with
every style and logged the mean result per style, alongside its situation
(src/selfplay/style_league.situation). A linear model per style predicts
that result from the situation, so the choice is instant - no playouts.

  python -m src.ai.style_select      # fit -> config/style_select.npz
  python -m src.ai.style_select --labels [data/style_league/labels.jsonl]
      # fit on the low-noise labels (src/selfplay/style_labels.py): ridge on
      # situation / situation_rich / encode_state, chosen by held-out regret
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


def situation_rich(gs, pid: int) -> list:
    """situation() plus what the opponents hold, the race, the current
    conflict and the markets (Imperium row, tech stacks, skill row)."""
    from src.ai.playstyle import owned
    from src.selfplay.style_league import situation
    x = situation(gs, pid)
    p = gs.players[pid]
    opps = [q for q in gs.players if q.id != pid]
    oo = [owned(gs, q.id) for q in opps]
    bl = getattr(gs, "bl", None)
    c = gs.current_conflict
    row = [getattr(card, "cost", 0) or 0 for card in gs.imperium_row]
    tech_prices = []
    if bl is not None:
        tech_prices = [bl.tech_price(pid, st[0]) for st in bl.tech_stacks
                       if st and not bl.has(pid, st[0].name)]
    vps = [q.victory_points for q in opps]
    x += [
        sum(o["high_council"] for o in oo) / 3.0,
        sum(min(o["commanders"], 4) for o in oo) / 12.0,
        sum(min(o["techs"], 5) for o in oo) / 15.0,
        max(min(o["techs"], 5) for o in oo) / 5.0,
        sum(1 for v in vps if v > p.victory_points) / 3.0,
        (max(vps) - p.victory_points) / 5.0,
        (c.conflict_level if c is not None else 0) / 3.0,
        gs.troops_in_conflict.get(pid, 0) / 10.0,
        max(gs.troops_in_conflict.get(q.id, 0) for q in opps) / 10.0,
        float(np.mean(row)) / 8.0 if row else 0.0,
        max(row) / 8.0 if row else 0.0,
        float(np.mean(tech_prices)) / 8.0 if tech_prices else 1.0,
        len(p.intrigue_cards) / 5.0,
        p.water / 5.0,
        float(gs.first_player == pid),
    ]
    return x


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


FEATS = {"x": "situation", "xr": "situation_rich", "enc": "encode_state"}


def _features(gs, pid: int, feats: str) -> np.ndarray:
    if feats == "enc":
        from src.ai.features import encode_state
        return np.asarray(encode_state(gs, pid), dtype=float)
    if feats == "xr":
        return np.asarray(situation_rich(gs, pid), dtype=float)
    from src.selfplay.style_league import situation
    return np.asarray(situation(gs, pid), dtype=float)


def fit_labels(path: Optional[str] = None, save: bool = True) -> Dict:
    """Fit on the low-noise labels (src/selfplay/style_labels.py): ridge per
    feature set and penalty, chosen by held-out regret (5 folds, split by
    game so one game's situations never sit on both sides)."""
    path = path or os.path.join(ROOT, "data", "style_league", "labels.jsonl")
    rows = [json.loads(l) for l in open(path)]
    names = sorted(rows[0]["means"])
    Y = np.array([[r["means"][k] for k in names] for r in rows])
    games = np.array([r["game"] for r in rows])
    fold = games % 5
    out = {"rows": len(rows), "styles": names}
    best_style = Y.mean(0).argmax()
    out["oracle_mean"] = round(float(Y.max(1).mean()), 4)
    out["always_" + names[best_style]] = round(float(Y[:, best_style].mean()), 4)
    out["always_regret"] = round(float((Y.max(1) - Y[:, best_style]).mean()), 4)
    # how often each style is the best one, by round
    rnd = np.array([r["round"] for r in rows])
    out["best_share"] = {k: round(float((Y.argmax(1) == i).mean()), 3) for i, k in enumerate(names)}
    out["best_by_round"] = {int(t): {k: round(float((Y[rnd == t].argmax(1) == i).mean()), 2)
                                     for i, k in enumerate(names)} for t in sorted(set(rnd))}
    results = []
    for feats in ("x", "xr", "enc"):
        X = np.array([r[feats] for r in rows], dtype=float)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xb = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
        for l2 in (1.0, 10.0, 100.0, 1000.0):
            reg, got = [], []
            for f in range(5):
                tr, va = fold != f, fold == f
                A = Xb[tr]
                W = np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ Y[tr])
                pick = (Xb[va] @ W).argmax(1)
                v = Y[va][np.arange(va.sum()), pick]
                got.append(v)
                reg.append(Y[va].max(1) - v)
            results.append((float(np.concatenate(reg).mean()), feats, l2,
                            float(np.concatenate(got).mean())))
    results.sort()
    out["cv"] = [{"feats": f, "l2": l, "regret": round(r, 4), "mean": round(g, 4)}
                 for r, f, l, g in results[:6]]
    _, feats, l2, _ = results[0]
    X = np.array([r[feats] for r in rows], dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xb = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    W = np.linalg.solve(Xb.T @ Xb + l2 * np.eye(Xb.shape[1]), Xb.T @ Y)
    if save:
        np.savez(MODEL, W=W, names=np.array(names), feats=np.array(feats), mu=mu, sd=sd)
    out["chosen"] = {"feats": feats, "l2": l2}
    return out


class StyleSelectAgent:
    name = "styleselect"

    def __init__(self, seed: Optional[int] = None, model: Optional[str] = None):
        from src.ai.agents import HeuristicAgent
        from src.ai.style_search import load_styles
        # model "random": a uniformly random style each round (control for
        # what switching alone is worth)
        self.random = np.random.default_rng(seed) if model == "random" else None
        z = np.load(MODEL if model in (None, "random") else model, allow_pickle=False)
        self.W, self.names = z["W"], [str(n) for n in z["names"]]
        # older models: raw situation(), no standardisation
        self.feats = str(z["feats"]) if "feats" in z.files else "x"
        self.mu = z["mu"] if "mu" in z.files else 0.0
        self.sd = z["sd"] if "sd" in z.files else 1.0
        styles = load_styles()
        self.play = {k: HeuristicAgent(seed=seed, tuning=styles[k]) for k in self.names}
        self.current, self.round = None, -1
        self.history = []

    def reset(self) -> None:
        self.current, self.round, self.history = None, -1, []

    def select_action(self, gs, pid, valid):
        if gs.round != self.round and gs.phase.value == "player_turns" \
                and gs.player_in_reveal_buy is None:
            self.round = gs.round
            x = (_features(gs, pid, self.feats) - self.mu) / self.sd
            pred = np.append(x, 1.0) @ self.W
            if self.random is not None:
                pred = self.random.random(len(self.names))
            self.current = self.names[int(np.argmax(pred))]
            self.history.append({"round": gs.round, "style": self.current,
                                 "means": {k: round(float(v), 3) for k, v in zip(self.names, pred)}})
        return self.play[self.current or self.names[0]].select_action(gs, pid, valid)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--labels":
        print(json.dumps(fit_labels(sys.argv[2] if len(sys.argv) > 2 else None), indent=1))
    else:
        print(fit())
