"""
Bloodlines playstyles: which objective a player is building toward.

Players in the video games spend their solari / spice on four objectives:

  swordmaster   3rd agent (Swordmaster space; first buyer pays more)
  high_council  council seat (5 solari; every tech -1 spice afterwards)
  commanders    Sardaukar commanders (2 solari; a board recruit = a skill)
  techs         tech tiles (spice, on green spaces)

Early on players commit to one or two of them; by rounds 6-7 most hold some
of everything (reports/playstyles.json, video_scrape/playstyles.py).

`style_state(gs, pid)` is a compact description of one player's position
with respect to those objectives (what they hold, what they can afford,
what the market offers). `ObjectiveModel` is a multinomial logistic model,
fitted on the video games, of WHICH objective a human pursues next given
that position (winners weighted up). The heuristic uses its probabilities
to lean toward the objective that fits its situation, so four AIs at one
table can end up on four different playstyles, and an early Swordmaster
frees solari for commanders / the council the same way it does for humans.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

OBJECTIVES = ("swordmaster", "high_council", "commanders", "techs", "none")
SKILLS = ("Hardy", "Driven", "Charismatic", "Canny", "Fierce", "Loyal", "Desperate")
SWORD_SKILLS = ("Canny", "Fierce", "Loyal")
HC_COST = 5
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "config", "objective_model.npz")


def owned(gs, pid: int) -> Dict[str, int]:
    """What a player holds of each objective (for labelling progress)."""
    p = gs.players[pid]
    bl = getattr(gs, "bl", None)
    cmd = 0
    if bl is not None:
        cmd = (getattr(p, "commanders_garrison", 0) + getattr(p, "commanders_supply", 0)
               + bl.commanders_in_conflict.get(pid, 0))
    return {"swordmaster": int(bool(p.has_swordmaster)),
            "high_council": int(bool(p.has_councilor)),
            "commanders": cmd,
            "techs": len(getattr(p, "techs", []) or [])}


def style_state(gs, pid: int) -> np.ndarray:
    """Position features for the objective model and the value/policy nets."""
    from src.game.gameState import MAX_ROUNDS
    p = gs.players[pid]
    bl = getattr(gs, "bl", None)
    o = owned(gs, pid)
    sm_cost = gs._swordmaster_cost()
    skills = getattr(p, "skills", set()) or set()
    row_new = on_board = affordable_techs = 0
    cheapest_tech = 9
    if bl is not None:
        row_new = len(bl.row_skills(pid))
        on_board = sum(1 for v in bl.commander_on_space.values() if v)
        for st in bl.tech_stacks:
            if st and not bl.has(pid, st[0].name):
                price = bl.tech_price(pid, st[0])
                cheapest_tech = min(cheapest_tech, price)
                affordable_techs += int(p.spice >= price)
    opp_sm = sum(1 for q in gs.players if q.id != pid and q.has_swordmaster)
    return np.array([
        gs.round / float(MAX_ROUNDS),
        p.solari / 10.0,
        p.spice / 10.0,
        o["swordmaster"],
        o["high_council"],
        min(o["commanders"], 4) / 4.0,
        min(o["techs"], 5) / 5.0,
        len(skills) / 4.0,
        sum(1 for s in SWORD_SKILLS if s in skills) / 3.0,
        on_board / 6.0,
        row_new / 4.0,
        affordable_techs / 3.0,
        min(cheapest_tech, 8) / 8.0,
        float(p.solari >= sm_cost and not p.has_swordmaster),
        float(p.solari >= HC_COST and not p.has_councilor),
        opp_sm / 3.0,
        p.troops_garrison / 10.0,
        p.victory_points / 10.0,
    ], dtype=np.float32)


STYLE_STATE_DIM = 18


class ObjectiveModel:
    """softmax(W . [style_state, 1]) over OBJECTIVES."""

    def __init__(self, W: Optional[np.ndarray] = None):
        self.W = W if W is not None else np.zeros((STYLE_STATE_DIM + 1, len(OBJECTIVES)))

    def probs(self, x: np.ndarray) -> np.ndarray:
        z = np.append(x, 1.0) @ self.W
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
            l2: float = 1e-3, iters: int = 3000, lr: float = 0.5) -> dict:
        Xb = np.hstack([X, np.ones((len(X), 1))])
        Y = np.eye(len(OBJECTIVES))[y]
        w = w / w.sum()
        for _ in range(iters):
            Z = Xb @ self.W
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            g = Xb.T @ ((P - Y) * w[:, None]) + l2 * self.W
            self.W -= lr * g
        P = np.array([self.probs(x) for x in X])
        nll = float(-(w * np.log(P[np.arange(len(y)), y] + 1e-9)).sum())
        acc = float((w * (P.argmax(1) == y)).sum())
        return {"nll": nll, "acc": acc}

    def save(self, path: str = MODEL_PATH) -> None:
        np.savez(path, W=self.W, objectives=np.array(OBJECTIVES))

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> Optional["ObjectiveModel"]:
        if not os.path.exists(path):
            return None
        z = np.load(path, allow_pickle=False)
        if z["W"].shape != (STYLE_STATE_DIM + 1, len(OBJECTIVES)):
            return None
        return cls(z["W"])


_MODEL: Optional[ObjectiveModel] = None
_LOADED = False


def objective_probs(gs, pid: int) -> Optional[Dict[str, float]]:
    """P(next objective | position) from the fitted model (None if absent)."""
    global _MODEL, _LOADED
    if not _LOADED:
        _MODEL, _LOADED = ObjectiveModel.load(), True
    if _MODEL is None:
        return None
    return dict(zip(OBJECTIVES, _MODEL.probs(style_state(gs, pid))))


def action_objective(gs, pid: int, a) -> Optional[str]:
    """Which objective an action buys outright (None = not a purchase)."""
    from src.game.gameState import ActionType
    p = gs.players[pid]
    at = a.action_type
    if at == ActionType.AGENT_TURN:
        if a.space_name == "Swordmaster" and not p.has_swordmaster:
            return "swordmaster"
        if a.space_name == "High Council" and not p.has_councilor:
            return "high_council"
        return None
    if at == ActionType.RESOLVE_BL_CHOICE and a.choice:
        if a.choice.startswith("board:") or a.choice == "supply":
            return "commanders"
        if a.choice.startswith("stack:"):
            bl = getattr(gs, "bl", None)
            c = next((c for c in bl.pending if c.player_id == pid), None) if bl else None
            if c is not None and c.kind in ("tech_buy", "tech_offer"):
                return "techs"
    return None
