"""Learned Conflict forecast (user, 2026-10-18: the AI over-deploys and
deploys into fights it cannot place in).

The hand-made rival model in agents._opp_fight assumed a rival sends 45% of
the troops it could and ~2 Reveal swords; measured over 300 games it was
right in rounds 1-3 but by round 9 forecast 9 strength for rivals who ended
on 14 (src/selfplay/deploy_report.py). These models are fitted on recorded
games instead (src/selfplay/fight_data.py):

  enter   P(a rival with no units in the Conflict ends up in it)      logistic
  rival   a rival's final strength - visible strength, if in it       linear
          + its typical error (the spread)                            linear on |residual|
          (each fitted twice: rivals who have / have not revealed - the
          agents use the learned one only before the rival's Reveal)
  own     my final strength from swords (Reveal, Intrigues, skills)
          beyond what is visible now - I see my own hand              linear

Opponents use public information only. Plain NumPy, coefficients in
models_fight/forecast.json.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models_fight", "forecast.json")
_MODEL: Dict = {"m": None, "loaded": False}

PUBLIC = ("bias", "round", "level", "level3", "first_vp", "visible", "units", "in_conflict",
          "agents", "agents_sq", "garrison", "garrison_x_agents", "hand", "intrigues",
          "revealed", "hooks", "commanders", "rivals_visible_max", "rivals_agents")
PRIVATE = ("hand_swords", "combat_intrigue_swords")


def _visible(gs, q) -> float:
    return (2 * gs.troops_in_conflict.get(q, 0) + 3 * gs.sandworms_in_conflict.get(q, 0)
            + gs.swords_this_reveal.get(q, 0))


def public_features(gs, q) -> Dict[str, float]:
    p = gs.players[q]
    cc = gs.current_conflict
    units = gs.troops_in_conflict.get(q, 0) + gs.sandworms_in_conflict.get(q, 0)
    agents = max(0, p.agents_available)
    garrison = p.troops_garrison
    others = [r for r in range(gs.num_players) if r != q]
    lvl = getattr(cc, "conflict_level", 1) if cc is not None else 1
    first = getattr(cc, "first_place_reward", {}) or {} if cc is not None else {}
    return {
        "bias": 1.0, "round": gs.round, "level": lvl, "level3": float(lvl == 3),
        "first_vp": float(first.get("vp", 0) or 0), "visible": _visible(gs, q), "units": units,
        "in_conflict": float(units > 0), "agents": agents, "agents_sq": agents * agents,
        "garrison": garrison, "garrison_x_agents": min(garrison, 6) * agents,
        "hand": len(p.hand), "intrigues": min(len(p.intrigue_cards), 5),
        "revealed": float(q in gs.players_revealed),
        "hooks": float(bool(getattr(p, "has_maker_hooks", False))),
        "commanders": getattr(p, "commanders_garrison", 0) or 0,
        "rivals_visible_max": max((_visible(gs, r) for r in others), default=0.0),
        "rivals_agents": sum(max(0, gs.players[r].agents_available) for r in others),
    }


def private_features(gs, q) -> Dict[str, float]:
    from src.ai.agents import _own_combat_intrigue_swords
    p = gs.players[q]
    f = public_features(gs, q)
    f["hand_swords"] = float(sum(getattr(c, "swords", 0) or 0 for c in p.hand))
    f["combat_intrigue_swords"] = float(_own_combat_intrigue_swords(gs, q))
    return f


def vec(f: Dict[str, float], names) -> np.ndarray:
    return np.array([float(f.get(n, 0.0)) for n in names])


def model() -> Optional[Dict]:
    if not _MODEL["loaded"]:
        _MODEL["loaded"] = True
        try:
            with open(MODEL_PATH) as fh:
                m = json.load(fh)
            _MODEL["m"] = {k: (np.array(v) if isinstance(v, list) else v) for k, v in m.items()}
        except (OSError, ValueError):
            _MODEL["m"] = None
    return _MODEL["m"]


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def rival(gs, q) -> Optional[Tuple[float, float, float]]:
    """(P(in the fight at the end), expected final strength if in, spread),
    from the model for rivals who have / have not revealed yet."""
    m = model()
    if m is None:
        return None
    f = public_features(gs, q)
    tag = "rev" if f["revealed"] else "unrev"
    if "rival_w_" + tag not in m:
        return None
    x = vec(f, PUBLIC)
    p_in = 1.0 if f["units"] > 0 else _sig(float(x @ m["enter_w_" + tag]))
    add = max(0.0, float(x @ m["rival_w_" + tag]))
    sd = max(1.5, float(x @ m["rival_sd_w_" + tag]))
    return p_in, f["visible"] + add, sd


def own_extra(gs, pid) -> Optional[float]:
    """Strength from swords I can still expect beyond what is visible now."""
    m = model()
    if m is None:
        return None
    x = vec(private_features(gs, pid), PUBLIC + PRIVATE)
    return max(0.0, float(x @ m["own_w"]))
