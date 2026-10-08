"""
Per-action feature vectors for the policy head.

`encode_action(gs, pid, action, h_norm)` returns a small fixed-length vector
describing ONE candidate action in the current state. The policy model scores
every legal action with `f([state_features ; action_features]) -> logit` and
softmaxes over the legal set, so this only needs to capture what distinguishes
the candidates from each other (which space / which card / how many troops),
not re-encode the whole board — that's the state vector's job.

`h_norm` is the flat heuristic's score for this action, min-max normalised to
[0,1] across the current candidate set, passed in by the caller. Feeding it in
lets the policy start from "play like the tuned heuristic" and learn the
deviations that actually correlate with winning.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.game.gameState import GameState, GameAction, ActionType

_TYPES = (
    ActionType.AGENT_TURN, ActionType.REVEAL_TURN, ActionType.ACQUIRE_CARD,
    ActionType.ACQUIRE_RESERVE, ActionType.END_REVEAL, ActionType.PLAY_INTRIGUE,
    ActionType.RESOLVE_DEPLOY, ActionType.RESOLVE_TRASH,
    ActionType.RESOLVE_INFLUENCE, ActionType.RESOLVE_CONTRACT,
)  # everything else -> the trailing "other" slot

_FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")


def encode_action(gs: GameState, pid: int, a: GameAction,
                  h_norm: float = 0.0) -> np.ndarray:
    from src.game.board.board import (UPRISING_BOARD, COMBAT_SPACES,
                                      MAKER_SPACES, SPACE_MANDATORY_COSTS)
    p = gs.players[pid]
    at = a.action_type
    f: list = []

    # -- action-type one-hot (+ "other") -------------------------------------
    f += [1.0 if at == t else 0.0 for t in _TYPES]
    f.append(1.0 if at not in _TYPES else 0.0)

    # -- AGENT_TURN detail --------------------------------------------------
    ag = at == ActionType.AGENT_TURN
    fac_of_space = gs._faction_for_space(a.space_name) if ag and a.space_name else None
    cur_inf = p.influence.get(fac_of_space, 0) if fac_of_space else 0
    solari_here = 0.0
    if ag and a.space_name:
        for e in gs.get_space_effects_preview(a.space_name, a.space_option):
            v = e.get("solari", 0)
            if isinstance(v, (int, float)):
                solari_here += max(0.0, v)
    f += [
        1.0 if fac_of_space else 0.0,
        1.0 if (ag and a.space_name in COMBAT_SPACES and gs.current_conflict is not None) else 0.0,
        1.0 if (ag and a.space_name == "Swordmaster" and not p.has_swordmaster) else 0.0,
        1.0 if (ag and a.space_name == "High Council" and not p.has_councilor) else 0.0,
        min(1.0, solari_here / 4.0),
        1.0 if (fac_of_space and cur_inf == 1) else 0.0,     # this play crosses to friendship
        1.0 if (fac_of_space and cur_inf == 3) else 0.0,     # crosses to alliance
        1.0 if (ag and a.use_gather_intelligence) else 0.0,
        1.0 if (ag and a.use_infiltrate) else 0.0,
        1.0 if (ag and a.space_name in MAKER_SPACES) else 0.0,
    ]

    # -- ACQUIRE detail ---------------------------------------------------
    card = None
    if at == ActionType.ACQUIRE_CARD:
        card = next((c for c in gs.imperium_row
                     if c.name == a.acquire_card_name), None)
        if card is None and p.reserved_card is not None \
                and p.reserved_card.name == a.acquire_card_name:
            card = p.reserved_card
    elif at == ActionType.ACQUIRE_RESERVE:
        stack = (gs.reserve_spice_must_flow if a.reserve_type == "spice_must_flow"
                 else gs.reserve_prepare_the_way)
        card = stack[-1] if stack else None
    is_ptw = bool(card) and card.name == "Prepare the Way"
    ptw_owned = 0
    if is_ptw:
        ptw_owned = sum(1 for z in (p.deck, p.discard, p.hand, p.in_play)
                        for c in z if c.name == "Prepare the Way")
    has_fac_access = bool(card) and any(
        getattr(s, "value", s) in _FACTIONS
        for s in getattr(card, "access_symbols", ()))
    f += [
        0.0,                     # retired slot (was the tier-list grade)
        (card.persuasion / 6.0) if card else 0.0,
        (card.cost / 8.0) if card else 0.0,
        1.0 if has_fac_access else 0.0,
        1.0 if (bool(card) and card.name == "The Spice Must Flow") else 0.0,
        1.0 if is_ptw else 0.0,
        min(1.0, ptw_owned / 4.0),
    ]

    # -- deploy / influence / round context ------------------------------
    infl_fac = a.influence_faction if at == ActionType.RESOLVE_INFLUENCE else None
    infl_cur = p.influence.get(infl_fac, 0) if infl_fac else 0
    f += [
        min(1.0, a.deploy_count / 8.0) if at == ActionType.RESOLVE_DEPLOY else 0.0,
        1.0 if (infl_fac and infl_cur in (1, 3)) else 0.0,   # influence pick crosses a threshold
        float(np.clip(h_norm, 0.0, 1.0)),
        gs.round / 10.0,
    ]

    f += _bloodlines_action(gs, pid, a)
    return np.asarray(f, dtype=np.float32)


def _bloodlines_action(gs: GameState, pid: int, a: GameAction) -> list:
    """Which Bloodlines purchase a choice makes (2026-10-08): recruit from the
    board (with WHICH skill) or the supply, buy WHICH tech at what price, or
    decline. Lets the policy learn skill and tech preferences."""
    from src.ai.playstyle import SKILLS
    from src.game.bloodlines.techs import TECHS
    v = [0.0] * _BL_ACTION_DIM
    if a.action_type != ActionType.RESOLVE_BL_CHOICE or not a.choice:
        return v
    bl = getattr(gs, "bl", None)
    ch = a.choice
    if ch.startswith("board:"):
        v[0] = 1.0
        sk = ch.split(":", 1)[1]
    elif ch.startswith("skill:"):           # Plasteel Blades' bonus skill
        sk = ch.split(":", 1)[1]
    else:
        sk = None
    if ch == "supply":
        v[1] = 1.0
    if ch == "decline":
        v[3] = 1.0
    if sk in SKILLS:
        v[5 + SKILLS.index(sk)] = 1.0
    if ch.startswith("stack:") and bl is not None:
        c = next((c for c in bl.pending if c.player_id == pid), None)
        i = int(ch.split(":")[1])
        if c is not None and c.kind in ("tech_buy", "tech_offer") \
                and i < len(bl.tech_stacks) and bl.tech_stacks[i]:
            d = bl.tech_stacks[i][0]
            v[2] = 1.0
            v[4] = bl.tech_price(pid, d, c.discount) / 6.0
            names = [t.name for t in TECHS]
            if d.name in names:
                v[5 + len(SKILLS) + names.index(d.name)] = 1.0
    return v


def _bl_action_dim() -> int:
    from src.ai.playstyle import SKILLS
    from src.game.bloodlines.techs import TECHS
    return 5 + len(SKILLS) + len(TECHS)


_BL_ACTION_DIM = _bl_action_dim()

#  (len(_TYPES)+1) type one-hot  +  10 agent-turn  +  7 acquire  +  4 misc
#  + Bloodlines purchase block (recruit / skill / tech)
ACTION_FEATURE_DIM = (len(_TYPES) + 1) + 10 + 7 + 4 + _BL_ACTION_DIM
