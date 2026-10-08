"""
State -> fixed-length feature vector for the value network.

`encode_state(gs, perspective_pid)` returns a float32 np.ndarray of length
FEATURE_DIM.  Player blocks are rotated so the perspective player is block 0,
then opponents in turn order.  Games with < 4 players are zero-padded.
All values are roughly scaled into [0, ~2].

Layout (2026-09-07 card-identity expansion):
  4 * _PLAYER_FEATS      per-player board state (rotated, perspective first)
      _GLOBAL_FEATS       round / phase / conflict / row aggregates
      _DECK_FEATS         perspective player's own card-composition multi-hot
      _ROW_FEATS          which Imperium cards are buyable right now
      _OPP_DECK_FEATS     coarse deck-quality summary for each opponent
      _BL_FEATS           Bloodlines (2026-09-28): techs, Sardaukar
                          Commanders + skills, tech market, Bloodlines cards
                          owned / in the Row. All zero in non-Bloodlines games.
      _COMM_FEATS         community cards owned / in the Row
      _STYLE_FEATS        playstyle position (2026-10-08): the face-up
                          Sardaukar skill row + the perspective player's
                          objective state (Swordmaster / High Council /
                          commanders / techs held and affordable), see
                          src/ai/playstyle.py

The Bloodlines block is appended LAST so models trained before it existed
keep working: ValueModel / PolicyModel use only the first `dim` features.

The deck / row blocks are what let the net finally prefer *specific* strong
cards (Overthrow, TSMF, ...) over filler — everything before this was pure
aggregates, so the net was structurally card-blind and purchasing was driven
entirely by the flat heuristic.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

MAX_PLAYERS = 4
_PLAYER_FEATS = 38
_GLOBAL_FEATS = 34

_FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")
_PHASES = ("round_start", "player_turns", "combat", "makers", "recall", "game_over")
_CONTROLLED = ("Arrakeen", "Spice Refinery", "Imperial Basin")

# stable leader ordering for a normalised id feature
_LEADER_IDX = {
    "Gurney Halleck": 1, "Lady Jessica": 2, "Muad'Dib": 3, "Staban Tuek": 4,
    "Feyd-Rautha Harkonnen": 5, "Lady Margot Fenring": 6, "Princess Irulan": 7,
    "Lady Amber Metulli": 8, "Shaddam Corrino IV": 9,
}

# conflict resource-reward weights (vp / control handled separately)
_CC_RES_W = {"solari": 0.4, "spice": 0.5, "water": 0.7, "troops": 0.6,
             "draw": 0.5, "intrigue": 0.9, "spy": 0.8, "spy_special": 0.9,
             "influence_emperor": 1.2, "influence_spacing_guild": 1.2,
             "influence_bene_gesserit": 1.2, "influence_fremen": 1.2}

# --- card-identity index (lazily built to avoid import-order surprises) -----
_CARD_LIST: Tuple[str, ...] = ()
_CARD_INDEX: Dict[str, int] = {}
_N_IMPERIUM = 0
_RESERVE_NAMES = ("The Spice Must Flow", "Prepare the Way")



def _ensure_card_index() -> None:
    global _CARD_LIST, _CARD_INDEX, _N_IMPERIUM
    if _CARD_INDEX:
        return
    from src.data.card_definitions import create_imperium_cards
    imp = create_imperium_cards()
    _N_IMPERIUM = len(imp)
    _CARD_LIST = tuple(c.name for c in imp) + _RESERVE_NAMES
    _CARD_INDEX = {n: i for i, n in enumerate(_CARD_LIST)}


def _deck_feats_dim() -> int:
    _ensure_card_index()
    return len(_CARD_LIST)


def _row_feats_dim() -> int:
    _ensure_card_index()
    return _N_IMPERIUM


_OPP_DECK_PER = 3
_OPP_DECK_FEATS = _OPP_DECK_PER * (MAX_PLAYERS - 1)


def _feature_dim() -> int:
    return (MAX_PLAYERS * _PLAYER_FEATS + _GLOBAL_FEATS
            + _deck_feats_dim() + _row_feats_dim() + _OPP_DECK_FEATS
            + _bl_feats_dim() + _comm_feats_dim() + _style_feats_dim())


def _style_feats_dim() -> int:
    from src.ai.playstyle import SKILLS, STYLE_STATE_DIM
    return len(SKILLS) + STYLE_STATE_DIM


def _style_block(gs, pid: int) -> np.ndarray:
    from src.ai.playstyle import SKILLS, style_state
    v = np.zeros(_style_feats_dim(), dtype=np.float32)
    bl = getattr(gs, "bl", None)
    if bl is None:
        return v
    row = getattr(bl, "skill_row", None) or []
    for k, sk in enumerate(SKILLS):          # face-up skill tiles (0-2 copies)
        v[k] = row.count(sk) / 2.0
    v[len(SKILLS):] = style_state(gs, pid)
    return v


def _comm_feats_dim() -> int:
    _ensure_bl_index()
    return 2 * len(_COMM_CARDS)


def _comm_block(gs, pid: int) -> np.ndarray:
    """Community-pool cards owned / in the Row (zeros outside Bloodlines)."""
    _ensure_bl_index()
    v = np.zeros(_comm_feats_dim(), dtype=np.float32)
    if getattr(gs, "bl", None) is None:
        return v
    ix = {n: k for k, n in enumerate(_COMM_CARDS)}
    for name in _owned_card_names(gs.players[pid]):
        if name in ix:
            v[ix[name]] = min(3.0, v[ix[name]] + 0.5)
    for c in gs.imperium_row:
        if c.name in ix:
            v[len(_COMM_CARDS) + ix[c.name]] = 1.0
    return v


# --- Bloodlines block --------------------------------------------------------
_BL_SKILLS = ("Canny", "Charismatic", "Desperate", "Driven", "Fierce", "Hardy",
              "Loyal")
_BL_PER_SEAT = 4 + len(_BL_SKILLS)
_BL_TECHS: Tuple[str, ...] = ()
_BL_CARDS: Tuple[str, ...] = ()
_COMM_CARDS: Tuple[str, ...] = ()   # community-pool cards (appended block)


def _ensure_bl_index() -> None:
    global _BL_TECHS, _BL_CARDS
    if _BL_TECHS:
        return
    from src.game.bloodlines.techs import TECHS
    from src.game.bloodlines.cards import create_bloodlines_imperium_cards
    _BL_TECHS = tuple(t.name for t in TECHS)
    _BL_CARDS = tuple(c.name for c in create_bloodlines_imperium_cards())
    from src.game.bloodlines.cards import create_community_imperium_cards
    global _COMM_CARDS
    _COMM_CARDS = tuple(c.name for c in create_community_imperium_cards())


def _bl_feats_dim() -> int:
    _ensure_bl_index()
    return (1 + MAX_PLAYERS * _BL_PER_SEAT + 2 * len(_BL_TECHS) + 6
            + 2 * len(_BL_CARDS))


def _bl_block(gs, pid: int, order: List[int]) -> np.ndarray:
    _ensure_bl_index()
    v = np.zeros(_bl_feats_dim(), dtype=np.float32)
    bl = getattr(gs, "bl", None)
    if bl is None:
        return v
    from src.game.bloodlines.rules import COMMANDER_SPACES
    i = 0
    v[i] = 1.0
    i += 1
    for seat in range(MAX_PLAYERS):
        if seat < len(order):
            p = gs.players[order[seat]]
            v[i:i + 4] = (len(p.techs) / 5.0, p.commanders_garrison / 2.0,
                          bl.commanders_in_conflict.get(p.id, 0) / 2.0,
                          p.commanders_supply / 2.0)
            for k, sk in enumerate(_BL_SKILLS):
                v[i + 4 + k] = 1.0 if sk in p.skills else 0.0
        i += _BL_PER_SEAT
    tix = {n: k for k, n in enumerate(_BL_TECHS)}
    for t in gs.players[pid].techs:                   # my techs
        if t.name in tix:
            v[i + tix[t.name]] = 1.0
    i += len(_BL_TECHS)
    for stack in bl.tech_stacks:                      # buyable right now
        if stack and stack[0].name in tix:
            v[i + tix[stack[0].name]] = 1.0
    i += len(_BL_TECHS)
    for k, sp in enumerate(COMMANDER_SPACES):         # commanders still on board
        v[i + k] = 1.0 if bl.commander_on_space.get(sp) else 0.0
    i += 6
    cix = {n: k for k, n in enumerate(_BL_CARDS)}
    for name in _owned_card_names(gs.players[pid]):
        if name in cix:
            v[i + cix[name]] = min(3.0, v[i + cix[name]] + 0.5)
    i += len(_BL_CARDS)
    for c in gs.imperium_row:
        if c.name in cix:
            v[i + cix[c.name]] = 1.0
    return v


# public constant — resolved once at import
FEATURE_DIM = _feature_dim()


def _player_block(gs, pid: int) -> List[float]:
    p = gs.players[pid]
    won = gs.won_conflicts.get(pid, [])
    controlled = sum(1 for loc in _CONTROLLED if gs.controlled_by.get(loc) == pid)
    friendships = sum(1 for v in p.faction_friendships.values() if v)
    alliances = sum(1 for v in p.alliances.values() if v)

    cc = gs.current_conflict
    icon = getattr(getattr(cc, "battle_icon", None), "value", None)
    icon_match_here = 1.0 if (icon and icon != "wild"
                              and icon in getattr(p, "battle_icons", [])) else 0.0
    one_from_alliance = sum(1 for f in _FACTIONS if p.influence[f] == 3) / 4.0
    holds_alliance = [1.0 if gs.alliance_holder.get(f) == pid else 0.0
                      for f in _FACTIONS]
    deploy_budget = getattr(p, "deploy_budget_this_turn", 0) / 6.0

    return [
        p.victory_points / 12.0,
        p.solari / 15.0,
        p.spice / 15.0,
        p.water / 10.0,
        p.agents_available / 3.0,
        p.agents_total / 3.0,
        p.troops_garrison / 12.0,
        p.troops_supply / 12.0,
        gs.troops_in_conflict.get(pid, 0) / 12.0,
        gs.sandworms_in_conflict.get(pid, 0) / 4.0,
        gs.combat_strength.get(pid, 0) / 20.0,
        p.influence["emperor"] / 6.0,
        p.influence["spacing_guild"] / 6.0,
        p.influence["bene_gesserit"] / 6.0,
        p.influence["fremen"] / 6.0,
        friendships / 4.0,
        alliances / 4.0,
        p.spies_available / 3.0,
        sum(v for v in p.spies_on_board.values()) / 3.0,
        len(p.hand) / 8.0,
        (len(p.deck) + len(p.discard) + len(p.in_play)) / 20.0,
        1.0 if p.has_swordmaster else 0.0,
        1.0 if p.has_councilor else 0.0,
        1.0 if p.has_maker_hooks else 0.0,
        controlled / 3.0,
        len(won) / 6.0,
        len(p.intrigue_cards) / 6.0,
        1.0 if pid in gs.players_revealed else 0.0,
        len(p.contracts_active) / 4.0,
        _LEADER_IDX.get(getattr(p.leader, "name", None), 0) / 9.0,
        gs.persuasion_pool.get(pid, 0) / 15.0,
        # --- 2026-09-07 combat / influence "when it matters" context ---
        holds_alliance[0], holds_alliance[1], holds_alliance[2], holds_alliance[3],
        one_from_alliance,
        icon_match_here,
        deploy_budget,
    ]


def _global_block(gs, pid: int) -> List[float]:
    from src.game.gameState import MAX_ROUNDS
    from src.game.board.board import SHIELD_WALL_PROTECTED

    feats: List[float] = [gs.round / float(MAX_ROUNDS)]
    ph = gs.phase.value
    feats += [1.0 if ph == x else 0.0 for x in _PHASES]         # 6

    cc = gs.current_conflict
    r = cc.first_place_reward if cc else {}
    cc_res_val = sum(_CC_RES_W.get(k, 0.3) * v for k, v in r.items()
                     if isinstance(v, (int, float)) and k not in ("vp",))
    rival_worm = 1.0 if cc and any(
        gs.sandworms_in_conflict.get(q, 0) > 0
        for q in range(gs.num_players) if q != pid) else 0.0
    rival_troops = max((gs.troops_in_conflict.get(q, 0)
                        for q in range(gs.num_players) if q != pid), default=0)
    feats += [
        (cc.conflict_level / 3.0) if cc else 0.0,
        1.0 if (cc and "vp" in r) else 0.0,
        1.0 if (cc and getattr(cc, "location", None)) else 0.0,
        1.0 if (cc and getattr(cc, "battle_icon", None)) else 0.0,
        # reward magnitude / contestedness, not just "is there a vp"
        float(r.get("vp", 0)) / 3.0,
        1.0 if r.get("control") else 0.0,
        cc_res_val / 8.0,
        1.0 if (cc and getattr(cc, "location", None) in SHIELD_WALL_PROTECTED) else 0.0,
        rival_worm,
        rival_troops / 12.0,
        1.0 if gs.round >= 6 else 0.0,
    ]
    feats += [
        1.0 if gs.shield_wall_intact else 0.0,
        gs.maker_bonus_spice.get("Imperial Basin", 0) / 5.0,
        gs.maker_bonus_spice.get("Hagga Basin", 0) / 5.0,
        gs.maker_bonus_spice.get("Deep Desert", 0) / 5.0,
    ]
    row = gs.imperium_row
    avg_cost = (sum(c.cost for c in row) / len(row)) if row else 0.0
    max_pers = max((c.persuasion for c in row), default=0.0)
    max_swords = max((c.swords for c in row), default=0.0)
    n_faction_access = sum(1 for c in row if getattr(c, "access_symbols", ()))
    feats += [
        len(row) / 5.0,
        avg_cost / 8.0,
        len(gs.imperium_deck) / 20.0,
        len(gs.reserve_spice_must_flow) / 5.0,
        max_pers / 8.0,
        max_swords / 6.0,
        n_faction_access / 5.0,
        1.0 if gs.first_player == pid else 0.0,
        len(gs.conflict_deck) / 10.0,
        len(gs.contracts_on_board) / 2.0,
        1.0 if gs.game_over else 0.0,
    ]
    # two retired slots (were tier-list deck/Row quality); kept as zeros so
    # saved models keep their input width
    feats += [0.0, 0.0]
    while len(feats) < _GLOBAL_FEATS:
        feats.append(0.0)
    return feats[:_GLOBAL_FEATS]


def _owned_card_names(p) -> List[str]:
    out: List[str] = []
    for zone in (p.deck, p.hand, p.discard, p.in_play):
        out += [c.name for c in zone]
    return out


def _deck_block(gs, pid: int) -> np.ndarray:
    """Perspective player's own card composition — you always know what's in
    your deck even if not the order, so this is legal information."""
    _ensure_card_index()
    v = np.zeros(len(_CARD_LIST), dtype=np.float32)
    for name in _owned_card_names(gs.players[pid]):
        idx = _CARD_INDEX.get(name)
        if idx is not None:
            v[idx] += 1.0
    np.divide(v, 2.0, out=v)
    return np.clip(v, 0.0, 3.0)


def _row_block(gs) -> np.ndarray:
    _ensure_card_index()
    v = np.zeros(_N_IMPERIUM, dtype=np.float32)
    for c in gs.imperium_row:
        idx = _CARD_INDEX.get(c.name)
        if idx is not None and idx < _N_IMPERIUM:
            v[idx] = 1.0
    return v


def _opp_deck_block(gs, order: List[int]) -> List[float]:
    """Coarse deck-quality read on each opponent from PUBLIC zones only
    (discard + in_play): how many cards they've acquired, and the raw
    persuasion / swords sitting in those visible cards."""
    feats: List[float] = []
    for seat in range(1, MAX_PLAYERS):
        if seat >= len(order):
            feats += [0.0] * _OPP_DECK_PER
            continue
        p = gs.players[order[seat]]
        visible = list(p.discard) + list(p.in_play)
        total = len(p.deck) + len(p.hand) + len(p.discard) + len(p.in_play)
        feats += [
            max(0, total - 10) / 15.0,
            sum(getattr(c, "persuasion", 0) for c in visible) / 10.0,
            sum(getattr(c, "swords", 0) for c in visible) / 6.0,
        ]
    return feats


def encode_state(gs, perspective_pid: int) -> np.ndarray:
    n = gs.num_players
    order = [(perspective_pid + i) % n for i in range(n)]
    vec: List[float] = []
    for seat in range(MAX_PLAYERS):
        if seat < n:
            vec += _player_block(gs, order[seat])
        else:
            vec += [0.0] * _PLAYER_FEATS
    vec += _global_block(gs, perspective_pid)

    arr = np.concatenate([
        np.asarray(vec, dtype=np.float32),
        _deck_block(gs, perspective_pid),
        _row_block(gs),
        np.asarray(_opp_deck_block(gs, order), dtype=np.float32),
        _bl_block(gs, perspective_pid, order),
        _comm_block(gs, perspective_pid),
        _style_block(gs, perspective_pid),
    ])
    assert arr.shape[0] == FEATURE_DIM, (arr.shape, FEATURE_DIM)
    return arr
