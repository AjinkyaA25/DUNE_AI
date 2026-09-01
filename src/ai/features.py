"""
State -> fixed-length feature vector for the value network.

`encode_state(gs, perspective_pid)` returns a float32 np.ndarray of length
FEATURE_DIM.  Player blocks are rotated so the perspective player is block 0,
then opponents in turn order.  Games with < 4 players are zero-padded.
All values are roughly scaled into [0, ~1.5].
"""
from __future__ import annotations

from typing import List

import numpy as np

MAX_PLAYERS = 4
_PLAYER_FEATS = 30
_GLOBAL_FEATS = 24
FEATURE_DIM = MAX_PLAYERS * _PLAYER_FEATS + _GLOBAL_FEATS

_FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")
_PHASES = ("round_start", "player_turns", "combat", "makers", "recall", "game_over")
_CONTROLLED = ("Arrakeen", "Spice Refinery", "Imperial Basin")

# stable leader ordering for a normalised id feature
_LEADER_IDX = {
    "Gurney Halleck": 1, "Lady Jessica": 2, "Muad'Dib": 3, "Staban Tuek": 4,
    "Feyd-Rautha Harkonnen": 5, "Lady Margot Fenring": 6, "Princess Irulan": 7,
    "Lady Amber Metulli": 8, "Shaddam Corrino IV": 9,
}


def _player_block(gs, pid: int) -> List[float]:
    p = gs.players[pid]
    won = gs.won_conflicts.get(pid, [])
    controlled = sum(1 for loc in _CONTROLLED if gs.controlled_by.get(loc) == pid)
    friendships = sum(1 for v in p.faction_friendships.values() if v)
    alliances = sum(1 for v in p.alliances.values() if v)
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
    ]


def _global_block(gs, pid: int) -> List[float]:
    from src.game.gameState import MAX_ROUNDS
    feats: List[float] = [gs.round / float(MAX_ROUNDS)]
    ph = gs.phase.value
    feats += [1.0 if ph == x else 0.0 for x in _PHASES]         # 6

    cc = gs.current_conflict
    feats += [
        (cc.conflict_level / 3.0) if cc else 0.0,
        1.0 if (cc and "vp" in {**cc.first_place_reward}) else 0.0,
        1.0 if (cc and getattr(cc, "location", None)) else 0.0,
        1.0 if (cc and getattr(cc, "battle_icon", None)) else 0.0,
    ]
    feats += [
        1.0 if gs.shield_wall_intact else 0.0,
        gs.maker_bonus_spice.get("Imperial Basin", 0) / 5.0,
        gs.maker_bonus_spice.get("Hagga Basin", 0) / 5.0,
        gs.maker_bonus_spice.get("Deep Desert", 0) / 5.0,
    ]
    row = gs.imperium_row
    avg_cost = (sum(c.cost for c in row) / len(row)) if row else 0.0
    feats += [
        len(row) / 5.0,
        avg_cost / 8.0,
        len(gs.imperium_deck) / 20.0,
        len(gs.reserve_spice_must_flow) / 5.0,
        gs.persuasion_pool.get(pid, 0) / 15.0,
        1.0 if gs.first_player == pid else 0.0,
        len(gs.conflict_deck) / 10.0,
        len(gs.contracts_on_board) / 2.0,
        1.0 if gs.game_over else 0.0,
    ]
    # pad to _GLOBAL_FEATS
    while len(feats) < _GLOBAL_FEATS:
        feats.append(0.0)
    return feats[:_GLOBAL_FEATS]


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
    arr = np.asarray(vec, dtype=np.float32)
    assert arr.shape[0] == FEATURE_DIM, (arr.shape, FEATURE_DIM)
    return arr
