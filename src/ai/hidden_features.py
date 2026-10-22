"""What an all-seeing observer knows that the player doesn't: the input of the
privileged critic (user, 2026-10-21: "use information that the agent
shouldn't know ... as an evaluator against the policy").

Only the PPO critic sees this vector, and only during training; the policy
never does, so play stays honest. Seats are relative to the player deciding
(next, next+1, next+2).

    opponents' hands: card bag + count / swords / persuasion
    opponents' Intrigues: card bag + count / combat / plot / endgame
    opponents' next draws (top 5 of deck): swords / persuasion
    my next draws (top 5 of my deck): card bag + swords / persuasion
    the next two Conflict cards, the next 3 Imperium cards, the next Intrigue
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

_V: Dict[str, Dict[str, int]] = {}
TOP = 5                                     # "next hand" = top 5 of a deck


def _vocab() -> Dict[str, Dict[str, int]]:
    if not _V:
        from src.ai.ppo import _IDS, _build_ids
        from src.data.card_definitions import (conflict_level_1_pool, conflict_level_2_pool,
                                               conflict_level_3_pool)
        _build_ids()
        _V["card"] = _IDS["card"]
        _V["intrigue"] = _IDS["intrigue"]
        names = sorted({c.name for pool in (conflict_level_1_pool(), conflict_level_2_pool(),
                                            conflict_level_3_pool()) for c in pool})
        _V["conflict"] = {n: i + 1 for i, n in enumerate(names)}
    return _V


def _bag(names, vocab: Dict[str, int]) -> np.ndarray:
    """Counts per identity (slot 0 = unknown name)."""
    v = np.zeros(len(vocab) + 1, np.float32)
    for n in names:
        v[vocab.get(n, 0)] += 1.0
    return v


def _timing(ic) -> set:
    ts = ic.timing if isinstance(ic.timing, (set, frozenset, tuple, list)) else (ic.timing,)
    return {getattr(t, "value", t) for t in ts}


def _hand_stats(cards) -> List[float]:
    return [len(cards), sum(getattr(c, "swords", 0) or 0 for c in cards) / 4.0,
            sum(getattr(c, "persuasion", 0) or 0 for c in cards) / 4.0]


def encode_hidden(gs, pid: int) -> np.ndarray:
    V = _vocab()
    n = len(gs.players)
    parts = []
    for j in range(1, 4):
        if j >= n:
            parts.append(np.zeros(len(V["card"]) + len(V["intrigue"]) + 2 + 3 + 4 + 2, np.float32))
            continue
        q = gs.players[(pid + j) % n]
        ints = list(q.intrigue_cards)
        tm = [_timing(ic) for ic in ints]
        parts += [_bag((c.name for c in q.hand), V["card"]), np.array(_hand_stats(q.hand), np.float32),
                  _bag((ic.name for ic in ints), V["intrigue"]),
                  np.array([len(ints), sum("combat" in t for t in tm), sum("plot" in t for t in tm),
                            sum("endgame" in t for t in tm)], np.float32),
                  np.array(_hand_stats(q.deck[-TOP:])[1:], np.float32)]
    me = gs.players[pid]
    top = me.deck[-TOP:]
    parts += [_bag((c.name for c in top), V["card"]), np.array(_hand_stats(top)[1:], np.float32)]
    nxt = list(gs.conflict_deck[:2]) if gs.conflict_deck else []
    for k in range(2):
        v = np.zeros(len(V["conflict"]) + 2, np.float32)
        if k < len(nxt):
            v[V["conflict"].get(nxt[k].name, 0)] = 1.0
            v[-1] = getattr(nxt[k], "conflict_level", 0) / 3.0
        parts.append(v)
    parts.append(_bag((c.name for c in gs.imperium_deck[:3]), V["card"]))
    parts.append(_bag((ic.name for ic in gs.intrigue_deck[:1]), V["intrigue"]))
    return np.concatenate(parts).astype(np.float32)


def hidden_dim() -> int:
    from src.data.card_definitions import setup_game
    gs = setup_game(num_players=4, seed=0, neutral_leaders=True, use_bloodlines=True)
    return int(encode_hidden(gs, 0).shape[0])
