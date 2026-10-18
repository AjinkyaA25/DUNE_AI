"""Opponent buying power from public information.

Every card a player owns was bought in the open, and their discard pile and
played cards are face up, so the cards that can be in their hand right now
(hand + deck) are known as a set; only which of them are in hand is hidden.
That gives each opponent's Reveal persuasion as a distribution: sample hands
from that pool, send the weakest cards out with the agents they still have,
reveal the rest, add persuasion already gained and the High Council's +2.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

N_SAMPLES = 64
# persuasion not in the cards' printed value (reveal effects, spaces still to
# be visited): 0.54 on average over 2,512 predictions (40 AI games); drawn as
# Poisson so it is lumpy like the real thing (a +2 space or nothing)
EXTRA = 0.55
_CACHE: Dict[tuple, np.ndarray] = {}


def reveal_persuasion_samples(gs, q: int) -> np.ndarray:
    """Samples of player q's persuasion on this round's Reveal (public info only)."""
    p = gs.players[q]
    base = gs.persuasion_pool.get(q, 0) + (2 if p.has_councilor else 0)
    pool = [c.persuasion or 0 for c in p.hand + p.deck]
    h = len(p.hand)
    k = max(0, h - p.agents_available)               # cards left to reveal
    key = (q, gs.round, base, h, k, tuple(sorted(pool)))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    rng = np.random.default_rng(len(pool) * 1009 + h * 31 + k)
    extra = rng.poisson(EXTRA, N_SAMPLES)
    if not pool or k == 0 or h == 0:
        out = float(base) + extra
    else:
        arr = np.asarray(pool, float)
        hands = np.stack([rng.choice(arr, size=min(h, len(arr)), replace=False)
                          for _ in range(N_SAMPLES)])
        hands.sort(axis=1)
        out = base + extra + hands[:, -k:].sum(axis=1)  # agents take the weakest
    if len(_CACHE) > 4096:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def p_afford(gs, q: int, cost: float) -> float:
    return float((reveal_persuasion_samples(gs, q) >= cost).mean())


def p_buys(gs, q: int, card, values: Dict[str, float]) -> float:
    """Chance q buys `card` from the row this round: q can afford it and no
    card q values more is affordable too (a rough exclusive-pick model).
    `values`: q's value for each row card by name."""
    samples = reveal_persuasion_samples(gs, q)
    v = values.get(card.name, 0.0)
    if v <= 0:
        return 0.0
    ok = samples >= card.cost
    for other in gs.imperium_row:
        if other.name != card.name and values.get(other.name, 0.0) > v:
            ok &= samples < other.cost
    return float(ok.mean())
