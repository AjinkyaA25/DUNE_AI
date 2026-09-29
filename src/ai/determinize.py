"""
Determinization for Information-Set MCTS.

`GameState.clone()` is a full, omniscient deep copy -- it carries the exact
order of every hidden deck and the exact contents of every opponent's hand.
Searching directly on that clone would let the search "see" hidden
information no real player has (opponents' hands, what's about to be drawn).

`determinize(gs, pid, rng)` returns a clone where everything *observable* to
`pid` is preserved exactly, and everything hidden from `pid` is re-randomized
into one consistent possible world:

  - Each opponent's hand + personal deck (their content is not itself hidden
    -- every card they own was bought publicly from the Imperium Row -- only
    which of those owned cards are currently in hand vs. still in their deck,
    and in what draw order, is hidden) is pooled and reshuffled, preserving
    the opponent's true hand size and deck size.
  - Intrigue cards are drawn from a shared deck and are the one genuinely
    hidden *identity*: the shared intrigue deck plus every opponent's hidden
    intrigue hand (pid's own intrigue hand is exactly known and left alone;
    intrigue_discard is face-up/public and also left alone) is pooled and
    reshuffled, preserving each opponent's true intrigue-hand size.
  - The Imperium Row replenishment deck and the future Conflict deck have
    fully known contents (everything else is accounted for in public piles)
    but hidden order, so they are simply shuffled in place.
  - The RNG is reseeded so repeated determinizations (and the simulation that
    follows) don't replay a correlated sequence from the original clone.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def determinize(gs, pid: int, rng: np.random.Generator):
    gs2 = gs.clone()
    gs2.rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))

    # --- opponents' hand/deck: known composition, hidden split + order ---
    for p in gs2.players:
        if p.id == pid:
            continue
        pool = p.hand + p.deck
        if pool:
            idx = gs2.rng.permutation(len(pool))
            pool = [pool[i] for i in idx]
        n_hand = len(p.hand)
        p.hand = pool[:n_hand]
        p.deck = pool[n_hand:]

    # --- intrigue: shared deck + every opponent's hidden hand ---
    pool = list(gs2.intrigue_deck)
    opp_counts = []
    for p in gs2.players:
        if p.id == pid:
            continue
        opp_counts.append((p, len(p.intrigue_cards)))
        pool.extend(p.intrigue_cards)
    if pool:
        idx = gs2.rng.permutation(len(pool))
        pool = [pool[i] for i in idx]
    cursor = 0
    for p, n in opp_counts:
        p.intrigue_cards = pool[cursor:cursor + n]
        cursor += n
    gs2.intrigue_deck = pool[cursor:]

    # --- known contents, hidden order ---
    gs2.rng.shuffle(gs2.imperium_deck)
    gs2.rng.shuffle(gs2.conflict_deck)

    return gs2
