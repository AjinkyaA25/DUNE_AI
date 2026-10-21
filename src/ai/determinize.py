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
  - pid's OWN deck: contents known, draw order hidden, so it is shuffled too
    (2026-10-10: it was left in true order, so every playout knew exactly
    which cards pid would draw next round).
  - The Conflict deck is built in level order (I, then II, then III; see
    create_conflict_deck), so it is shuffled within each level, never across.
  - Bloodlines: the tech tiles under each stack's top (face-down) are pooled
    and dealt back keeping every stack's size; the skill deck is shuffled.
  - The RNG is reseeded so repeated determinizations (and the simulation that
    follows) don't replay a correlated sequence from the original clone.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def determinize(gs, pid: int, rng: np.random.Generator):
    gs2 = gs.clone()
    gs2.rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    gs2._shuffle_salt = int(rng.integers(1, 2**31 - 1))    # future reshuffles unknown too

    # --- opponents' hand/deck: known composition, hidden split + order ---
    for p in gs2.players:
        if p.id == pid:
            if p.deck:
                gs2.rng.shuffle(p.deck)      # own draw order is hidden too
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
    deck = gs2.conflict_deck                 # level order: shuffle within levels
    for lvl in sorted({c.conflict_level for c in deck}):
        idx = [i for i, c in enumerate(deck) if c.conflict_level == lvl]
        cards = [deck[i] for i in idx]
        gs2.rng.shuffle(cards)
        for i, c in zip(idx, cards):
            deck[i] = c

    bl = getattr(gs2, "bl", None)
    if bl is not None:
        under = [t for st in bl.tech_stacks for t in st[1:]]
        gs2.rng.shuffle(under)
        k = 0
        for st in bl.tech_stacks:
            n = len(st) - 1
            if n > 0:
                st[1:] = under[k:k + n]
                k += n
        gs2.rng.shuffle(bl.skill_deck)

    return gs2
