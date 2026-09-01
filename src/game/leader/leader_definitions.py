"""
The 9 Dune Imperium: Uprising Leaders.

⚠️  ABILITY TEXT IS APPROXIMATE.  These reconstructions are functional and
    balanced-ish but should be verified against the physical Leader cards.
    Each Leader's `notes` field flags what to check.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.game.effects import EffectResolver
from src.game.leader.leader import Leader


def _fx(effect: Dict):
    """Return a hook that resolves a fixed effect dict for the player."""
    def hook(player, gs):
        EffectResolver.resolve_single_effect(dict(effect), player, gs)
    return hook


# ---------------------------------------------------------------------------
# Leader factory
# ---------------------------------------------------------------------------

def create_leaders() -> List[Leader]:
    leaders: List[Leader] = []

    # ── Gurney Halleck (1 icon) ────────────────────────────────────────────
    def gurney_agent_placed(player, gs, space):
        from src.game.board.board import UPRISING_BOARD
        if UPRISING_BOARD[space].is_combat_space and player.leader.once_per_turn("gurney"):
            EffectResolver.resolve_single_effect({"troops": 1}, player, gs)
    leaders.append(Leader(
        "Gurney Halleck", 1,
        on_agent_placed=gurney_agent_placed,
        signet=_fx({"troops": 2}),
        notes="APPROX: passive should let you pay spice for +combat strength; "
              "signet grants troops.",
    ))

    # ── Lady Jessica (3 icons) ────────────────────────────────────────────
    def jessica_card_played(player, gs, card):
        from src.game.cards.card import CardTag
        if card.has_tag(CardTag.BENE_GESSERIT) and player.leader.once_per_turn("jessica"):
            EffectResolver.resolve_single_effect({"draw": 1}, player, gs)
    leaders.append(Leader(
        "Lady Jessica", 3,
        on_card_played=jessica_card_played,
        signet=_fx({"persuasion": 2}),
        notes="APPROX: passive = first Bene Gesserit card each turn draws a card.",
    ))

    # ── Muad'Dib (3 icons) ────────────────────────────────────────────────
    def muaddib_reveal(player, gs):
        if not player.hand:                    # revealed an empty-ish hand
            EffectResolver.resolve_single_effect({"solari": 2, "draw": 1}, player, gs)
    leaders.append(Leader(
        "Muad'Dib", 3,
        on_reveal=muaddib_reveal,
        signet=_fx({"water": 1, "spice": 1}),
        notes="APPROX: passive = Reveal with no cards left in hand -> 2 solari + draw.",
    ))

    # ── Staban Tuek (2 icons) ─────────────────────────────────────────────
    leaders.append(Leader(
        "Staban Tuek", 2,
        on_setup=_fx({"spice": 2}),
        on_agent_placed=(lambda player, gs, space: (
            EffectResolver.resolve_single_effect({"spice": 1}, player, gs)
            if space in {"Hagga Basin", "Deep Desert", "Imperial Basin"}
            and player.leader.once_per_turn("tuek") else None)),
        signet=_fx({"spice": 2}),
        notes="APPROX: starts +2 spice; +1 spice first Maker space each turn; "
              "signet +2 spice. (Real: may spend spice as solari.)",
    ))

    # ── Feyd-Rautha Harkonnen (2 icons) ──────────────────────────────────
    leaders.append(Leader(
        "Feyd-Rautha Harkonnen", 2,
        on_combat_win=(lambda player, gs, conflict:
                       EffectResolver.resolve_single_effect({"intrigue": 1}, player, gs)),
        signet=_fx({"intrigue": 1}),
        notes="APPROX: win a Conflict -> draw Intrigue; signet draws Intrigue. "
              "(Real: Feyd token / remove opponent troop mechanic.)",
    ))

    # ── Lady Margot Fenring (2 icons) ────────────────────────────────────
    leaders.append(Leader(
        "Lady Margot Fenring", 2,
        on_spy_placed=_fx({"persuasion": 1}),
        signet=(lambda player, gs:
                gs.add_pending_spy_placement(player.id, 1, allow_occupied=True)),
        notes="APPROX: placing a Spy grants +1 persuasion; signet places a Spy "
              "(may share an occupied post).",
    ))

    # ── Princess Irulan (2 icons) ────────────────────────────────────────
    leaders.append(Leader(
        "Princess Irulan", 2,
        on_intrigue_gained=_fx({"solari": 1}),
        signet=_fx({"intrigue": 1}),
        notes="APPROX: gain 1 solari whenever you gain an Intrigue card; "
              "signet draws an Intrigue.",
    ))

    # ── Lady Amber Metulli (1 icon) ─────────────────────────────────────
    def amber_reveal(player, gs):
        allies = sum(1 for v in player.alliances.values() if v)
        if allies:
            gs.gain_persuasion(player.id, allies)
    leaders.append(Leader(
        "Lady Amber Metulli", 1,
        on_reveal=amber_reveal,
        signet=_fx({"persuasion": 2}),
        notes="APPROX: Reveal -> +1 persuasion per Alliance token; signet +2 persuasion.",
    ))

    # ── Shaddam Corrino IV (CHOAM only) ─────────────────────────────────
    def shaddam_setup(player, gs):
        # Real: set aside the two Sardaukar contracts for Shaddam.
        for _ in range(2):
            if gs.contract_bank:
                player.take_contract(gs.contract_bank.pop(0))
    leaders.append(Leader(
        "Shaddam Corrino IV", 2,
        on_setup=shaddam_setup,
        signet=_fx({"solari": 2}),
        notes="APPROX: starts with 2 contracts; signet +2 solari. CHOAM only.",
    ))

    return leaders


ALL_LEADERS: Dict[str, Leader] = {ld.name: ld for ld in create_leaders()}


def get_leader(name: str) -> Leader:
    # Return a FRESH copy so per-turn flags don't leak between games.
    fresh = {ld.name: ld for ld in create_leaders()}
    if name not in fresh:
        raise KeyError(f"Unknown leader '{name}'. Options: {sorted(fresh)}")
    return fresh[name]


def assign_leaders(gs, leaders: Optional[List[str]] = None) -> None:
    """Attach a Leader to each player. `leaders` = explicit names, else random."""
    pool = create_leaders()
    if not gs.use_choam:
        pool = [ld for ld in pool if ld.name != "Shaddam Corrino IV"]

    if leaders is None:
        gs.rng.shuffle(pool)
        chosen = pool[: gs.num_players]
    else:
        by_name = {ld.name: ld for ld in create_leaders()}
        chosen = [by_name[n] for n in leaders]

    for player, leader in zip(gs.players, chosen):
        player.leader = leader
