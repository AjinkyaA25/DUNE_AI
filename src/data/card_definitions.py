"""
Card and conflict definitions for Dune Imperium: Uprising.

STARTER + IMPERIUM cards use the REAL Uprising names, costs, access icons, tags,
persuasion and swords (verified against the card images / rulebook).  Conditional
"card text" effects are modelled approximately — each such card carries a `notes`
string describing what was simplified.  See src/data/uprising_cards.json for the
source extraction.

Conflicts, contracts and intrigue are still placeholder data (a later pass).
"""
from __future__ import annotations
from typing import List

from src.game.cards.card import Card, CardType, AccessSymbol, CardTag
from src.game.combat.conflict import Conflict, BattleIcon
from src.game.intrigue.intrigue import IntrigueCard, IntrigueTiming
from src.game.gameState import GameState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c(name: str, card_type: CardType = CardType.IMPERIUM, cost: int = 0) -> Card:
    return Card(name, card_type, cost)


_ACCESS = {
    "city": AccessSymbol.CITY, "desert": AccessSymbol.DESERT,
    "landsraad": AccessSymbol.LANDSRAAD, "emperor": AccessSymbol.EMPEROR,
    "spacing_guild": AccessSymbol.SPACING, "bene_gesserit": AccessSymbol.BENE,
    "fremen": AccessSymbol.FREMEN, "spy": AccessSymbol.SPY,
}
_TAG = {
    "emperor": CardTag.EMPEROR, "spacing_guild": CardTag.SPACING_GUILD,
    "bene_gesserit": CardTag.BENE_GESSERIT, "fremen": CardTag.FREMEN,
}


def _make(name, cost, ctype, access=(), tags=(), agent=None, reveal=None,
          persuasion=0, swords=0, acquire=None, trash=None, notes="") -> Card:
    c = _c(name, ctype, cost)
    for a in access:
        c.add_access_symbol(_ACCESS[a])
    for t in tags:
        c.add_tag(_TAG[t])
    if agent:
        c.agent_effects = [dict(agent)]
    reveal = dict(reveal or {})
    res = {k: reveal.pop(k) for k in ("solari", "spice", "water") if k in reveal}
    if res:
        c.reveal_resources = res
    if reveal:
        c.reveal_effects = [reveal]
    c.persuasion = persuasion
    c.swords = swords
    if acquire:
        c.acquire_effects = [dict(acquire)]
    if trash:
        c.trash_effects = [dict(trash)]
    if notes:
        c.notes = notes
    return c


# ---------------------------------------------------------------------------
# Starter deck  (10 cards — each player receives an identical copy)
# ---------------------------------------------------------------------------

def create_starter_cards() -> List[Card]:
    """The real 10-card Uprising starting deck (rulebook component list)."""
    S = CardType.STARTER
    cards: List[Card] = []
    cards += [_make("Convincing Argument", 0, S, persuasion=2) for _ in range(2)]
    cards += [_make("Dagger", 0, S, access=["landsraad"], swords=1) for _ in range(2)]
    cards += [_make("Dune, the Desert Planet", 0, S, access=["desert"], persuasion=1)
              for _ in range(2)]
    cards.append(_make("Diplomacy", 0, S,
                       access=["emperor", "spacing_guild", "bene_gesserit", "fremen"],
                       persuasion=1))
    cards.append(_make("Reconnaissance", 0, S, access=["city"], persuasion=1))
    cards.append(_make("Seek Allies", 0, S,
                       access=["emperor", "spacing_guild", "bene_gesserit", "fremen"],
                       agent={"trash_self": 1},
                       notes="Agent: trash this card (deck thinning)."))
    # Signet Ring's agent box triggers the Leader's Signet ability (GameState hook).
    cards.append(_make("Signet Ring", 0, S, access=["landsraad", "city", "desert"],
                       persuasion=1))
    return cards  # 10 cards


# ---------------------------------------------------------------------------
# Imperium deck — the real 58 Uprising Imperium cards.
# Effects use the standard effect-dict vocabulary + approximate condition keys
# (if_fremen_bond, if_spy_recalled, if_agent_to_maker, if_tag_other_<t>,
#  if_influence_<f>_<n>, if_alliance_<f>, if_any_alliance, if_contracts_<n>,
#  if_councilor, if_swordmaster, if_faction_agent, if_spies_<n>).
# `notes` flags anything simplified from the printed text.
# ---------------------------------------------------------------------------

I = CardType.IMPERIUM
R = CardType.RESERVE


def create_imperium_cards() -> List[Card]:
    m = _make
    return [
        m("Arrakis Revolt", 6, I, access=["city"], tags=["fremen"],
          agent={"if_hooks": {"pay_spice_spawn_sandworm": 2}}, persuasion=1, swords=3,
          notes="Agent: if you have Maker Hooks, you MAY spend 2 spice to summon a "
                "sandworm and (optionally) detonate the Shield Wall. Auto-taken when "
                "affordable + a Conflict is up; wall broken only if the Conflict is "
                "at a protected location."),
        m("Bene Gesserit Operative", 3, I, access=["bene_gesserit"], tags=["bene_gesserit"],
          agent={"spy": 1}, persuasion=1, reveal={"if_spies_2": {"persuasion": 2}}),
        m("Branching Path", 3, I, access=["bene_gesserit", "city"], tags=["bene_gesserit"],
          agent={"if_alliance_bene_gesserit":
                 {"trash_intrigue_for": {"spice": 2, "intrigue": 1}}}, persuasion=2),
        m("Calculus of Power", 3, I, access=["city", "spy"], tags=["emperor"],
          agent={"trash": 1}, persuasion=2,
          reveal={"if_tag_other_emperor": {"swords": 3}},
          notes="Reveal swords require trashing an Emperor card in play — approximated "
                "as: another Emperor card in play."),
        m("Captured Mentat", 5, I, access=["landsraad", "desert"],
          agent={"discard_then": {"draw": 1, "intrigue": 1}}, persuasion=1,
          reveal={"influence_swap": 1},
          notes="Agent: MAY discard a card to draw a card + gain an Intrigue "
                "(accept/decline choice). Reveal: lose 1 Influence on any Faction, "
                "gain 1 on any Faction (same Faction allowed); auto-chosen to "
                "reallocate a wasted cube, else no-op."),
        m("Cargo Runner", 3, I, access=["landsraad", "city", "desert"], tags=["spacing_guild"],
          agent={"if_contracts_2": {"draw": 1}, "if_contracts_4": {"draw": 1}},
          persuasion=1),
        m("Chani, Clever Tactician", 5, I, access=["fremen", "city", "desert"], tags=["fremen"],
          agent={"if_units_in_conflict_3": {"intrigue": 1}}, persuasion=2,
          reveal={"if_fremen_bond": {"persuasion": 2}},
          notes="Reveal also: MAY retreat 2 troops from the Conflict to garrison "
                "for 4 swords — optional/situational, not modelled."),
        m("Corrinth City", 6, I, access=["emperor", "landsraad"], tags=["emperor"],
          agent={"discard_pay_vp": {"discard": 2, "solari": 5, "vp": 1}}, persuasion=0,
          reveal={"hc_seat_or_solari": 5},
          notes="Agent: MAY discard 2 cards + spend 5 solari -> 1 VP (accept/decline). "
                "Reveal: gain 5 solari, OR spend 5 solari to take a High Council seat "
                "(auto: takes the seat if you have none and can pay)."),
        m("Covert Operation", 3, I, access=["spy"],
          agent={"opponents_discard": 1}, reveal={"spy": 2}),
        m("Dangerous Rhetoric", 3, I, access=["landsraad", "spy"],
          agent={"influence_any": 1, "trash_self": 1}, persuasion=1, swords=1,
          notes="Agent: gain 1 Influence with a Faction of your choice, then trash this card."),
        m("Delivery Agreement", 5, I, access=["city"], tags=["spacing_guild"],
          agent={"discard_then": {"contract": 1}}, persuasion=0,
          reveal={"choose_by_contracts": {"n": 4, "yes": {"vp": 1}, "no": {"spice": 1}}},
          notes="No reveal persuasion. Reveal: gain 1 spice, OR — if you have "
                "completed 4+ contracts — 1 VP instead (auto-takes the VP). "
                "Agent: MAY discard a card to take a contract."),
        m("Desert Power", 6, I, access=["desert"], tags=["fremen"],
          agent={"if_agent_to_maker": {"spice": 2}}, persuasion=0,
          reveal={"worm_or_persuasion": {"persuasion": 2, "water": 1, "sandworm": 1}},
          notes="Reveal: gain 2 persuasion, OR — with Maker Hooks — spend 1 water "
                "to summon a sandworm into the Conflict (needs the Shield Wall down "
                "/ Conflict not behind it). Worm auto-taken only when already "
                "committed to that Conflict, else the 2 persuasion."),
        m("Desert Survival", 2, I, access=["desert"], tags=["fremen"],
          agent={"trash": 1}, persuasion=1, swords=1),
        m("Double Agent", 3, I, access=["landsraad", "city", "desert"],
          tags=["emperor", "spacing_guild"],
          agent={"spy_special": 1}, persuasion=1, swords=1,
          notes="'Spy on the space you visited' approximated as a free special Spy placement."),
        m("Ecological Testing Station", 3, I, access=["fremen", "city"], tags=["fremen"],
          agent={"pay_then": {"cost": {"water": 2}, "draw": 2}},
          persuasion=1, reveal={"if_fremen_bond": {"water": 1}},
          notes="Agent: MAY spend 2 water to draw 2 cards (auto-taken when you "
                "have 2+ water). Reveal: 1 persuasion, +1 water if Fremen bond."),
        m("Fedaykin Stilltent", 2, I, access=["desert"], tags=["fremen"],
          agent={"if_agent_to_maker": {"troops": 1}}, reveal={"water": 1}),
        m("Guild Envoy", 3, I,
          access=["emperor", "spacing_guild", "bene_gesserit", "fremen"], tags=["spacing_guild"],
          agent={"discard_then_if_sg": {"draw": 2}}, persuasion=1,
          notes="Agent: mandatory — discard a card; if it was a Spacing Guild "
                "card, draw 2 cards (auto-discards a spare SG card to earn this)."),
        m("Guild Spy", 3, I, access=["spy", "spacing_guild"], tags=["spacing_guild"],
          agent={"discard_then": {"draw": 1}}, persuasion=2, acquire={"spy": 1},
          notes="Acquire: place a Spy. Agent modelled as 'MAY discard a card to "
                "draw a card'; the printed upgrade — if you discarded a Spacing "
                "Guild card this turn, draw a card + gain an Intrigue instead — "
                "needs discard-tracking, not modelled. Reveal: 2 persuasion; the "
                "printed bonus (if you acquired The Spice Must Flow this Reveal "
                "turn, +1 influence with each Faction whose post holds your Spy) "
                "is not modelled."),
        m("Hidden Missive", 2, I, access=["landsraad"], tags=["bene_gesserit"],
          agent={"if_influence_bene_gesserit_2": {"troops": 1, "draw": 1}},
          persuasion=1, swords=1),
        m("Imperial Spymaster", 2, I, access=["emperor", "spy"], tags=["emperor"],
          agent={"if_spy_recalled": {"intrigue": 1}}, persuasion=1, swords=1),
        m("In High Places", 5, I, access=["emperor", "bene_gesserit"],
          tags=["emperor", "bene_gesserit"], acquire={"spy": 1},
          agent={"if_tag_other_bene": {"spy": 1, "draw": 1}},
          persuasion=2,
          reveal={"recall_spies_persuasion": {"count": 2, "persuasion": 3}},
          notes="Acquire: place a Spy. Agent: with another Bene Gesserit card in "
                "play, gain a Spy + draw a card. Reveal: 2 persuasion, plus MAY "
                "recall 2 Spies this turn for +3 persuasion (5 total); auto-taken "
                "when 2+ Spies are on the board."),
        m("Interstellar Trade", 7, I, access=["landsraad", "city", "desert"],
          tags=["spacing_guild"], acquire={"contract": 1},
          reveal={"persuasion_per_contract": 1},
          notes="Acquire: take a contract. Reveal: 1 persuasion per completed "
                "contract (no other reveal effect)."),
        m("Junction Headquarters", 6, I,
          access=["landsraad", "city", "desert"], tags=["spacing_guild"],
          agent={"if_alliance_spacing_guild":
                 {"trash_intrigue_for": {"pay_then": {"cost": {"spice": 2}, "vp": 1}}}},
          reveal={"persuasion": 1, "water": 1, "troops": 1},
          notes="Agent: with the Spacing Guild Alliance, MAY trash an Intrigue "
                "card and spend 2 spice to gain 1 VP."),
        m("Leadership", 5, I, access=["desert", "fremen"], tags=["fremen"],
          agent={"draw_per_sandworm_in_conflict": 1}, persuasion=2, swords=1,
          reveal={"swords_per_other_revealed_card": 1},
          notes="Agent: draw 1 per sandworm you have in the Conflict. Reveal: "
                "2 persuasion + 1 sword, +1 more sword per other card revealed."),
        m("Long Live the Fighters", 7, I, access=["fremen", "city"], tags=["fremen"],
          agent={"look_top3_draw_discard_trash": 1}, persuasion=2, swords=3,
          notes="Agent: look at top 3 of your deck — draw 1, discard 1, trash 1 "
                "(all mandatory; auto keeps the strongest / trashes the weakest)."),
        m("Maker Keeper", 2, I, access=["city"], tags=["bene_gesserit", "fremen"],
          agent={"if_influence_bene_gesserit_2": {"water": 1},
                 "if_influence_fremen_2": {"spice": 1}},
          persuasion=2),
        m("Maula Pistol", 3, I, access=["city", "desert"], tags=["fremen"],
          agent={"draw": 1}, persuasion=1, swords=1),
        m("Northern Watermaster", 3, I, access=["city"], tags=["fremen"],
          agent={"water": 1}, persuasion=1, reveal={"if_fremen_bond": {"spice": 2}}),
        m("Overthrow", 8, I,
          access=["emperor", "spacing_guild", "bene_gesserit", "fremen"],
          agent={"if_faction_agent": {"influence_faction_visited": 2}},
          persuasion=2, swords=2, reveal={"troops": 1}, acquire={"intrigue": 1},
          notes="Agent: gain 2 influence with the Faction whose space you visited "
                "(the space's normal +1 still applies). Acquire: gain an Intrigue."),
        m("Paracompass", 4, I, access=["city"],
          agent={"solari": 2, "if_influence_bene_gesserit_2": {"draw": 1}},
          reveal={"if_councilor": {"persuasion": 2,
                                   "if_swordmaster": {"persuasion": 1}}},
          notes="Agent: 2 solari, +1 draw if 2+ Bene Gesserit influence. Reveal: "
                "with a High Council seat, 2 persuasion (+1 more if you also have "
                "your Swordmaster)."),
        m("Prepare the Way", 2, I, access=["landsraad", "city"], tags=["bene_gesserit"],
          agent={"if_influence_bene_gesserit_2": {"solari": 1}}, persuasion=2),
        m("Price is No Object", 6, I, access=["emperor", "bene_gesserit"],
          tags=["emperor", "bene_gesserit"],
          agent={"acquire_with_solari": {"max_cost": 6}},
          persuasion=2, reveal={"solari": 2}, acquire={"solari": 2},
          notes="Agent: MAY acquire an Imperium Row card by paying Solari equal "
                "to its cost instead of Persuasion (auto-buys the best affordable; "
                "max_cost 6 unverified). Reveal: 2 persuasion + 2 solari. "
                "Acquire: gain 2 solari."),
        m("Priority Contracts", 6, I, access=["landsraad", "desert"], tags=["spacing_guild"],
          agent={"contract": 1}, persuasion=0,
          reveal={"choose_by_contracts": {"n": 4, "yes": {"vp": 1}, "no": {"spice": 2}}},
          notes="No reveal persuasion. Reveal: gain 2 spice, OR — with 4+ "
                "contracts completed — 1 VP instead (auto-takes the VP)."),
        m("Public Spectacle", 4, I, access=["spy"], tags=["emperor"],
          agent={"if_spy_recalled": {"influence_any": 1}}, persuasion=1,
          reveal={"spy": 1}),
        m("Rebel Supplier", 3, I, access=["city"], tags=["fremen"],
          agent={"if_spy_recalled": {"troops": 2}}, persuasion=0, swords=1,
          reveal={"spice": 1}),
        m("Reliable Informant", 2, I, access=["spacing_guild"], tags=["spacing_guild"],
          agent={"spy_posts": {"count": 1, "posts": ["Emperor Post",
                 "Bene Gesserit Post", "Fremen Post"]}},
          persuasion=1, reveal={"solari": 1},
          notes="Agent: place a Spy on the Emperor, Bene Gesserit, or Fremen "
                "observation post only. Reveal: 1 persuasion + 1 solari."),
        m("Sardaukar Coordination", 4, I, access=["emperor", "landsraad"], tags=["emperor"],
          agent={"deploy_recruited": 1}, persuasion=2,
          reveal={"swords_per_emperor_card": 1},
          notes="Agent: MAY deploy any number of troops you recruited this turn "
                "to the Conflict (no cap). Reveal: 2 persuasion, +1 sword per "
                "Emperor card in play if you have another Emperor card besides this one."),
        m("Sardaukar Soldier", 1, I, access=["city"], tags=["emperor"],
          persuasion=1, swords=1, trash={"intrigue": 1},
          notes="When trashed: gain 1 Intrigue card."),
        m("Shishakli", 4, I, access=["city", "desert"], tags=["fremen"],
          agent={"trash_then": {"draw": 1}}, swords=2,
          reveal={"if_fremen_bond": {"influence_fremen": 1}},
          notes="Agent: MAY trash a card; if you do, draw a card."),
        m("Smuggler's Harvester", 1, I, access=["desert"], tags=["spacing_guild"],
          agent={"if_agent_to_maker": {"spice": 1}}, persuasion=1),
        m("Smuggler's Haven", 4, I, access=["spacing_guild", "desert"], tags=["spacing_guild"],
          agent={"pay_then": {"cost": {"spice": 4}, "vp": 1}},
          persuasion=1, reveal={"if_spy_at_maker_post": {"spice": 2}},
          notes="Agent: MAY convert 4 spice into 1 VP (accept/decline choice). "
                "Reveal: 1 persuasion, +2 spice if you have a Spy on a post that "
                "borders a Maker board space."),
        m("Southern Elders", 4, I, access=["bene_gesserit", "fremen"],
          tags=["bene_gesserit", "fremen"],
          agent={"if_tag_other_bene": {"troops": 2}},
          reveal={"water": 1, "if_fremen_bond": {"persuasion": 2}}),
        m("Space-Time Folding", 1, I, access=["spacing_guild"], tags=["spacing_guild"],
          agent={"discard_then_sg": {"base": {"draw": 1}, "sg_bonus": {"draw": 1}}},
          persuasion=1,
          notes="Agent: MAY discard a card to draw a card; if the discarded card "
                "was a Spacing Guild card, draw 1 extra (auto-discards a spare "
                "Guild card when one is in hand)."),
        m("Spacing Guild's Favor", 5, I, access=["spacing_guild", "desert"],
          tags=["spacing_guild"], agent={"draw": 1}, persuasion=2,
          reveal={"pay_then": {"cost": {"spice": 3}, "influence_choice": 1}},
          notes="Reveal: 2 persuasion, plus MAY spend 3 spice for 1 influence "
                "with a Faction of your choice."),
        m("Spy Network", 2, I, tags=["emperor", "spacing_guild"],
          acquire={"spy": 1}, persuasion=1,
          reveal={"if_spies_2": {"recall_spy_then": {"intrigue": 1}}},
          notes="No access icons (reveal-only card). Acquire: place a Spy. "
                "Reveal: 1 persuasion; with 2+ Spies on the board, MAY recall one "
                "for an Intrigue card."),
        m("Steersman", 8, I, access=["spacing_guild", "landsraad", "city", "desert"],
          tags=["spacing_guild"], acquire={"influence_spacing_guild": 1},
          agent={"draw": 1, "uplift": 1}, persuasion=2, reveal={"spice": 2},
          notes="Acquire: 1 Spacing Guild influence. Agent: draw a card and "
                "recall (uplift) one of your other Agents from the board."),
        m("Stilgar, the Devoted", 6, I, access=["city", "desert", "fremen"], tags=["fremen"],
          agent={"troops": 2},
          reveal={"if_tag_fremen": {"persuasion": 2}, "if_tag_fremen_count_2": {"persuasion": 2}},
          notes="Reveal: 2 persuasion per Fremen card in play — approximated as tiers."),
        m("Strike Fleet", 5, I, access=["spy"], acquire={"spy": 1},
          agent={"if_spy_recalled": {"troops": 3}}, persuasion=1, swords=3,
          notes="Acquire: place a Spy."),
        m("Subversive Advisor", 5, I, access=["spy"], acquire={"spy": 1},
          agent={"trash_self": 1,
                 "if_faction_agent": {"influence_faction_visited": 2}}, persuasion=1,
          notes="Acquire: place a Spy. Agent: a one-time Overthrow — trash this "
                "card, and (if you visited a Faction space) gain 2 influence with "
                "that Faction."),
        m("The Beast's Spoils", 3, I, access=["city"], tags=["emperor"],
          agent={"beast_spoils": 1}, swords=3,
          notes="Agent: one reward per distinct face-up battle icon among the "
                "Conflict cards you have won — Desert Mouse -> 1 spice, "
                "Crysknife -> trash a card, Ornithopter -> recruit 1 troop "
                "(Wild counts as all three)."),
        m("Thumper", 3, I, access=["desert"], tags=["fremen"],
          agent={"if_agent_to_maker": {"spice": 2}}, persuasion=1, reveal={"spice": 1},
          notes="'Double the bonus spice harvested' approximated as +2 spice at a Maker space."),
        m("Treacherous Maneuver", 5, I,
          access=["emperor", "spacing_guild", "bene_gesserit", "fremen"], tags=["emperor"],
          agent={"trash_pair_emperor_influence": 1},
          persuasion=1, reveal={"influence_emperor": 1},
          notes="Agent: if you have another Emperor card in play, trash BOTH it "
                "and Treacherous Maneuver, then gain 1 influence with the Faction "
                "whose space you visited. Reveal: 1 persuasion + 1 Emperor influence."),
        m("Tread in Darkness", 4, I, access=["landsraad", "city", "desert"],
          tags=["bene_gesserit"],
          agent={"if_tag_other_bene": {"trash": 1, "draw": 1}}, persuasion=2, swords=1),
        m("Truthtrance", 4, I,
          access=["emperor", "spacing_guild", "bene_gesserit", "fremen"], tags=["bene_gesserit"],
          persuasion=1, notes="Complex Intrigue-prediction text not modelled; flat persuasion 1."),
        m("Undercover Asset", 2, I, access=["landsraad", "city", "desert"],
          tags=["emperor", "spacing_guild"],
          agent={"ignore_influence_gates": 1},
          reveal={"spy_or_swords": {"spy": 1, "swords": 2}},
          notes="Agent: ignore board-space Influence requirements this turn. "
                "Reveal: choose 1 Spy OR 2 swords (takes swords when contesting "
                "the Conflict, else the Spy)."),
        m("Unswerving Loyalty", 1, I, tags=["fremen"],
          persuasion=1,
          reveal={"troops": 1, "if_fremen_bond": {"deploy_or_retreat": 1}},
          notes="Reveal: 1 persuasion + 1 troop; with a Fremen bond, choose to "
                "deploy 1 troop OR retreat 1 troop."),
        m("Weirding Woman", 1, I, access=["city", "desert"], tags=["bene_gesserit"],
          agent={"if_tag_other_bene": {"return_self_to_hand": 1}}, persuasion=1, swords=1),
        m("Wheels Within Wheels", 2, I, access=["spy"], tags=["emperor", "spacing_guild"],
          agent={"if_influence_emperor_2": {"solari": 2},
                 "if_influence_spacing_guild_2": {"spice": 1}},
          persuasion=1, reveal={"spy": 1}),
    ]  # 58 imperium cards


# ---------------------------------------------------------------------------
# Reserve cards — always available.  Prepare the Way lives in the imperium
# deck in Uprising (built above with card_type RESERVE); The Spice Must Flow
# has its own reserve stack.
# ---------------------------------------------------------------------------

def create_reserve_prepare_the_way(_count: int = 0) -> List[Card]:
    # Uprising: Prepare the Way is part of the Imperium deck, not a reserve stack.
    return []


def create_reserve_spice_must_flow(count: int = 5) -> List[Card]:
    return [_make("The Spice Must Flow", 9, CardType.RESERVE, access=["desert"],
                  tags=["spacing_guild"], acquire={"spice": 1},
                  reveal={"spice": 1, "vp": 1}) for _ in range(count)]


# ---------------------------------------------------------------------------
# Conflict deck — the REAL Uprising conflicts (from conflicts/tier 1-3 images).
#
# Names, battle icons and locations are read directly from the card art.
# Reward values are read from the art where legible and otherwise reconstructed
# in the standard Uprising style — treat the exact numbers as ~approximate.
#
# Deck construction (rulebook): shuffle the 4 Level I and take 1; shuffle the
# 10 Level II and take 5; shuffle the 4 Level III and take all 4.  Final deck
# top->bottom: [1 Level I][5 Level II][4 Level III]  -> a 10-round game.
# ---------------------------------------------------------------------------

_CR = BattleIcon.CRYSKNIFE
_DM = BattleIcon.DESERT_MOUSE
_OR = BattleIcon.ORNITHOPTER
_WILD = BattleIcon.WILD


def _cf(name, level, r1, r2, r3=None, icon=None, location=None):
    return Conflict(name=name, conflict_level=level, first_place_reward=r1,
                    second_place_reward=r2, third_place_reward=r3,
                    battle_icon=icon, location=location)


def conflict_level_1_pool() -> List[Conflict]:
    return [
        _cf("Skirmish (Crysknife)", 1,
            {"influence_any": 1}, {"intrigue": 1, "solari": 1}, {"solari": 1}, _CR),
        _cf("Skirmish (Desert Mouse)", 1,
            {"water": 2}, {"solari": 3}, {"solari": 2}, _DM),
        _cf("Skirmish (Ornithopter)", 1,
            {"intrigue": 1, "solari": 1}, {"intrigue": 1, "solari": 2},
            {"intrigue": 1}, _OR),
        _cf("Skirmish (Wild)", 1,
            {"trash": 1}, {"water": 1, "solari": 1}, {"solari": 2}, _WILD),
    ]


def conflict_level_2_pool() -> List[Conflict]:
    return [
        _cf("Siege of Arrakeen", 2,
            {"control": True, "solari": 2, "troops": 2},
            {"solari": 4, "troops": 1}, {"solari": 3}, _CR, "Arrakeen"),
        _cf("CHOAM Security", 2,
            {"spice": 1, "troops": 1, "solari": 1},
            {"water": 1, "solari": 2, "troops": 2},
            {"intrigue": 1, "troops": 1}, _CR),
        _cf("Secure Imperial Basin", 2,
            {"control": True, "spice": 2, "troops": 1},
            {"water": 2, "troops": 1}, {"water": 1, "troops": 1}, _DM, "Imperial Basin"),
        _cf("Protect the Sietches", 2,
            {"influence_any": 1, "water": 1, "troops": 1},
            {"spice": 3, "troops": 1}, {"spice": 2}, _DM),
        _cf("Shadow Contest", 2,
            {"influence_any": 1, "intrigue": 1},
            {"intrigue": 1, "spice": 1, "troops": 1},
            {"spice": 1, "troops": 1}, _OR),
        _cf("Spice Freighters", 2,
            {"may_pay_spice_for_vp": {"cost": 3, "vp": 1}},
            {"water": 1, "spice": 1, "solari": 1},
            {"spice": 1, "solari": 1}, _CR),
        _cf("Seize Spice Refinery", 2,
            {"control": True, "water": 1, "spice": 2},
            {"intrigue": 1, "spice": 1, "troops": 1}, {"spice": 2}, _CR, "Spice Refinery"),
        _cf("Storms in the South", 2,
            {"solari": 1, "water": 1, "spice": 2},
            {"intrigue": 2, "solari": 2}, {"intrigue": 1, "solari": 2}, _WILD),
        _cf("Test of Loyalty", 2,
            {"influence_any": 1, "water": 1, "solari": 2},
            {"solari": 4, "troops": 1}, {"solari": 3}, _OR),
        _cf("Trade Dispute", 2,
            {"spice": 1, "water": 1, "trash": 1},
            {"water": 1, "spice": 1, "trash": 1},
            {"water": 1, "troops": 1}, _DM),
    ]


def conflict_level_3_pool() -> List[Conflict]:
    return [
        _cf("Battle for Arrakeen", 3,
            {"vp": 1, "control": True,
             "may_pay_troops_for_vp": {"cost": 2, "vp": 1}},
            {"draw": 1, "spice": 1, "solari": 3}, {"spice": 2, "solari": 2},
            _CR, "Arrakeen"),
        _cf("Battle for Imperial Basin", 3,
            {"vp": 1, "control": True,
             "may_pay_spice_for_vp": {"cost": 4, "vp": 1}},
            {"spice": 5}, {"spice": 3}, _OR, "Imperial Basin"),
        _cf("Battle for Spice Refinery", 3,
            {"vp": 1, "control": True,
             "may_pay_solari_for_vp": {"cost": 6, "vp": 1}},
            {"intrigue": 1, "spice": 3}, {"spice": 3}, _DM, "Spice Refinery"),
        _cf("Propaganda", 3,
            {"influence_any": 2},
            {"intrigue": 1, "spice": 3}, {"spice": 3}, _WILD),
    ]


def create_conflict_deck(rng=None) -> List[Conflict]:
    import numpy as _np
    r = rng or _np.random.default_rng()
    lvl1 = conflict_level_1_pool()
    lvl2 = conflict_level_2_pool()
    lvl3 = conflict_level_3_pool()
    r.shuffle(lvl1)
    r.shuffle(lvl2)
    r.shuffle(lvl3)
    return lvl1[:1] + lvl2[:5] + lvl3[:4]        # top -> bottom


# ---------------------------------------------------------------------------
# Intrigue deck — the real Uprising Intrigue cards (from intrigue_cards/ images).
# Names + timing read from the art; effects use the standard vocabulary +
# approximate keys (see src/game/effects.py). `notes`-worthy simplifications are
# in comments below. 3 Rise-of-Ix tech cards are excluded (engine has no Tech).
# ---------------------------------------------------------------------------

_PLOT = IntrigueTiming.PLOT
_COMBAT = IntrigueTiming.COMBAT
_END = IntrigueTiming.ENDGAME
_PC = (IntrigueTiming.PLOT, IntrigueTiming.COMBAT)
_PE = (IntrigueTiming.PLOT, IntrigueTiming.ENDGAME)
_CE = (IntrigueTiming.COMBAT, IntrigueTiming.ENDGAME)


def create_intrigue_deck() -> List[IntrigueCard]:
    IC = IntrigueCard
    return [
        # ── Plot ────────────────────────────────────────────────────────────
        IC("Change Allegiances", _PLOT, [{"influence_swap": 1},
            {"pay_then": {"cost": {"spice": 3}, "influence_choice": 1}}]),
        IC("Sietch Ritual", _PLOT, [{"discard_then": {"influence_choice": 1}}]),
        IC("Unexpected Allies", _PLOT,
           [{"water_summon": {"cost": 2, "sandworm": 1, "deploy": 2}}]),
        IC("Cunning", _PLOT, [{"draw": 1},
            {"pay_then": {"cost": {"spice": 3}, "trash": 1, "draw": 1}}]),
        IC("Buy Access", _PLOT, [{"pay_then": {"cost": {"solari": 5}, "influence_choice": 2}}]),
        IC("Call to Arms", _PLOT, [{"troops": 2}]),   # approx: flat 2 troops
        IC("Councilor's Ambition", _PLOT, [{"if_councilor": {"water": 2}}]),
        IC("Depart for Arrakis", _PLOT, [{"pay_then": {"cost": {"spice": 2}, "troops": 3}},
            {"if_influence_spacing_guild_3": {"draw": 1}}]),
        IC("Detonation", _PLOT, [{"break_shield_wall": 1, "deploy": 4}]),
        IC("Distraction", _PLOT, [{"spy_special": 1}]),   # approx (real: triggered)
        IC("Imperium Politics", _PLOT, [{"pay_then": {"cost": {"solari": 1}, "influence_choice": 1}}]),
        IC("Inspire Awe", _PLOT, [{"acquire_free": {"max_cost": 3}}]),
        IC("Intelligence Report", _PLOT, [{"if_spies_2": {"draw": 2}}]),
        IC("Leverage", _PLOT, [{"spice": 1, "solari": 1}]),   # approx: "if gained spice"
        IC("Manipulate", _PLOT, [{"refresh_imperium_row": 1}]),
        IC("Market Opportunity", _PLOT, [{"market_convert": 1}]),
        IC("Mercenaries", _PLOT, [{"pay_then": {"cost": {"solari": 3}, "intrigue": 1, "troops": 2}}]),
        IC("Opportunism", _PLOT, [{"opportunism_vp": 1}]),
        IC("Poison Snooper", _PLOT, [{"peek_top": 1}]),
        IC("Seize Production", _PLOT, [{"solari": 2}]),   # Sardaukar-Commander option dropped
        IC("Shaddam's Favor", _PLOT, [{"troops": 1, "if_influence_emperor_3": {"solari": 3}}]),
        IC("Special Mission", _PLOT, [{"spy": 1}]),   # recall+wall option dropped
        IC("Strategic Stockpiling", _PLOT, [{"pay_then": {"cost": {"spice": 5}, "vp": 1}}]),  # uncertain
        IC("Adaptive Tactics", _PLOT, [{"pay_then": {"cost": {"spice": 1}, "troops": 1, "trash": 1}}]),
        IC("Bribery", _PLOT, [{"pay_then": {"cost": {"solari": 2}, "influence_choice": 1}}]),
        IC("Coercive Negotiation", _PLOT, [{"contract": 1}]),   # approx (real: triggered)
        IC("Emperor's Invitation", _PLOT, [{"draw": 1}]),   # 2nd option unclear
        IC("False Orders", _PLOT, [{"spy": 1}]),
        IC("Honor Guard", _PLOT, [{"troops": 1}]),
        IC("Insider Information", _PLOT, [{"recall_spy_then": {"trash": 1, "draw": 1}}]),
        IC("Sleeper Unit", _PLOT, [{"recall_spy_then": {"troops": 2}}]),
        # ── Combat ──────────────────────────────────────────────────────────
        IC("Questionable Methods", _COMBAT, [{"lose_influence_any": 1, "swords": 5}]),
        IC("Devour", _COMBAT, [{"swords": 2, "if_sandworm_in_conflict": {"swords": 2, "trash": 1}}]),
        IC("Find Weakness", _COMBAT, [{"swords": 2}, {"recall_spy_then": {"swords": 3}}]),
        IC("Go to Ground", _COMBAT, [{"retreat": 2}]),
        IC("Impress", _COMBAT, [{"swords": 2, "acquire_free": {"max_cost": 3}}]),
        IC("Reach Agreement", _COMBAT, [{"retreat": 2}]),
        IC("Return the Favor", _COMBAT, [{"swords": 1, "swords_per_friendship": 1}]),
        IC("Ripples in the Sand", _COMBAT, [{"swords": 3, "if_sandworm_in_conflict": {"intrigue": 1}}]),
        IC("Spice is Power", _COMBAT, [{"pay_then": {"cost": {"spice": 3}, "swords": 5}}]),
        IC("Spring the Trap", _COMBAT, [{"recall_spies_swords": {"count": 2, "swords": 7}}]),
        IC("Tactical Option", _COMBAT, [{"swords": 2}]),
        IC("Desert Support", _COMBAT, [{"pay_then": {"cost": {"water": 1}, "swords": 5}}]),
        # ── Dual timing ─────────────────────────────────────────────────────
        IC("Counterattack", _PC, [{"deploy": 2}, {"if_opp_combat_intrigue": {"swords": 4}}]),
        IC("Backed by CHOAM", _PC, [{"lose_influence_any": 1, "solari": 4},
            {"if_contracts_2": {"swords": 4}}]),
        IC("Contingency Plan", _PC, [{"choose_by_combat": {"combat": {"swords": 3},
                                                           "else": {"solari": 2}}}]),
        IC("Tenuous Bond", _PC, [{"influence_swap": 1},
            {"if_opp_combat_intrigue": {"swords": 4}}]),
        IC("Grasp Arrakis", _CE, [{"if_units_in_conflict": {"swords": 3}},
            {"flip_conflicts_vp": {"count": 2, "vp": 1}}]),
        IC("Crysknife", _PE, [{"flip_conflicts_vp": {"icon": "crysknife", "count": 1, "vp": 1}}]),
        IC("Desert Mouse", _PE, [{"flip_conflicts_vp": {"icon": "desert_mouse", "count": 1, "vp": 1}}]),
        IC("Ornithopter", _PE, [{"flip_conflicts_vp": {"icon": "ornithopter", "count": 1, "vp": 1}}]),
        # ── Endgame ─────────────────────────────────────────────────────────
        IC("CHOAM Profits", _END, [{"if_contracts_4": {"vp": 1}}]),
        IC("Secure Spice Trade", _END, [{"if_tsmf_2": {"vp": 1, "spice": 2}}]),
        IC("Shadow Alliance", _END, [{"if_shadow_alliance": {"vp": 1}}]),
    ]


# ---------------------------------------------------------------------------
# Full game setup
# ---------------------------------------------------------------------------

def setup_game(
    num_players: int = 2,
    seed: int = None,
    use_choam: bool = True,
    leaders=None,
) -> GameState:
    """
    Return a fully initialized GameState with round 1 started and hands dealt.

    use_choam=True enables the CHOAM contract module (Accept Contract /
    Dutiful Service take contracts; harvest / board-space contracts resolve).
    `leaders` is an optional list of leader names (see src/game/leader);
    None → deal distinct random leaders.
    """
    if not (2 <= num_players <= 4):
        raise ValueError("num_players must be 2–4")

    gs = GameState(num_players=num_players, seed=seed, use_choam=use_choam)

    gs.setup_conflict_deck(create_conflict_deck(gs.rng))
    gs.setup_imperium_deck(
        create_imperium_cards(),
        create_reserve_prepare_the_way(),
        create_reserve_spice_must_flow(),
    )
    gs.setup_intrigue_deck(create_intrigue_deck())
    if use_choam:
        from src.game.contract.contract_definitions import create_uprising_contracts
        gs.setup_choam_contracts(create_uprising_contracts())
    gs.setup_player_starting_decks(create_starter_cards())

    # Leaders
    from src.game.leader.leader_definitions import assign_leaders
    assign_leaders(gs, leaders)

    # Begin round 1 (deals 5 cards to each player, flips first conflict)
    gs.start_new_round()

    # All leaders start with 1 water, 0 solari, 0 spice (then leader setup tweaks)
    for player in gs.players:
        player.water  = 1
        player.solari = 0
        player.spice  = 0
    for player in gs.players:
        if getattr(player, "leader", None) is not None:
            player.leader.apply_setup(player, gs)

    return gs
