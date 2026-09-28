"""
Bloodlines Imperium and Intrigue cards, transcribed from the TTS mod's card
art (Workshop 3522149839). Added to the game when use_bloodlines=True.

Several Bloodlines intrigues were already in the base intrigue deck
(Adaptive Tactics, Coercive Negotiation, Desert Support, Emperor's
Invitation, False Orders, Grasp Arrakis, Insider Information, Return the
Favor, Ripples in the Sand, Sleeper Unit, Tenuous Bond); only the missing ones
are here.

Effects use the engine vocabulary (effects.py) plus Bloodlines keys handled
in bloodlines/rules.py:
  command            reveal: fires once your reveal persuasion reaches 6
  bl_choose          {"options": [...], "both_if": cond} - pick one option
  bl_trash_card      name - trash that card from play (an optional
                     "trash this card ->" inside a bl_choose option)
  if_bl_<cond>       commander_in_conflict, gained_spice_2,
                     completed_contract, techs_2, techs_3, garrison_4,
                     other_bene, endgame_techs_3, endgame_water_3, not_endgame
  bl_after_turn      resolve at the end of this Agent turn (after contracts)
  bl_* (misc)        see Bloodlines.resolve_effect

Icon legend used when reading the cards: helmet = Emperor, red infinity =
Spacing Guild, purple = Bene Gesserit, blue target = Fremen, green pentagon =
Landsraad, blue circle = City, orange triangle = Spice Trade (desert),
eye = Spy, orange token = CHOAM contract, crossed blades = deploy,
black "?" diamond = influence with any faction.
"""
from __future__ import annotations

from typing import List

from src.data.card_definitions import _make, _PLOT, _COMBAT
from src.game.cards.card import CardType
from src.game.intrigue.intrigue import IntrigueCard, IntrigueTiming

I = CardType.IMPERIUM
ALL_ICONS = ["emperor", "spacing_guild", "bene_gesserit", "fremen",
             "landsraad", "city", "desert"]
INFLUENCE_ALL = {"influence_emperor": 1, "influence_spacing_guild": 1,
                 "influence_bene_gesserit": 1, "influence_fremen": 1}


def _may(name: str, **reward) -> dict:
    """'Trash this card -> reward' as an optional choice."""
    return {"bl_choose": {"options": [{}, {"bl_trash_card": name, **reward}]}}


def create_bloodlines_imperium_cards() -> List:
    m = _make
    cards = [
        m("Arrakis Observer", 3, I, access=["city", "desert"], tags=["spacing_guild"],
          agent={"bl_discard_for": {"reward": {"spy": 1}, "tag": "spacing_guild",
                                    "bonus": {"spice": 2}}},
          persuasion=1, reveal={"recall_spy_then": {"swords": 3}},
          notes="Agent: MAY discard a card -> place a Spy; +2 spice if it was a "
                "Spacing Guild card. Reveal: 1 persuasion AND MAY recall a Spy "
                "-> 3 swords."),
        m("Bombast", 1, I, access=["landsraad"], tags=["emperor"], persuasion=1,
          reveal={"command": {"solari": 3, "bl_trash_card": "Bombast"}},
          notes="Command: 3 solari and trash this card."),
        m("CHOAM Demands", 6, I, access=["landsraad", "city", "desert"],
          tags=["spacing_guild"], agent={"bl_complete_contract": 1},
          reveal={"if_contracts_4": _may("CHOAM Demands", **INFLUENCE_ALL)},
          notes="Agent: complete one of your contracts. Reveal: with 4+ "
                "completed contracts, MAY trash this -> +1 influence with each "
                "faction."),
        m("Command Center", 3, I, access=["emperor", "city"], tags=["emperor"],
          agent={"if_influence_emperor_2": {"troops": 1}}, persuasion=1,
          reveal={"bl_choose": {"options": [{}, {"bl_retreat": 2, "persuasion": 2}]}},
          notes="Reveal: MAY retreat two troops -> +2 persuasion."),
        m("Corrupt Bureaucrat", 4, I, access=["spacing_guild", "landsraad", "spy"],
          tags=["spacing_guild"], agent={"if_spy_recalled": {"contract": 1}},
          persuasion=2,
          notes="When this card is discarded: 3 solari (Card.bl_on_discard)."),
        m("Delivery Logistics", 2, I, tags=["spacing_guild"],
          reveal={"bl_choose": {"options": [{"persuasion": 1}, {"contract": 1}]}},
          notes="Has the Agent icons of all your incomplete contracts "
                "(GameState.can_send_agent)."),
        m("Disruption Tactics", 2, I, access=["fremen", "desert"], tags=["fremen"],
          agent={"bl_force_retreat": 1}, persuasion=1,
          reveal=_may("Disruption Tactics", grant_deploy=0),
          notes="Agent: force an enemy troop to retreat (auto-targets the "
                "strongest opponent). Reveal: MAY trash this -> deploy."),
        m("Eliminate Allies", 2, I, access=["spy"], tags=["emperor"],
          agent={"trash": 1}, persuasion=1, swords=1, trash={"troops": 2},
          notes="When this card is trashed: 2 troops."),
        m("Elite Forces", 3, I, access=["emperor", "spacing_guild"],
          tags=["emperor", "spacing_guild"],
          agent={"bl_trash_from_hand": {"tag": "emperor",
                                        "bonus": {"intrigue": 1, "troops": 1,
                                                  "grant_deploy": 0}}},
          persuasion=1, swords=1,
          notes="Agent: MAY trash a card from hand; if it's an Emperor card: "
                "Intrigue + troop + deploy."),
        m("Engineered Miracle", 3, I, access=["fremen", "desert"],
          tags=["bene_gesserit"], agent={"discard_then": {"water": 1}},
          persuasion=1,
          reveal={"command": _may("Engineered Miracle",
                                  acquire_free={"max_cost": 99})},
          notes="Command: MAY trash this -> acquire a card from the Imperium Row."),
        m("Fremen War Name", 4, I, access=["fremen", "desert"], tags=["fremen"],
          agent={"if_bl_gained_spice_2": {"troops": 1, "draw": 1}}, persuasion=2,
          reveal={"if_fremen_bond": {"swords": 2}}),
        m("Holy War", 5, I, access=["emperor", "spacing_guild", "bene_gesserit",
                                    "landsraad"], tags=["fremen"],
          agent={"bl_opponents_lose_troop": 1}, persuasion=1,
          reveal={"troops": 1, "if_fremen_bond": {"grant_deploy": 0}},
          notes="Agent: each opponent loses one troop. NOT modelled: each "
                "opponent spying on that space must move the Spy."),
        m("I Believe", 3, I, access=["fremen", "city"], tags=["fremen"],
          agent={"discard_then": {"draw": 1}}, persuasion=1,
          reveal={"command": {"troops": 2}}),
        m("Imperial Throneship", 7, I, access=ALL_ICONS, tags=["emperor"],
          acquire={"influence_emperor": 1}, agent={"intrigue": 1}, persuasion=2,
          reveal={"if_bl_garrison_4": {"persuasion": 1, "solari": 3}},
          notes="Reveal: with 4+ garrisoned units, +1 persuasion and 3 solari."),
        m("Intelligence Training", 3, I, access=["landsraad", "city"],
          tags=["emperor"], acquire={"spy": 1}, persuasion=1, swords=1,
          reveal={"command": {"spy": 1}}),
        m("Ixian Ambassador", 4, I, access=["landsraad"], agent={"spice": 1},
          persuasion=1, reveal={"if_bl_techs_2": {"influence_any": 1}}),
        m("Litany Against Fear", 3, I, tags=["bene_gesserit"], persuasion=2,
          notes="At the start of your turn: MAY put this into play -> draw a "
                "card and pass your turn (ACTIVATE_TECH 'card:Litany Against "
                "Fear')."),
        m("Mercantile Affairs", 5, I, access=["bene_gesserit", "city", "desert", "spy"],
          tags=["bene_gesserit"], acquire={"contract": 1},
          agent={"bl_after_turn": {"if_bl_completed_contract": {"intrigue": 1}}},
          persuasion=2,
          notes="Agent: if you completed a contract this turn, an Intrigue "
                "(checked at the end of the turn, after contract completion)."),
        m("Pivotal Gambit", 3, I, access=["fremen", "city"], tags=["fremen"],
          agent=_may("Pivotal Gambit", troops=1,
                     bl_conflict_bonus={"influence_any": 1}),
          persuasion=1, swords=2,
          notes="Agent: MAY trash this -> troop AND add 'influence any' to the "
                "first place reward of this Conflict."),
        m("Pointing the Way", 6, I, access=["fremen", "city", "desert"],
          tags=["fremen"], agent={"if_sandworm_in_conflict": {"intrigue": 1}},
          persuasion=1, swords=2, reveal={"command": {"influence_any": 1}}),
        m("Possible Futures", 8, I, access=["landsraad", "city", "desert"],
          tags=["bene_gesserit", "fremen"], acquire={"water": 1},
          agent={"bl_choose": {"options": [{"influence_any": 1}, {"troops": 2}],
                               "both_if": "other_bene"}},
          persuasion=2, reveal={"water": 1},
          notes="Agent: influence OR 2 troops; both with another Bene "
                "Gesserit card in play."),
        m("Quash Rebellion", 5, I, access=["emperor", "spacing_guild", "landsraad"],
          tags=["emperor"], agent={"solari": 2}, swords=2,
          reveal={"if_bl_commander_in_conflict": {"persuasion": 2}}),
        m("Ruthless Leadership", 4, I, access=["city", "desert"], tags=["emperor"],
          agent={"if_bl_commander_in_conflict": {"trash": 2}}, persuasion=1,
          swords=1, reveal={"command": {"grant_deploy": 0}}),
        m("Sandwalk", 1, I, access=["desert"], tags=["fremen"],
          agent={"if_bl_gained_spice_2": {"draw": 1}}, persuasion=1, swords=1,
          reveal={"if_fremen_bond": {"persuasion": 1}}),
        m("Sardaukar Standard", 4, I, access=["emperor", "city"], tags=["emperor"],
          persuasion=2, reveal={"troops": 1}, trash={"bl_free_commander": 1},
          notes="When trashed: acquire and recruit a Sardaukar Commander from "
                "the bank (no skill)."),
        m("Shrouded Counsel", 4, I, access=["spy"], tags=["bene_gesserit"],
          agent={"intrigue": 1}, persuasion=1, reveal={"command": {"trash": 1}}),
        m("Southern Faith", 5, I, access=["fremen", "city"],
          tags=["bene_gesserit", "fremen"],
          agent={"bl_choose": {"options": [
              {"draw": 1}, {"if_bl_other_bene": {"influence_bene_gesserit": 1}}]}},
          persuasion=1, swords=2, reveal={"command": {"spice": 2}}),
        m("Urgent Shigawire", 2, I, access=["bene_gesserit", "city"],
          tags=["bene_gesserit"], agent={"bl_shigawire": 1}, persuasion=1,
          notes="Agent: the next Bene Gesserit card you play this round has all "
                "Agent icons and adds 'draw a card' to its Agent box."),
    ]
    for c in cards:
        if c.name == "Corrupt Bureaucrat":
            c.bl_on_discard = {"solari": 3}
    return cards


_CE = (IntrigueTiming.COMBAT, IntrigueTiming.ENDGAME)
_PE = (IntrigueTiming.PLOT, IntrigueTiming.ENDGAME)


def create_bloodlines_intrigues() -> List[IntrigueCard]:
    IC = IntrigueCard
    return [
        IC("Battlefield Research", _CE, [
            {"choose_by_combat": {"combat": {"retreat_for": {
                "min": 1, "max": 2, "reward": {"bl_tech_offer": 1}}}, "else": {}}},
            {"if_bl_endgame_techs_3": {"vp": 1}}]),
        IC("Honor Guard", _PLOT, [{"troops": 1, "bl_commander_discount": 1}]),
        IC("Rapid Engineering", _PLOT, [{"bl_choose": {"options": [
            {"discard_then": {"bl_tech_offer": 1}},
            {"if_bl_techs_3": {"influence_any": 2}}]}}]),
        IC("Sacred Pools", _PE, [
            {"if_bl_not_endgame": {"discard_then": {"water": 1}}},
            {"if_bl_endgame_water_3": {"vp": 1}}]),
        IC("Seize Production", _PLOT, [{"bl_choose": {"options": [
            {"solari": 2}, {"if_bl_commander_in_conflict": {"spice": 2}}]}}]),
        IC("The Strong Survive", _COMBAT, [{"bl_choose": {"options": [
            {"swords": 3}, {"bl_retreat": 1, "trash": 1}]}}]),
        IC("Withdrawal Agreement", _COMBAT, [{"bl_choose": {"options": [
            {}, {"bl_retreat": 3, "influence_any": 1}]}}]),
    ]
