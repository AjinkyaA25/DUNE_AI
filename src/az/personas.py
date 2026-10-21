"""Playstyle personas for self-play: each seat leans toward one plan.

A persona adds a bias (in logit units) to some moves. Self-play plays the
biased choice, but the policy targets recorded from search stay unbiased,
and every game's result trains the value head, so the network learns which
of these plans actually wins rather than copying the lean.

  sword     - get the Swordmaster (3rd agent) early; collect solari for it
  deck      - build persuasion: buy expensive cards, trash Daggers and
              weak starters, take the High Council seat (+2 persuasion)
  resource  - collect spice / solari / water and turn them into troops
              (Heighliner, Sardaukar, Research Station); deploy just
              enough to win the Conflict, not more
  faction   - influence: faction spaces, cards and choices that move the
              tracks toward 2 / 4 influence and alliances
  smf       - High Council + strong cards + The Spice Must Flow: takes the
              High Council seat and Assembly Hall for persuasion, buys the
              strongest cards it can (persuasion / draw first), draws cards,
              thins Daggers, and buys TSMF over other cards
  commanders- actively seeks Sardaukar commanders: commander spaces, recruits
              (board and supply), Sardaukar High Command, solari to pay for them
  (faction  - also chases the expensive influence cards - Interstellar Trade,
              Overthrow, Steersman, ... - by drawing cards and going to
              Assembly Hall / High Council for the persuasion to buy them)

The set used in self-play comes from AZ_PERSONAS (comma list; src.az.train
--personas), default the first four.
"""
from __future__ import annotations

import os
import re
from typing import List

import numpy as np

_INFL = re.compile(r"'(influence_(?:any|choice|swap|faction_visited|emperor|spacing_guild|"
                   r"bene_gesserit|fremen)|uplift)'")
DRAW_SPACES = {"Arrakeen", "Research Station", "Fremkit", "Espionage", "Imperial Privilege",
               "Accept Contract"}
PERSUASION_SPACES = {"Assembly Hall"}          # +1 persuasion on your Reveal turn


def influence_card(card) -> bool:
    """A card that moves you up a faction track (not just one that checks it)."""
    e = str(getattr(card, "agent_effects", "")) + str(getattr(card, "reveal_effects", "")) \
        + str(getattr(card, "acquire_effects", ""))
    return bool(_INFL.search(e))


def draws(card) -> bool:
    return "'draw'" in str(getattr(card, "agent_effects", "")) + str(getattr(card, "reveal_effects", ""))

PERSONAS = tuple(os.environ.get("AZ_PERSONAS", "sword,deck,resource,faction").split(","))

FACTION_SPACES = {"Sardaukar", "Dutiful Service", "Heighliner", "Deliver Supplies",
                  "Espionage", "Secrets", "Desert Tactics", "Fremkit"}
SOLARI_SPACES = {"Spice Refinery", "Shipping", "Arrakeen"}
RESOURCE_SPACES = {"Spice Refinery", "Shipping", "Imperial Basin", "Hagga Basin", "Deep Desert",
                   "Sietch Tabr", "Deliver Supplies", "Gather Support"}
TROOP_SPACES = {"Heighliner", "Sardaukar", "Research Station", "Arrakeen", "Desert Tactics"}
TRASH_FIRST = {"Dagger": 2.0, "Dune, the Desert Planet": 0.8, "Reconnaissance": 0.6,
               "Seek Allies": 0.4, "Prepare the Way": 0.6}
FACTION_KEYS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")


def _row_card(gs, name):
    return next((c for c in gs.imperium_row if c.name == name), None)


def _strength_need(gs, pid, avail):
    """Troops to deploy to lead the Conflict by one troop (units on board only)."""
    tic = gs.troops_in_conflict
    worms = getattr(gs, "sandworms_in_conflict", {})
    mine = 2 * tic.get(pid, 0) + 3 * worms.get(pid, 0)
    opp = max((2 * tic.get(q, 0) + 3 * worms.get(q, 0) for q in range(gs.num_players) if q != pid),
              default=0)
    return int(np.clip((opp - mine) // 2 + 1, 0, avail))


def bias(gs, pid: int, acts: List, persona: str) -> np.ndarray:
    from src.game.gameState import ActionType as T
    p = gs.players[pid]
    out = np.zeros(len(acts), np.float32)
    for i, a in enumerate(acts):
        t = a.action_type
        if t.name == "RESOLVE_CHOICE" and a.choice and a.choice.startswith(("recruit:", "techbuy:")):
            import dataclasses as _dc
            a = _dc.replace(a, action_type=T.RESOLVE_BL_CHOICE,
                            choice=a.choice[8:] if a.choice.startswith("recruit:") else "stack:" + a.choice[8:])
            t = a.action_type
        b = 0.0
        if persona == "sword":
            if not p.has_swordmaster:
                if t == T.AGENT_TURN and a.space_name == "Swordmaster":
                    b += 3.0
                elif t == T.PAY_SWORDMASTER:
                    b += 3.0
                elif t == T.AGENT_TURN and a.space_name in SOLARI_SPACES:
                    b += 1.0                      # the 8 solari
                elif t == T.AGENT_TURN and a.space_name == "High Council":
                    b -= 0.8                      # save the solari
                elif t == T.RESOLVE_BL_CHOICE and a.choice and \
                        (a.choice.startswith("board:") or a.choice == "supply"):
                    b -= 0.8                      # a commander costs solari too
                elif t == T.ACQUIRE_CARD:
                    c = _row_card(gs, a.acquire_card_name)
                    if c is not None and "solari" in str(c.agent_effects) + str(c.reveal_effects):
                        b += 0.5
        elif persona == "deck":
            if t == T.ACQUIRE_CARD:
                c = _row_card(gs, a.acquire_card_name)
                if c is not None:
                    pers, dr = getattr(c, "persuasion", 0), draws(c)
                    b += 0.1 * c.cost + 0.4 * pers + (0.6 if dr else 0.0) - (1.0 if pers == 0 and not dr else 0.0)
            elif t == T.ACQUIRE_RESERVE and a.reserve_type == "prepare_the_way":
                b -= 0.4
            elif t == T.RESOLVE_TRASH:
                b += TRASH_FIRST.get(a.trash_card_name, -0.8 if a.trash_card_name is None else 0.0)
            elif t == T.AGENT_TURN and a.space_name == "High Council" and not p.has_councilor:
                b += 1.0
            elif t == T.AGENT_TURN and a.space_name == "Desert Tactics":
                b += 0.5                          # trash a card
        elif persona == "resource":
            # value of this Conflict for 1st place: VP on the card counts most
            c = gs.current_conflict
            first = (c.first_place_reward or {}) if c is not None else {}
            vp_fight = ("vp" in first or "may_pay_spice_for_vp" in first
                        or "may_pay_solari_for_vp" in first or "may_recall_spies_for_vp" in first)
            if t == T.AGENT_TURN and a.space_name in RESOURCE_SPACES:
                b += 0.6
            elif t == T.AGENT_TURN and a.space_name in TROOP_SPACES:
                b += 0.5 + (0.6 if vp_fight else 0.0)
            elif t == T.AGENT_TURN and a.space_name in ("Spice Refinery", "Arrakeen", "Imperial Basin") \
                    and vp_fight:
                b += 0.4
            elif t == T.RESOLVE_DEPLOY:
                avail = max(x.deploy_count for x in acts if x.action_type == T.RESOLVE_DEPLOY)
                b -= 0.4 * abs(a.deploy_count - _strength_need(gs, pid, avail))
        elif persona == "faction":
            # an influence card in the Row too expensive to buy yet: go get persuasion
            big = [c for c in gs.imperium_row if influence_card(c) and c.cost >= 6]
            if t == T.AGENT_TURN and a.space_name in FACTION_SPACES:
                b += 0.8
            if t == T.AGENT_TURN and big:
                if a.space_name in DRAW_SPACES:
                    b += 0.6
                elif a.space_name in PERSUASION_SPACES:
                    b += 0.8
                elif a.space_name == "High Council" and not p.has_councilor:
                    b += 0.8
            elif t == T.RESOLVE_INFLUENCE and a.influence_faction:
                lvl = p.influence.get(a.influence_faction, 0)
                b += 0.6 if lvl in (1, 3) else 0.2   # reaching 2 (VP) / 4 (alliance race)
            elif t == T.ACQUIRE_CARD:
                c = _row_card(gs, a.acquire_card_name)
                if c is not None and influence_card(c):
                    b += 1.0 + (0.5 if c.cost >= 6 else 0.0)
                elif c is not None and draws(c):
                    b += 0.3
        elif persona == "smf":
            if t == T.ACQUIRE_RESERVE and a.reserve_type == "spice_must_flow":
                b += 3.0
            elif t == T.ACQUIRE_RESERVE and a.reserve_type == "prepare_the_way" and gs.round >= 4:
                b -= 0.6
            elif t == T.ACQUIRE_CARD:
                c = _row_card(gs, a.acquire_card_name)
                if c is not None:
                    # persuasion and draw are what reach 9 for TSMF; a card with
                    # neither (0 persuasion on reveal, no draw) works against it
                    pers, dr = getattr(c, "persuasion", 0), draws(c)
                    # strong cards when it can afford them (cost), weighted to
                    # persuasion and draw, which is what reaches 9 for TSMF
                    b += 0.12 * c.cost + 0.5 * pers + (0.8 if dr else 0.0)                         - (1.2 if pers == 0 and not dr else 0.0)
            elif t == T.AGENT_TURN and a.space_name == "High Council" and not p.has_councilor:
                b += 2.0                          # +2 persuasion every Reveal
            elif t == T.AGENT_TURN and a.space_name in DRAW_SPACES:
                b += 0.6
            elif t == T.AGENT_TURN and a.space_name in PERSUASION_SPACES:
                b += 0.7
            elif t == T.RESOLVE_TRASH and a.trash_card_name == "Dagger":
                b += 1.0
        elif persona == "commanders":
            from src.game.bloodlines.rules import COMMANDER_SPACES
            bl = gs.bl
            cost = bl.commander_cost(pid) if bl else 99
            if t == T.RESOLVE_CHOICE and a.choice == "recruit:supply":
                b += 2.0                          # re-recruit on the Reveal turn
            elif t == T.RESOLVE_BL_CHOICE and a.choice:
                if a.choice.startswith("board:") or a.choice == "supply":
                    b += 2.0
                elif a.choice.startswith("stack:"):
                    _d = bl.tech_at(a.player_id, int(a.choice.split(":")[1]))[0]
                    st = [_d] if _d is not None else []
                    if st and st[0].name in ("Sardaukar High Command", "Plasteel Blades"):
                        b += 1.5
            elif t == T.AGENT_TURN and a.space_name in COMMANDER_SPACES:
                b += 1.0 if p.solari >= cost else 0.3
            elif t == T.AGENT_TURN and a.space_name in SOLARI_SPACES and p.solari < cost:
                b += 0.7
            elif t == T.RESOLVE_DEPLOY and a.deploy_cmd:
                b += 0.3                          # commanders fight and keep skills on
        out[i] = b
    return out


def assign(rng) -> List[str]:
    """Personas for seats 0-3 this game (a random order, one of each)."""
    return [PERSONAS[i] for i in rng.permutation(len(PERSONAS))]
