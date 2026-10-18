"""Coach hints: the user's gameplay advice as a move bias for every seat.

Like a persona (src/az/personas.py) the bias tilts the move that is played;
search targets stay unbiased and the moves the net would not have picked are
credited by what follows, so the network keeps only the advice that wins.

User (2026-10-14), on what the AlphaZero AI gets wrong:
  * doesn't value the Swordmaster or a High Council seat
  * doesn't buy The Spice Must Flow from round ~6 on
  * spends too much on combat and gives up faction influence it could score
    with influence cards; short on influence -> buy Panopticon (spies, cards,
    a troop every round)
  * late Conflicts (round 7+) only score the base 1-2 VP: it doesn't collect
    the spice / solari / spies the card's "pay X -> VP" needs (e.g. go to
    Imperial Basin for the 4 spice Battle for Imperial Basin asks)
  * over-deploys to win; very heavy on fighting
  * trashes the strong cards it bought (In High Places, Desert Survival)
    instead of Daggers / Dune, the Desert Planet
  * doesn't break the Shield Wall: with Maker Hooks it should (Sietch Tabr,
    Detonation) so worms reach the protected round 7+ VP battles and double
    them, instead of using Detonation to deploy troops
"""
from __future__ import annotations

from typing import List

import numpy as np

FACTION_SPACE = {"Sardaukar": "emperor", "Dutiful Service": "emperor",
                 "Heighliner": "spacing_guild", "Deliver Supplies": "spacing_guild",
                 "Espionage": "bene_gesserit", "Secrets": "bene_gesserit",
                 "Desert Tactics": "fremen", "Fremkit": "fremen"}
SPICE_SPACES = {"Imperial Basin", "Hagga Basin", "Deep Desert"}
# Trash weak starters, never the cards you paid for (user, 2026-10-14: the AI
# was trashing In High Places / Desert Survival and keeping its Daggers).
TRASH_VALUE = {"Dagger": 2.0, "Dune, the Desert Planet": 1.4, "Reconnaissance": 1.0,
               "Prepare the Way": 0.9, "Convincing Argument": 0.4, "Seek Allies": 0.4,
               "Signet Ring": -2.5, "Diplomacy": -2.0}
BOUGHT_CARD_TRASH = -3.0
SOLARI_SPACES = {"Spice Refinery", "Shipping", "Arrakeen"}


def _gives_influence(card) -> bool:
    eff = list(getattr(card, "agent_effects", []) or []) + list(getattr(card, "reveal_effects", []) or [])
    return any("influence" in str(e) for e in eff)


def _conversion_need(gs, pid):
    """('spice' | 'solari', shortfall) if this Conflict's 1st place lets you
    pay for an extra VP and you are short of it; else None."""
    c = gs.current_conflict
    if c is None:
        return None
    r = c.first_place_reward or {}
    p = gs.players[pid]
    for key, res in (("may_pay_spice_for_vp", "spice"), ("may_pay_solari_for_vp", "solari")):
        if key in r:
            short = r[key].get("cost", 0) - getattr(p, res)
            if short > 0:
                return res, short
    return None


def _contending(gs, pid) -> bool:
    tic = gs.troops_in_conflict
    mine = tic.get(pid, 0)
    return mine > 0 and mine >= max((tic.get(q, 0) for q in range(gs.num_players) if q != pid), default=0)


def bias(gs, pid: int, acts: List) -> np.ndarray:
    from src.az.personas import _strength_need
    from src.game.board.board import UPRISING_BOARD
    from src.game.gameState import ActionType as T
    p = gs.players[pid]
    r = gs.round
    low = [f for f, v in p.influence.items() if v <= 1]
    need = _conversion_need(gs, pid) if r >= 6 else None
    cf = gs.current_conflict
    first = (cf.first_place_reward or {}) if cf is not None else {}
    vp_conflict = any(k in first for k in ("vp", "may_pay_spice_for_vp", "may_pay_solari_for_vp",
                                           "may_recall_spies_for_vp"))
    out = np.zeros(len(acts), np.float32)
    deploys = [a.deploy_count for a in acts if a.action_type == T.RESOLVE_DEPLOY]
    want = _strength_need(gs, pid, max(deploys)) if deploys else 0
    sm_cost = 6 if any(q.has_swordmaster for q in gs.players) else 8
    from src.game.board.board import SHIELD_WALL_PROTECTED
    protected_now = gs.shield_wall_intact and cf is not None and cf.location in SHIELD_WALL_PROTECTED
    # Breaking the wall (Sietch Tabr, Detonation, Unexpected Allies, Forbidden
    # Weapons) pays when you have hooks: a lot when this Conflict is walled and
    # worth VP, some when later walled Conflicts (the round 7+ VP battles) are
    # coming - and much more if you are the only one with hooks, since an open
    # wall lets every hooked rival worm in too.
    rivals_hooked = sum(1 for q in gs.players if q.id != pid and q.has_maker_hooks)
    exclusive = (1.5, 1.0, 0.6, 0.4)[min(rivals_hooked, 3)]
    wall_worth = 0.0
    if gs.shield_wall_intact and p.has_maker_hooks:
        if protected_now:
            wall_worth = (1.6 + (1.0 if vp_conflict else 0.0) + 0.3 * (cf.conflict_level or 1)) * exclusive
        elif r >= 4:
            wall_worth = (0.6 + 0.3 * (r - 4)) * exclusive
    pc = gs.pending_choice_for(pid) if hasattr(gs, "pending_choice_for") else None
    for i, a in enumerate(acts):
        t, b = a.action_type, 0.0
        if t.name == "RESOLVE_CHOICE" and a.choice and a.choice.startswith(("recruit:", "techbuy:")):
            import dataclasses as _dc
            a = _dc.replace(a, action_type=T.RESOLVE_BL_CHOICE,
                            choice=a.choice[8:] if a.choice.startswith("recruit:") else "stack:" + a.choice[8:])
            t = a.action_type
        if t == T.AGENT_TURN and a.space_name:
            s = a.space_name
            # Swordmaster by round 6 at the latest (user): save solari for it
            saving = not p.has_swordmaster and r <= 6 and p.solari < sm_cost
            if s == "Swordmaster" and not p.has_swordmaster:
                b += 3.0 if r <= 6 else 1.0
            elif s == "High Council" and not p.has_councilor:
                # Swordmaster first: don't spend its solari on the seat
                b += (-0.6 if saving and p.solari - 5 < sm_cost else 1.2) if r <= 7 else 0.4
            if saving:
                if s == "Shipping" and p.spice >= 3:
                    b += 1.4                              # 3 spice -> 5 solari
                elif s == "Spice Refinery" and p.spice >= 1:
                    b += 1.1                              # 1 spice -> 4 solari
                elif s in SOLARI_SPACES:
                    b += 0.8
                elif s in SPICE_SPACES and p.spice < 3 and p.solari + 4 < sm_cost:
                    b += 0.6                              # spice to sell for solari
                elif s == "Gather Support":
                    b -= 0.8                              # costs the Swordmaster solari
            f = FACTION_SPACE.get(s)
            if f and p.influence.get(f, 0) in (1, 3):
                b += 0.8                                  # one step from 2 VP / alliance
            elif f and p.influence.get(f, 0) == 0 and r <= 5:
                b += 0.4                                  # 2 on every track by round 5
            if need and ((need[0] == "spice" and s in SPICE_SPACES)
                         or (need[0] == "solari" and s in SOLARI_SPACES)) and _contending(gs, pid):
                b += 1.0                                  # collect what the Conflict's VP costs
            sp = UPRISING_BOARD.get(s)
            if sp is not None and sp.is_combat_space and not f and r == 1:
                b -= 0.4                                  # don't over-fight the opening
            if r >= 6 and sp is not None and sp.is_combat_space and vp_conflict:
                b += 0.5                                  # late: fight where the card has VP
            if s == "Sietch Tabr":
                if a.space_option == "shield_wall":
                    b += wall_worth
                elif a.space_option == "hooks":
                    # hooks you lack are worth taking; a second time it's only a
                    # troop + water, so open the wall instead
                    b += 0.5 if not p.has_maker_hooks else -0.6 * (wall_worth > 0)
            if a.space_option == "sandworm":
                b += 1.2 if vp_conflict else 0.5          # a worm doubles the Conflict reward
        elif t == T.ACQUIRE_RESERVE and a.reserve_type == "spice_must_flow" and r >= 5:
            b += 1.5 if r < 7 else 2.0                    # a VP now beats a card that pays later
        elif t == T.ACQUIRE_CARD and low:
            c = next((x for x in gs.imperium_row if x.name == a.acquire_card_name), None)
            if c is not None and _gives_influence(c):
                b += 0.6
        elif t == T.RESOLVE_INFLUENCE and a.influence_faction:
            if p.influence.get(a.influence_faction, 0) in (1, 3):
                b += 0.8
        elif t == T.RESOLVE_BL_CHOICE and gs.bl and a.choice and a.choice.startswith("stack:"):
            _d = gs.bl.tech_at(a.player_id, int(a.choice.split(":")[1]))[0]
            st = [_d] if _d is not None else []
            if st and st[0].name == "Panopticon" and len(low) >= 2:
                b += 1.5
            elif st and st[0].name == "Forbidden Weapons" and wall_worth:
                b += 0.6 * wall_worth                     # its acquire may break the wall
        elif t == T.RESOLVE_DEPLOY:
            b -= 0.4 * abs(a.deploy_count - want)        # just enough to lead
        elif t == T.RESOLVE_BL_CHOICE and a.choice and a.choice.endswith(":Loyal"):
            # Loyal needs 3+ Emperor influence; far from it, it's a wasted skill
            e = p.influence.get("emperor", 0)
            b += 0.0 if e >= 3 else (-0.5 if e == 2 else -1.5)
        elif t == T.RESOLVE_CHOICE and pc is not None and wall_worth:
            eff = next((e for k, _l, e in pc.options if k == a.choice), None)
            if eff is not None and "break_shield_wall" in str(eff):
                b += wall_worth                           # Detonation: open the wall, don't just deploy
        elif t == T.PLAY_INTRIGUE and a.intrigue_card_name == "Unexpected Allies" and gs.shield_wall_intact:
            # its own worm needs no hooks: big when this walled Conflict is worth VP
            b += (1.5 + (1.0 if vp_conflict else 0.0)) if protected_now and _contending(gs, pid) else 0.0
            b += 0.5 * wall_worth
        elif (t == T.RESOLVE_BL_CHOICE and a.choice and (a.choice.startswith("board:") or a.choice == "supply")
              and not p.has_swordmaster and r <= 6 and p.solari - 2 < sm_cost):
            b -= 0.8                                      # a commander would spend the Swordmaster solari
        elif t == T.RESOLVE_TRASH and a.trash_card_name:
            n = a.trash_card_name
            if n == "The Spice Must Flow":
                # its VP is kept once acquired; keep 2 only with Secure Spice Trade
                sst = any(ic.name == "Secure Spice Trade" for ic in p.intrigue_cards)
                b += -1.5 if sst else 1.0
            else:
                b += TRASH_VALUE.get(n, BOUGHT_CARD_TRASH)
        elif (t == T.RESOLVE_OPTIONAL and a.accept_optional and not p.has_swordmaster and r <= 6
              and any(getattr(op, "player_id", pid) == pid and (op.cost or {}).get("solari", 0) > 0
                      and getattr(op, "label", "") != "conflict_vp"
                      for op in gs.pending_optional_payments)):
            b -= 0.6                                      # keep the solari for the Swordmaster
        elif (t == T.RESOLVE_OPTIONAL and a.accept_optional and any(
                getattr(op, "player_id", pid) == pid and "break_shield_wall" in str(getattr(op, "reward", ""))
                for op in gs.pending_optional_payments)):
            b += 1.0 + wall_worth                         # e.g. Unexpected Allies: pay the water
        elif t == T.RESOLVE_OPTIONAL and a.accept_optional and any(
                getattr(op, "label", "") == "conflict_vp" and getattr(op, "player_id", pid) == pid
                for op in gs.pending_optional_payments):
            b += 0.5                                      # take the Conflict's extra VP
        out[i] = b
    return out
