"""
Agents for Dune Imperium: Uprising.

  RandomAgent       - uniform over non-NO_OP legal actions
  HeuristicAgent    - fast hand-crafted scoring of each legal action (no search)
  GreedyValueAgent  - 1-ply lookahead: clone, apply, evaluate with a ValueModel
                      (falls back to the heuristic state value if no model)

All agents expose `select_action(gs, pid, valid_actions) -> GameAction`.
`temperature > 0` turns the argmax into a softmax sample (for self-play).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

import numpy as np

from src.game.gameState import GameState, GameAction, ActionType, MAX_ROUNDS
from src.ai.features import encode_state
from src.ai.value_model import ValueModel
from src.ai.opening_book import OpeningBook

# rough marginal value of one unit of each resource (VP-equivalent * 10)
# Faction influence is S-tier: at high-level play every alliance eventually
# gets claimed and 1-2 are actively fought over (a player overtaking another
# at the 4-tier is a 2-VP swing, not a 1-VP gain) — priced well above a plain
# resource accordingly, see `_influence_gain_value`.
_RES_VALUE = {
    "vp": 10.0, "solari": 0.55, "spice": 0.65, "water": 0.55,
    "troops": 0.8, "draw": 1.1, "intrigue": 1.3, "persuasion": 0.8,
    "maker_hooks": 1.5, "uplift": 3.0, "spy": 1.2, "spy_special": 1.4,
    "influence_emperor": 2.2, "influence_spacing_guild": 2.2,
    "influence_bene_gesserit": 2.2, "influence_fremen": 2.2,
    "sandworm": 3.0, "sandworm_maker_space": 3.0, "contract": 1.5,
}
_INF_KEYS = ("influence_emperor", "influence_spacing_guild",
             "influence_bene_gesserit", "influence_fremen")
_FACTION_OF = {
    "influence_emperor": "emperor", "influence_spacing_guild": "spacing_guild",
    "influence_bene_gesserit": "bene_gesserit", "influence_fremen": "fremen",
}
_FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")

# Board spaces gated behind 2+ influence with a specific faction — reaching
# friendship with these unlocks real, durable board access (Sietch Tabr for
# combat + hooks + water, Shipping for solari, Imperial Privilege to cycle
# cards), which is exactly why friendships are worth chasing EARLY, not just
# as a step toward an eventual alliance.
_FRIENDSHIP_GATE_BONUS = {"fremen": 2.5, "spacing_guild": 2.0, "emperor": 2.0}


def _faction_access_strength(gs: GameState, pid: int, fac: str) -> float:
    """
    How much of this player's own deck (deck/discard/hand/in_play) actually
    grants access to `fac` — i.e. how cheaply/repeatably they can keep
    sending an Agent there. A player holding Stilgar the Devoted, Chani, and
    Ecological Testing Station can visit Fremen spaces almost every round,
    which makes committing to the Fremen alliance far more realistic than
    for a player with no Fremen-access cards at all — this should drive
    which alliance gets pursued, rather than treating all four uniformly.
    """
    p = gs.players[pid]
    all_cards = list(p.deck) + list(p.discard) + list(p.hand) + list(p.in_play)
    if not all_cards:
        return 0.0
    n = 0
    for c in all_cards:
        for sym in getattr(c, "access_symbols", ()):
            if getattr(sym, "value", sym) == fac:
                n += 1
                break
    return n / max(1, len(all_cards))


def _game_urgency(gs: GameState, pid: int) -> float:
    """
    Scales VP-chasing behavior up as the game nears its likely end. Winning
    is a race: whoever crosses 10 VP first ends the game outright (checked
    right after Makers + Recall each round) and denies everyone else the
    chance to also cross it that round. High-level play treats round 6+ as
    that race, chasing friendships/alliances/combat wins hard through
    whichever avenue the player's build supports — the game should not
    routinely run to round 9-10 for someone who started at 1 VP to reach 10.

    Returns 1.0 in the early game, rising toward ~3.5 by round 9-10 or as
    soon as ANY player (not just this one — a rival closing in matters too,
    since it compresses everyone's remaining window) is closing in on 10 VP.
    """
    round_component = max(0.0, gs.round - 4) / 6.0            # 0 @R4 -> 1.0 @R10
    best_vp = max((q.victory_points for q in gs.players), default=0)
    vp_component = max(0.0, best_vp - 5) / 5.0                 # 0 @<=5 -> 1.0 @10
    return 1.0 + 2.5 * max(round_component, vp_component)


def _influence_gain_value(gs: GameState, pid: int, fac: str, amt: float) -> float:
    """
    Value of adding `amt` influence to one faction track — or, if fac == "any"
    (Dangerous Rhetoric etc.), the best of the four choices, since the player
    picks whichever faction the gain helps most.

    Crossing to friendship (2) or alliance (4) is priced as securing a real VP;
    overtaking a RIVAL who already holds the alliance is priced even higher —
    it's a 2-VP swing (they lose it, you gain it), which is the exact
    "1-2 alliances get fought over" dynamic at high-level play. The
    VP-securing bonuses scale with `_game_urgency` — a friendship/alliance
    that finishes the game this round is worth far more than the same gain
    on round 2 — EXCEPT crossing to friendship also carries a flat,
    urgency-independent bonus on gated factions (Fremen/Spacing Guild/
    Emperor) for the board access it unlocks (Sietch Tabr / Shipping /
    Imperial Privilege), which is valuable throughout the game and
    especially early, since there are more rounds left to use it.

    Alliance pursuit is additionally scaled by `_faction_access_strength` —
    committing to a faction whose spaces the player's own deck lets them
    visit repeatedly is a far more realistic path to 4+ influence than one
    they have no access-card support for, so that archetype should emerge
    from what the player has actually bought, not be priced identically
    across all four factions.
    """
    if fac == "any":
        return max(_influence_gain_value(gs, pid, f, amt) for f in _FACTIONS)
    p = gs.players[pid]
    cur = p.influence.get(fac, 0)
    urgency = _game_urgency(gs, pid)
    access = _faction_access_strength(gs, pid, fac)
    v = amt * _RES_VALUE.get(f"influence_{fac}", 2.2)
    if cur < 2 <= cur + amt:
        v += 7.0 * urgency                            # friendship secured = 1 VP
        v += _FRIENDSHIP_GATE_BONUS.get(fac, 0.0)     # durable board access unlocked
    if cur < 4 <= cur + amt:
        alliance_mult = 1.0 + 1.5 * access
        holder = gs.alliance_holder.get(fac)
        if holder is not None and holder != pid:
            v += 10.0 * urgency * alliance_mult       # overtake a rival's alliance: ~2 VP swing
        else:
            v += 7.5 * urgency * alliance_mult        # first to claim the alliance = 1 VP
    elif cur >= 4 and gs.alliance_holder.get(fac) == pid:
        # already hold it: pushing further defends against a close rival
        if any(q.influence.get(fac, 0) >= cur - 1 for q in gs.players if q.id != pid):
            v += 1.5 * amt * urgency
    return v


# ---------------------------------------------------------------------------

_WALL_BREAK_KEYS = ("break_shield_wall", "destroy_shield_wall", "may_break_shield_wall")
_SANDWORM_KEYS = ("sandworm", "sandworm_maker_space")


def _effect_value(gs: GameState, pid: int, eff: dict) -> float:
    v = 0.0
    for k, amt in eff.items():
        if not isinstance(amt, (int, float)):
            continue
        if k in _FACTION_OF:
            v += _influence_gain_value(gs, pid, _FACTION_OF[k], amt)
        elif k == "influence_any":
            v += _influence_gain_value(gs, pid, "any", amt)
        elif k in _WALL_BREAK_KEYS:
            v += _wall_break_value(gs, pid) * amt
        elif k in _SANDWORM_KEYS:
            v += _sandworm_value(gs, pid) * amt
        else:
            v += _RES_VALUE.get(k, 0.3) * amt
    return v


def _flatten(eff: dict) -> dict:
    """Best-effort flatten of nested conditional/choice sub-effects for scoring."""
    out = {}
    for k, v in eff.items():
        if isinstance(v, dict) and (k.startswith("if_") or k in (
                "pay_then", "recall_spy_then", "discard_then", "choose_by_combat")):
            inner = v.get("combat") or v.get("else") or v
            for kk, vv in (inner.items() if isinstance(inner, dict) else []):
                if isinstance(vv, (int, float)):
                    out[kk] = out.get(kk, 0) + vv
        elif isinstance(v, (int, float)):
            out[k] = out.get(k, 0) + v
    return out


def _cond_weight(gs: GameState, pid: int, key: str) -> float:
    """
    0..1 'how likely/true is this condition for me right now' — used to value
    a card's CONDITIONAL agent/reveal effects at buy time, so a card whose
    only payoff sits behind an alliance/influence threshold the player doesn't
    have isn't priced the same as an unconditional card of the same cost
    (e.g. Junction Headquarters without the Spacing Guild Alliance).
    """
    p = gs.players[pid]
    if key.startswith("alliance_"):
        return 1.0 if p.alliances.get(key[9:]) else 0.10
    if key == "any_alliance":
        return 1.0 if any(p.alliances.values()) else 0.10
    if key.startswith("influence_"):                # influence_fremen_2
        fac, _, n = key[10:].rpartition("_")
        need = int(n) if n.isdigit() else 2
        cur = p.influence.get(fac, 0)
        if cur >= need:
            return 1.0
        return max(0.10, 0.40 - 0.10 * (need - cur))
    if key.startswith("tag_other_"):
        return 0.30                                  # rarely true yet, some upside
    if key.startswith("contracts_"):
        need = int(key[10:]) if key[10:].isdigit() else 2
        return 1.0 if len(p.contracts_completed) >= need else 0.15
    if key.startswith("spies_"):
        need = int(key[6:]) if key[6:].isdigit() else 2
        return 1.0 if sum(p.spies_on_board.values()) >= need else 0.30
    if key.startswith("units_in_conflict_"):
        return 0.30                                  # depends on future combat state
    if key == "fremen_bond":
        return 1.0 if p.influence.get("fremen", 0) >= 2 else 0.30
    if key == "councilor":
        return 1.0 if p.has_councilor else 0.15
    if key == "swordmaster":
        return 1.0 if p.has_swordmaster else 0.15
    return 0.35                                      # unknown / per-turn flag: modest default


def _card_effect_value(gs: GameState, pid: int, eff: dict, w: float = 1.0) -> float:
    """
    Recursively value a card's agent/reveal effect dict, discounting
    conditional sub-effects by how likely their condition is to hold
    (see `_cond_weight`) instead of either counting them at full value or
    silently ignoring them.
    """
    v = 0.0
    for k, sub in eff.items():
        if k.startswith("if_") and isinstance(sub, dict):
            v += _card_effect_value(gs, pid, sub, w * _cond_weight(gs, pid, k[3:]))
        elif k == "pay_then" and isinstance(sub, dict):
            cost = sub.get("cost", {})
            p = gs.players[pid]
            affordable = all(getattr(p, ck, 0) >= cv for ck, cv in cost.items())
            reward = {kk: vv for kk, vv in sub.items() if kk != "cost"}
            v += _card_effect_value(gs, pid, reward, w * (0.85 if affordable else 0.30))
        elif k == "choose_by_combat" and isinstance(sub, dict):
            best = max(_card_effect_value(gs, pid, sub.get("combat", {}), w),
                       _card_effect_value(gs, pid, sub.get("else", {}), w))
            v += 0.6 * best                          # situational — partial credit
        elif k == "choose_by_contracts" and isinstance(sub, dict):
            have = len(gs.players[pid].contracts_completed)
            need = sub.get("n", 4)
            branch = sub.get("yes") if have >= need else sub.get("no")
            if isinstance(branch, dict):
                v += _card_effect_value(gs, pid, branch, w)
        elif k == "persuasion_per_contract" and isinstance(sub, (int, float)):
            n = len(gs.players[pid].contracts_completed)
            v += w * _RES_VALUE.get("persuasion", 0.8) * n * sub
        elif k in ("discard_then", "discard_then_sg", "discard_then_if_sg",
                  "recall_spy_then") and isinstance(sub, dict):
            inner = sub.get("base", sub)
            v += 0.7 * _card_effect_value(gs, pid, inner, w)
        elif k == "trash_intrigue_for" and isinstance(sub, dict):
            has_intrigue = bool(gs.players[pid].intrigue_cards)
            v += _card_effect_value(gs, pid, sub, w * (0.7 if has_intrigue else 0.25))
        elif isinstance(sub, (int, float)):
            if k in _FACTION_OF:
                v += w * _influence_gain_value(gs, pid, _FACTION_OF[k], sub)
            elif k == "influence_any":
                v += w * _influence_gain_value(gs, pid, "any", sub)
            elif k in _WALL_BREAK_KEYS:
                v += w * _wall_break_value(gs, pid) * sub
            elif k in _SANDWORM_KEYS:
                v += w * _sandworm_value(gs, pid) * sub
            else:
                v += w * _RES_VALUE.get(k, 0.3) * sub
        # other nested dicts (unrecognized structure) contribute nothing,
        # same as before — but recognized ones are no longer invisible.
    return v


def _acquire_card_value(gs: GameState, pid: int, card) -> float:
    """
    Shared valuation for buying a card, whether from the Imperium Row
    (ACQUIRE_CARD) or a reserve stack (ACQUIRE_RESERVE — The Spice Must Flow /
    Prepare the Way).  Previously ACQUIRE_RESERVE was a flat score that never
    looked at the card at all, so TSMF's guaranteed 1 VP on acquire (worth
    10.0 in _RES_VALUE) was invisible; and ACQUIRE_CARD ignored every card's
    `acquire_effects` (free Spies, contracts, intrigue, influence, VP...) —
    both are folded in here now.
    """
    s = 1.5 * card.persuasion + 0.9 * card.swords
    for e in (getattr(card, "agent_effects", []) +
              getattr(card, "reveal_effects", []) +
              getattr(card, "acquire_effects", [])):
        s += 0.8 * _card_effect_value(gs, pid, e)
    # Faction-access is a DURABLE asset — but only cheaply usable with a 3rd
    # Agent (Swordmaster): with only 2 Agents, spending one on a faction space
    # is a much bigger opportunity cost against board-space value, so the
    # access is worth much less until you have Swordmaster.
    p = gs.players[pid]
    fac_mult = 1.0 if p.has_swordmaster else 0.35
    for sym in getattr(card, "access_symbols", ()):
        fac = getattr(sym, "value", sym)
        if fac in _FACTION_OF.values():
            cur = p.influence.get(fac, 0)
            s += fac_mult * (1.2 + (2.0 if cur == 1 else 0.0)
                             + (1.5 if cur == 3 else 0.0))
    # NOTE: no flat "expensive = good" bonus — a costly card whose payoff sits
    # behind a condition you don't meet (e.g. Junction Headquarters without
    # the Spacing Guild Alliance) is priced by what it does for you right now.
    s -= 0.15 * card.cost
    # TSMF is a late-game VP dump, not a card that compounds into future turns
    # like a Row card does — rarely worth it before round 4; from then on it
    # competes normally (i.e. only wins out when the Row has nothing better).
    if card.name == "The Spice Must Flow" and gs.round < 4:
        s *= 0.15
    return s


def _intrigue_value(gs: GameState, pid: int, ic) -> float:
    """How good is it to play this Intrigue right now?"""
    p = gs.players[pid]
    in_combat = gs.phase.name == "COMBAT"
    from src.game.intrigue.intrigue import IntrigueTiming
    ts = ic.timing if isinstance(ic.timing, (set, tuple, list, frozenset)) else (ic.timing,)
    v = 0.0
    for e in ic.effects:
        fe = _flatten(e)
        v += _effect_value(gs, pid, fe)
        v += 0.9 * fe.get("swords", 0)          # swords matter in combat
    if IntrigueTiming.COMBAT in ts:
        if in_combat:
            worth = _conflict_worth(gs, pid)
            mine = gs.combat_strength.get(pid, 0)
            opp = max((gs.combat_strength.get(q, 0)
                       for q in range(gs.num_players) if q != pid), default=0)
            # most valuable when a swing would flip the placing
            v += 1.5 * worth if abs(mine - opp) <= 5 else 0.3 * worth
        else:
            v -= 3.0                            # save combat intrigues for combat
    if IntrigueTiming.ENDGAME in ts and not gs.game_over:
        v -= 5.0                                # never waste an endgame card early
    if list(ts) == [IntrigueTiming.PLOT] and not in_combat:
        v -= 0.6                                # small bias to hold plot intrigues
    return v


def _conflict_worth(gs: GameState, pid: int) -> float:
    """
    How valuable is winning the current Conflict for this player. Scaled by
    `_game_urgency` — the same reward is worth much more to chase once the
    game is in its round 6+ scoring race, which is what should make combat
    (worm stomps especially) escalate into the 20+-strength contests seen in
    high-level round 7-10 play instead of staying flat all game.

    Two situational bumps on top of the raw reward:
    - A rival who already has a sandworm in this Conflict would DOUBLE their
      reward by winning it — contesting them (with troops, swords, or your
      own worm) denies that double, so it's worth more than the face reward.
    - Winning would complete a battle-icon match you're already holding half
      of (an immediate, guaranteed 1 VP) — that's worth chasing even when
      the printed reward itself is modest.
    """
    cc = gs.current_conflict
    if cc is None:
        return 0.0
    r = cc.first_place_reward
    v = {1: 0.5, 2: 1.8, 3: 2.8}.get(cc.conflict_level, 1.0)
    v += 3.0 * r.get("vp", 0)
    if r.get("control"):
        v += 1.6
    v += sum(_RES_VALUE.get(k, 0.3) * n for k, n in r.items()
             if isinstance(n, (int, float)))
    # resource -> VP conversions the winner may take (Battle for Imperial Basin
    # = pay 4 spice, Battle for Spice Refinery = pay 6 solari, Battle for
    # Arrakeen = recall 2 Spies).  Worth a near-full VP when the player can
    # already pay right now, a fraction of one when they'd still have to find
    # the resource before combat resolves.  A worm makes each convertible
    # twice (engine `times`) — priced in via the doubled reward when a worm is
    # actually committed, not speculatively here.
    p_conv = gs.players[pid]
    for k, val in r.items():
        if not isinstance(val, dict):
            continue
        if k in ("may_pay_spice_for_vp", "may_pay_solari_for_vp"):
            res = "spice" if "spice" in k else "solari"
            cost, cvp = val.get("cost", 3), val.get("vp", 1)
            v += (2.8 if getattr(p_conv, res, 0) >= cost else 1.0) * cvp
        elif k == "may_pay_troops_for_vp":
            cost, cvp = val.get("cost", 1), val.get("vp", 1)
            have = gs.troops_in_conflict.get(pid, 0)
            v += (2.4 if have >= cost else 1.0) * cvp
        elif k == "may_recall_spies_for_vp":
            cnt, cvp = int(val.get("count", 2)), val.get("vp", 1)
            have = sum(p_conv.spies_on_board.values())
            v += (2.4 if have >= cnt else 0.7) * cvp
    # a controlled-location Conflict the player already holds is worth defending
    if cc.location and gs.controlled_by.get(cc.location) == pid:
        v += 1.0
    if any(gs.sandworms_in_conflict.get(q, 0) > 0
           for q in range(gs.num_players) if q != pid):
        v *= 1.5                                  # deny a rival's worm-doubled reward
    icon = getattr(cc.battle_icon, "value", None)
    if icon and icon != "wild" and icon in gs.players[pid].battle_icons:
        v += 4.0                                  # winning completes a guaranteed 1 VP match
    return v * _game_urgency(gs, pid)


def _wall_break_value(gs: GameState, pid: int) -> float:
    """
    Breaking the Shield Wall is only good for YOU if it unlocks a sandworm on
    a Conflict worth winning that you can actually reach with Maker Hooks —
    otherwise it just as often hands the same worm access to a rival, and
    the more rivals who already have hooks of their own, the worse an
    unprepared wall-break is (you've simply opened the door for them).
    Scaled by `_game_urgency`: a wall-break that opens a worm stomp in round
    7-9 (the deck's shield-wall-detonating Intrigues exist for exactly this)
    is one of the highest-leverage plays in the game and should be taken
    eagerly rather than treated as a minor incidental bonus.
    """
    from src.game.board.board import SHIELD_WALL_PROTECTED
    p = gs.players[pid]
    cc = gs.current_conflict
    if not gs.shield_wall_intact:
        return 0.0
    if (p.has_maker_hooks and cc is not None
            and cc.location in SHIELD_WALL_PROTECTED
            and _conflict_worth(gs, pid) > 1.5):
        return 2.5 * _game_urgency(gs, pid)
    rival_hooks = sum(1 for q in gs.players if q.id != pid and q.has_maker_hooks)
    return -0.5 - 0.4 * rival_hooks


def _sandworm_value(gs: GameState, pid: int) -> float:
    """
    A sandworm in the Conflict doubles the winner's reward (except control /
    battle-icon), so summoning one is only worth what winning THIS Conflict
    is actually worth — not a flat resource value. A worm grabbed when the
    prize is trivial (or there's no live Conflict at all) is a wasted
    maker-space visit better spent taking the spice instead; the same worm
    when the prize is large, or when a rival already has one in there and
    beating them denies their double, is one of the best plays available
    (`_conflict_worth` already prices both of those situations in).
    """
    worth = _conflict_worth(gs, pid)
    if worth <= 0.5:
        return 0.6                                # bank hooks/board-state for later
    return 0.5 * worth


def _spy_post_value(gs: GameState, pid: int, post: Optional[str]) -> float:
    """
    Values placing a Spy at a specific observation post by what it actually
    unlocks — sending an Agent there later via the Spy icon without
    recalling, or eventually Infiltrating (bypass an occupied space) /
    Gathering Intelligence (draw a card) from it. Prefers posts bordering a
    live Conflict's combat space, a faction space for a faction worth
    pursuing, or a space this player is otherwise gated out of — a spy on
    the Landsraad Post (High Council/Swordmaster/Imperial Privilege) is not
    the same asset as one on a post bordering a single minor space.
    """
    if not post:
        return 0.5
    from src.game.board.board import UPRISING_BOARD, OBSERVATION_POST_CONNECTIONS
    spaces = OBSERVATION_POST_CONNECTIONS.get(post, ())
    if not spaces:
        return 0.5
    best = 0.0
    for sp_name in spaces:
        sp = UPRISING_BOARD.get(sp_name)
        if sp is None:
            continue
        v = sum(_effect_value(gs, pid, eff)
                for eff in gs.get_space_effects_preview(sp_name))
        if sp.is_combat_space and gs.current_conflict is not None:
            v += 0.6 * _conflict_worth(gs, pid)
        fac = gs._faction_for_space(sp_name)
        if fac:
            v += 0.5 * _influence_gain_value(gs, pid, fac, 1)
        best = max(best, v)
    # a spy is a durable, reusable asset (repeated Spy-icon sends, or a
    # later Infiltrate/Gather Intelligence) — a floor plus the best thing
    # it currently reaches.
    return 1.0 + 0.5 * best


def _combat_build_strength(gs: GameState, pid: int) -> float:
    """
    Rough measure of how combat-oriented this player's deck actually is —
    sword icons on reveal and troop-granting card effects across everything
    they own (deck/discard/hand/in-play) — so a player who has bought into
    Strike Fleet / Stilgar the Devoted / sword-heavy reveals leans harder
    into contesting Conflicts than one whose garrison just happens to be
    similarly sized this instant.
    """
    p = gs.players[pid]
    all_cards = list(p.deck) + list(p.discard) + list(p.hand) + list(p.in_play)
    if not all_cards:
        return 0.0
    swords = sum(getattr(c, "swords", 0) for c in all_cards)
    troop_cards = sum(
        1 for c in all_cards
        for e in (getattr(c, "agent_effects", []) + getattr(c, "reveal_effects", []))
        if isinstance(e, dict) and "troops" in e
    )
    return (swords + 1.5 * troop_cards) / max(1, len(all_cards))


def heuristic_state_value(gs: GameState, pid: int) -> float:
    """Cheap position eval in ~[0,1] — P(this player is doing well)."""
    p = gs.players[pid]
    others = [q for q in gs.players if q.id != pid]
    my = p.victory_points
    best_opp = max((q.victory_points for q in others), default=0)
    score = 1.5 * (my - best_opp)
    score += 0.10 * (p.solari + p.spice) + 0.08 * p.water
    score += 0.25 * p.troops_garrison + 0.5 * gs.troops_in_conflict.get(pid, 0)
    score += 0.15 * (len(p.deck) + len(p.discard) + len(p.in_play))
    score += 0.9 * sum(min(i, 4) for i in p.influence.values())
    score += 1.2 * p.agents_total + 1.5 * (1 if p.has_councilor else 0)
    score += 1.0 * len(p.intrigue_cards) + 3.0 * len(gs.won_conflicts.get(pid, []))
    return 1.0 / (1.0 + math.exp(-0.06 * score))


# ---------------------------------------------------------------------------

class Agent:
    name = "agent"

    def reset(self) -> None:
        pass

    def select_action(self, gs: GameState, pid: int,
                      valid_actions: List[GameAction]) -> GameAction:
        raise NotImplementedError


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_action(self, gs, pid, valid_actions):
        pool = [a for a in valid_actions if a.action_type != ActionType.NO_OP]
        return self.rng.choice(pool or valid_actions)


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, seed: Optional[int] = None,
                 opening_book: Optional[OpeningBook] = None,
                 temperature: float = 0.0):
        self.rng = random.Random(seed)
        self.book = opening_book if opening_book is not None else OpeningBook.default()
        self.temperature = temperature

    # -- per-action score ----------------------------------------------

    def score(self, gs: GameState, pid: int, a: GameAction) -> float:
        p = gs.players[pid]
        at = a.action_type
        s = 0.0

        if at == ActionType.AGENT_TURN:
            from src.game.board.board import UPRISING_BOARD, SPACE_MANDATORY_COSTS
            sp = UPRISING_BOARD[a.space_name]
            for eff in gs.get_space_effects_preview(a.space_name, a.space_option):
                s += _effect_value(gs, pid, eff)
                if "destroy_shield_wall" in eff:
                    s += _wall_break_value(gs, pid)
            fac = gs._faction_for_space(a.space_name)
            if fac:
                s += _influence_gain_value(gs, pid, fac, 1)
            cost = SPACE_MANDATORY_COSTS.get(a.space_name, {})
            s -= sum(_RES_VALUE.get(k, 0.4) * v for k, v in cost.items())
            # High Council (+2 persuasion every future Reveal turn) and
            # Swordmaster (a 3rd Agent every future round) are durable,
            # compounding structural investments, not one-off payoffs — a
            # flat bonus badly underprices them once other actions (combat,
            # friendships) got urgency-scaled up: measured self-play showed
            # Swordmaster legally available and unclaimed ~4x per player-
            # game on average, but taken only ~10% of those times, because
            # a static +7.0 routinely loses to an inflated late-game combat
            # or friendship score even though grabbing the 3rd Agent by
            # round 3 is worth far more than either. Price by how many
            # rounds remain to actually benefit from it — huge early,
            # negligible in the last round or two.
            remaining_rounds = max(1, MAX_ROUNDS - gs.round)
            if a.space_name == "High Council" and not p.has_councilor:
                s += min(15.0, 1.5 * remaining_rounds)
            if a.space_name == "Swordmaster" and not p.has_swordmaster:
                s += min(22.0, 2.5 * remaining_rounds)
            if sp.is_combat_space and gs.current_conflict is not None:
                s += _conflict_worth(gs, pid) * (1.0 + 0.3 * min(p.troops_garrison, 4)
                                                  + 0.5 * _combat_build_strength(gs, pid))
            if a.space_option == "pay_spice" and p.spice < 3:
                s -= 1.0
            if a.use_gather_intelligence:
                s += 0.8
            s += 4.0 * self.book.bonus(gs, pid, a)
            # discourage wasting the last agent on a weak play
            s += 0.5

        elif at == ActionType.REVEAL_TURN:
            # revealing is fine once agent plays are weak; slight positive base
            s = 1.0 + 0.4 * len(p.hand)

        elif at == ActionType.ACQUIRE_CARD:
            card = next((c for c in gs.imperium_row if c.name == a.acquire_card_name), None)
            if card:
                s += _acquire_card_value(gs, pid, card)
                s += 5.0 * self.book.bonus(gs, pid, a)

        elif at == ActionType.ACQUIRE_RESERVE:
            stack = (gs.reserve_spice_must_flow if a.reserve_type == "spice_must_flow"
                     else gs.reserve_prepare_the_way)
            card = stack[-1] if stack else None
            if card:
                s += _acquire_card_value(gs, pid, card)

        elif at == ActionType.PLAY_INTRIGUE:
            ic = next((c for c in p.intrigue_cards
                       if c.name == a.intrigue_card_name), None)
            if ic:
                s += _intrigue_value(gs, pid, ic)

        elif at == ActionType.END_REVEAL:
            s = -0.5 + (2.0 if gs.persuasion_pool.get(pid, 0) < 2 else -1.0)

        elif at == ActionType.COMBAT_PASS:
            s = 0.0

        elif at == ActionType.RESOLVE_DEPLOY:
            worth = _conflict_worth(gs, pid)           # already urgency-scaled
            urgency = _game_urgency(gs, pid)
            n = a.deploy_count
            my = gs.troops_in_conflict.get(pid, 0) + n
            opp_visible = max((gs.troops_in_conflict.get(q, 0)
                               for q in range(gs.num_players) if q != pid), default=0)
            # Hidden information: opponents still in this Conflict may hold
            # Combat Intrigues (extra swords) we can't see. Pad the assumed
            # opposing strength so the AI doesn't cut margins razor-thin
            # against a field that can swing after we've committed.
            hidden_swords = sum(min(len(q.intrigue_cards), 3)
                                 for q in gs.players if q.id != pid)
            opp = opp_visible + 0.6 * hidden_swords
            # value each committed troop by the reward at stake; bonus for
            # actually pulling ahead of / catching the leader
            s = n * (0.15 + 0.7 * worth)
            if worth > 0.5:
                if my > opp:
                    s += 1.2
                if 0 < opp - my <= 2:
                    s += 0.8 * worth
                # Extra strength beyond the visible leader is insurance
                # against hidden intrigues and a hedge against being sniped,
                # not waste — the slack before it's penalized grows with how
                # urgent (late-game / high-stakes) the race is, and further
                # with the reward itself: overcommitting is fine, even good,
                # for a Conflict that scores multiple VP outright.
                slack = 3 + 2 * (urgency - 1.0) + 0.5 * worth
                if my - opp > slack:
                    s -= 0.5 * (my - opp - slack) / urgency
            s -= 0.35 * n / urgency                     # troops have holding value early,
                                                         # ~nothing once the game's a scoring race
            s -= 0.20 * n * max(0, 3 - gs.round)       # early troops are precious
            if p.troops_garrison - n < 1 and worth < 2.0:
                s -= 1.5                                # don't empty the garrison cheaply

        elif at == ActionType.RESOLVE_TRASH:
            # thin the weak starter cards; keep bought cards
            if a.trash_card_name in ("Reconnaissance", "Diplomacy",
                                     "Dune, the Desert Planet"):
                s = 2.0
            elif a.trash_card_name == "Dagger":
                # Dagger + Signet Ring are the ONLY 2 Landsraad-access cards
                # in the whole 10-card starter deck. Trashing both copies of
                # Dagger (as if it were dead-weight like Reconnaissance)
                # starves the player's own access to Swordmaster/High
                # Council for many rounds — measured self-play showed the
                # average first chance at Swordmaster didn't arrive until
                # round ~6 specifically because of this. Keep at least one
                # around until the structural payoffs it unlocks are secured.
                s = 2.0 if (p.has_swordmaster and p.has_councilor) else -2.0
            elif a.trash_card_name is None:
                s = 0.5
            else:
                s = -1.0

        elif at == ActionType.RESOLVE_INFLUENCE:
            s = _influence_gain_value(gs, pid, a.influence_faction, 1)

        elif at == ActionType.RESOLVE_CONTRACT:
            ct = (gs.contracts_on_board[a.contract_index]
                  if a.contract_index < len(gs.contracts_on_board) else None)
            s = 1.0
            if ct is not None:
                s += sum(_effect_value(gs, pid, {k: v}) for k, v in ct.rewards.items()
                         if isinstance(v, (int, float)))
                if getattr(ct, "contract_type", None) is not None and \
                        ct.contract_type.value == "immediate":
                    s += 1.5

        elif at == ActionType.RESOLVE_SPY:
            s = _spy_post_value(gs, pid, a.spy_post_name)
        elif at == ActionType.RESOLVE_UPLIFT:
            s = 2.0
        elif at == ActionType.RESOLVE_INTRIGUE_TRASH:
            s = 1.0
        elif at == ActionType.RESOLVE_OPTIONAL:
            op = next((o for o in gs.pending_optional_payments
                       if o.player_id == pid), None)
            if op is None:
                s = 0.0
            elif not a.accept_optional:
                s = 0.2                                   # declining is free
            else:
                gain = _effect_value(gs, pid, op.reward)
                cost = sum(_RES_VALUE.get(k, 0.3) * v for k, v in op.cost.items())
                cost += 1.1 * op.discard                  # a discarded card ~ 1 draw
                s = gain - cost
        elif at == ActionType.NO_OP:
            s = -5.0
        return s

    def select_action(self, gs, pid, valid_actions):
        scores = [self.score(gs, pid, a) for a in valid_actions]
        if self.temperature <= 0:
            best = max(range(len(valid_actions)), key=lambda i: scores[i])
            return valid_actions[best]
        return _softmax_pick(valid_actions, scores, self.temperature, self.rng)


# --- crucial-combat detection + exploiter agent ---------------------------

_BULLY_TROOP_SPACES = {"Heighliner", "Sardaukar", "Gather Support",
                       "Research Station", "Deliver Supplies",
                       "Desert Tactics", "Fremkit"}


def _is_crucial_combat(gs: GameState, pid: int) -> bool:
    """A Conflict whose outcome swings the game: a Level III (round 7-10)
    battle, one that scores a VP outright, one that would complete this
    player's battle-icon match (a guaranteed VP), or one a rival already
    has a sandworm in (winning denies their doubled reward)."""
    cc = gs.current_conflict
    if cc is None:
        return False
    r = cc.first_place_reward
    if cc.conflict_level == 3 or r.get("vp", 0) or "control" in r:
        return True
    icon = getattr(cc.battle_icon, "value", None)
    if icon and icon != "wild" and icon in gs.players[pid].battle_icons:
        return True
    return any(gs.sandworms_in_conflict.get(q, 0) > 0
               for q in range(gs.num_players) if q != pid)


class CombatBullyAgent(HeuristicAgent):
    """Plays like the HeuristicAgent EXCEPT it goes all-in to win any *crucial*
    Conflict (`_is_crucial_combat`): floods troops via Heighliner / recruit
    spaces, dumps sword & deploy Intrigues, and deploys its whole garrison
    with no overcommit penalty and a large bonus for out-massing the field.
    A training sparring partner: value agents that under-deploy to maximize
    economy get their crucial combats stolen and lose, so self-play learns
    that maxing strength on the combats that decide games pays off."""

    name = "bully"

    def score(self, gs: GameState, pid: int, a: GameAction) -> float:
        s = super().score(gs, pid, a)
        if not _is_crucial_combat(gs, pid):
            return s
        p = gs.players[pid]
        urg = _game_urgency(gs, pid)
        at = a.action_type
        if at == ActionType.RESOLVE_DEPLOY:
            n = a.deploy_count
            my = gs.troops_in_conflict.get(pid, 0) + n
            opp = max((gs.troops_in_conflict.get(q, 0)
                       for q in range(gs.num_players) if q != pid), default=0)
            # pure "win this decisively" — no holding value, no overcommit cost
            s = n * (0.4 + 1.1 * urg)
            if my > opp:
                s += 3.0
            if my > opp + 3:
                s += 2.0
            if p.troops_garrison - n < 1 and n > 0:
                s += 1.0                              # emptying the garrison is fine here
        elif at == ActionType.AGENT_TURN:
            if a.space_name in _BULLY_TROOP_SPACES:
                s += 3.0 * urg
            for eff in gs.get_space_effects_preview(a.space_name, a.space_option):
                if isinstance(eff, dict) and any(
                        k in eff for k in ("troops", "deploy", "grant_deploy")):
                    s += 1.5 * urg
        elif at == ActionType.PLAY_INTRIGUE:
            ic = next((c for c in p.intrigue_cards
                       if c.name == a.intrigue_card_name), None)
            if ic:
                bonus = 0.0
                for e in ic.effects:
                    fe = _flatten(e)
                    for k in ("swords", "deploy", "grant_deploy"):
                        val = fe.get(k, 0)
                        bonus += val if isinstance(val, (int, float)) else 1
                s += 1.6 * bonus
        elif at == ActionType.COMBAT_PASS:
            s -= 4.0
        return s


class GreedyValueAgent(Agent):
    name = "value"

    def __init__(self, model: Optional[ValueModel] = None,
                 rollout: Optional[HeuristicAgent] = None,
                 branch_cap: int = 14, temperature: float = 0.0,
                 seed: Optional[int] = None,
                 opening_book: Optional[OpeningBook] = None,
                 heuristic_weight: float = 0.35):
        self.model = model
        self.rollout = rollout or HeuristicAgent(seed=seed, opening_book=opening_book)
        self.branch_cap = branch_cap
        self.temperature = temperature
        self.heuristic_weight = heuristic_weight   # anchor lookahead to the heuristic prior
        self.rng = random.Random(seed)
        self.book = self.rollout.book

    def _leaf(self, gs: GameState, pid: int) -> float:
        if gs.game_over:
            return 1.0 if gs.winner == pid else 0.0
        if self.model is not None:
            return self.model.predict(encode_state(gs, pid))
        return heuristic_state_value(gs, pid)

    def select_action(self, gs, pid, valid_actions):
        if self.model is None:
            # No value network yet -> just be the (well-tuned) heuristic.
            return self.rollout.select_action(gs, pid, valid_actions)
        acts = [a for a in valid_actions if a.action_type != ActionType.NO_OP] \
            or valid_actions
        if len(acts) == 1:
            return acts[0]
        # heuristic prior over every candidate (also used to prune)
        hs = [self.rollout.score(gs, pid, a) for a in acts]
        if len(acts) > self.branch_cap:
            keep = sorted(range(len(acts)), key=lambda i: hs[i],
                          reverse=True)[: self.branch_cap]
            acts = [acts[i] for i in keep]
            hs = [hs[i] for i in keep]
        lo, hi = min(hs), max(hs)
        span = (hi - lo) or 1.0
        vals = []
        for a, h in zip(acts, hs):
            g2 = gs.clone()
            try:
                g2.step(a)
                leaf = self._leaf(g2, pid)
            except Exception:
                leaf = -1.0
            h_norm = (h - lo) / span                     # 0..1
            vals.append(leaf
                        + self.heuristic_weight * h_norm
                        + 0.03 * self.book.bonus(gs, pid, a))
        if self.temperature <= 0:
            return acts[max(range(len(acts)), key=lambda i: vals[i])]
        return _softmax_pick(acts, vals, self.temperature, self.rng)


def _softmax_pick(actions, scores, temp, rng):
    m = max(scores)
    exps = [math.exp((s - m) / max(temp, 1e-6)) for s in scores]
    tot = sum(exps)
    r = rng.random() * tot
    acc = 0.0
    for a, e in zip(actions, exps):
        acc += e
        if r <= acc:
            return a
    return actions[-1]


# ---------------------------------------------------------------------------

def make_agent(spec: str, seed: Optional[int] = None,
               opening_book: Optional[OpeningBook] = None) -> Agent:
    """
    spec: 'random' | 'heuristic' | 'heuristic:T<temp>' |
          'bully' | 'bully:T<temp>' (crucial-combat exploiter sparring partner) |
          'value' | 'value:<model.npz>' | 'value:<model.npz>:T<temp>' |
          'value:<model.npz>:T<temp>:HW<heuristic_weight>'

    `HW<x>` overrides GreedyValueAgent's heuristic_weight (default 0.35) —
    how much the 1-ply lookahead's own trained value prediction gets pulled
    back toward the flat heuristic's prior. At the default, the heuristic
    (which has no notion of a discounted/speed-aware target) can swamp a
    value net trained specifically to prefer faster wins; lower it to let
    the net's own signal actually drive action selection.
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "heuristic":
        temp = 0.0
        for p in parts[1:]:
            if p.startswith("T"):
                temp = float(p[1:])
        return HeuristicAgent(seed=seed, opening_book=opening_book, temperature=temp)
    if kind == "bully":
        temp = 0.0
        for p in parts[1:]:
            if p.startswith("T"):
                temp = float(p[1:])
        return CombatBullyAgent(seed=seed, opening_book=opening_book, temperature=temp)
    if kind == "value":
        model, temp, hw = None, 0.0, 0.35
        for p in parts[1:]:
            if p.startswith("T"):
                temp = float(p[1:])
            elif p.startswith("HW"):
                hw = float(p[2:])
            elif p:
                model = ValueModel.load(p)
        return GreedyValueAgent(model=model, temperature=temp, seed=seed,
                                opening_book=opening_book, heuristic_weight=hw)
    raise ValueError(f"Unknown agent spec: {spec!r}")
