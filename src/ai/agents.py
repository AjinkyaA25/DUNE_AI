"""
Agents for Dune Imperium: Uprising.

  RandomAgent       - uniform over non-NO_OP legal actions
  HeuristicAgent    - fast hand-crafted scoring of each legal action (no search)
  GreedyValueAgent  - 1-ply lookahead: clone, apply, evaluate with a ValueModel
                      (falls back to the heuristic state value if no model)
  ISMCTSAgent       - multi-ply Information-Set MCTS: determinizes hidden info
                      per simulation, searches only my own decision nodes,
                      rolls opponents out with the heuristic (see determinize.py)

All agents expose `select_action(gs, pid, valid_actions) -> GameAction`.
`temperature > 0` turns the argmax into a softmax sample (for self-play).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.game.gameState import GameState, GameAction, ActionType, MAX_ROUNDS, INFLUENCE_4_BONUS
from src.ai.features import encode_state
from src.ai.value_model import ValueModel
from src.ai.opening_book import OpeningBook
from src.ai.determinize import determinize

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
# Tunable knobs (fitted to human games by video_scrape/tune_heuristic.py).
# Defaults reproduce the hand-set heuristic exactly; a HeuristicAgent with a
# `tuning` dict applies its own values at the start of every score() call.
_BASE_RES_VALUE = dict(_RES_VALUE)
TUNE_DEFAULTS = {
    # Spy posts (user, 2026-10-16: "doesn't know which spy spots are better"):
    # overall scale, worth of each extra space a post covers, and the weight
    # of a bordering combat space while a Conflict is live
    "spy_post": 0.5, "spy_multi": 0.35, "spy_combat": 0.6,
    # troops kept for the next Conflict; how much Intrigue cards are worth
    "deploy_hold": 0.4, "intrigue_love": 1.0,
    # tech_love: multiplies what owning a tech tile is worth (tech persona)
    "tech_love": 1.0,
    # contract_stack: bonus per extra contract one Agent move completes, and
    # for taking a contract that completes where a held one does (persona)
    "contract_stack": 0.0,
    # leader_play: how much the Leader passives that depend on where an Agent
    # goes count (Jessica's flip, Reverend Mother repeats); 0 = old behaviour
    "leader_play": 1.0,
    # multipliers on _RES_VALUE entries
    "res_contract": 1.0, "res_spy": 1.0, "res_intrigue": 1.0, "res_draw": 1.0,
    "res_solari": 1.0, "res_spice": 1.0, "res_water": 1.0, "res_troops": 1.0,
    # other levers
    "sm_solari": 2.0,        # per solari toward Swordmaster (Spice Refinery etc.)
    "combat": 1.0,           # combat-space bonus multiplier
    "influence": 1.0,        # faction influence value multiplier
    "reveal_bias": 0.0,      # added to the Reveal turn's score
    "ptw_tax": 4.0,          # Prepare the Way penalty per extra copy
    "faction_space": 0.0,    # flat bonus for sending an agent to a faction space
    # saturation: a resource gain is worth 1 / (1 + sat * held / 5) of its
    # value, so 0 = flat (the hand-set heuristic) and 1 halves it at 5 held
    "sat_solari": 0.0, "sat_spice": 0.0, "sat_water": 0.0,
    "sat_troops": 0.0, "sat_intrigue": 0.0,
    # holding value: a troop kept in the garrison / an intrigue kept in hand
    # is worth this multiple of its default future value (fight_ev model)
    "troop_hold": 1.0, "ci_hold": 1.0, "plot_hold": 0.6,
    # user, 2026-10-17: troops sent into a Conflict where you finish out of
    # the rewards (4th of 4) are lost for nothing - each one expected to be
    # lost that way costs this many troop-values (fight_ev). 0 = off
    "out_of_rewards": 0.0,
    # fight_learned: weight of the rival forecast fitted on recorded games
    # (src/ai/fight_forecast.py) for rivals who have NOT revealed yet, blended
    # with the hand-made rule (0 = rule only, 1 = learned only); revealed
    # rivals keep the rule (as accurate there). fight_own: 1 = my own swords
    # still to come from the fitted model instead of 0.7 x hand swords
    "fight_learned": 0.0, "fight_own": 0.0,
    # intrigue_count: opponents' Intrigues valued at the average Conflict
    # swords of the cards still unseen (played ones are public until the
    # reshuffle), x this weight; 0 = the flat 0.6 swords per card
    "intrigue_count": 0.0,
    # fight_calib: 1 = "will this rival finish ahead of me?" odds calibrated
    # on real outcomes (models_fight/calibration.json); 0 = plain normal curve
    "fight_calib": 0.0,
    # deploy_wait: cost per troop deployed while I still have Agents to place
    # and rivals have not revealed, x the share of rivals still hidden
    "deploy_wait": 0.0,
    # combat_ev: value a combat-space Agent move by the best deploy it opens
    # (fight_ev with the calibrated odds) x this, instead of the flat
    # combat x Conflict worth x garrison rule; 0 = the flat rule
    "combat_ev": 0.0,
    # judgements instead of flat rules (user, 2026-10-18): tsmf_judge = TSMF
    # as 1 VP now minus the deck dilution over the draws left; hc_judge = the
    # High Council seat as +2 persuasion x what it buys x Reveals left; 0 = old rule
    "tsmf_judge": 0.0, "hc_judge": 0.0,
    # play to the Intrigue hand (user, 2026-10-19): ci_plan = Combat Intrigue
    # swords in hand count toward the strength a combat-space Agent move can
    # reach (many Combat Intrigues -> join more fights); inf_plan = influence
    # the held Intrigues will add counts toward the next friendship / alliance
    # (faction-bump Intrigues -> score alliances); eg_plan = gains toward a
    # held endgame Intrigue's condition (Secure Spice Trade: 2 TSMF, CHOAM
    # Profits: 4 contracts, Sacred Pools: 3 water, Strategic Stockpiling:
    # 5 spice, an icon card: win that icon) earn its VP x the chance they add
    "ci_plan": 0.0, "inf_plan": 0.0, "eg_plan": 0.0,
    # prep_draw (user, 2026-10-19: round 6 sets up round 7): from round 6, a
    # move that draws cards is also valued by how it changes the next round's
    # hand - drawing the deck down so the bought cards get shuffled in
    "prep_draw": 0.0,
    # spice_judge (user, 2026-10-19: "the AI hates picking up spice ... spice
    # buys techs"): spice is also worth the tech it gets you to - the best
    # market tech's surplus once affordable, part of it while saving up,
    # x the chance of reaching a tech space before the end; 0 = flat value
    "spice_judge": 0.0,
    # tech_judge: Panopticon / CHOAM Transports / Gene-Locked Vault / Delivery
    # Bay / Ornithopter Fleet valued by the player's situation (blend weight;
    # 0 = the old flat numbers) - src/ai/bloodlines_heuristic.SITUATIONAL
    "tech_judge": 0.0,
    # tech_deny: + this x half of what the tile is worth to the rival most
    # likely to buy it soon (tiles are unique: rush the ones in demand)
    "tech_deny": 0.0,
    # tech_learned: blend toward the tile's measured gain in this situation
    # (models_tech/tech_values.json from src.selfplay.tech_study --fit)
    "tech_learned": 0.0,
    # persuasion economy (user, 2026-10-17: enough persuasion to buy TSMF
    # every round late): hc_rush = extra value per round left for the High
    # Council seat (+2 persuasion each Reveal); starter_trash = extra value
    # for trashing a weak starter card (Dagger, Convincing Argument, ...)
    "hc_rush": 0.0, "starter_trash": 0.0,
    "fight_model": 1,        # 0 = the old deploy / combat-intrigue scoring (for A/B)
    # deck_model 1: value a buy by how it changes the average card you draw
    # for the rest of the game (dilution), not in isolation; deck_k scales it
    "deck_model": 0, "deck_k": 1.0,
    # buy_plan: agent moves are also valued by the best card the reveal-turn
    # persuasion they leave you can buy (keep persuasion cards for the reveal,
    # play Dagger to Assembly Hall, ...); trash_dagger: Daggers are normal
    # trash targets while another Landsraad-access card remains
    "buy_plan": 0.0, "trash_dagger": 0,
    # trash_model: trash by what thinning is worth to THIS deck (see
    # _trash_value) instead of a fixed list of weak starters
    "trash_model": 0,
    # cmd_choice: pick troops vs Sardaukar commanders on a deploy by skills
    # (0 = old fixed rule, commanders first; A/B baseline only)
    "cmd_choice": 1,
    # opp_pool: opponents' reveal swords from the cards that can still be in
    # their hand (owned minus discard / in play, all public) instead of a flat 2
    "opp_pool": 0,
    # deny: weight on taking a row card that opponents still to Reveal this
    # round are likely to afford and want (buying power from public info)
    "deny": 0.0,
    # hl_keep: late game (round 7+), decline a Conflict's "spice -> VP" when
    # paying would drop below the 5 spice a Heighliner costs next round
    "hl_keep": 0.0,
    "wall_future": 0.0,
    # tsmf_goal: strong players acquire ~2 The Spice Must Flow a game. While
    # short of 2, bonus a TSMF buy, value persuasion on bought cards, and
    # value agent moves that keep 9+ persuasion for the Reveal turn
    "tsmf_goal": 0.0,
    # smf_plan: value per agent move of how close it keeps the Reveal turn to
    # the 9 persuasion for TSMF, counting cards it draws (_smf_plan_value);
    # use early./mid./late. versions to learn WHEN to aim for TSMF
    "smf_plan": 0.0,
    # pers_w: extra value per persuasion point on a card (buying it, keeping
    # it out of the trash) - persuasion is what reaches 9 for TSMF
    "pers_w": 0.0,
    # card_play: an agent move also counts the played card's own agent box
    # (Imperial Spymaster's Intrigue on a Spy recall, In High Places' Spy +
    # draw, ...) and what that card would have added on the Reveal turn, so
    # the AI picks WHICH card to send. spy_recall: recalling a Spy (Gather
    # Intelligence / Infiltrate) costs the Spy's board value. 0 = old scoring.
    # 2026-10-05 four-way, tuned heuristic, 3,840 games: card_play 1 won 27.5%
    # vs 22.6-23.6% for the old scoring; spy_recall at 1.0 / 0.3 LOST win share
    # (a card now beats keeping the Spy), so it stays off.
    "card_play": 1.0, "spy_recall": 0.0,
    # contracts: agent moves count the reward of each contract they complete
    # (board space / Harvest), and taking a contract is valued by its reward x
    # the chance of finishing it. maker_bonus: a maker space's accumulated
    # bonus spice counts toward its value. 0 = old scoring.
    # 2026-10-06 four-way, 3,840 games each: contracts 2 won 27.6% vs 22-25%
    # for 0 (3 = 27.2%, 1 = 24-26%); contracts completed per player 0.86 ->
    # 1.75. maker_bonus showed no gain (23.9% vs 24.2%), so it stays off.
    "contracts": 2.0, "maker_bonus": 0.0,
    # skills (Bloodlines): combat strength counts Sardaukar sword skills, and
    # a commander recruit is valued by switching the held skills on for this
    # round's combat (commanders return to supply after every combat).
    # 0 = old scoring, else a multiplier on skill value. 2026-10-06 sweep
    # (four-way, 3,840 games each): win share rose steadily from 1 to ~40 and
    # plateaued at 40-80; 40 vs 0 head-to-head = 30-33% vs 18%. Re-recruits
    # per player 0.85 -> 3.0, combats with skills on 1.5 -> 3.2 of ~7.3.
    "skills": 40.0,
    # race: VP is worth as much early as late (urgency floor 1 + 2.5 x race),
    # so the AI banks friendships / alliances / combat VP as soon as it can.
    # sm_first: while Swordmaster is unbought (4+ rounds left), a commander's
    # solari cost is priced at the Swordmaster rate (sm_solari per solari).
    "race": 0.0, "sm_first": 0.0,
    # From the Consules "5 ways to play Bloodlines" guide:
    # hc_bl: the High Council also discounts every future tech by 1 spice and
    # makes Command (6+ persuasion) fire far more often — worth more to a
    # player who owns Command cards / techs.
    # opp_cmdr: an opponent's solari is hidden combat strength (2 solari + an
    # agent at a commander space = a 2-strength unit plus their skill swords).
    "hc_bl": 0.0, "opp_cmdr": 0.0,
    # style (2026-10-08): lean toward the Bloodlines objective (Swordmaster /
    # High Council / commanders / techs) that human players - winners
    # weighted up - pursue from this position (config/objective_model.npz,
    # src/ai/playstyle.py). A purchase of objective o gets style * P(o); an
    # agent move that opens that purchase (commander space with a recruit
    # affordable, green space with a tech affordable) gets half. 0 = off.
    "style": 0.0,
    # fixated playstyles (2026-10-09): a flat bonus for buying one objective
    # outright (half for an agent move that opens the purchase), for the
    # style league (src/selfplay/style_league.py). buy_mult scales every card
    # purchase; tsmf adds to The Spice Must Flow.
    "fix_swordmaster": 0.0, "fix_high_council": 0.0,
    "fix_commanders": 0.0, "fix_techs": 0.0,
    "buy_mult": 1.0, "tsmf": 0.0,
    # VP gap vs humans (2026-10-11: humans take 1.8 TSMF / 1.0 alliances per
    # game, the AI 0.6 / 0.7). tsmf_early: 1 = no round<4 discount on TSMF;
    # alliance: multiplier on the alliance-crossing bonus; alliance_step:
    # extra value for the 3rd influence point (one step from an alliance).
    "tsmf_early": 0.0, "alliance": 1.0, "alliance_step": 0.0,
    # worm_model 1 (2026-10-11): a sandworm DOUBLES the winner's reward, so
    # the conflict is valued doubled once the player has a worm in it, a worm
    # is valued by the expected-reward change it makes (3 strength + the
    # double), never on a protected conflict behind an intact Shield Wall, and
    # breaking the wall is valued by the worm it opens for you minus the ones
    # it opens for rivals with Maker Hooks. 0 = old scoring; 2 = only the
    # summon / wall values; 3 = only the doubled conflict value.
    "worm_model": 0.0,
    # deck_plan (2026-10-11): play the strategy the owned cards call for.
    # Cards gated on a condition (Bene Gesserit friendship, 2 Spies on the
    # board, an alliance, Fremen Bond, another Bene Gesserit card in play)
    # make meeting it worth (value of the gated effect) x (plays of those
    # cards left): influence toward the faction, Spies, and buying cards
    # with the tag / faction access all gain that much. Cards that give
    # influence for a faction count as access to it (alliance pursuit).
    # 0 = off, else a multiplier.
    "deck_plan": 0.0,
    # worm_bonus (2026-10-12): flat bonus for a maker-space sandworm summon.
    # The worm audit (src/selfplay/worm_audit.py, 1,946 decisions played out
    # 64x each way) found summoning beat the heuristic's best other move by
    # +0.02-0.05 win share whenever the heuristic rated that move up to ~10
    # points above the worm, and was a coin flip beyond that.
    "worm_bonus": 0.0,
}
# Situational tuning: a key "<phase>.<knob>" overrides <knob> in that phase
# only. Phases follow how the game is played: early = rounds 1-3 (build
# resources), mid = 4-5, late = 6+ (convert resources and techs into VP
# through faction influence and combat).
PHASES = ("early", "mid", "late")


def phase_of(rnd: int) -> str:
    return "early" if rnd <= 3 else "mid" if rnd <= 5 else "late"


def split_phases(tuning: Optional[dict]) -> Dict[str, dict]:
    """{"early": full knob dict, "mid": ..., "late": ...} from a tuning dict
    that may hold phase-prefixed keys."""
    tuning = tuning or {}
    base = {**TUNE_DEFAULTS, **{k: v for k, v in tuning.items() if "." not in k}}
    out = {}
    for ph in PHASES:
        out[ph] = {**base, **{k.split(".", 1)[1]: v for k, v in tuning.items()
                              if k.startswith(ph + ".")}}
    return out


_TUNE = dict(TUNE_DEFAULTS)

# value a knob that is off (0) starts from when a mutation / Leader weight turns it on
SEED_ON = {"maker_bonus": 0.5, "alliance": 0.5, "contract_stack": 1.0, "out_of_rewards": 1.0,
           "hc_rush": 0.5, "starter_trash": 1.0, "smf_plan": 2.0, "tsmf_goal": 2.0,
           "pers_w": 0.3, "buy_plan": 1.0, "fight_learned": 0.5, "intrigue_count": 0.5,
           "deploy_wait": 0.5, "combat_ev": 1.0, "tsmf_judge": 1.0, "hc_judge": 1.0, "deck_model": 1,
           "ci_plan": 1.0, "inf_plan": 0.5, "eg_plan": 1.0, "prep_draw": 1.0, "spice_judge": 1.0,
           "tech_judge": 1.0, "tech_deny": 0.5,
           "tech_learned": 1.0}


def _apply_tuning(t: dict) -> None:
    global _TUNE, _RES_VALUE
    _TUNE = t
    res = dict(_BASE_RES_VALUE)
    for k, v in t.items():
        if k.startswith("res_"):
            name = k[4:]
            for key in list(res):
                if key == name or (name == "spy" and key.startswith("spy")):
                    res[key] = _BASE_RES_VALUE[key] * v
    # the Intrigue-loving persona: every Intrigue card (gained, stolen, kept) is worth more
    res["intrigue"] = res.get("intrigue", 1.3) * t.get("intrigue_love", 1.0)
    _RES_VALUE = res


_INF_KEYS = ("influence_emperor", "influence_spacing_guild",
             "influence_bene_gesserit", "influence_fremen")
_FACTION_OF = {
    "influence_emperor": "emperor", "influence_spacing_guild": "spacing_guild",
    "influence_bene_gesserit": "bene_gesserit", "influence_fremen": "fremen",
}
_FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")

# Set per-call at the top of HeuristicAgent.score() from `self.faction_focus`
# (single-threaded, no reentrancy — the module-level helpers below read it
# within that same score() call). `heuristic:ff0` turns it off for A/B.
_FACTION_FOCUS = True
_INFLUENCE_BASE = 4.5      # per-point value of influence when _FACTION_FOCUS
                          # (~0.5 VP: every point is a down payment on the next
                          #  friendship/alliance threshold, not dead weight)

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
    dp_on = _TUNE.get("deck_plan", 0.0)
    for c in all_cards:
        if any(getattr(sym, "value", sym) == fac for sym in getattr(c, "access_symbols", ())):
            n += 1
        elif dp_on:
            fl = {}
            for e in (getattr(c, "agent_effects", []) or []) + (getattr(c, "reveal_effects", []) or []):
                for kk, vv in _flatten(e).items():
                    fl[kk] = fl.get(kk, 0) + vv
            if fl.get("influence_" + fac, 0) > 0:
                n += 1
            elif fl.get("influence_any", 0) > 0:
                n += 0.5
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

    NOTE (2026-09-10): raising the early floor here (tested up to ~3.0, i.e.
    near-max urgency all game) does NOT move the winner's VP-velocity curve —
    the early game is throughput-limited, not valuation-limited, and the R7
    jump is the conflict deck (level-2 rounds 2-6 award no direct VP; level-3
    from R7 awards 2). See training_status memory.
    """
    round_component = max(0.0, gs.round - 4) / 6.0            # 0 @R4 -> 1.0 @R10
    best_vp = max((q.victory_points for q in gs.players), default=0)
    vp_component = max(0.0, best_vp - 5) / 5.0                 # 0 @<=5 -> 1.0 @10
    race = min(1.0, _TUNE.get("race", 0.0))
    return 1.0 + 2.5 * max(round_component, vp_component, race)


def _influence_gain_value(gs: GameState, pid: int, fac: str, amt: float, _shift: int = 0) -> float:
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
    k_plan = _TUNE.get("inf_plan", 0.0)
    if k_plan and not _shift:
        pend = _pending_influence(gs, pid, fac)
        if pend:
            # the held Intrigues' influence lands too: blend in the value of
            # this gain on top of it (it may now finish a friendship / alliance)
            k_plan = min(1.0, k_plan)
            return ((1.0 - k_plan) * _influence_gain_value(gs, pid, fac, amt, _shift=-1)
                    + k_plan * _influence_gain_value(gs, pid, fac, amt, _shift=pend))
    p = gs.players[pid]
    cur = min(6, p.influence.get(fac, 0) + max(0, _shift))
    amt = int(round(amt))
    new = min(6, cur + amt)
    urgency = _game_urgency(gs, pid)
    access = _faction_access_strength(gs, pid, fac)

    if not _FACTION_FOCUS:
        v = amt * _RES_VALUE.get(f"influence_{fac}", 2.2)
        if cur < 2 <= new:
            v += 7.0 * urgency
            v += _FRIENDSHIP_GATE_BONUS.get(fac, 0.0)
        if cur < 4 <= new:
            alliance_mult = (1.0 + 1.5 * access) * _TUNE.get("alliance", 1.0)
            holder = gs.alliance_holder.get(fac)
            v += (10.0 if (holder is not None and holder != pid) else 7.5) * urgency * alliance_mult
            v += _effect_value(gs, pid, dict(INFLUENCE_4_BONUS[fac]))   # 4th-space reward
        elif cur >= 4 and gs.alliance_holder.get(fac) == pid \
                and any(q.influence.get(fac, 0) >= cur - 1 for q in gs.players if q.id != pid):
            v += 1.5 * amt * urgency
        return v + _dp_influence_bonus(gs, pid, fac, cur, new)

    # faction_focus: value each influence point by how it moves you toward the
    # NEXT unreached threshold, so the AI spreads to four friendships (4 clean
    # VP) instead of hoarding one track to 5-6 (measured: winner's influence
    # sits ~[4.8, 2.9, 1.8, 0.9] — tracks 3 & 4 stranded below friendship = ~2
    # wasted VP). A point that overshoots a threshold with nothing to finish is
    # nearly dead; a point one step from a threshold you can realistically
    # reach is a ~0.5-VP down payment.
    rounds_left = max(1, MAX_ROUNDS - gs.round)
    reach = min(1.0, (0.45 + 2.2 * access) * rounds_left / 3.0)   # can I finish this track?
    v = 0.0
    if cur < 2 <= new:                                # friendship crossed = clean 1 VP
        v += 5.5 * urgency + _FRIENDSHIP_GATE_BONUS.get(fac, 0.0)
    if cur < 4 <= new:                                # alliance crossed
        alliance_mult = (1.0 + 1.5 * access) * _TUNE.get("alliance", 1.0)
        holder = gs.alliance_holder.get(fac)
        v += (10.0 if (holder is not None and holder != pid) else 7.5) * urgency * alliance_mult
        # the 4th space's own reward (Fremen water / BG Intrigue / Guild 3 solari / Emperor Spy)
        v += _effect_value(gs, pid, dict(INFLUENCE_4_BONUS[fac]))
    # per-point "down payment" on points that DON'T cross a threshold this gain
    for pos in range(cur + 1, new + 1):
        if pos == 2 or pos == 4:
            continue                                  # the crossing point itself, already scored
        if pos < 2:
            v += 4.0 * reach                          # 0->1: heading to friendship
        elif pos < 4:
            # 2->3: only worth it if committed to the alliance (access + a rival
            # not already parked ahead of you) — otherwise it's the classic
            # stranded overshoot
            committed = reach * (0.7 if gs.alliance_holder.get(fac) in (None, pid) else 1.1)
            v += 3.0 * min(1.0, committed)
            if pos == 3:
                v += _TUNE.get("alliance_step", 0.0) * reach
        else:
            v += 0.4                                  # 5,6: essentially dead
    if cur >= 4 and gs.alliance_holder.get(fac) == pid \
            and any(q.influence.get(fac, 0) >= cur - 1 for q in gs.players if q.id != pid):
        v += 1.5 * amt * urgency                      # defend a contested alliance
    return v * _TUNE["influence"] + _dp_influence_bonus(gs, pid, fac, cur, new)


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
        elif k == "contract" and amt > 0 and _TUNE.get("contracts", 0):
            v += _take_contract_value(gs, pid) * amt
            v += _endgame_goal_bonus(gs, pid, "contracts", 0.6 * amt)
        elif k in _WALL_BREAK_KEYS:
            v += _wall_break_value(gs, pid, with_worm=any(x in eff for x in _SANDWORM_KEYS)) * amt
        elif k in _SANDWORM_KEYS:
            v += _sandworm_value(gs, pid, wall_open=any(x in eff for x in _WALL_BREAK_KEYS)) * amt
        elif k == "trash" and amt > 0 and _TUNE.get("trash_model", 0):
            v += 0.6 * _best_trash_value(gs, pid) * amt
        else:
            if k == "spy" and amt > 0:
                v += _dp_spy_bonus(gs, pid) * amt
            if k in ("water", "spice") and amt > 0:
                v += _endgame_goal_bonus(gs, pid, k, amt)
            if k == "spice":
                v += _spice_judge_delta(gs, pid, amt)
            val = _RES_VALUE.get(k, 0.3)
            sat = _TUNE.get("sat_" + k, 0.0)
            if sat and amt > 0:
                val /= 1.0 + sat * _held(gs.players[pid], k) / 5.0
            v += val * amt
    return v


_SPICE_CACHE: Dict[tuple, list] = {}


def _spice_potential(gs: GameState, pid: int, x: float) -> float:
    """spice_judge: what holding x spice is worth toward the market techs."""
    bl = getattr(gs, "bl", None)
    if bl is None:
        return 0.0
    p = gs.players[pid]
    tops = tuple(st[0].name if st else "" for st in bl.tech_stacks)
    key = (id(gs), id(_TUNE), pid, gs.round, tops, len(getattr(p, "techs", []) or []))
    offers = _SPICE_CACHE.get(key)
    if offers is None:
        from src.ai.bloodlines_heuristic import tech_buy_value
        offers = []
        for st in bl.tech_stacks:
            if st:
                surplus = tech_buy_value(gs, pid, st[0])
                if surplus > 0:
                    offers.append((bl.tech_price(pid, st[0]), surplus))
        if len(_SPICE_CACHE) > 4096:
            _SPICE_CACHE.clear()
        _SPICE_CACHE[key] = offers
    if not offers:
        return 0.0
    reach = min(0.9, 0.35 * _rounds_left(gs))          # a tech space visit still to come
    return reach * max(sv * (1.0 if x >= price else 0.5 * x / max(1, price)) for price, sv in offers)


def _spice_judge_delta(gs: GameState, pid: int, amt: float) -> float:
    k = _TUNE.get("spice_judge", 0.0)
    if not k or not amt:
        return 0.0
    have = gs.players[pid].spice
    return k * (_spice_potential(gs, pid, max(0, have + amt)) - _spice_potential(gs, pid, have))


def _held(p, k: str) -> int:
    if k == "troops":
        return p.troops_garrison
    if k == "intrigue":
        return len(p.intrigue_cards)
    return getattr(p, k, 0) or 0


def _flatten(eff: dict) -> dict:
    """Best-effort flatten of nested conditional/choice sub-effects for scoring."""
    out = {}
    if isinstance(eff.get("bl_choose"), dict):          # Bloodlines: best option
        opts = [_flatten(o) for o in eff["bl_choose"].get("options", [])]
        best = max(opts, key=lambda o: sum(v for v in o.values()
                                           if isinstance(v, (int, float))), default={})
        for kk, vv in best.items():
            out[kk] = out.get(kk, 0) + vv
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


# Known-at-play-time condition values (set while scoring an agent move).
_PLAY_CTX: Optional[dict] = None


def _cond_weight(gs: GameState, pid: int, key: str) -> float:
    """
    0..1 'how likely/true is this condition for me right now' — used to value
    a card's CONDITIONAL agent/reveal effects at buy time, so a card whose
    only payoff sits behind an alliance/influence threshold the player doesn't
    have isn't priced the same as an unconditional card of the same cost
    (e.g. Junction Headquarters without the Spacing Guild Alliance).
    """
    p = gs.players[pid]
    if _PLAY_CTX is not None and key in _PLAY_CTX:
        return _PLAY_CTX[key]
    if key.startswith("bl_"):
        from src.ai.bloodlines_heuristic import cond_weight
        return cond_weight(gs, pid, key[3:])
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
        if _TUNE.get("deck_plan", 0.0):
            tag = next((t for t, sh in _TAG_SHORT.items() if sh == key[10:]), None)
            owned = [c for z in (p.deck, p.discard, p.hand, p.in_play) for c in z]
            if tag and owned:
                share = sum(1 for c in owned if any(getattr(x, "value", x) == tag
                                                    for x in getattr(c, "tags", ()))) / len(owned)
                return max(0.10, min(0.95, 1.0 - (1.0 - share) ** 4))
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
        elif k == "command" and isinstance(sub, dict):
            from src.ai.bloodlines_heuristic import P_COMMAND
            v += _card_effect_value(gs, pid, sub, w * P_COMMAND)
        elif k == "bl_choose" and isinstance(sub, dict):
            opts = [_card_effect_value(gs, pid, o, w) for o in sub.get("options", [])]
            v += (sum(opts) if sub.get("both_if") else 0.9 * max(opts, default=0.0))
        elif k in ("bl_discard_for", "bl_trash_from_hand") and isinstance(sub, dict):
            v += 0.6 * _card_effect_value(gs, pid, sub.get("reward") or sub.get("bonus", {}), w)
        elif k in ("bl_after_turn", "bl_conflict_bonus") and isinstance(sub, dict):
            v += 0.5 * _card_effect_value(gs, pid, sub, w)
        elif k == "trash_intrigue_for" and isinstance(sub, dict):
            has_intrigue = bool(gs.players[pid].intrigue_cards)
            v += _card_effect_value(gs, pid, sub, w * (0.7 if has_intrigue else 0.25))
        elif isinstance(sub, (int, float)):
            if k in _FACTION_OF:
                v += w * _influence_gain_value(gs, pid, _FACTION_OF[k], sub)
            elif k == "influence_any":
                v += w * _influence_gain_value(gs, pid, "any", sub)
            elif k == "contract" and sub > 0 and _TUNE.get("contracts", 0):
                v += w * _take_contract_value(gs, pid) * sub
            elif k in _WALL_BREAK_KEYS:
                v += w * _wall_break_value(gs, pid, with_worm=any(x in eff for x in _SANDWORM_KEYS)) * sub
            elif k in _SANDWORM_KEYS:
                v += w * _sandworm_value(gs, pid, wall_open=any(x in eff for x in _WALL_BREAK_KEYS)) * sub
            else:
                from src.ai.bloodlines_heuristic import BL_RES_VALUE
                v += w * _RES_VALUE.get(k, BL_RES_VALUE.get(k, 0.3)) * sub
                if k == "spy" and sub > 0:
                    v += w * _dp_spy_bonus(gs, pid) * sub
        # other nested dicts (unrecognized structure) contribute nothing,
        # same as before — but recognized ones are no longer invisible.
    return v


def _deck_total(p) -> int:
    """Every card the player owns, across all zones — their working deck size."""
    return len(p.deck) + len(p.discard) + len(p.hand) + len(p.in_play)


# A lean deck (you cycle it faster, so your good cards come up more often) is
# worth defending. Past this many cards, marginal filler should stop looking
# like a buy while genuine upgrades still clear the bar. The starting deck is
# 10; the intent is ~4-6 good acquisitions offset by trashing weak starters,
# landing around a 12-14 card deck rather than the ~18 the flat scorer built.
_DECK_TARGET = 12


def _dilution_penalty(gs: GameState, p) -> float:
    """Score to subtract from a prospective buy for bloating the deck past the
    lean target. Ramps hard per card over the target, held near full strength
    through the deck-cycling part of the game and fading only in the last
    couple of rounds (where a late VP grab like TSMF shouldn't be suppressed)."""
    over = _deck_total(p) - _DECK_TARGET
    if over <= 0:
        return 0.0
    fade = max(0.0, min(1.0, (MAX_ROUNDS - 1 - gs.round) / 3.0))
    return 0.9 * fade * over


def _faction_cards_owned(p, fac: str) -> int:
    """How many cards this player already owns that grant access to `fac`."""
    n = 0
    for zone in (p.deck, p.discard, p.hand, p.in_play):
        for c in zone:
            for sym in getattr(c, "access_symbols", ()):
                if getattr(sym, "value", sym) == fac:
                    n += 1
                    break
    return n


# --- deck_plan: the strategy the owned cards call for -----------------------
_DP_CACHE: dict = {}
_DP_BUSY = False
_TAG_SHORT = {"bene_gesserit": "bene", "fremen": "fremen", "emperor": "emperor",
              "spacing_guild": "spacing"}


def _walk_conds(gs: GameState, pid: int, eff, out: dict, w: float) -> None:
    if not isinstance(eff, dict):
        return
    for k, sub in eff.items():
        if not isinstance(sub, dict):
            continue
        if k.startswith("if_"):
            out[k[3:]] = out.get(k[3:], 0.0) + w * _card_effect_value(gs, pid, sub)
        _walk_conds(gs, pid, sub, out, w)


def _deck_plan(gs: GameState, pid: int) -> dict:
    """What the owned cards ask for: {condition: value per play of the
    effects gated on it} and how often each owned card is still played."""
    global _DP_BUSY
    p = gs.players[pid]
    owned = [c for z in (p.deck, p.discard, p.hand, p.in_play) for c in z]
    key = (id(_TUNE), pid, gs.round, tuple(sorted(c.name for c in owned)),
           tuple(sorted(p.influence.items())), sum(p.spies_on_board.values()))
    hit = _DP_CACHE.get(key)
    if hit is not None:
        return hit
    if len(_DP_CACHE) > 20000:
        _DP_CACHE.clear()
    conds: dict = {}
    _DP_BUSY = True
    try:
        for c in owned:
            for e in (getattr(c, "agent_effects", []) or []) + (getattr(c, "reveal_effects", []) or []):
                _walk_conds(gs, pid, e, conds, 0.8)
    finally:
        _DP_BUSY = False
    plays = max(0.0, 9.5 - gs.round) * 5.0 / max(1, len(owned))
    dp = {"conds": conds, "plays": plays, "n": len(owned),
          "tags": {t: sum(1 for c in owned if any(getattr(x, "value", x) == t
                                                  for x in getattr(c, "tags", ())))
                   for t in _TAG_SHORT}}
    _DP_CACHE[key] = dp
    return dp


def _dp_influence_bonus(gs: GameState, pid: int, fac: str, cur: int, new: int) -> float:
    """deck_plan: influence that meets (or moves toward) a faction condition
    the player's own cards are gated on."""
    k = _TUNE.get("deck_plan", 0.0)
    if not k or _DP_BUSY or new <= cur:
        return 0.0
    dp = _deck_plan(gs, pid)
    tot = 0.0
    for cond, u in dp["conds"].items():
        if cond == "alliance_" + fac:
            need = 4
        elif cond.startswith("influence_" + fac + "_"):
            n = cond.rsplit("_", 1)[1]
            need = int(n) if n.isdigit() else 2
        elif cond == "fremen_bond" and fac == "fremen":
            need = 2
        else:
            continue
        if cur >= need:
            continue
        gain = u * dp["plays"]
        tot += gain if new >= need else 0.5 * gain * (new - cur) / need
    return k * tot


def _dp_spy_bonus(gs: GameState, pid: int) -> float:
    """deck_plan: one more Spy on the board, for cards that pay off with
    2+ Spies (Bene Gesserit Operative, Spy Network) or on recalling one."""
    k = _TUNE.get("deck_plan", 0.0)
    if not k or _DP_BUSY:
        return 0.0
    dp = _deck_plan(gs, pid)
    have = sum(gs.players[pid].spies_on_board.values())
    tot = 0.0
    for cond, u in dp["conds"].items():
        if cond.startswith("spies_"):
            n = cond[6:]
            need = int(n) if n.isdigit() else 2
            if have < need:
                tot += u * dp["plays"] / (need - have)
        elif cond == "spy_recalled":
            tot += u * dp["plays"] * (0.5 if have == 0 else 0.15)
    return k * tot


def _dp_buy_bonus(gs: GameState, pid: int, card) -> float:
    """deck_plan: a bought card that feeds owned cards' conditions - its tag
    (another Bene Gesserit card in play) or access to a faction whose
    friendship they need."""
    k = _TUNE.get("deck_plan", 0.0)
    if not k:
        return 0.0
    dp = _deck_plan(gs, pid)
    p = gs.players[pid]
    hand_share = min(1.0, 4.0 / max(1, dp["n"]))      # chance it shares a turn with one
    v = 0.0
    for t in getattr(card, "tags", ()):
        u = dp["conds"].get("tag_other_" + _TAG_SHORT.get(getattr(t, "value", t), "?"), 0.0)
        v += 0.5 * hand_share * u * dp["plays"]
    for sym in getattr(card, "access_symbols", ()):
        fac = getattr(sym, "value", sym)
        if fac in _FACTIONS:
            cur = p.influence.get(fac, 0)
            v += 0.3 * _dp_influence_bonus(gs, pid, fac, cur, cur + 1) / k
    return k * v


def _card_play_quality(gs: GameState, pid: int, card) -> float:
    """What a card is worth each time it is drawn and played: persuasion,
    swords, agent + reveal effects, faction access (flat). The same
    yardstick for owned cards and Row cards."""
    q = (1.5 + _TUNE.get("pers_w", 0.0)) * (card.persuasion or 0) + 0.9 * (card.swords or 0)
    for e in getattr(card, "agent_effects", []) + getattr(card, "reveal_effects", []):
        q += 0.8 * _card_effect_value(gs, pid, e)
    q += 0.6 * sum(1 for s_ in getattr(card, "access_symbols", ())
                   if getattr(s_, "value", s_) in _FACTION_OF.values())
    return q


def _deck_buy_value(gs: GameState, pid: int, card) -> float:
    """deck_model: immediate acquire effects + (draws of this card left in
    the game) x (its quality - the average quality of the cards you own).
    Filler below your deck's average scores negative, so 'buy nothing' or
    'one strong card' beats bloating the deck."""
    p = gs.players[pid]
    owned = [c for z in (p.deck, p.discard, p.hand, p.in_play) for c in z]
    mean = sum(_card_play_quality(gs, pid, c) for c in owned) / max(1, len(owned))
    draws_left = max(0.0, 9.5 - gs.round) * 5.0          # ~5 cards a round
    my_draws = draws_left / (len(owned) + 1)
    v = my_draws * (_card_play_quality(gs, pid, card) - mean) * _TUNE.get("deck_k", 1.0)
    for e in getattr(card, "acquire_effects", []):
        v += _card_effect_value(gs, pid, e)
    if card.name == "The Spice Must Flow" and gs.round < 4:
        v *= 0.15
    return v


def _tsmf_judgement(gs: GameState, pid: int, card) -> float:
    """The Spice Must Flow judged (user, 2026-10-18): 1 VP now, priced by how
    close the game is to its end, MINUS a weak card in the deck = the times
    it will still be drawn x how much worse it plays than the deck's average
    card. Early that costs a lot of draws; late almost none."""
    p = gs.players[pid]
    owned = [c for z in (p.deck, p.discard, p.hand, p.in_play) for c in z]
    mean = sum(_card_play_quality(gs, pid, c) for c in owned) / max(1, len(owned))
    draws_left = max(0.0, 9.5 - gs.round) * 5.0          # ~5 cards a round
    my_draws = draws_left / (len(owned) + 1)
    dilution = my_draws * max(0.0, mean - _card_play_quality(gs, pid, card))
    return _reward_worth(gs, pid, {"vp": 1}) - dilution


def _hc_judgement(gs: GameState, pid: int) -> float:
    """The High Council seat judged: +2 persuasion on every Reveal left, each
    worth 2 x what a point of persuasion buys - the Row cards' average value
    per persuasion of cost (an average: the Row changes before most of those
    Reveals, so one card that +2 happens to unlock today is not the price)."""
    per = [max(0.0, _acquire_card_value(gs, pid, c)) / c.cost for c in gs.imperium_row if (c.cost or 0) > 0]
    reveals_left = max(0, 10 - gs.round) + (0 if pid in gs.players_revealed else 1)
    return reveals_left * 2.0 * (sum(per) / len(per) if per else 0.0)


def _acquire_card_value(gs: GameState, pid: int, card, lean: bool = False) -> float:
    if _TUNE.get("deck_model", 0):
        return _deck_buy_value(gs, pid, card)
    """
    Shared valuation for buying a card, whether from the Imperium Row
    (ACQUIRE_CARD) or a reserve stack (ACQUIRE_RESERVE — The Spice Must Flow /
    Prepare the Way).  Previously ACQUIRE_RESERVE was a flat score that never
    looked at the card at all, so TSMF's guaranteed 1 VP on acquire (worth
    10.0 in _RES_VALUE) was invisible; and ACQUIRE_CARD ignored every card's
    `acquire_effects` (free Spies, contracts, intrigue, influence, VP...) —
    both are folded in here now.
    """
    s = (1.5 + _TUNE.get("pers_w", 0.0)) * card.persuasion + 0.9 * card.swords
    for e in (getattr(card, "agent_effects", []) +
              getattr(card, "reveal_effects", []) +
              getattr(card, "acquire_effects", [])):
        s += 0.8 * _card_effect_value(gs, pid, e)
    # Faction-access cards are the early-game VP engine. Every turn you play one
    # on a faction space is +1 influence ~= 0.5 VP (friendship progress), and a
    # deck with several access cards for one faction is what actually lets you
    # push a track to friendship/alliance and keep showing up in that faction's
    # conflicts (battle-icon pairs). Price the access as a recurring asset, with
    # diminishing returns once you already hold a few for that faction and a
    # softer (not 0.35) discount before Swordmaster.
    p = gs.players[pid]
    seen_fac = set()
    for sym in getattr(card, "access_symbols", ()):
        fac = getattr(sym, "value", sym)
        if fac not in _FACTION_OF.values() or fac in seen_fac:
            continue
        seen_fac.add(fac)
        cur = p.influence.get(fac, 0)
        if cur >= 4:
            continue                              # track maxed — access adds little
        if _FACTION_FOCUS:
            base = 3.5                            # ~0.5 VP x ~1.5 future influence gains, discounted
            if cur == 1:
                base += 2.5                       # a play here crosses to friendship
            elif cur == 3:
                base += 3.0                       # a play here crosses to alliance
            owned = _faction_cards_owned(p, fac)
            base *= max(0.4, 1.0 - 0.28 * owned)  # 3rd+ copy for one faction is filler
            if not p.has_swordmaster:
                base *= 0.75
            s += base
        else:
            fac_mult = 1.0 if p.has_swordmaster else 0.35
            s += fac_mult * (1.2 + (2.0 if cur == 1 else 0.0)
                             + (1.5 if cur == 3 else 0.0))
    # NOTE: no flat "expensive = good" bonus — a costly card whose payoff sits
    # behind a condition you don't meet (e.g. Junction Headquarters without
    # the Spacing Guild Alliance) is priced by what it does for you right now.
    s -= 0.15 * card.cost
    # TSMF is a late-game VP dump, not a card that compounds into future turns
    # like a Row card does — rarely worth it before round 4; from then on it
    # competes normally (i.e. only wins out when the Row has nothing better).
    if card.name == "The Spice Must Flow" and gs.round < 4 and not _TUNE.get("tsmf_early", 0)             and not _TUNE.get("tsmf_judge", 0):
        s *= 0.15
    if card.name == "The Spice Must Flow" and _TUNE.get("tsmf_judge", 0):
        s = _TUNE["tsmf_judge"] * _tsmf_judgement(gs, pid, card)

    # Prepare the Way is a cheap tempo/ramp card, not a payload: the 2nd copy is
    # worth far less than the first (same conditional solari, same Landsraad/BG
    # access you already have) and a 3rd/4th is dead weight that only dilutes
    # the deck. The flat scorer was draining the whole 8-card reserve stack
    # every game (buyer win-rate well below fair). Tax each copy past the
    # first — always on, since PTW-spam is a losing pattern regardless of
    # deck-size philosophy.
    if card.name == "Prepare the Way":
        owned = sum(1 for z in (p.deck, p.discard, p.hand, p.in_play)
                    for c in z if c.name == "Prepare the Way")
        if owned >= 1:
            s -= _TUNE["ptw_tax"] + 2.0 * (owned - 1)

    # Deck-dilution pressure. Bites every card equally: once the deck is
    # bloated, a filler card goes negative and gets skipped, while a strong
    # upgrade still clears the bar. This is what turns "buy something every reveal" into
    # "buy only when it's an actual upgrade, otherwise bank the turn".
    if lean:
        s -= _dilution_penalty(gs, p)
    s += _dp_buy_bonus(gs, pid, card)
    return s


def _ci_card_swords(ic) -> float:
    """Swords one Intrigue card adds in a Conflict (0 if it isn't a Combat Intrigue)."""
    from src.game.intrigue.intrigue import IntrigueTiming
    ts = ic.timing if isinstance(ic.timing, (set, tuple, list, frozenset)) else (ic.timing,)
    if IntrigueTiming.COMBAT not in ts:
        return 0.0
    sw = 0.0
    for e in ic.effects:
        fe = _flatten(e)
        v = fe.get("swords", 0)
        sw += v if isinstance(v, (int, float)) else 2.0
        for sub in fe.values():
            if isinstance(sub, dict) and isinstance(sub.get("swords"), (int, float)):
                sw += sub["swords"]
    return sw if sw else 2.0                            # deploy / pull-troop combat cards ~= 2


_UNSEEN_CACHE: Dict[tuple, Tuple[float, float]] = {}


def _unseen_intrigue_swords(gs: GameState, viewer: Optional[int]) -> Tuple[float, float]:
    """(mean, variance) of the Conflict swords on one Intrigue card `viewer`
    cannot see (user, 2026-10-18: a played Intrigue is gone until the deck is
    reshuffled - track what is left). Unseen = the Intrigue deck + the other
    players' hands: the discard pile is public and my own hand is known."""
    hands = tuple(len(q.intrigue_cards) for q in gs.players)
    key = (id(gs), viewer, len(gs.intrigue_deck), len(gs.intrigue_discard), hands)
    hit = _UNSEEN_CACHE.get(key)
    if hit is not None:
        return hit
    pool = list(gs.intrigue_deck)
    for q in gs.players:
        if q.id != viewer:
            pool += q.intrigue_cards
    if not pool:
        out = (0.0, 0.0)
    else:
        xs = [_ci_card_swords(c) for c in pool]
        m = sum(xs) / len(xs)
        out = (m, sum((x - m) ** 2 for x in xs) / len(xs))
    if len(_UNSEEN_CACHE) > 4096:
        _UNSEEN_CACHE.clear()
    _UNSEEN_CACHE[key] = out
    return out


def _own_combat_intrigue_swords(gs: GameState, pid: int) -> float:
    """Swords this player could still add to the CURRENT combat from Combat
    Intrigues sitting in their hand (playable only with >=1 unit in the
    Conflict). Lets the deploy scorer see 'an opponent under-deployed and I'm
    holding Find Weakness — commit the troops and take this' instead of pricing
    the deploy off raw troops alone."""
    from src.game.intrigue.intrigue import IntrigueTiming
    total = 0.0
    for ic in gs.players[pid].intrigue_cards:
        ts = ic.timing if isinstance(ic.timing, (set, tuple, list, frozenset)) else (ic.timing,)
        if IntrigueTiming.COMBAT not in ts:
            continue
        sw = 0.0
        for e in ic.effects:
            fe = _flatten(e)
            v = fe.get("swords", 0)
            sw += v if isinstance(v, (int, float)) else 2.0
            for sub in fe.values():                     # e.g. retreat_for -> {swords: N}
                if isinstance(sub, dict) and isinstance(sub.get("swords"), (int, float)):
                    sw += sub["swords"]
        total += sw if sw else 2.0                      # deploy / pull-troop combat cards ~= 2
    return min(total, 8.0)


# ---------------------------------------------------------------------------
# Fight model: expected reward from the current Conflict given my strength.
# Strength = 2 per troop + 3 per sandworm + swords (needs 1+ unit). Opponents
# are modelled as their visible strength plus what they can still add (troops
# they could deploy with agents left, reveal swords, combat intrigues in
# hand), with spread growing with that potential. Deploy and combat-intrigue
# decisions value the CHANGE in expected reward, against the value of
# keeping the troop / card for later.
# ---------------------------------------------------------------------------

def _phi(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / 1.41421356))


def _rounds_left(gs: GameState) -> int:
    return max(0, 10 - gs.round)


def _reward_worth(gs: GameState, pid: int, r) -> float:
    if not r:
        return 0.0
    v = 3.0 * r.get("vp", 0)
    v += sum(_RES_VALUE.get(k, 0.3) * n for k, n in r.items()
             if isinstance(n, (int, float)) and k != "vp")
    return v * _game_urgency(gs, pid)


def _skill_swords(gs: GameState, pid: int, assume_commander: bool = False) -> float:
    """Sardaukar sword skills (Canny / Fierce / Loyal) — they count only while
    one of the player's commanders is in the Conflict. `assume_commander`:
    as if one were sent in now."""
    bl = getattr(gs, "bl", None)
    if bl is None or not _TUNE.get("skills", 0) or not getattr(gs.players[pid], "skills", None):
        return 0.0
    if bl.commanders_in_conflict.get(pid, 0) > 0:
        return float(bl.strength_bonus(pid))
    if not assume_commander:
        return 0.0
    bl.commanders_in_conflict[pid] = 1
    try:
        return float(bl.strength_bonus(pid))
    finally:
        bl.commanders_in_conflict[pid] = 0


def _opp_fight(gs: GameState, q: int, viewer: Optional[int] = None):
    """(expected final strength, spread, can_place) of opponent q, as player
    `viewer` sees it. can_place may be a probability (learned forecast:
    P(in the fight))."""
    mu, sd, can = _opp_fight_rule(gs, q, viewer)
    lam = _TUNE.get("fight_learned", 0)
    if lam and gs.phase.name != "COMBAT" and q not in gs.players_revealed:
        from src.ai.fight_forecast import rival
        r = rival(gs, q)
        if r is not None:
            p_in, lmu, lsd = r
            lam = min(1.0, lam)
            return (mu + lam * (lmu - mu), sd + lam * (lsd - sd),
                    float(can) + lam * (p_in - float(can)))
    return mu, sd, can


def _opp_fight_rule(gs: GameState, q: int, viewer: Optional[int] = None):
    p = gs.players[q]
    units = gs.troops_in_conflict.get(q, 0) + gs.sandworms_in_conflict.get(q, 0)
    vis = 2 * gs.troops_in_conflict.get(q, 0) + 3 * gs.sandworms_in_conflict.get(q, 0) \
        + gs.swords_this_reveal.get(q, 0)
    pot = 0.0
    if gs.phase.name != "COMBAT":
        if p.agents_available > 0 and p.troops_garrison > 0:
            pot += 2.0 * min(p.troops_garrison, 2 + p.agents_available) * 0.45
        if q not in gs.players_revealed:
            if _TUNE.get("opp_pool", 0):
                # public: q's owned cards minus discard / played / trashed =
                # what hand + deck hold. Cards left to reveal = hand minus
                # the ones still going out with agents.
                pool = p.hand + p.deck
                if pool:
                    mean = sum(getattr(c, "swords", 0) or 0 for c in pool) / len(pool)
                    pot += mean * max(0, len(p.hand) - p.agents_available)
            else:
                pot += 2.0                               # reveal swords, on average
    ci_var = 0.0
    if _TUNE.get("intrigue_count", 0):
        # each of q's Intrigues is an unseen card: its swords on average, from
        # what is left after the played (public) ones
        m, v = _unseen_intrigue_swords(gs, viewer)
        k = len(p.intrigue_cards)
        pot += _TUNE["intrigue_count"] * k * m
        ci_var = k * v
    else:
        pot += 1.2 * min(len(p.intrigue_cards), 3) * 0.5     # combat intrigues maybe
    vis += _skill_swords(gs, q)
    bl = getattr(gs, "bl", None)
    if bl is not None and _TUNE.get("opp_cmdr", 0) and gs.phase.name != "COMBAT"             and p.agents_available > 0 and p.solari >= bl.commander_cost(q)             and (p.commanders_supply > 0 or any(bl.commander_on_space.values()))             and not bl.commanders_in_conflict.get(q, 0):
        pot += _TUNE["opp_cmdr"] * 0.4 * (2.0 + _skill_swords(gs, q, assume_commander=True))
    if bl is not None and gs.phase.name != "COMBAT" and getattr(p, "commanders_garrison", 0) > 0             and not bl.commanders_in_conflict.get(q, 0) and p.agents_available > 0:
        pot += 0.5 * _skill_swords(gs, q, assume_commander=True)
    can = units > 0 or (gs.phase.name != "COMBAT" and p.agents_available > 0
                        and p.troops_garrison > 0)
    if ci_var:
        return vis + pot, (1.5 + 0.9 * (pot ** 0.5)) + 0.5 * ci_var ** 0.5, can
    return vis + pot, 1.5 + 0.9 * (pot ** 0.5), can


_CALIB: Dict[int, object] = {}


def _p_ahead(mu: float, sd: float, can, my_strength: float, revealed: bool = False) -> float:
    """P(this rival finishes ahead of me). fight_calib on: the odds fitted on
    real outcomes (src/selfplay/fight_calib.py) - the AI ends ~2.5 stronger
    than it estimates, and Reveal / Intrigue swings make the extremes less
    sure - for the forecast blend in use (nearest fight_learned weight), with
    separate odds against a rival who has already revealed (user, 2026-10-18:
    only Combat Intrigues are left to come, so deploy what the fight needs)."""
    z = (mu - my_strength) / sd
    ver = int(_TUNE.get("fight_calib", 0))
    if ver:
        # fight_calib N > 1: the refit on Best AI's own play, calibration_vN.json
        # (src.selfplay.evolve_loop --recalib)
        if ver not in _CALIB:
            try:
                import json, os
                name = "calibration.json" if ver == 1 else f"calibration_v{ver}.json"
                path = os.path.join(os.path.dirname(__file__), "..", "..", "models_fight", name)
                with open(path) as fh:
                    _CALIB[ver] = {float(k): v for k, v in json.load(fh).items()}
            except (OSError, ValueError):
                _CALIB[ver] = None
        m = _CALIB[ver]
        if m:
            c = m[min(m, key=lambda k: abs(k - min(1.0, _TUNE.get("fight_learned", 0.0))))]
            c = c.get("rev" if revealed else "unrev", c)
            u = max(-30.0, min(30.0, c["a"] * z + c["b"]))
            return float(can) / (1.0 + math.exp(-u))
    return _phi(z) * float(can)


def _combat_space_ev(gs: GameState, pid: int, a) -> float:
    """What an Agent on combat space `a` is worth for the Conflict: the best
    deploy it opens (2 + the troops the space recruits), priced like the
    deploy decision itself - change in expected reward minus the troops kept.
    Against rivals who have revealed the odds are near-certain (user,
    2026-10-18: a late Agent - the Swordmaster's 3rd - deploys knowing what
    the fight needs)."""
    p = gs.players[pid]
    rec = sum(e["troops"] for e in gs.get_space_effects_preview(a.space_name, a.space_option)
              if isinstance(e, dict) and isinstance(e.get("troops"), (int, float)) and e["troops"] > 0)
    avail = int(min(p.troops_garrison + rec, 2 + rec))
    # ci_plan: the Combat Intrigues in hand go in too once a unit is there
    ci = _TUNE.get("ci_plan", 0.0) * _own_combat_intrigue_swords(gs, pid) if _TUNE.get("ci_plan", 0.0) else 0.0
    mine0, u0 = _my_strength(gs, pid, add_swords=ci)
    ev0 = fight_ev(gs, pid, mine0, u0)
    best = 0.0
    for n in range(1, avail + 1):
        mine, u = _my_strength(gs, pid, add_troops=n, add_swords=ci)
        keep = max(0, n - rec) * _troop_hold_value(gs)          # recruited troops weren't in the garrison
        best = max(best, fight_ev(gs, pid, mine, u) - ev0 - keep
                   - n * _TUNE.get("deploy_hold", 0.4) * min(1.0, _rounds_left(gs) / 2.0))
    return best


def fight_ev(gs: GameState, pid: int, my_strength: float, my_units: int,
             worm: Optional[bool] = None) -> float:
    """Expected value of this Conflict's rewards for pid at this strength."""
    cc = gs.current_conflict
    if cc is None or my_units <= 0:
        return 0.0
    k2 = 2.0 if _worm_doubled(gs, pid, worm) else 1.0
    rewards = [_conflict_worth(gs, pid, worm),
               k2 * _reward_worth(gs, pid, cc.second_place_reward),
               k2 * _reward_worth(gs, pid, cc.third_place_reward) if gs.num_players >= 4 else 0.0]
    dist = [1.0]                         # P(k opponents ahead of me)
    for q in range(gs.num_players):
        if q == pid:
            continue
        mu, sd, can = _opp_fight(gs, q, pid)
        pa = _p_ahead(mu, sd, can, my_strength, q in gs.players_revealed)
        nd = [0.0] * (len(dist) + 1)
        for k, pk in enumerate(dist):
            nd[k] += pk * (1 - pa)
            nd[k + 1] += pk * pa
        dist = nd
    ev = sum(pk * (rewards[k] if k < len(rewards) else 0.0) for k, pk in enumerate(dist))
    pen = _TUNE.get("out_of_rewards", 0.0)
    if pen:
        # finishing below every rewarded place: the units in the fight are wasted
        paid = sum(1 for r in rewards if r > 0)
        p_out = sum(dist[paid:])
        ev -= pen * p_out * my_units * _RES_VALUE["troops"] * min(1.0, _rounds_left(gs) / 3.0)
    return ev


def _my_strength(gs: GameState, pid: int, add_troops: int = 0, add_swords: float = 0.0,
                 add_cmd: Optional[int] = None):
    t = gs.troops_in_conflict.get(pid, 0) + add_troops
    w = gs.sandworms_in_conflict.get(pid, 0)
    sw = gs.swords_this_reveal.get(pid, 0) + add_swords
    extra = None
    if _TUNE.get("fight_own", 0) and gs.phase.name != "COMBAT":
        from src.ai.fight_forecast import own_extra
        extra = own_extra(gs, pid)              # Reveal + Combat Intrigue swords to come
    if extra is not None:
        sw += extra
    elif gs.phase.name != "COMBAT" and pid not in gs.players_revealed:
        sw += sum(getattr(c, "swords", 0) or 0 for c in gs.players[pid].hand) * 0.7
    # add_cmd: how many of the added units are commanders (None: commanders first)
    sends_cmd = (add_cmd > 0) if add_cmd is not None else (
        add_troops > 0 and getattr(gs.players[pid], "commanders_garrison", 0) > 0)
    sw += _skill_swords(gs, pid, assume_commander=sends_cmd)
    return 2 * t + 3 * w + sw, t + w


_DENY_CACHE: Dict[tuple, Dict[str, float]] = {}


def _deny_value(gs: GameState, pid: int, card, lean: bool = False) -> float:
    """Taking `card` keeps it from opponents who still Reveal this round:
    sum over them of P(they'd buy it) x their value for it. Their buying
    power comes from public info only (src/ai/opp_model.py)."""
    from src.ai.opp_model import p_buys
    if gs.phase.name == "COMBAT":
        return 0.0
    tot = 0.0
    for q in range(gs.num_players):
        if q == pid or q in gs.players_revealed:
            continue
        qp = gs.players[q]
        key = (id(gs), gs.round, q, len(qp.deck), len(qp.discard), len(qp.hand),
               tuple(c.name for c in gs.imperium_row))
        vals = _DENY_CACHE.get(key)
        if vals is None:
            vals = {c.name: _acquire_card_value(gs, q, c, lean) for c in gs.imperium_row}
            if len(_DENY_CACHE) > 2048:
                _DENY_CACHE.clear()
            _DENY_CACHE[key] = vals
        tot += p_buys(gs, q, card, vals) * max(0.0, vals.get(card.name, 0.0))
    return tot


def _cmd_deploy_adj(gs: GameState, pid: int, a: GameAction) -> float:
    """Troops or commanders: the first commander in the Conflict switches the
    reveal skills on (sword skills are counted in _my_strength), but every
    commander sent goes to the commander supply after the fight and costs
    a re-recruit, where a plain troop is just a troop."""
    c = a.deploy_cmd
    bl = getattr(gs, "bl", None)
    if not c or bl is None:
        return 0.0
    p = gs.players[pid]
    s = 0.0
    if not bl.commanders_in_conflict.get(pid, 0) and pid not in gs.players_revealed:
        sk = getattr(p, "skills", set())
        s += (_RES_VALUE["troops"] * ("Hardy" in sk) + _RES_VALUE["spice"] * ("Driven" in sk)
              + _RES_VALUE["persuasion"] * ("Charismatic" in sk) + 1.5 * ("Desperate" in sk))
    s -= c * 0.4 * _RES_VALUE["solari"] * bl.commander_cost(pid) * min(1.0, _rounds_left(gs) / 3.0)
    return s


def _card_swords(ic) -> float:
    sw = 0.0
    for e in ic.effects:
        fe = _flatten(e)
        v = fe.get("swords", 0)
        sw += v if isinstance(v, (int, float)) else 2.0
        for sub in fe.values():
            if isinstance(sub, dict) and isinstance(sub.get("swords"), (int, float)):
                sw += sub["swords"]
    return sw


def _troop_hold_value(gs: GameState) -> float:
    return _RES_VALUE["troops"] * _TUNE.get("troop_hold", 1.0) * min(1.0, _rounds_left(gs) / 3.0)


def _tsmf_short(gs: GameState, pid: int) -> int:
    """TSMF copies still missing from the 2-a-game goal (0 once the stack is empty)."""
    if not gs.reserve_spice_must_flow:
        return 0
    return max(0, 2 - getattr(gs.players[pid], "tsmf_acquired", 0))


_STARTERS = frozenset(("Reconnaissance", "Diplomacy", "Dune, the Desert Planet",
                       "Dagger", "Convincing Argument", "Seek Allies", "Signet Ring"))


def _draw_persuasion(gs: GameState, pid: int) -> float:
    """Expected persuasion of one card drawn now (deck, or the discard that
    gets shuffled in when the deck is empty)."""
    p = gs.players[pid]
    pool = p.deck or p.discard
    if not pool:
        return 0.0
    return sum((c.persuasion or 0) for c in pool) / len(pool)


def _move_draws(gs: GameState, a, played) -> float:
    """Cards agent move `a` draws: the space's effects and the played card's agent box."""
    n = 0.0
    effs = list(gs.get_space_effects_preview(a.space_name, a.space_option))
    effs += list(getattr(played, "agent_effects", []) or [])
    for e in effs:
        if isinstance(e, dict) and isinstance(e.get("draw"), (int, float)):
            n += e["draw"]
    return n


def _reveal_persuasion_after(gs: GameState, pid: int, a, draws: bool = False) -> float:
    """Persuasion this player can expect on their Reveal turn if they make
    agent move `a` now: the cards left in hand (the lowest-persuasion ones
    go out with the remaining agents), persuasion already gained this round,
    the space's reveal persuasion (Assembly Hall, High Council) and the
    played card's agent-box persuasion. draws: also the cards the move draws,
    at the deck's average persuasion."""
    p = gs.players[pid]
    hand = list(p.hand)
    played = next((c for c in hand if c.name == a.card_name), None)
    if played is not None:
        hand.remove(played)
    pers = sorted((c.persuasion or 0) for c in hand)
    left_agents = max(0, p.agents_available - 1)
    keep = pers[left_agents:]                        # spend the weakest as agents
    P = sum(keep) + gs.persuasion_pool.get(pid, 0)
    if p.has_councilor or a.space_name == "High Council":
        P += 2
    for eff in gs.get_space_effects_preview(a.space_name, a.space_option):
        if a.space_name != "High Council":
            P += eff.get("persuasion", 0) if isinstance(eff, dict) else 0
    for e in getattr(played, "agent_effects", []) or []:
        if isinstance(e, dict) and isinstance(e.get("persuasion"), (int, float)):
            P += e["persuasion"]
    if draws:
        P += _move_draws(gs, a, played) * _draw_persuasion(gs, pid)
    return P


def _smf_plan_value(gs: GameState, pid: int, a) -> float:
    """How far agent move `a` gets this player toward the 9 persuasion a
    The Spice Must Flow costs on the Reveal turn (user, 2026-10-18: build
    persuasion - High Council, draw spaces, keeping strong cards for the
    Reveal - to buy TSMF). 0 below 5 persuasion, 1 at 9+; nothing once the
    TSMF stack is empty."""
    if not gs.reserve_spice_must_flow:
        return 0.0
    P = _reveal_persuasion_after(gs, pid, a, draws=True)
    return max(0.0, min(1.0, (P - 5.0) / 4.0))


def _best_buy_value(gs: GameState, pid: int, P: float) -> float:
    best = 0.0
    cands = list(gs.imperium_row)
    for stack in (gs.reserve_prepare_the_way, gs.reserve_spice_must_flow):
        if stack:
            cands.append(stack[-1])
    for c in cands:
        if (c.cost or 0) <= P:
            best = max(best, _acquire_card_value(gs, pid, c))
    return best


def _landsraad_access_owned(p) -> int:
    n = 0
    for zone in (p.deck, p.discard, p.hand, p.in_play):
        for c in zone:
            if any(getattr(sy, "value", sy) == "landsraad" for sy in getattr(c, "access_symbols", ())):
                n += 1
    return n


_STRONG_Q = 4.0      # beats every starter card (Diplomacy is 3.9)
_TRASH_BUSY = False


def _trash_quality(gs: GameState, pid: int, card) -> float:
    """_card_play_quality plus the board access it ignores (Landsraad, city,
    desert), so Signet Ring ranks above Reconnaissance and Dune."""
    q = _card_play_quality(gs, pid, card)
    q += 0.3 * sum(1 for s_ in getattr(card, "access_symbols", ())
                   if getattr(s_, "value", s_) in ("landsraad", "city", "desert"))
    if card.name == "Signet Ring":                 # trashing it ends the Leader's ability
        q += 0.5 * _signet_value(gs, pid, None) * max(0, MAX_ROUNDS - gs.round)
    return q


def _trash_values(gs: GameState, pid: int) -> Dict[str, float]:
    """trash_model: {card name: value of trashing it} for the player's cards.
    Thinning card X means every other card is drawn more often for the rest
    of the game. Over D draws left in a deck of n cards that is worth
    (D / n) x (mean quality of the other cards - quality of X). It only pays
    once the deck holds strong cards you will keep drawing, so a deck with
    none gets a quarter of it, rising to full at two."""
    global _TRASH_BUSY
    p = gs.players[pid]
    owned = [c for z in (p.deck, p.discard, p.hand, p.in_play) for c in z]
    if len(owned) < 2 or _TRASH_BUSY:
        return {}
    _TRASH_BUSY = True                # card quality values a card's own "trash" effect
    try:
        qs = [_trash_quality(gs, pid, c) for c in owned]
    finally:
        _TRASH_BUSY = False
    tot = sum(qs)
    n_strong = sum(1 for q in qs if q >= _STRONG_Q)
    draws_left = max(0.0, 9.5 - gs.round) * 5.0
    last_landsraad = _landsraad_access_owned(p) < 2
    k = _TUNE.get("trash_model", 0)
    n_tsmf = sum(1 for c in owned if c.name == "The Spice Must Flow")
    tsmf_payoff = (any(i.name == "Secure Spice Trade" for i in p.intrigue_cards)
                   or any(t.name == "Holtzman Engine" for t in getattr(p, "techs", ())))
    out: Dict[str, float] = {}
    for card, qx in zip(owned, qs):
        if card.name in out:
            continue
        if card.name in ("Signet Ring", "Diplomacy"):
            out[card.name] = -1.0   # leader ability / the only lasting faction access
            continue
        if last_landsraad and any(getattr(sy, "value", sy) == "landsraad"
                                  for sy in getattr(card, "access_symbols", ())):
            out[card.name] = -2.0   # the last way to Swordmaster / High Council
            continue
        if card.name == "The Spice Must Flow" and tsmf_payoff:
            if n_tsmf <= 2:
                out[card.name] = -3.0   # Secure Spice Trade / Holtzman Engine need 2 in the deck
                continue
        mean_o = (tot - qx) / (len(owned) - 1)
        strong = n_strong - (1 if qx >= _STRONG_Q else 0)
        gate = 0.25 + 0.75 * min(1.0, strong / 2.0)
        if card.name == "The Spice Must Flow":
            gate = 1.0          # its VP came on acquire; the card itself is 1 spice
        v = gate * draws_left / len(owned) * (mean_o - qx)
        if card.name == "The Spice Must Flow" and not tsmf_payoff:
            v += 1.0
        if (getattr(card, "cost", 0) or 0) > 0 and card.name not in ("The Spice Must Flow", "Prepare the Way"):
            v -= 2.0     # a bought card: the play-quality yardstick misses conditional effects
        if card in p.hand and pid not in gs.players_revealed:
            v -= _card_reveal_worth(gs, pid, card)    # it won't be revealed this round
        out[card.name] = v * k
        if card.name in _STARTERS and card.name not in ("Signet Ring", "Diplomacy"):
            out[card.name] += _TUNE.get("starter_trash", 0.0)
    return out


def _trash_value(gs: GameState, pid: int, name: Optional[str]) -> float:
    if name is None:
        return 0.0
    return _trash_values(gs, pid).get(name, -1.0)


def _best_trash_value(gs: GameState, pid: int) -> float:
    """What a 'you may trash a card' effect is worth: the best trash now."""
    return max([0.0] + list(_trash_values(gs, pid).values()))


def _influence_loss_value(gs: GameState, pid: int, fac: str) -> float:
    """Cost of losing 1 influence = the worth of regaining it from one below
    (2 -> 1 loses a friendship VP; 1 -> 0 or 3 -> 2 costs little)."""
    p = gs.players[pid]
    cur = p.influence.get(fac, 0)
    if cur <= 0:
        return 0.0
    p.influence[fac] = cur - 1
    try:
        return _influence_gain_value(gs, pid, fac, 1)
    finally:
        p.influence[fac] = cur


def _choice_option_value(gs: GameState, pid: int, eff: dict) -> float:
    """Value of one option of a 'choose' decision (src/game/choices.py):
    its normal effects, any nested choice at its best, and the '_' keys."""
    from src.game import choices as CH
    p = gs.players[pid]
    v = 0.0
    normal = {k: x for k, x in eff.items() if not k.startswith("_") and k not in CH.CHOICE_KEYS}
    fe = _flatten(normal)
    v += _effect_value(gs, pid, fe) + 0.9 * fe.get("swords", 0)
    for k in CH.CHOICE_KEYS:
        if k in eff:
            built = CH.build(gs, p, k, eff[k])
            if built:
                v += max(_choice_option_value(gs, pid, o[2]) for o in built[2])
    lf = eff.get("_leader_fx")
    if lf and lf[0] in ("lose_troops", "lose_conflict_troops") and gs.current_conflict is not None:
        n = int(lf[1])
        from_conflict = n if lf[0] == "lose_conflict_troops" else max(0, n - p.troops_garrison)
        v -= 2.5 * from_conflict                 # 2 strength each, maybe a lost placement
    for q in eff.get("_recall_posts", []) + ([eff["_recall_post"]] if "_recall_post" in eff else []):
        v -= 0.8 * _spy_post_value(gs, pid, q)          # keep the better-placed Spies
    if lf and lf[0] in ("trash_in_play", "trash_hand", "trash_zone"):
        v += _trash_value(gs, pid, lf[2] if lf[0] == "trash_zone" else lf[1])
    if lf and lf[0] == "jessica_flip":
        v += _TUNE.get("leader_play", 1.0) * _jessica_flip_worth(gs, pid)
    if lf and lf[0] == "rm_repeat":                # the space's effects once more
        v += _space_repeat_value(gs, pid, lf[1], lf[2] if len(lf) > 2 else None)
    if lf and lf[0] == "lose_split" and gs.current_conflict is not None:
        v -= 2.5 * int(lf[2])                    # troops taken out of the Conflict
    if "_v" in eff:                      # Leader / Twisted option: its value hint
        hint = _flatten(dict(eff["_v"]))
        v += _effect_value(gs, pid, hint) + 0.9 * hint.get("swords", 0)
    if "_swap_intrigue" in eff:          # Imperial Privilege: swap a weak Intrigue
        ic = next((c for c in p.intrigue_cards if c.name == eff["_swap_intrigue"]), None)
        if ic is not None:
            v += 0.8 * (0.55 - max(-2.0, _intrigue_value(gs, pid, ic)))   # 0.55 = avg Intrigue
    if eff.get("_draw_top") and p.deck:
        v += 0.5 if p.deck[-1].name in _STARTERS else 1.4
    if eff.get("_trash_top") and p.deck:
        v += 1.4 if p.deck[-1].name in _STARTERS else -1.0
    if eff.get("_peek_after_shuffle"):
        v += 1.0
    if "_acquire_named" in eff:
        card = next((c for c in gs.imperium_row if c.name == eff["_acquire_named"]["card"]), None)
        if card is not None:
            v += _acquire_card_value(gs, pid, card) + (0.8 if eff["_acquire_named"].get("to_hand") else 0.0)
    same = "_lose_influence" in eff and eff.get("_gain_influences") == [eff["_lose_influence"]]
    for f in ([] if same else eff.get("_gain_influences", [])):
        v += _influence_gain_value(gs, pid, f, 1)
    for f in [] if same else (eff.get("_lose_influences", [])
                              + ([eff["_lose_influence"]] if "_lose_influence" in eff else [])):
        v -= _influence_loss_value(gs, pid, f)
    ld = getattr(p, "leader", None)
    lf_ = eff.get("_lose_influence")
    if (ld is not None and ld.name == "Steersman Y'rkoon" and lf_ is not None
            and eff.get("_gain_influences") == [lf_] and p.influence.get(lf_, 0) == 2
            and ld.state.get("nav_next", 0) < len(ld.state.get("nav", []))):
        v += 3.0                    # 2 -> 1 -> 2 reaches 2 again: plays the next Navigation card
    if "_manipulate" in eff:
        card = next((c for c in gs.imperium_row if c.name == eff["_manipulate"]), None)
        if card is not None:
            v += 0.4 * _acquire_card_value(gs, pid, card)
    if eff.get("_retreat_n"):
        n = eff["_retreat_n"]
        mine = gs.combat_strength.get(pid, 0)
        best_opp = max((gs.combat_strength.get(q, 0) for q in range(gs.num_players) if q != pid), default=0)
        v += 0.35 * n if mine + 2 < best_opp else -0.6 * n * min(2.0, _conflict_worth(gs, pid))
    for k, x in (eff.get("_pay") or {}).items():
        v -= _RES_VALUE.get(k, 0.5) * x
    if eff.get("_recall_spy"):
        v -= 1.2 * eff["_recall_spy"]
    if "_trash_discard" in eff:
        card = next((c for c in p.discard if c.name == eff["_trash_discard"]), None)
        if card is not None:
            v += 1.2 if card.name in _STARTERS else -0.35 * (card.cost or 0)
    if "_discard_named" in eff:
        card = next((c for c in p.hand if c.name == eff["_discard_named"]), None)
        if card is not None:
            v -= 0.3 + 0.3 * (card.persuasion + card.swords) + 0.4 * len(getattr(card, "agent_effects", []) or [])
    if "_take_contract" in eff:
        i = int(eff["_take_contract"])
        if i < len(gs.contract_bank):
            ct = gs.contract_bank[i]
            if _TUNE.get("contracts", 0):
                v += _take_contract_value(gs, pid, ct)
            else:
                v += 1.5 + sum(_effect_value(gs, pid, {k: x}) for k, x in ct.rewards.items()
                               if isinstance(x, (int, float)))
    if eff.get("_deploy_n"):
        v += 0.8 * eff["_deploy_n"] * min(2.0, _conflict_worth(gs, pid)) / 2.0
    if eff.get("_grant_emperor_access"):
        v += 0.8
    if eff.get("_city_spy"):
        v += 1.2
    return v


def _effect_with_choices_value(gs: GameState, pid: int, e: dict) -> float:
    """Value of an intrigue effect, counting each 'choose' at its best option."""
    from src.game import choices as CH
    if "choose_by_combat" in e:
        spec = e["choose_by_combat"]
        in_fight = gs.phase.name == "COMBAT" and (gs.troops_in_conflict.get(pid, 0)
                                                  + gs.sandworms_in_conflict.get(pid, 0)) > 0
        branch = spec.get("combat" if in_fight else "else") or {}
        e = {**{k: x for k, x in e.items() if k != "choose_by_combat"}, **branch}
    return _choice_option_value(gs, pid, e)


def _endgame_chance(gs: GameState, pid: int, cond: str) -> float:
    """Chance the endgame condition holds when the game ends (techs_3, water_3...)."""
    p = gs.players[pid]
    try:
        what, need = cond.rsplit("_", 1)
        need = int(need)
    except ValueError:
        return 0.5
    have = len(getattr(p, "techs", []) or []) if what == "techs" else getattr(p, what, 0)
    if not isinstance(have, (int, float)):
        return 0.5
    if have >= need:
        return 0.95
    short = need - have
    left = _rounds_left(gs)
    return max(0.05, min(0.8, 0.45 * left / (2.0 * short)))


def _endgame_hold_value(gs: GameState, pid: int, ic) -> float:
    """What keeping an Intrigue with an endgame VP part is worth: P(qualify) x
    the VP (Battlefield Research with 3 techs is a near-certain point)."""
    v = 0.0
    vp1 = _effect_value(gs, pid, {"vp": 1})
    for e in ic.effects:
        if not isinstance(e, dict):
            continue
        for k, sub in e.items():
            if k.startswith("if_bl_endgame_") and isinstance(sub, dict) and sub.get("vp"):
                v += _endgame_chance(gs, pid, k[len("if_bl_endgame_"):]) * sub["vp"] * (vp1 + 1.0)
    return v


_PEND_INF_KEYS = {"influence_choice": None, "influence_any": None,
                  "influence_emperor_or_spacing": ("emperor", "spacing_guild"),
                  "influence_bene_or_fremen": ("bene_gesserit", "fremen")}


def _pending_influence(gs: GameState, pid: int, fac: str) -> int:
    """Influence on track `fac` the player's held Plot Intrigues can add
    (Buy Access, Imperium Politics, Sietch Ritual, ...)."""
    n = 0.0
    for ic in gs.players[pid].intrigue_cards:
        for e in ic.effects:
            fe = _flatten(e) if isinstance(e, dict) else {}
            for k, v in fe.items():
                if not isinstance(v, (int, float)) or v <= 0:
                    continue
                if k in _PEND_INF_KEYS:
                    facs = _PEND_INF_KEYS[k]
                    if facs is None or fac in facs:
                        n += v
                elif _FACTION_OF.get(k) == fac:
                    n += v
    return int(min(2, n))


def _endgame_goals(gs: GameState, pid: int) -> list:
    """(what, need, vp) for every endgame condition on a held Intrigue."""
    out = []
    for ic in gs.players[pid].intrigue_cards:
        for e in ic.effects:
            if not isinstance(e, dict):
                continue
            for k, sub in e.items():
                if k == "if_tsmf_2":
                    out.append(("tsmf", 2, 1.4))              # 1 VP + 2 spice
                elif k.startswith("if_contracts_") and ic.name == "CHOAM Profits":
                    out.append(("contracts", int(k[len("if_contracts_"):]), 1.0))
                elif k == "if_bl_endgame_water_3":
                    out.append(("water", 3, 1.0))
                elif k == "if_bl_endgame_techs_3":
                    out.append(("techs", 3, 1.0))
                elif k == "spice_or_match_icon" and isinstance(sub, dict):
                    out.append(("icon:" + sub.get("icon", ""), 1, 1.0))
                elif k == "pay_then" and isinstance(sub, dict) and sub.get("vp")                         and (sub.get("cost") or {}).get("spice"):
                    out.append(("spice", int(sub["cost"]["spice"]), float(sub["vp"])))
    return out


def _goal_have(gs: GameState, pid: int, what: str) -> int:
    p = gs.players[pid]
    if what == "tsmf":
        return sum(1 for pile in (p.deck, p.hand, p.discard, p.in_play)
                   for c in pile if c.name == "The Spice Must Flow")
    if what == "contracts":
        return len(getattr(p, "contracts_completed", []) or [])
    if what == "techs":
        return len(getattr(p, "techs", []) or [])
    return int(getattr(p, what, 0) or 0)


def _goal_chance(have: float, need: int, left: int) -> float:
    if have >= need:
        return 0.95
    return max(0.05, min(0.8, 0.45 * left / (2.0 * (need - have))))


def _endgame_goal_bonus(gs: GameState, pid: int, what: str, amt: float) -> float:
    """eg_plan: gaining `amt` of `what` raises the chance a held endgame
    Intrigue pays -> that change x its VP."""
    k = _TUNE.get("eg_plan", 0.0)
    if not k or amt <= 0:
        return 0.0
    goals = [g for g in _endgame_goals(gs, pid) if g[0] == what]
    if not goals:
        return 0.0
    left = _rounds_left(gs)
    have = _goal_have(gs, pid, what)
    vp1 = _effect_value(gs, pid, {"vp": 1})
    return k * sum(vp * vp1 * (_goal_chance(have + amt, need, left) - _goal_chance(have, need, left))
                   for _, need, vp in goals)


def _next_hand_quality(gs: GameState, pid: int, extra_draws: float = 0.0) -> float:
    """prep_draw: expected quality of next round's 5-card hand. Cards left in
    the deck come first; once it runs out, everything else (discard, cards in
    play and in hand, which go to the discard at Recall) is shuffled in."""
    p = gs.players[pid]
    deck = list(p.deck)
    rest = list(p.discard) + list(p.in_play) + list(p.hand)
    if not deck and not rest:
        return 0.0
    q = {}

    def qual(c):
        if c.name not in q:
            q[c.name] = _card_play_quality(gs, pid, c)
        return q[c.name]

    left = max(0.0, len(deck) - extra_draws)
    m_deck = sum(qual(c) for c in deck) / len(deck) if deck else 0.0
    m_all = sum(qual(c) for c in deck + rest) / len(deck + rest)
    from_deck = min(5.0, left)
    # drawn cards land in hand -> discard, so the reshuffle sees the whole deck
    return from_deck * m_deck + (5.0 - from_deck) * m_all


def _intrigue_value(gs: GameState, pid: int, ic) -> float:
    """How good is it to play this Intrigue right now?"""
    p = gs.players[pid]
    in_combat = gs.phase.name == "COMBAT"
    from src.game.intrigue.intrigue import IntrigueTiming
    ts = ic.timing if isinstance(ic.timing, (set, tuple, list, frozenset)) else (ic.timing,)
    v = 0.0
    for e in ic.effects:
        v += _effect_with_choices_value(gs, pid, e)   # swords counted inside
    if IntrigueTiming.COMBAT in ts:
        if in_combat and not _TUNE.get("fight_model", 1):
            worth = _conflict_worth(gs, pid)
            mine = gs.combat_strength.get(pid, 0)
            opp = max((gs.combat_strength.get(q, 0)
                       for q in range(gs.num_players) if q != pid), default=0)
            v += 1.5 * worth if abs(mine - opp) <= 5 else 0.3 * worth
        elif in_combat:
            # replace the flat sword value with what the swords change: the
            # expected Conflict reward, minus keeping the card for later
            sw = _card_swords(ic)
            mine, units = _my_strength(gs, pid)
            gain = fight_ev(gs, pid, mine + sw, units) - fight_ev(gs, pid, mine, units)
            hold = 0.45 * sw * _TUNE.get("ci_hold", 1.0) * min(1.0, _rounds_left(gs) / 3.0)
            # (the flat sword value counted above is replaced by `gain`)
            v += gain - (0.9 + _RES_VALUE.get("swords", 0.3)) * sw - hold
        else:
            v -= 3.0                            # save combat intrigues for combat
    if IntrigueTiming.ENDGAME in ts and not gs.game_over:
        v -= max(5.0, _endgame_hold_value(gs, pid, ic))   # never waste an endgame card early
    if list(ts) == [IntrigueTiming.PLOT] and not in_combat:
        v -= _TUNE.get("plot_hold", 0.6)        # keeping a plot intrigue has value
    if _TUNE.get("contracts", 0) and not in_combat \
            and getattr(p, "visited_maker_this_turn", False):
        # spice that tops this turn's harvest up to a held Harvest contract
        add = sum(e.get("spice", 0) for e in ic.effects
                  if isinstance(e, dict) and isinstance(e.get("spice"), (int, float)))
        have = getattr(p, "harvest_spice_this_turn", 0)
        for ct in p.contracts_active:
            if ct.contract_type.value == "harvest" and add > 0 and \
                    have < ct.trigger_condition.get("min_spice", 3) <= have + add:
                v += _TUNE["contracts"] * _contract_reward_value(gs, pid, ct)
    return v


def _worm_doubled(gs: GameState, pid: int, worm: Optional[bool]) -> bool:
    """Is pid's reward doubled (a sandworm of theirs in the Conflict)? Only
    under worm_model; `worm` forces the answer (valuing a summon)."""
    m = _TUNE.get("worm_model", 0)
    if not m:
        return False
    if worm is not None:
        return worm
    if m == 2:                       # 2: summon / wall values only
        return False
    return gs.sandworms_in_conflict.get(pid, 0) > 0


def _worms_blocked(gs: GameState) -> bool:
    from src.game.board.board import SHIELD_WALL_PROTECTED
    cc = gs.current_conflict
    return cc is None or (gs.shield_wall_intact and cc.location in SHIELD_WALL_PROTECTED)


def _worm_gain(gs: GameState, pid: int, n: float = 1.0, wall_open: bool = False) -> float:
    """worm_model: expected-reward change from n sandworms of pid's joining
    this Conflict (3 strength each, and the doubled reward)."""
    if gs.current_conflict is None:
        return 0.0
    if _worms_blocked(gs) and not wall_open:
        return 0.0
    mine, u = _my_strength(gs, pid)
    had = gs.sandworms_in_conflict.get(pid, 0) > 0
    ev0 = fight_ev(gs, pid, mine, u, worm=had) if u > 0 else 0.0
    ev1 = fight_ev(gs, pid, mine + 3 * n, u + n, worm=True)
    return max(0.0, ev1 - ev0)


def _conflict_worth(gs: GameState, pid: int, worm: Optional[bool] = None) -> float:
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
    doubled = _worm_doubled(gs, pid, worm)
    v = {1: 0.5, 2: 1.8, 3: 2.8}.get(cc.conflict_level, 1.0)
    if r.get("control"):
        v += 1.6
    flat = 3.0 * r.get("vp", 0) + sum(_RES_VALUE.get(k, 0.3) * n for k, n in r.items()
                                      if isinstance(n, (int, float)))
    v += flat * (2.0 if doubled else 1.0)
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
            if doubled:                           # the worm's 2nd application
                v += (2.8 if getattr(p_conv, res, 0) >= 2 * cost else 0.6) * cvp
        elif k == "may_pay_troops_for_vp":
            cost, cvp = val.get("cost", 1), val.get("vp", 1)
            have = gs.troops_in_conflict.get(pid, 0)
            v += (2.4 if have >= cost else 1.0) * cvp
            if doubled:
                v += (2.4 if have >= 2 * cost else 0.6) * cvp
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
    # Battle icons. The conflict's icon is known at round start, so the AI can
    # aim its combat at the conflicts that build toward icon-pair VP, not just
    # the ones that print a VP.
    icon = getattr(cc.battle_icon, "value", None)
    held = gs.players[pid].battle_icons
    if icon == "wild":
        v += 1.8                                  # pairs with anything at Endgame
    elif icon and icon in held:
        v += 4.0                                  # completes a guaranteed 1 VP now
    elif icon and _TUNE.get("eg_plan", 0.0) and any(
            g[0] == "icon:" + icon for g in _endgame_goals(gs, pid)):
        v += 1.6 + 2.4 * min(1.0, _TUNE["eg_plan"])   # the held icon Intrigue pairs it at Endgame
    elif icon:
        # first icon of this type — a half-pair. ~0.5 VP in expectation once you
        # win another of the same; worth more to a player who already wins
        # combat regularly (more icons in hand = more chances the next win pairs).
        v += 1.6 + 0.8 * min(2, sum(1 for x in held if x != "wild"))
    return v * _game_urgency(gs, pid)


def _wall_break_value(gs: GameState, pid: int, with_worm: bool = False) -> float:
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
    # wall_future (user, 2026-10-14): with Maker Hooks from round 5 on, opening
    # the wall pays later too - the round 7+ VP battles (Imperial Basin,
    # Arrakeen, Spice Refinery) are the protected ones, and a worm doubles them
    wf = _TUNE.get("wall_future", 0.0)
    if wf and p.has_maker_hooks and gs.round >= 5:
        later = wf * (1.0 + 0.3 * (gs.round - 4))
        if cc is None or cc.location not in SHIELD_WALL_PROTECTED:
            return later
        return later + 2.5 * _game_urgency(gs, pid)
    if _TUNE.get("worm_model", 0) in (1, 2) and cc is not None:
        if cc.location not in SHIELD_WALL_PROTECTED:
            # nothing to open this round; later protected conflicts open for
            # everyone alike
            return 0.3 if p.has_maker_hooks else -0.5 - 0.4 * sum(
                1 for q in gs.players if q.id != pid and q.has_maker_hooks)
        mine = _worm_gain(gs, pid, wall_open=True) if (
            p.has_maker_hooks and p.agents_available > 0) or with_worm else 0.0
        theirs = sum(_worm_gain(gs, q.id, wall_open=True) for q in gs.players
                     if q.id != pid and q.has_maker_hooks and q.agents_available > 0)
        return 0.8 * mine - 0.4 * theirs
    if (p.has_maker_hooks and cc is not None
            and cc.location in SHIELD_WALL_PROTECTED
            and _conflict_worth(gs, pid) > 1.5):
        return 2.5 * _game_urgency(gs, pid)
    rival_hooks = sum(1 for q in gs.players if q.id != pid and q.has_maker_hooks)
    return -0.5 - 0.4 * rival_hooks


def _sandworm_value(gs: GameState, pid: int, wall_open: bool = False) -> float:
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
    if getattr(gs.players[pid], "no_deploy_this_turn", False):
        return 0.0                                # Shaddam's Signet turn: the worm can't come
    if _TUNE.get("worm_model", 0) in (1, 2):
        return _worm_gain(gs, pid, wall_open=wall_open)
    worth = _conflict_worth(gs, pid)
    if worth <= 0.5:
        return 0.6                                # bank hooks/board-state for later
    return 0.5 * worth


def _card_reveal_worth(gs: GameState, pid: int, card) -> float:
    """What a card in hand adds if it stays for the Reveal turn."""
    v = 0.8 * (card.persuasion or 0) + 0.4 * (card.swords or 0)
    for e in getattr(card, "reveal_effects", []) or []:
        v += 0.6 * _card_effect_value(gs, pid, e)
    return v


# Rough worth of each Leader's Signet Ring ability (src/game/leader/LEADERS.md),
# as an effect dict the heuristic already knows how to value.
_SIGNET_VALUE = {
    "Gurney Halleck": {"troops": 1}, "Feyd-Rautha Harkonnen": {"spy": 1},
    "Lady Amber Metulli": {"solari": 1}, "Lady Jessica": {"intrigue": 1},
    "Reverend Mother Jessica": {"water": 0.6}, "Lady Margot Fenring": {"spy": 1},
    "Muad'Dib": {"draw": 1}, "Princess Irulan": {"spice": 1}, "Shaddam Corrino IV": {"solari": 1, "troops": 1},
    "Staban Tuek": {"spy": 1, "solari": 1}, "Chani": {"draw": 1}, "Count Hasimir Fenring": {"solari": 1},
    "Duncan Idaho": {"swords": 2}, "Esmar Tuek": {"spice": 1}, "Gaius Helen Mohiam": {"spy": 1},
    "Kota Odax of Ix": {"spice": 1}, "Liet Kynes": {"solari": 0.7}, "Piter De Vries": {"troops": 0.6},
}


def _signet_value(gs: GameState, pid: int, space: str) -> float:
    p = gs.players[pid]
    ld = getattr(p, "leader", None)
    if ld is None or ld.signet is None:
        return 0.0
    eff = dict(_SIGNET_VALUE.get(ld.name, {}))
    if ld.name == "Duncan Idaho" and gs.current_conflict is None:
        eff = {}
    if ld.name == "Lady Amber Metulli" and any(p.alliances.values()):
        eff["spice"] = 1
    if ld.name == "Liet Kynes":                  # depends on where the Agent goes
        from src.game.board.board import SPACE_ICONS
        icon = SPACE_ICONS.get(space)
        eff = {"city": {"solari": 1}, "desert": {"spice": 1}}.get(icon, {})
        if icon == "landsraad" and p.influence.get("emperor", 0) >= 2:
            eff = {"water": 1}
    if ld.name == "Feyd-Rautha Harkonnen" and ld.state.get("feyd") == "end":
        eff = {}
    fe = _flatten(eff)
    v = _effect_value(gs, pid, fe) + 0.9 * fe.get("swords", 0)
    if ld.name == "Lady Jessica" and p.spice >= 1 and _rounds_left(gs) >= 1:
        v += _TUNE.get("leader_play", 1.0) * 0.5 * _effect_value(gs, pid, {"draw": 1})
    return v


def _space_repeat_value(gs: GameState, pid: int, space: str, option=None) -> float:
    """Worth of the effects printed on a space (Reverend Mother repeats them)."""
    return sum(_effect_value(gs, pid, e) for e in gs.get_space_effects_preview(space, option))


def _jessica_flip_worth(gs: GameState, pid: int) -> float:
    """Becoming Reverend Mother: a water-paid repeat of Bene Gesserit / Fremen
    spaces each turn from now on (beyond the memories' draws and troops)."""
    return min(4.0, 0.7 * _rounds_left(gs))


def _leader_space_value(gs: GameState, pid: int, a) -> float:
    """Leader passives that depend on where the Agent goes (user, 2026-10-17:
    the AI rarely took Jessica to the Bene Gesserit, so she seldom flipped)."""
    p = gs.players[pid]
    ld = getattr(p, "leader", None)
    if ld is None:
        return 0.0
    from src.game.board.board import SPACE_ICONS, SPACE_MANDATORY_COSTS
    icon = SPACE_ICONS.get(a.space_name)
    if ld.name == "Lady Jessica" and not ld.state.get("flipped"):
        n = ld.state.get("memories", 0)
        if icon == "bene_gesserit" and n > 0:
            return _effect_value(gs, pid, {"draw": n, "troops": n}) + _jessica_flip_worth(gs, pid)
        return 0.0
    if (ld.name == "Reverend Mother Jessica" and icon in ("bene_gesserit", "fremen")
            and not ld._per_turn_flags.get("rm")):
        need = 1 + SPACE_MANDATORY_COSTS.get(a.space_name, {}).get("water", 0)
        if p.water >= need:
            rep = _space_repeat_value(gs, pid, a.space_name, a.space_option) - _RES_VALUE.get("water", 1.0)
            return max(0.0, rep)
    return 0.0


def _card_agent_play_value(gs: GameState, pid: int, a) -> float:
    """
    Agent move: the played card's agent box at its real (known now) condition
    values, minus the Reveal value it gives up relative to the cheapest card
    in hand to send. Relative, so agent-vs-Reveal balance is unchanged and
    only the choice of card moves.
    """
    global _PLAY_CTX
    from src.game.board.board import MAKER_SPACE_OPTIONS
    p = gs.players[pid]
    card = next((c for c in p.hand if c.name == a.card_name), None)
    if card is None:
        return 0.0
    ctx = {"spy_recalled": 1.0 if (a.use_gather_intelligence or a.use_infiltrate) else 0.0,
           "agent_to_maker": 1.0 if a.space_name in MAKER_SPACE_OPTIONS else 0.0,
           "faction_agent": 1.0 if gs._faction_for_space(a.space_name) else 0.0}
    for short, tag in (("bene", "bene_gesserit"), ("fremen", "fremen"),
                       ("emperor", "emperor"), ("spacing", "spacing_guild")):
        others = sum(1 for c in p.in_play
                     if any(getattr(t, "value", t) == tag for t in getattr(c, "tags", ())))
        ctx["tag_other_" + short] = 1.0 if others else 0.0
    _PLAY_CTX = ctx
    try:
        v = sum(_card_effect_value(gs, pid, e) for e in card.agent_effects or [])
        if card.name == "Signet Ring":
            v += _signet_value(gs, pid, a.space_name)
    finally:
        _PLAY_CTX = None
    lost = _card_reveal_worth(gs, pid, card)
    cheapest = min(_card_reveal_worth(gs, pid, c) for c in p.hand)
    return v - (lost - cheapest)


def _spy_recall_cost(gs: GameState, pid: int, a) -> float:
    """A recalled Spy is a board asset gone: its post value, fading in the
    last rounds (fewer turns left to use it)."""
    from src.game.board.board import SPACE_TO_OBSERVATION_POSTS
    p = gs.players[pid]
    posts = [q for q in SPACE_TO_OBSERVATION_POSTS.get(a.space_name, set())
             if p.has_spy_at(q)]
    n = int(bool(a.use_gather_intelligence)) + int(bool(a.use_infiltrate))
    if not posts or not n:
        return 0.0
    fade = max(0.2, min(1.0, (MAX_ROUNDS - gs.round) / 3.0))
    per = sum(_spy_post_value(gs, pid, q) for q in posts[:n]) / min(n, len(posts))
    return 0.6 * fade * per * n


# --- CHOAM contracts --------------------------------------------------------

def _contract_cards_owned(p) -> int:
    """Owned cards that pay off per completed contract (Interstellar Trade,
    Delivery Agreement, Cargo Runner, ...)."""
    n = 0
    for z in (p.deck, p.discard, p.hand, p.in_play):
        for c in z:
            for e in (getattr(c, "agent_effects", []) or []) + (getattr(c, "reveal_effects", []) or []):
                if isinstance(e, dict) and any(
                        k == "persuasion_per_contract" or k == "choose_by_contracts"
                        or k.startswith("if_contracts_") for k in e):
                    n += 1
                    break
    return n


def _contract_reward_value(gs: GameState, pid: int, ct) -> float:
    """Value of completing `ct` now: its rewards, plus the completed-contract
    count itself (cards that scale with it)."""
    v = 0.0
    for k, x in ct.rewards.items():
        if isinstance(x, (int, float)):
            v += _effect_value(gs, pid, {k: x})
        elif isinstance(x, dict):
            v += 0.6 * _card_effect_value(gs, pid, x)
    return v + 0.4 + 0.8 * _contract_cards_owned(gs.players[pid])


def _contract_finish_chance(gs: GameState, pid: int, ct) -> float:
    """Rough chance of completing a contract taken now."""
    kind = ct.contract_type.value
    if kind == "immediate":
        return 1.0 if (not ct.requires_intrigue()
                       or gs.players[pid].intrigue_cards) else 0.0
    rounds_left = max(0, MAX_ROUNDS - gs.round)
    if kind == "acquire_card":                     # The Spice Must Flow
        return min(0.9, 0.3 + 0.2 * rounds_left) if gs.round >= 3 else 0.6
    per_round = 0.5 if kind == "board_space" else 0.3
    return min(0.9, per_round * (rounds_left + 0.3))


def _take_contract_value(gs: GameState, pid: int, ct=None) -> float:
    """Taking a contract: a specific one, or the best one on the board (else
    the 2 solari you get when none are left)."""
    if ct is not None:
        v = _contract_finish_chance(gs, pid, ct) * _contract_reward_value(gs, pid, ct)
        stack = _TUNE.get("contract_stack", 0.0)
        key = _contract_key(ct)
        if stack and key is not None:            # completes together with a held one
            v += stack * sum(_contract_key(c) == key for c in gs.players[pid].contracts_active)
        return v
    pool = list(getattr(gs, "contracts_on_board", []) or [])
    if not getattr(gs, "use_choam", False) or not pool:
        return 2 * _RES_VALUE["solari"]
    return max(_take_contract_value(gs, pid, c) for c in pool)


def _turn_spice_estimate(gs: GameState, pid: int, a) -> float:
    """Spice this turn would total if agent move `a` is made: already gained
    this turn + the space (incl. maker bonus) + the played card's agent box."""
    p = gs.players[pid]
    n = getattr(p, "harvest_spice_this_turn", 0)
    for eff in gs.get_space_effects_preview(a.space_name, a.space_option):
        x = eff.get("spice", 0) if isinstance(eff, dict) else 0
        n += x if isinstance(x, (int, float)) and x > 0 else 0
    n += gs.maker_bonus_spice.get(a.space_name, 0)
    card = next((c for c in p.hand if c.name == a.card_name), None)
    for e in (getattr(card, "agent_effects", []) or []) if card else []:
        x = e.get("spice", 0) if isinstance(e, dict) else 0
        n += x if isinstance(x, (int, float)) and x > 0 else 0
        if isinstance(e.get("if_agent_to_maker"), dict):
            n += e["if_agent_to_maker"].get("spice", 0)
    return n


def _contract_completion_value(gs: GameState, pid: int, a) -> float:
    """Agent move: rewards of the held contracts it completes."""
    from src.game.board.board import MAKER_SPACES
    p = gs.players[pid]
    if not getattr(gs, "use_choam", False) or not p.contracts_active:
        return 0.0
    v = 0.0
    spice = None
    done = 0
    for ct in p.contracts_active:
        kind = ct.contract_type.value
        if kind == "board_space":
            if ct.trigger_condition.get("board_space") == a.space_name:
                v += _contract_reward_value(gs, pid, ct)
                done += 1
        elif kind == "harvest" and a.space_name in MAKER_SPACES:
            if spice is None:
                spice = _turn_spice_estimate(gs, pid, a)
            if spice >= ct.trigger_condition.get("min_spice", 3):
                v += _contract_reward_value(gs, pid, ct)
                done += 1
    if done >= 2:
        v += 2.0 * _TUNE.get("contract_stack", 0.0) * (done - 1)
    return v


def _contract_key(ct):
    """Where a contract completes: its board space, or 'harvest'."""
    kind = ct.contract_type.value
    if kind == "board_space":
        return ct.trigger_condition.get("board_space")
    return kind if kind == "harvest" else None


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
    vals = []
    for sp_name in spaces:
        sp = UPRISING_BOARD.get(sp_name)
        if sp is None:
            continue
        v = sum(_effect_value(gs, pid, eff)
                for eff in gs.get_space_effects_preview(sp_name))
        if sp.is_combat_space and gs.current_conflict is not None:
            v += _TUNE.get("spy_combat", 0.6) * _conflict_worth(gs, pid)
        fac = gs._faction_for_space(sp_name)
        if fac:
            v += 0.5 * _influence_gain_value(gs, pid, fac, 1)
        occ = gs.agent_on_space.get(sp_name)
        if occ is not None and occ != pid:
            v += 0.5                   # taken this round: the Spy is the way in (Infiltrate)
        vals.append(v)
    if not vals:
        return 0.5
    vals.sort(reverse=True)
    # a spy is a durable, reusable asset (repeated Spy-icon sends, or a later
    # Infiltrate / Gather Intelligence): a floor, plus the best space it
    # reaches, plus part of every other space it also covers (flexibility).
    score = vals[0] + _TUNE.get("spy_multi", 0.35) * sum(vals[1:])
    return 1.0 + _TUNE.get("spy_post", 0.5) * score


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
    if getattr(gs, "bl", None) is not None:
        from src.ai.bloodlines_heuristic import state_bonus
        score += state_bonus(gs, pid)
    return 1.0 / (1.0 + math.exp(-0.06 * score))


# ---------------------------------------------------------------------------

class Agent:
    name = "agent"

    def reset(self) -> None:
        pass

    def select_action(self, gs: GameState, pid: int,
                      valid_actions: List[GameAction]) -> GameAction:
        raise NotImplementedError


def _style_bonus(gs: GameState, pid: int, a: GameAction,
                 fixed: Optional[dict] = None) -> float:
    """P(objective | position) for the objective this action buys (or half
    of it for an agent move that opens the purchase); see the style knob.
    `fixed` replaces the probabilities with a fixed weight per objective
    (the fixated playstyles)."""
    from src.ai.playstyle import action_objective, objective_probs
    probs = fixed if fixed is not None else objective_probs(gs, pid)
    if probs is None:
        return 0.0
    obj = action_objective(gs, pid, a)
    if obj is not None:
        return probs[obj]
    if a.action_type != ActionType.AGENT_TURN or not a.space_name:
        return 0.0
    from src.game.bloodlines.rules import COMMANDER_SPACES, GREEN_SPACES
    bl = gs.bl
    b = 0.0
    if a.space_name in COMMANDER_SPACES and bl.recruit_options(pid, a.space_name):
        b += 0.5 * probs["commanders"]
    if a.space_name in GREEN_SPACES and bl.buyable_stacks(pid):
        b += 0.5 * probs["techs"]
    return b


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
                 temperature: float = 0.0, lean_deck: bool = False,
                 sm_boost: bool = True, faction_focus: bool = True,
                 bloodlines: bool = True, tuning: Optional[dict] = None,
                 near: float = 0.0):
        self.rng = random.Random(seed)
        # near > 0 (`heuristic:N<x>`): pick uniformly among the moves scored
        # within x points of the best - many Dune moves are about equally
        # good, so the AI shouldn't always play the single top score
        self.near = near
        self.tuning = {**TUNE_DEFAULTS, **(tuning or {})}
        self._phase_tuning = split_phases(tuning)
        # per-Leader weight multipliers, "L.<Leader>.<knob>": x (evolved per Leader)
        self._leader_mult: Dict[str, Dict[str, float]] = {}
        for k, v in (tuning or {}).items():
            if k.startswith("L.") and k.count(".") >= 2:
                _, ld, knob = k.split(".", 2)
                self._leader_mult.setdefault(ld, {})[knob] = float(v)
        self._leader_cache: Dict[tuple, dict] = {}
        # bloodlines=False (`heuristic:nobl`): decline every tech / commander /
        # activation - the baseline for checking the Bloodlines scoring pays off
        self.bloodlines = bloodlines
        self.book = opening_book if opening_book is not None else OpeningBook.default()
        self.temperature = temperature
        # sm_boost (2026-09-10, default ON): price Swordmaster + the solari that
        # buys it aggressively so the AI actually secures the 3rd agent in the
        # opening. `heuristic:sm0` reverts to the old min(22, 2.5r) pricing.
        self.sm_boost = sm_boost
        # faction_focus (2026-09-10, default ON): value faction-access cards as a
        # recurring +1-influence (~0.5 VP) engine when buying, so the AI builds a
        # deck that can actually push a track to friendship/alliance and keep
        # showing up in that faction's icon conflicts. `heuristic:ff0` disables.
        self.faction_focus = faction_focus
        # lean_deck (2026-09-09, EXPERIMENTAL, default OFF): adds a deck-dilution
        # penalty + a "bank the turn" option so the AI stops buying filler every
        # reveal and targets a ~13-15 card deck. Arena-tested against the flat
        # scorer: a lone lean player in a field of greedy buyers goes ~0.81x
        # fair when tuned aggressively and ~1.0x when gentle — the current
        # heuristic can't exploit a lean deck (Uprising has almost no on-demand
        # trashing, and unspent persuasion is wasted, so "buy less" just means
        # "weaker board"). Kept as a lever for the value-net retrain path, where
        # deck-quality features could let the net actually use it. Enable with
        # the `heuristic:lean` spec. The PTW-spam tax is separate and always on.
        self.lean_deck = lean_deck

    # -- per-action score ----------------------------------------------

    def score(self, gs: GameState, pid: int, a: GameAction) -> float:
        p = gs.players[pid]
        ld = getattr(p, "leader", None)
        if (a.action_type == ActionType.AGENT_TURN and a.card_name == "Signet Ring"
                and ld is not None and ld.name == "Shaddam Corrino IV"
                and not getattr(p, "no_deploy_this_turn", False)):
            # his Signet turn deploys nothing: no Conflict, worm or deploy value
            p.no_deploy_this_turn = True
            try:
                s = self._score_core(gs, pid, a)
            finally:
                p.no_deploy_this_turn = False
        else:
            s = self._score_core(gs, pid, a)
        if getattr(gs, "bl", None) is not None:
            k = _TUNE.get("style", 0.0)
            if k:
                s += k * _style_bonus(gs, pid, a)
            fix = {o: _TUNE.get("fix_" + o, 0.0) for o in
                   ("swordmaster", "high_council", "commanders", "techs")}
            if any(fix.values()):
                s += _style_bonus(gs, pid, a, fixed=fix)
        if a.action_type in (ActionType.ACQUIRE_CARD, ActionType.ACQUIRE_RESERVE):
            m = _TUNE.get("buy_mult", 1.0)
            if m != 1.0 and s > 0:
                s *= m
            is_tsmf = (a.acquire_card_name == "The Spice Must Flow"
                       or a.reserve_type == "spice_must_flow")
            if _TUNE.get("tsmf", 0.0) and is_tsmf:
                s += _TUNE["tsmf"]
            g = _TUNE.get("tsmf_goal", 0.0)
            if g:
                short = _tsmf_short(gs, pid)
                if is_tsmf:
                    s += g * short / 2.0
                elif short and gs.round <= 7 and a.action_type == ActionType.ACQUIRE_CARD:
                    card = next((c for c in gs.imperium_row if c.name == a.acquire_card_name), None)
                    if card is not None:      # persuasion that funds a 9-cost buy later
                        s += 0.1 * g * (card.persuasion or 0) * min(short, 1.5)
        if a.trash_from == "hand" and a.trash_card_name and pid not in gs.players_revealed:
            # a copy still in hand can be played or revealed this round: trash
            # one from the discard / play area first
            c = next((c for c in gs.players[pid].hand if c.name == a.trash_card_name), None)
            if c is not None:
                s -= 0.3 + 0.25 * ((c.persuasion or 0) + (c.swords or 0))
        if a.deploy_cmd is not None and not _TUNE.get("cmd_choice", 1):
            # A/B baseline: the old fixed rule, commanders deploy first
            if a.deploy_cmd != min(a.deploy_count, gs.players[pid].commanders_garrison):
                s -= 100.0
        return s

    def _score_core(self, gs: GameState, pid: int, a: GameAction) -> float:
        global _FACTION_FOCUS
        _FACTION_FOCUS = self.faction_focus
        ph = phase_of(gs.round)
        t = self._phase_tuning[ph]
        ld = getattr(gs.players[pid], "leader", None)
        if self._leader_mult and ld is not None and ld.name in self._leader_mult:
            key = (ph, ld.name)
            lt = self._leader_cache.get(key)
            if lt is None:
                lt = dict(t)
                for knob, m in self._leader_mult[ld.name].items():
                    lt[knob] = (lt.get(knob) or SEED_ON.get(knob, 0.0)) * m
                self._leader_cache[key] = lt
            t = lt
        if _TUNE is not t:
            _apply_tuning(t)
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
                s += _influence_gain_value(gs, pid, fac, 1) + _TUNE["faction_space"]
            cost = SPACE_MANDATORY_COSTS.get(a.space_name, {})
            s -= sum(_RES_VALUE.get(k, 0.4) * v for k, v in cost.items())
            if cost.get("spice"):
                s += _spice_judge_delta(gs, pid, -cost["spice"])
            # High Council (+2 persuasion every future Reveal turn) and
            # Swordmaster (a 3rd Agent every future round) are durable,
            # compounding structural investments, not one-off payoffs. The
            # value is roughly (worth of one extra agent-play) x (rounds left)
            # — a 3rd agent from round 3 is 7 extra plays, one of the biggest
            # swings in the game and the backbone of contesting combat every
            # round. Measured at the old min(22, 2.5r): the Swordmaster action
            # was legal + affordable + reachable 591 times and DECLINED 523 of
            # them (88%) because a single round's urgency-inflated combat or
            # friendship out-scored it — even though it pays that back every
            # subsequent round. Priced now to win the agent-slot in the
            # opening/midgame and taper as fewer rounds remain to benefit.
            remaining_rounds = max(1, MAX_ROUNDS - gs.round)
            _sm = self.sm_boost
            if a.space_name == "High Council" and not p.has_councilor and _TUNE.get("hc_judge", 0):
                s += _TUNE["hc_judge"] * _hc_judgement(gs, pid)
            elif a.space_name == "High Council" and not p.has_councilor:
                s += min(20.0, 2.5 * remaining_rounds) if _sm else min(15.0, 1.5 * remaining_rounds)
                s += _TUNE.get("hc_rush", 0.0) * remaining_rounds
                if _TUNE.get("hc_bl", 0) and getattr(gs, "bl", None) is not None:
                    from src.ai.bloodlines_heuristic import high_council_bl_value
                    s += _TUNE["hc_bl"] * high_council_bl_value(gs, pid)
            if a.space_name == "Swordmaster" and not p.has_swordmaster:
                s += (min(30.0, 4.0 * remaining_rounds) + 2.0) if _sm \
                    else min(22.0, 2.5 * remaining_rounds)
            # Building toward the 3rd agent: while Swordmaster is still unbought,
            # solari that closes the gap to its cost is worth well above face
            # value — this is what makes Spice Refinery (and other solari
            # spaces) a real opening priority instead of a ~1-point play.
            if _sm and not p.has_swordmaster and remaining_rounds >= 4:
                from src.game.board.board import (SWORDMASTER_COST_FIRST,
                                                  SWORDMASTER_COST_AFTER)
                sm_cost = (SWORDMASTER_COST_AFTER
                           if any(q.has_swordmaster for q in gs.players)
                           else SWORDMASTER_COST_FIRST)
                gap = sm_cost - p.solari
                if gap > 0:
                    solari_here = sum(
                        e["solari"] for e in
                        gs.get_space_effects_preview(a.space_name, a.space_option)
                        if isinstance(e.get("solari"), (int, float)) and e["solari"] > 0)
                    if solari_here:
                        s += _TUNE["sm_solari"] * min(solari_here, gap)
            if (sp.is_combat_space and gs.current_conflict is not None
                    and not getattr(p, "no_deploy_this_turn", False)):
                if _TUNE.get("combat_ev", 0):
                    s += _TUNE["combat_ev"] * _combat_space_ev(gs, pid, a)
                else:
                    s += _TUNE["combat"] * _conflict_worth(gs, pid) * (1.0 + 0.3 * min(p.troops_garrison, 4)
                                                      + 0.5 * _combat_build_strength(gs, pid))
            if _TUNE.get("leader_play", 1.0):
                s += _TUNE.get("leader_play", 1.0) * _leader_space_value(gs, pid, a)
            if a.space_name == "Tuek's Sietch":
                s += max(_effect_value(gs, pid, {"spice": 1}), _effect_value(gs, pid, {"draw": 1}))
                if not _TUNE.get("maker_bonus", 0):
                    s += _effect_value(gs, pid, {"spice": gs.maker_bonus_spice.get(a.space_name, 0)})
                if p.leader is not None and p.leader.name == "Esmar Tuek":
                    s += _effect_value(gs, pid, {"solari": 1})
                elif any(q.leader is not None and q.leader.name == "Esmar Tuek" for q in gs.players):
                    s -= 0.5                     # Esmar gets an Intrigue card
            if a.space_option == "sandworm":
                s += _TUNE.get("worm_bonus", 0.0)
            if a.space_option == "pay_spice" and p.spice < 3:
                s -= 1.0
            if a.use_gather_intelligence:
                s += 0.8
            if _TUNE.get("card_play", 0):
                s += _TUNE["card_play"] * _card_agent_play_value(gs, pid, a)
            if _TUNE.get("spy_recall", 0):
                s -= _TUNE["spy_recall"] * _spy_recall_cost(gs, pid, a)
            if _TUNE.get("contracts", 0):
                s += _TUNE["contracts"] * _contract_completion_value(gs, pid, a)
            if _TUNE.get("maker_bonus", 0) and gs.maker_bonus_spice.get(a.space_name, 0):
                s += _TUNE["maker_bonus"] * _effect_value(
                    gs, pid, {"spice": gs.maker_bonus_spice[a.space_name]})
            if self.bloodlines and getattr(gs, "bl", None) is not None:  # tech / commander it opens
                from src.ai.bloodlines_heuristic import agent_space_bonus
                s += agent_space_bonus(gs, pid, a.space_name)
            s += 4.0 * self.book.bonus(gs, pid, a)
            # plan the buy: value the reveal-turn purchase this move leaves
            if _TUNE.get("buy_plan", 0.0) and gs.round <= 8:
                s += _TUNE["buy_plan"] * 0.3 * _best_buy_value(
                    gs, pid, _reveal_persuasion_after(gs, pid, a))
            if _TUNE.get("smf_plan", 0.0):
                s += _TUNE["smf_plan"] * _smf_plan_value(gs, pid, a)
            if _TUNE.get("prep_draw", 0.0) and gs.round >= 6:
                played = next((c for c in p.hand if c.name == a.card_name), None)
                nd = _move_draws(gs, a, played)
                if nd:
                    s += _TUNE["prep_draw"] * (_next_hand_quality(gs, pid, nd) - _next_hand_quality(gs, pid))
            g = _TUNE.get("tsmf_goal", 0.0)
            if g and gs.round >= 3 and _tsmf_short(gs, pid)                     and _reveal_persuasion_after(gs, pid, a) >= 9:
                s += 0.5 * g          # this move keeps a TSMF buy on the Reveal turn
            # discourage wasting the last agent on a weak play
            s += 0.5

        elif at == ActionType.REVEAL_TURN:
            # revealing is fine once agent plays are weak; slight positive base
            s = 1.0 + 0.4 * len(p.hand) + _TUNE["reveal_bias"]

        elif at == ActionType.ACQUIRE_CARD:
            card = next((c for c in gs.imperium_row if c.name == a.acquire_card_name), None)
            if card:
                s += _acquire_card_value(gs, pid, card, self.lean_deck)
                s += 5.0 * self.book.bonus(gs, pid, a)
                if _TUNE.get("deny", 0.0):
                    s += _TUNE["deny"] * _deny_value(gs, pid, card, self.lean_deck)

        elif at == ActionType.ACQUIRE_RESERVE:
            stack = (gs.reserve_spice_must_flow if a.reserve_type == "spice_must_flow"
                     else gs.reserve_prepare_the_way)
            card = stack[-1] if stack else None
            if card:
                s += _acquire_card_value(gs, pid, card, self.lean_deck)
                if a.reserve_type == "spice_must_flow":
                    s += _endgame_goal_bonus(gs, pid, "tsmf", 1)

        elif at == ActionType.PLAY_INTRIGUE:
            ic = next((c for c in p.intrigue_cards
                       if c.name == a.intrigue_card_name), None)
            if ic:
                s += _intrigue_value(gs, pid, ic)

        elif at == ActionType.END_REVEAL:
            pool = gs.persuasion_pool.get(pid, 0)
            s = -0.5 + (2.0 if pool < 2 else -1.0)
            if _TUNE.get("deck_model", 0):
                s = 0.0               # stopping is neutral: buys must beat your deck
            # Banking a lean turn is a legitimate choice. If the deck is already
            # a healthy size and nothing affordable in the Row or reserves is an
            # actual upgrade (its dilution-adjusted value is weak), stopping now
            # beats spending persuasion just to bloat the deck with filler.
            if self.lean_deck and _deck_total(p) >= 14:
                best = 0.0
                cands = list(gs.imperium_row)
                for stack in (gs.reserve_prepare_the_way, gs.reserve_spice_must_flow):
                    if stack:
                        cands.append(stack[-1])
                for c in cands:
                    if c.cost <= pool:
                        best = max(best, _acquire_card_value(gs, pid, c, self.lean_deck))
                if best < 4.5:
                    s += 3.0

        elif at == ActionType.COMBAT_PASS:
            s = 0.0

        elif at == ActionType.RESOLVE_DEPLOY and not _TUNE.get("fight_model", 1):
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
            # OUR OWN Combat Intrigues are known swords we can add once we have
            # a unit in the fight — count them toward our reachable strength so
            # we actually commit to a Conflict we're holding the tools to win.
            own_ci = _own_combat_intrigue_swords(gs, pid)
            my_reachable = my + own_ci
            # value each committed troop by the reward at stake; bonus for
            # actually pulling ahead of / catching the leader
            s = n * (0.15 + 0.7 * worth)
            if worth > 0.5:
                if my > opp:
                    s += 1.2
                if 0 < opp - my <= 2:
                    s += 0.8 * worth
                # An under-deployed field + Combat Intrigues in hand = a
                # crucial Conflict we can steal. Reward committing enough
                # troops to (a) make the intrigues playable and (b) get
                # my_reachable over the top.
                already_in = gs.troops_in_conflict.get(pid, 0)
                if own_ci > 0 and _is_crucial_combat(gs, pid):
                    if already_in == 0 and n > 0:
                        s += 1.5 * worth               # get a foot in the door so CIs fire
                    if my < opp and my_reachable >= opp:
                        s += 1.8 * worth * urgency     # this deploy makes it winnable
                    elif my_reachable < opp and n > 0:
                        s += 0.5 * worth               # still building toward it
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
            if a.deploy_cmd:
                s += _cmd_deploy_adj(gs, pid, a) + 0.3 * worth * (0.0 if gs.bl.commanders_in_conflict.get(
                    pid, 0) else _skill_swords(gs, pid, assume_commander=True))

        elif at == ActionType.RESOLVE_DEPLOY:
            # deploy only what changes the expected Conflict reward by more
            # than the troops are worth kept in the garrison (troops in the
            # Conflict go back to supply after the fight)
            n = a.deploy_count
            mine0, u0 = _my_strength(gs, pid)
            mine, u = _my_strength(gs, pid, add_troops=n, add_cmd=a.deploy_cmd)
            own_ci = _own_combat_intrigue_swords(gs, pid) * 0.6
            if _TUNE.get("fight_own", 0) and gs.phase.name != "COMBAT":
                own_ci = 0.0                    # already in the learned own-sword forecast
            ev = fight_ev(gs, pid, mine + own_ci, u)
            ev0 = fight_ev(gs, pid, mine0 + own_ci, u0)
            s = (ev - ev0) - n * _troop_hold_value(gs) + _cmd_deploy_adj(gs, pid, a)
            if p.troops_garrison - n <= 0 and n > 0 and _rounds_left(gs) > 1:
                s -= 0.3                                # keep some flexibility
            # troops held back fight the NEXT Conflict (user, 2026-10-16: the AI
            # max-deploys too often): each one deployed gives that up
            if n > 0 and _rounds_left(gs) >= 1:
                s -= n * _TUNE.get("deploy_hold", 0.4) * min(1.0, _rounds_left(gs) / 2.0)
            # committing blind (user, 2026-10-18): with Agents still to place,
            # troops can follow on a later turn - after rivals reveal, when
            # only their Combat Intrigues are left to come. Deploying against
            # rivals who have all revealed costs nothing extra (the Swordmaster's
            # 3rd Agent is what buys that wait).
            w = _TUNE.get("deploy_wait", 0.0)
            if w and n > 0 and gs.phase.name != "COMBAT" and p.agents_available > 0                     and pid not in gs.players_revealed:
                rivals = [q for q in range(gs.num_players) if q != pid]
                hidden = sum(1 for q in rivals if q not in gs.players_revealed) / max(1, len(rivals))
                s -= w * n * _RES_VALUE["troops"] * hidden

        elif at == ActionType.RESOLVE_TRASH and _TUNE.get("trash_model", 0):
            s = _trash_value(gs, pid, a.trash_card_name)

        elif at == ActionType.RESOLVE_TRASH:
            # thin the weak starter cards; keep bought cards
            if a.trash_card_name in ("Reconnaissance", "Dune, the Desert Planet"):
                s = 2.0
            elif a.trash_card_name == "Diplomacy":
                # the only starter that keeps reaching faction spaces (Seek
                # Allies trashes itself): influence and VP — keep it.
                # trash_diplomacy=1 restores the old "weak starter" score.
                s = 2.0 if _TUNE.get("trash_diplomacy", 0) else 0.0
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
                if _TUNE.get("trash_dagger", 0):
                    # 0 persuasion: trash it while another Landsraad card remains
                    s = 1.8 if _landsraad_access_owned(p) >= 2 else -2.0
            elif a.trash_card_name is None:
                s = 0.5
            else:
                s = -1.0

        elif at == ActionType.RESOLVE_INFLUENCE:
            s = _influence_gain_value(gs, pid, a.influence_faction, 1)

        elif at == ActionType.RESOLVE_CONTRACT:
            aside = getattr(gs, "shaddam_contracts", {}).get(pid, [])
            if a.contract_index >= 100:
                k = a.contract_index - 100
                ct = aside[k] if k < len(aside) else None
            else:
                ct = (gs.contracts_on_board[a.contract_index]
                      if a.contract_index < len(gs.contracts_on_board) else None)
            s = 1.0
            if ct is not None and _TUNE.get("contracts", 0):
                s += _take_contract_value(gs, pid, ct)
            elif ct is not None:
                s += sum(_effect_value(gs, pid, {k: v}) for k, v in ct.rewards.items()
                         if isinstance(v, (int, float)))
                if getattr(ct, "contract_type", None) is not None and \
                        ct.contract_type.value == "immediate":
                    s += 1.5

        elif at == ActionType.RESOLVE_SPY:
            s = _spy_post_value(gs, pid, a.spy_post_name)
            if a.choice and a.choice.startswith("recall:"):     # the Spy that comes back
                s -= 0.8 * _spy_post_value(gs, pid, a.choice[7:])
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
                if a.choice and a.choice.startswith("recall:"):  # which Spies come back
                    cost += sum(0.8 * _spy_post_value(gs, pid, q) - _RES_VALUE.get("recall_spy", 0.3)
                                for q in a.choice[7:].split("|"))
                s = gain - cost
                k = _TUNE.get("hl_keep", 0.0)
                pl = gs.players[pid]
                if (k and op.label == "conflict_vp" and op.cost.get("spice")
                        and gs.round >= 7 and _rounds_left(gs) >= 1
                        and pl.spice >= 5 > pl.spice - op.cost["spice"]):
                    s -= k                                # keep the 5 for a Heighliner
        elif at == ActionType.RESOLVE_CHOICE and a.choice and a.choice.startswith(("recruit:", "techbuy:")):
            from src.ai.bloodlines_heuristic import score_bl_choice
            ch = a.choice[8:] if a.choice.startswith("recruit:") else "stack:" + a.choice[8:]
            s = score_bl_choice(gs, pid, ch) if self.bloodlines else -1.0
        elif at == ActionType.RESOLVE_CHOICE:
            pc = gs.pending_choice_for(pid)
            opt = next((o for o in pc.options if o[0] == a.choice), None) if pc else None
            s = _choice_option_value(gs, pid, opt[2]) if opt else 0.0
        elif at == ActionType.RESOLVE_BL_CHOICE:
            from src.ai.bloodlines_heuristic import score_bl_choice
            if self.bloodlines:
                s = score_bl_choice(gs, pid, a.choice)
            else:
                s = 1.0 if a.choice == "decline" else 0.0
        elif at == ActionType.ACTIVATE_TECH:
            from src.ai.bloodlines_heuristic import score_activation
            s = score_activation(gs, pid, a.choice) if self.bloodlines else -1.0
        elif at == ActionType.NO_OP:
            s = -5.0
        return s

    def select_action(self, gs, pid, valid_actions):
        scores = [self.score(gs, pid, a) for a in valid_actions]
        if self.near > 0:
            top = max(scores)
            pool = [a for a, s in zip(valid_actions, scores)
                    if s >= top - self.near and a.action_type != ActionType.NO_OP]
            if pool:
                return self.rng.choice(pool)
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
            if a.deploy_cmd:
                s += _cmd_deploy_adj(gs, pid, a) + 0.5 * (0.0 if gs.bl.commanders_in_conflict.get(
                    pid, 0) else _skill_swords(gs, pid, assume_commander=True))
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
                 heuristic_weight: float = 0.35,
                 policy=None, policy_weight: float = 0.25):
        self.model = model
        self.rollout = rollout or HeuristicAgent(seed=seed, opening_book=opening_book)
        self.branch_cap = branch_cap
        self.temperature = temperature
        self.heuristic_weight = heuristic_weight   # anchor lookahead to the heuristic prior
        # policy head (AWR-trained P(action|state)): a prior that can favour a
        # compounding move (Swordmaster, an alliance push) that 1-ply value
        # lookahead can't see, because it learned the move->win correlation
        # across whole games. Added to each candidate's score as
        # policy_weight * log pi(a|s).
        self.policy = policy
        self.policy_weight = policy_weight
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
        h_norms = [(h - lo) / span for h in hs]

        pol_logp = [0.0] * len(acts)
        if self.policy is not None:
            try:
                from src.ai.action_features import encode_action
                af = np.stack([encode_action(gs, pid, a, hn)
                               for a, hn in zip(acts, h_norms)])
                pol = self.policy.policy(encode_state(gs, pid), af)
                pol_logp = list(np.log(np.clip(pol, 1e-6, 1.0)))
            except Exception:
                pol_logp = [0.0] * len(acts)

        vals = []
        for a, h_norm, plp in zip(acts, h_norms, pol_logp):
            g2 = gs.clone()
            try:
                g2.step(a)
                leaf = self._leaf(g2, pid)
            except Exception:
                leaf = -1.0
            vals.append(leaf
                        + self.heuristic_weight * h_norm
                        + self.policy_weight * plp
                        + 0.03 * self.book.bonus(gs, pid, a))
        if self.temperature <= 0:
            return acts[max(range(len(acts)), key=lambda i: vals[i])]
        return _softmax_pick(acts, vals, self.temperature, self.rng)


class _ISNode:
    """One of *my* decision points in the search tree, keyed by action repr."""
    __slots__ = ("children", "visits", "value_sum")

    def __init__(self):
        self.children: dict = {}
        self.visits = 0
        self.value_sum = 0.0

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class ISMCTSAgent(Agent):
    """
    Single-observer Information-Set MCTS.

    Dune Imperium is hidden-information (opponent hands, deck order) and
    4-player-adversarial, so this isn't textbook single-agent MCTS:

      - Each simulation starts by calling `determinize()` on the real state,
        producing one consistent guess at the currently-hidden information.
        This is what keeps the search honest -- it can't peek at opponents'
        actual hands or the actual next card off any deck.
      - The search tree only has nodes at *my* (`pid`'s) decision points.
        Opponent turns are not searched (their hidden info differs across
        determinizations, so a shared tree over their choices wouldn't mean
        anything) -- they're played out directly by `rollout` (the same
        heuristic used elsewhere as a fast default policy / opponent model).
      - Each simulation runs to one of: the game actually ending (exact
        1.0/0.0 backprop), a move cap (safety valve), or `max_my_decisions`
        of *my own* choices deep (beyond which `_leaf` -- the value net or
        heuristic state value -- estimates the outcome instead of continuing).
      - Root action choice is the "robust child": most-visited action, which
        is the standard, variance-robust choice (rather than highest raw Q).
    """
    name = "ismcts"

    def __init__(self, model: Optional[ValueModel] = None,
                 rollout: Optional[HeuristicAgent] = None,
                 n_simulations: int = 48, c_uct: float = 1.4,
                 max_my_decisions: int = 6, move_cap: int = 300,
                 branch_cap: int = 14,
                 seed: Optional[int] = None,
                 opening_book: Optional[OpeningBook] = None):
        self.model = model
        self.rollout = rollout or HeuristicAgent(seed=seed, opening_book=opening_book)
        self.n_simulations = n_simulations
        self.c_uct = c_uct
        self.max_my_decisions = max_my_decisions
        self.move_cap = move_cap
        # Cap each of *my* decision nodes to the top-`branch_cap` actions by the
        # heuristic prior, same as GreedyValueAgent -- without this, a wide
        # decision (>branch_cap legal actions) burns the whole simulation
        # budget on one-rollout-each expansion and never reaches real UCB
        # comparison between actions.
        self.branch_cap = branch_cap
        self.rng = np.random.default_rng(seed)
        self.book = self.rollout.book

    def _leaf(self, gs: GameState, pid: int) -> float:
        if gs.game_over:
            return 1.0 if gs.winner == pid else 0.0
        if self.model is not None:
            return self.model.predict(encode_state(gs, pid))
        return heuristic_state_value(gs, pid)

    def select_action(self, gs, pid, valid_actions):
        acts = [a for a in valid_actions if a.action_type != ActionType.NO_OP] \
            or valid_actions
        if len(acts) == 1:
            return acts[0]

        root = _ISNode()
        for _ in range(self.n_simulations):
            gs_det = determinize(gs, pid, self.rng)
            self._simulate(gs_det, pid, root, my_decisions=0, moves=0)

        key_to_action = {repr(a): a for a in acts}
        best_action, best_visits = acts[0], -1
        for key, child in root.children.items():
            if key in key_to_action and child.visits > best_visits:
                best_visits = child.visits
                best_action = key_to_action[key]
        return best_action

    def _simulate(self, gs: GameState, pid: int, node: _ISNode,
                  my_decisions: int, moves: int) -> float:
        if gs.game_over:
            return 1.0 if gs.winner == pid else 0.0
        if moves >= self.move_cap:
            return self._leaf(gs, pid)

        cur = gs.player_in_reveal_buy
        if cur is None:
            cur = gs.get_current_player_id()
        valid = gs.get_valid_actions(cur)

        if cur != pid:
            action = self.rollout.select_action(gs, cur, valid)
            gs.step(action)
            return self._simulate(gs, pid, node, my_decisions, moves + 1)

        acts = [a for a in valid if a.action_type != ActionType.NO_OP] or valid
        if len(acts) == 1:
            gs.step(acts[0])
            return self._simulate(gs, pid, node, my_decisions, moves + 1)
        if my_decisions >= self.max_my_decisions:
            return self._leaf(gs, pid)

        hs = [self.rollout.score(gs, pid, a) for a in acts]
        if len(acts) > self.branch_cap:
            keep = sorted(range(len(acts)), key=lambda i: hs[i],
                          reverse=True)[: self.branch_cap]
            acts = [acts[i] for i in keep]
            hs = [hs[i] for i in keep]

        keys = [repr(a) for a in acts]
        untried = [k for k in keys if k not in node.children]
        if untried:
            untried_set = set(untried)
            key, action, _ = max(
                (t for t in zip(keys, acts, hs) if t[0] in untried_set),
                key=lambda t: t[2])
            node.children[key] = _ISNode()
        else:
            total = sum(node.children[k].visits for k in keys)
            log_total = math.log(total + 1.0)

            def ucb(k):
                c = node.children[k]
                if c.visits == 0:
                    return float("inf")
                return c.q() + self.c_uct * math.sqrt(log_total / c.visits)

            key = max(keys, key=ucb)
            action = next(a for a, kk in zip(acts, keys) if kk == key)

        gs.step(action)
        child = node.children[key]
        value = self._simulate(gs, pid, child, my_decisions + 1, moves + 1)
        child.visits += 1
        child.value_sum += value
        return value


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
    spec: 'random' | 'heuristic' | 'heuristic:T<temp>' | 'heuristic:lean' |
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
    # Re-join a Windows drive-letter split, e.g. "value:C:/models/v.npz" splits
    # to ["value", "C", "/models/v.npz"] — "C" + "/models/v.npz" -> "C:/models/v.npz".
    merged, i = [], 0
    while i < len(parts):
        if (len(parts[i]) == 1 and parts[i].isalpha() and i + 1 < len(parts)
                and parts[i + 1][:1] in ("/", "\\")):
            merged.append(parts[i] + ":" + parts[i + 1])
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    parts = merged
    kind = parts[0]
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "heuristic":
        temp = 0.0
        lean = False
        sm_boost = True
        faction_focus = True
        bloodlines = True
        tuning = None
        near = 0.0
        for p in parts[1:]:
            if p.startswith("T"):
                temp = float(p[1:])
            elif p.startswith("N") and p[1:].replace(".", "", 1).isdigit():
                near = float(p[1:])
            elif p in ("lean", "L1"):
                lean = True
            elif p == "sm0":
                sm_boost = False
            elif p == "ff0":
                faction_focus = False
            elif p == "nobl":
                bloodlines = False
            elif p.startswith("tuned="):
                import json as _json
                with open(p[6:], encoding="utf-8") as _f:
                    tuning = _json.load(_f)
        return HeuristicAgent(seed=seed, opening_book=opening_book,
                              temperature=temp, lean_deck=lean, sm_boost=sm_boost,
                              faction_focus=faction_focus, bloodlines=bloodlines,
                              tuning=tuning, near=near)
    if kind == "az":
        # az[:S<sims>][:net=<path.npz>]: AlphaZero net + search (src/az/agent.py)
        from src.az.agent import AZAgent
        kw = {"sims": 32, "net_path": None}
        for p in parts[1:]:
            if p.startswith("S"):
                kw["sims"] = int(p[1:])
            elif p.startswith("net="):
                kw["net_path"] = p[4:]
            elif p.startswith("style="):
                kw["style"] = p[6:]
            elif p == "coach" or p.startswith("coach="):
                kw["coach"] = float(p[6:]) if "=" in p else 1.0
        return AZAgent(seed=seed, **kw)
    if kind == "styleselect":
        # styleselect[:<model.npz>] (default config/style_select.npz)
        from src.ai.style_select import StyleSelectAgent
        return StyleSelectAgent(seed=seed, model=parts[1] if len(parts) > 1 else None)
    if kind == "stylesearch":
        # stylesearch[:M<playouts per style>][:MG<margin>][:search]: picks a
        # playstyle (config/styles/) each round by playing the game out
        from src.ai.style_search import StyleSearchAgent
        kw = {"m": 16, "margin": 0.05}
        base = "heuristic"
        for p in parts[1:]:
            if p.startswith("MG"):
                kw["margin"] = float(p[2:])
            elif p.startswith("M"):
                kw["m"] = int(p[1:])
            elif p == "search":
                base = "search"
        return StyleSearchAgent(seed=seed, base=base,
                                search_kw={"k": 5, "m": 24, "margin": 0.02}, **kw)
    if kind == "search":
        from src.ai.search import RoundSearchAgent
        kw = {"k": 5, "m": 24, "margin": 0.02, "horizon": "end"}
        for p in parts[1:]:
            if p.startswith("MG"):
                kw["margin"] = float(p[2:])
            elif p.startswith("KB"):
                kw["kb"] = int(p[2:])
            elif p.startswith("K"):
                kw["k"] = int(p[1:])
            elif p.startswith("M"):
                kw["m"] = int(p[1:])
            elif p == "round":
                kw["horizon"] = "round"
            elif p == "B":
                kw["buys"] = True
            elif p == "SH":
                kw["halving"] = True
            elif p.startswith("tuned="):
                import json as _json
                with open(p[6:], encoding="utf-8") as _f:
                    kw["tuning"] = _json.load(_f)
        return RoundSearchAgent(seed=seed, opening_book=opening_book, **kw)
    if kind == "ppo":
        # ppo[:net=<npz>][:tuned=<json>][:greedy][:T<temp>][:K<n>] (src/ai/ppo.py)
        from src.ai.ppo import PPOAgent
        kw = {}
        for p in parts[1:]:
            if p.startswith("net="):
                kw["net"] = p[4:]
            elif p == "greedy":
                kw["greedy"] = True
            elif p.startswith("T") and p[1:].replace(".", "", 1).isdigit():
                kw["temperature"] = float(p[1:])
            elif p.startswith("K") and p[1:].isdigit():
                kw["k"] = int(p[1:])
            elif p.startswith("tuned="):
                import json as _json
                with open(p[6:], encoding="utf-8") as _f:
                    kw["tuning"] = _json.load(_f)
        return PPOAgent(seed=seed, opening_book=opening_book, **kw)
    if kind == "look":
        # look[:D<depth>][:K<k>][:M<m>][:LW<look_w>][:ev=<npz>][:tuned=<json>] (src/ai/lookahead.py)
        from src.ai.lookahead import LookaheadAgent
        kw = {}
        for p in parts[1:]:
            if p.startswith("EK"):
                kw["end_k"] = int(p[2:])
            elif p.startswith("EM"):
                kw["end_m"] = int(p[2:])
            elif p.startswith("E") and p[1:].isdigit():
                kw["end_round"] = int(p[1:])
            elif p.startswith("LW"):
                kw["look_w"] = float(p[2:])
            elif p.startswith("D"):
                kw["depth"] = int(p[1:])
            elif p.startswith("K"):
                kw["k"] = int(p[1:])
            elif p.startswith("M"):
                kw["m"] = int(p[1:])
            elif p.startswith("ev="):
                kw["evaluator"] = p[3:]
            elif p.startswith("tuned="):
                import json as _json
                with open(p[6:], encoding="utf-8") as _f:
                    kw["tuning"] = _json.load(_f)
        return LookaheadAgent(seed=seed, opening_book=opening_book, **kw)
    if kind == "bully":
        temp = 0.0
        for p in parts[1:]:
            if p.startswith("T"):
                temp = float(p[1:])
        return CombatBullyAgent(seed=seed, opening_book=opening_book, temperature=temp)
    if kind == "value":
        # value:<value.npz>[:<policy.npz>][:T<temp>][:HW<hw>][:PW<policy_weight>]
        # the 2nd .npz-ending part (or any part starting 'PP') is the policy head.
        model, policy, temp, hw, pw = None, None, 0.0, 0.35, 0.25
        for p in parts[1:]:
            if p.startswith("T"):
                temp = float(p[1:])
            elif p.startswith("HW"):
                hw = float(p[2:])
            elif p.startswith("PW"):
                pw = float(p[2:])
            elif p.startswith("PP"):
                from src.ai.policy_model import PolicyModel
                policy = PolicyModel.load(p[2:])
            elif p.endswith(".npz") and model is not None:
                from src.ai.policy_model import PolicyModel
                policy = PolicyModel.load(p)
            elif p:
                model = ValueModel.load(p)
        return GreedyValueAgent(model=model, temperature=temp, seed=seed,
                                opening_book=opening_book, heuristic_weight=hw,
                                policy=policy, policy_weight=pw)
    if kind == "ismcts":
        # ismcts[:<value.npz>][:N<n_simulations>][:D<max_my_decisions>][:C<c_uct>]
        model, n_sims, max_dec, c_uct = None, 48, 6, 1.4
        for p in parts[1:]:
            if p.startswith("N"):
                n_sims = int(p[1:])
            elif p.startswith("D"):
                max_dec = int(p[1:])
            elif p.startswith("C"):
                c_uct = float(p[1:])
            elif p:
                model = ValueModel.load(p)
        return ISMCTSAgent(model=model, n_simulations=n_sims, c_uct=c_uct,
                           max_my_decisions=max_dec, seed=seed,
                           opening_book=opening_book)
    raise ValueError(f"Unknown agent spec: {spec!r}")
