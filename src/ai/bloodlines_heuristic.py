"""
Heuristic valuation of Bloodlines content for HeuristicAgent (and everything
built on it: CombatBully, the rollout policy, the policy head's h_norm).

Units match agents._RES_VALUE (10 = 1 VP). Recurring effects are priced as
(value per round) x (rounds likely left) x (chance they fire), which is what
makes an engine tech worth buying early and not late.

Entry points used by agents.py:
  score_bl_choice(agent_ctx, gs, pid, action)   RESOLVE_BL_CHOICE
  score_activation(gs, pid, name)                ACTIVATE_TECH
  agent_space_bonus(gs, pid, space)              extra value of a green /
                                                 commander space (tech buy,
                                                 commander recruit it opens)
  state_bonus(gs, pid)                           heuristic_state_value term
  BL_RES_VALUE, cond_weight(), extend card-effect scoring for bl_* keys
"""
from __future__ import annotations

from typing import Dict

from src.game.gameState import MAX_ROUNDS

SWORD = 0.5          # one sword on a reveal, averaged over rounds you fight
P_COMMAND = 0.45     # chance a reveal reaches 6 persuasion (Command)
P_IN_CONFLICT = 0.5  # chance a commander is in the Conflict on a given reveal
SKILL_SWORD = 0.7    # a sword from a skill (recurring, stacks with others)

# flat values for Bloodlines effect keys (effects.py vocabulary extensions)
BL_RES_VALUE: Dict[str, float] = {
    "bl_tech_offer": 1.5, "bl_free_commander": 2.0, "bl_force_retreat": 0.8,
    "bl_opponents_lose_troop": 1.2, "bl_commander_discount": 0.3,
    "bl_shigawire": 0.8, "bl_complete_contract": 2.0, "bl_retreat": -0.5,
    "grant_deploy": 0.5, "swords": SWORD, "bl_recover_bene": 1.2,
}

_WEAK = ("Reconnaissance", "Diplomacy", "Dune, the Desert Planet",
         "Convincing Argument")


def rounds_left(gs) -> int:
    """Rounds in which a recurring effect will still fire (games usually end
    by round 8-9, rarely all 10)."""
    return max(1, min(MAX_ROUNDS, 9) - gs.round)


def cond_weight(gs, pid: int, key: str) -> float:
    """0..1 chance a Bloodlines condition (if_bl_<key>) holds."""
    p = gs.players[pid]
    bl = gs.bl
    if key == "commander_in_conflict":
        if bl and bl.commanders_in_conflict.get(pid, 0):
            return 1.0
        has = (p.commanders_garrison + p.commanders_supply) > 0 if bl else False
        return 0.45 if has else 0.15
    if key.startswith("techs_"):
        need = int(key[6:])
        have = len(getattr(p, "techs", []))
        return 1.0 if have >= need else max(0.1, 0.45 - 0.15 * (need - have))
    if key == "gained_spice_2":
        return 0.4
    if key == "garrison_4":
        return 1.0 if p.troops_garrison >= 4 else 0.35
    if key == "other_bene":
        return 0.3
    if key == "completed_contract":
        return 0.3
    if key == "not_endgame":
        return 1.0
    if key.startswith("endgame_"):
        return 0.5
    return 0.35


# ---------------------------------------------------------------------------
# techs
# ---------------------------------------------------------------------------
def _effects_value(gs, pid, effects) -> float:
    from src.ai.agents import _card_effect_value
    return sum(_card_effect_value(gs, pid, e) for e in effects)


def tech_value(gs, pid: int, d) -> float:
    """Worth of owning tech `d` from now on (before paying for it)."""
    p = gs.players[pid]
    R = rounds_left(gs)
    v = _effects_value(gs, pid, d.acquire)
    v += 0.9 * R * _effects_value(gs, pid, d.reveal)
    v += P_COMMAND * R * _effects_value(gs, pid, d.command)
    v += R * _effects_value(gs, pid, d.round_start)
    v += 0.3 * R * _effects_value(gs, pid, d.on_win_conflict)
    v += 0.35 * R * _effects_value(gs, pid, d.on_complete_contract)
    a = d.activation
    if a:
        from src.ai.agents import _RES_VALUE
        gain = _effects_value(gs, pid, a.get("effects", []))
        cost = sum(_RES_VALUE.get(k, 0.5) * n for k, n in a.get("cost", {}).items())
        cost += 1.1 * a.get("discard", 0)
        net = max(0.0, gain - cost)
        v += net if a.get("trash_self") else 0.7 * R * net
    fac_le1 = sum(1 for f in p.influence if p.influence[f] <= 1)
    special = {
        "Artillery": 1.2 * SWORD * R * P_IN_CONFLICT * 2,
        "Restricted Ordnance": 4 * SWORD * R * (0.9 if p.has_councilor else 0.35),
        "Forbidden Weapons": 3 * SWORD * R * 0.6 - 0.8 * R,
        "Disposal Facility": 0.8 * R * P_COMMAND,
        "Navigation Chamber": 0.55 * R,
        "Ornithopter Fleet": 7.0,
        "Chaumurky": 2.0,
        "Spaceport": 0.35 * R,
        "Glowglobes": 0.4,
        "Gene-Locked Vault": 1.0,
        "Servo-Receivers": 0.6 * R,
        "Suspensor Suits": 0.6 * R,
        "Sardaukar High Command": 0.3 * R,
        "Plasteel Blades": 3.0,
        "Rapid Dropships": 0.4 * R,
        "Invasion Ships": 0.3 * R,
        "Spy Satellites": 10.0 * 0.6 * fac_le1,
        "Panopticon": 1.5 * fac_le1,
        "Memocorders": 10.0 * (0.7 if all(i >= 2 for i in p.influence.values())
                               else 0.15),
        "CHOAM Transports": 10.0 * (0.6 if len(p.contracts_completed) >= 2 else 0.2),
        "Holtzman Engine": 10.0 * (
            0.6 if any(c.name == "The Spice Must Flow"
                       for z in (p.deck, p.hand, p.discard, p.in_play) for c in z)
            else 0.25),
        "Advanced Data Analysis": -1.2,          # costs you a Spy
    }.get(d.name, 0.0)
    return v + special


def tech_buy_value(gs, pid: int, d, discount: int = 0) -> float:
    from src.ai.agents import _RES_VALUE
    price = gs.bl.tech_price(pid, d, discount)
    return 0.8 * tech_value(gs, pid, d) - _RES_VALUE["spice"] * price


def best_tech_buy(gs, pid: int, discount: int = 0) -> float:
    bl = gs.bl
    best = 0.0
    for i in bl.buyable_stacks(pid, discount):
        best = max(best, tech_buy_value(gs, pid, bl.tech_stacks[i][0], discount))
    return best


# ---------------------------------------------------------------------------
# commanders + skills
# ---------------------------------------------------------------------------
def _skills_on() -> float:
    """The `skills` knob: 0 = old scoring, else a multiplier on skill value."""
    from src.ai import agents
    return float(agents._TUNE.get("skills", 0))


def _p_fight(gs, pid: int) -> float:
    """Chance this player has units in a given round's Conflict."""
    from src.ai.agents import _combat_build_strength
    p = gs.players[pid]
    return max(0.35, min(0.85, 0.45 + 0.06 * min(p.troops_garrison, 5)
                         + 0.3 * _combat_build_strength(gs, pid)))


P_COMMANDER_READY = 0.6   # a commander re-recruited (or still home) for a combat


def _p_green(gs, pid: int) -> float:
    """Chance an Agent of this player is on a green (Landsraad) space when
    combat comes — Canny's condition — from the deck's Landsraad access."""
    from src.ai.agents import _deck_total
    p = gs.players[pid]
    n = sum(1 for z in (p.deck, p.discard, p.hand, p.in_play) for c in z
            if any(getattr(s, "value", s) == "landsraad"
                   for s in getattr(c, "access_symbols", ())))
    return min(0.9, 0.15 + 2.0 * n / max(1, _deck_total(p)))


def _p_loyal(gs, pid: int) -> float:
    e = gs.players[pid].influence.get("emperor", 0)
    return 1.0 if e >= 3 else 0.6 if e == 2 else 0.3 if e == 1 else 0.15


def skill_per_combat(gs, pid: int, skill: str) -> float:
    """What a skill adds in one combat where a commander is present."""
    sw = SKILL_SWORD
    return {
        "Hardy": 0.8,                                 # +1 troop on Reveal
        "Driven": 0.65,                               # +1 spice
        "Charismatic": 0.8 + 0.3 * P_COMMAND,         # +1 persuasion (Command)
        "Canny": 2 * sw * _p_green(gs, pid),
        "Fierce": 1.15 * sw,
        "Loyal": 2 * sw * _p_loyal(gs, pid),
    }.get(skill, 0.0)


def skills_active_now(gs, pid: int, extra_skill: str = "") -> float:
    """Value of switching the held skills (+ `extra_skill`) on for THIS
    round's combat by getting a commander into play now: sword skills priced
    by how they change the expected Conflict reward, the others at face."""
    from src.ai.agents import fight_ev, _my_strength
    bl, p = gs.bl, gs.players[pid]
    if gs.current_conflict is None:
        return 0.0
    if p.commanders_garrison > 0 or bl.commanders_in_conflict.get(pid, 0) > 0:
        return 0.0                                    # already switched on
    skills = set(p.skills) | ({extra_skill} if extra_skill else set())
    if not skills:
        return 0.0
    pf = _p_fight(gs, pid)
    v = sum(skill_per_combat(gs, pid, k) for k in skills
            if k in ("Hardy", "Driven", "Charismatic"))
    held = set(p.skills)
    if extra_skill:
        p.skills.add(extra_skill)
    try:
        bl.commanders_in_conflict[pid] = 1
        bonus = float(bl.strength_bonus(pid))
    finally:
        bl.commanders_in_conflict[pid] = 0
        p.skills.clear(); p.skills.update(held)
    if bonus:
        mine, units = _my_strength(gs, pid)
        units = max(1, units)
        mine = max(mine, 4.0)                         # roughly: you'll send some troops
        v += max(0.0, fight_ev(gs, pid, mine + bonus, units) - fight_ev(gs, pid, mine, units))
    return pf * v


def skill_value(gs, pid: int, skill: str) -> float:
    p = gs.players[pid]
    if _skills_on():
        R = rounds_left(gs)
        if skill == "Desperate":
            return _skills_on() * 3 * SWORD * 0.7     # one-shot
        return _skills_on() * skill_per_combat(gs, pid, skill) * R             * _p_fight(gs, pid) * P_COMMANDER_READY
    R = rounds_left(gs)
    f = R * P_IN_CONFLICT
    # sword skills add strength every round a commander fights (and stack),
    # so a sword counts for more here than a one-off reveal sword
    sw = SKILL_SWORD
    return {
        "Hardy": 0.8 * f,
        "Driven": 0.65 * f,
        "Charismatic": 0.8 * f,
        "Desperate": 3 * SWORD * 0.7,                 # one-shot, only if fighting
        "Canny": 2 * sw * f * 0.7,                    # needs an agent on green
        "Fierce": 1.2 * sw * f,
        "Loyal": 2 * sw * f * (0.9 if p.influence.get("emperor", 0) >= 3 else 0.35),
    }.get(skill, 0.0)


def commander_value(gs, pid: int, option: str) -> float:
    """RESOLVE_BL_CHOICE commander option: 'board:<skill>' or 'supply'."""
    from src.ai.agents import _RES_VALUE
    cost = gs.bl.commander_cost(pid)
    unit = 1.6          # a 2-strength unit in the garrison (a troop is 0.8)
    per_solari = _RES_VALUE["solari"]
    from src.ai import agents
    p = gs.players[pid]
    if agents._TUNE.get("sm_first", 0) and not p.has_swordmaster             and MAX_ROUNDS - gs.round >= 4:
        per_solari = max(per_solari, agents._TUNE.get("sm_solari", 2.0))
    v = unit - per_solari * cost
    skill = option.split(":", 1)[1] if option.startswith("board:") else ""
    if skill:
        v += skill_value(gs, pid, skill)
    if _skills_on():
        # commanders return to supply after every combat: a recruit now is
        # what switches the held skills on for this round's Conflict
        v += _skills_on() * skills_active_now(gs, pid, skill)
    return v


def best_commander(gs, pid: int, space: str) -> float:
    bl = gs.bl
    opts = bl.recruit_options(pid, space)
    board = max((commander_value(gs, pid, o) for o in opts if o.startswith("board")), default=0.0)
    supply = commander_value(gs, pid, "supply") if "supply" in opts else 0.0
    # both may be taken on one visit (2 solari each) when affordable
    if board > 0 and supply > 0 and gs.players[pid].solari >= 2 * bl.commander_cost(pid):
        return board + max(0.0, supply)
    return max(board, supply)


# ---------------------------------------------------------------------------
# action scoring
# ---------------------------------------------------------------------------
def agent_space_bonus(gs, pid: int, space: str) -> float:
    """Expected extra from the Bloodlines choices a space opens up."""
    bl = gs.bl
    if bl is None:
        return 0.0
    from src.game.bloodlines.rules import GREEN_SPACES, COMMANDER_SPACES
    v = 0.0
    if space in GREEN_SPACES:
        v += 0.7 * max(0.0, best_tech_buy(gs, pid))
    if space in COMMANDER_SPACES:
        v += 0.7 * max(0.0, best_commander(gs, pid, space))
    return v


def _trash_score(p, name: str) -> float:
    if name in _WEAK:
        return 2.0
    if name == "Dagger":
        return 2.0 if (p.has_swordmaster and p.has_councilor) else -2.0
    return -1.0


def score_bl_choice(gs, pid: int, choice: str) -> float:
    from src.ai.agents import _card_effect_value, _conflict_worth
    bl = gs.bl
    c = next((c for c in bl.pending if c.player_id == pid), None)
    if c is None:
        return 0.0
    p = gs.players[pid]
    if choice == "decline":
        return 0.2
    if c.kind in ("tech_buy", "tech_offer"):
        d = bl.tech_stacks[int(choice.split(":")[1])][0]
        return tech_buy_value(gs, pid, d, c.discount)
    if c.kind == "commander":
        return commander_value(gs, pid, choice)
    if c.kind == "plasteel":              # trash Plasteel Blades for a row skill
        return skill_value(gs, pid, choice.split(":", 1)[1])
    if c.kind == "desperate":
        return 3 * SWORD * 2.0 * min(2.0, _conflict_worth(gs, pid)) \
            - skill_value(gs, pid, "Desperate") * 0.5
    if c.kind in ("disposal", "trash_hand"):
        s = _trash_score(p, choice)
        if c.kind == "trash_hand" and c.tag:
            card = next((x for x in p.hand if x.name == choice), None)
            from src.game.cards.card import CardTag
            if card is not None and card.has_tag(CardTag(c.tag)):
                s += _card_effect_value(gs, pid, c.bonus)
        return s
    if c.kind == "effect":
        payload = c.payloads.get(choice, {})
        return 0.2 + _card_effect_value(gs, pid, payload) if payload else 0.2
    return 0.0


def score_activation(gs, pid: int, name: str) -> float:
    from src.ai.agents import _card_effect_value, _RES_VALUE
    from src.game.bloodlines.rules import LITANY
    if name == LITANY:
        # a free "pass" that draws: good when the hand is weak for agent plays
        return 1.2
    t = gs.bl._tile(pid, name)
    a = t.d.activation
    v = sum(_card_effect_value(gs, pid, e) for e in a.get("effects", []))
    v -= sum(_RES_VALUE.get(k, 0.5) * n for k, n in a.get("cost", {}).items())
    v -= 1.1 * a.get("discard", 0)
    flag = a.get("flag")
    if flag == "cycle_intrigues":
        v += 0.4 * len(gs.players[pid].intrigue_cards)
    if flag == "dropships":
        v += 1.0 if gs.current_conflict is not None else -1.0
    if flag == "ignore_blocking":
        v += 0.5
    if a.get("trash_self") and name == "Spy Satellites":
        v -= 0.6 * 10.0 * sum(1 for f in gs.players[pid].influence.values() if f <= 1) / 2
    return v


def state_bonus(gs, pid: int) -> float:
    """Extra terms for heuristic_state_value (same scale as its score)."""
    if getattr(gs, "bl", None) is None:
        return 0.0
    p = gs.players[pid]
    units = p.commanders_garrison + p.commanders_supply \
        + gs.bl.commanders_in_conflict.get(pid, 0)
    return 0.9 * len(p.techs) + 0.5 * units + 0.6 * len(p.skills)


def high_council_bl_value(gs, pid: int) -> float:
    """Bloodlines extras of a High Council seat: 1 spice off every future tech,
    and +2 persuasion per reveal pushing Command (6+) from ~P_COMMAND to ~0.8
    for the Command cards / techs this player owns."""
    from src.ai.agents import _RES_VALUE, _card_effect_value, _deck_total
    p = gs.players[pid]
    R = rounds_left(gs)
    v = _RES_VALUE["spice"] * 0.35 * R                 # ~1 tech per 3 rounds
    cmd = 0.0
    n = max(1, _deck_total(p))
    for z in (p.deck, p.discard, p.hand, p.in_play):
        for c in z:
            for e in getattr(c, "reveal_effects", []) or []:
                if isinstance(e, dict) and isinstance(e.get("command"), dict):
                    cmd += _card_effect_value(gs, pid, e["command"]) * min(1.0, 5.0 / n)
    for t in getattr(p, "techs", []):
        cmd += _effects_value(gs, pid, t.d.command)
    return v + (0.8 - P_COMMAND) * R * cmd
