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
def skill_value(gs, pid: int, skill: str) -> float:
    p = gs.players[pid]
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
    v = unit - _RES_VALUE["solari"] * cost
    if option.startswith("board:"):
        skill = option.split(":", 1)[1]
        if skill:
            v += skill_value(gs, pid, skill)
    return v


def best_commander(gs, pid: int, space: str) -> float:
    bl = gs.bl
    if gs.players[pid].bl_recruited_this_turn:
        return 0.0
    opts = bl.recruit_options(pid, space)
    return max((commander_value(gs, pid, o) for o in opts), default=0.0)


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
