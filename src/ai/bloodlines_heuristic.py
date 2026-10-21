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
    from src.ai import agents
    k = float(agents._TUNE.get("tech_judge", 0.0))
    if k and d.name in SITUATIONAL:
        # judge the tile by this player's situation instead of a flat number
        special = (1.0 - min(1.0, k)) * special + min(1.0, k) * SITUATIONAL[d.name](gs, pid, d)
    return v + special


# ---------------------------------------------------------------------------
# tech_judge (user, 2026-10-19): a tile's worth depends on the situation -
# Panopticon is a mistake with every friendship in hand and key when you
# won't visit the factions; CHOAM Transports needs contracts you can finish;
# the draw tiles buy a strong Imperium Row card this round; Ornithopter Fleet
# pays for icons you won't match otherwise. Each returns the tile's extra
# worth beyond its plain acquire / reveal effects (10 = 1 VP).
# ---------------------------------------------------------------------------
def _vp(gs, pid) -> float:
    from src.ai.agents import _effect_value
    return _effect_value(gs, pid, {"vp": 1})


def _p_raise(gs, pid, fac) -> float:
    """Chance this player still moves a faction track up before the end."""
    from src.ai.agents import _faction_access_strength
    return min(0.9, (0.25 + 1.2 * _faction_access_strength(gs, pid, fac)) * rounds_left(gs) / 4.0)


def _sit_panopticon(gs, pid, d) -> float:
    """Endgame +1 influence on every track at 0-1: a track ending at exactly
    1 becomes a friendship (1 VP); 0 -> 1 earns nothing."""
    p = gs.players[pid]
    ends_at_1 = 0.0
    for fac, cur in p.influence.items():
        up = _p_raise(gs, pid, fac)
        if cur == 1:
            ends_at_1 += 1.0 - up                      # stays at 1 unless visited
        elif cur == 0:
            ends_at_1 += 0.5 * up                      # reaches 1 (and stops there)
    return ends_at_1 * _vp(gs, pid)


def _sit_choam(gs, pid, d) -> float:
    """1 VP at 4+ completed contracts, a card per later completion: worth it
    only with contracts you can actually finish."""
    import math
    from src.ai.agents import _contract_finish_chance, _RES_VALUE
    p = gs.players[pid]
    done = len(p.contracts_completed)
    act = sum(_contract_finish_chance(gs, pid, ct) for ct in p.contracts_active)
    rate = done / max(1, gs.round - 1)                  # completions per round so far
    future = act + 1.0 + 0.7 * rate * rounds_left(gs)    # +1: the contract it gives
    proj = done + future
    p4 = 1.0 / (1.0 + math.exp(-(proj - 3.6) / 0.6))
    return p4 * _vp(gs, pid) + 0.8 * future * _RES_VALUE.get("draw", 1.1)


def _hand_persuasion(gs, pid) -> float:
    p = gs.players[pid]
    return (gs.persuasion_pool.get(pid, 0) + sum((c.persuasion or 0) for c in p.hand)
            + (2 if p.has_councilor else 0))


def _draw_now_value(gs, pid) -> float:
    """A card drawn now: what it adds to this round's buy (a strong Imperium
    Row card coming into reach) beyond a plain draw."""
    from src.ai.agents import _best_buy_value
    p = gs.players[pid]
    pool = p.deck or p.discard
    if not pool:
        return 0.0
    P = _hand_persuasion(gs, pid)
    base = _best_buy_value(gs, pid, P)
    # expectation over the card actually drawn (most draws add 0-1 persuasion)
    cnt = {}
    for c in pool:
        cnt[c.persuasion or 0] = cnt.get(c.persuasion or 0, 0) + 1
    gain = sum(n * (_best_buy_value(gs, pid, P + v) - base) for v, n in cnt.items()) / len(pool)
    return 0.3 * max(0.0, gain)              # same scale as buy_plan's reveal-buy value


def _sit_vault(gs, pid, d) -> float:
    """Gene-Locked Vault: an Intrigue or a card now, Intrigues safe from theft."""
    from src.ai.agents import _RES_VALUE
    draw = _RES_VALUE.get("draw", 1.1) + _draw_now_value(gs, pid)
    return max(_RES_VALUE.get("intrigue", 1.3), draw) + 0.15 * len(gs.players[pid].intrigue_cards)


def _sit_delivery_bay(gs, pid, d) -> float:
    """The acquire draw (already counted plainly) can buy a strong row card now."""
    return _draw_now_value(gs, pid)


def _sit_spaceport(gs, pid, d) -> float:
    return 2 * _draw_now_value(gs, pid) + 0.35 * rounds_left(gs)


def _sit_fleet(gs, pid, d) -> float:
    """Every icon becomes an Ornithopter: the held ones pair at once, and every
    later win pairs with any leftover. Worth most with unmatched icons you
    are unlikely to match by winning (and no icon Intrigue to match them)."""
    from src.ai.agents import _combat_build_strength, _endgame_goals
    p = gs.players[pid]
    icons = list(p.battle_icons)
    wild = sum(1 for i in icons if i == "wild")
    reals = [i for i in icons if i != "wild"]
    held = {g[0][5:] for g in _endgame_goals(gs, pid) if g[0].startswith("icon:")}
    p_win = min(0.6, (0.25 + 0.4 * _combat_build_strength(gs, pid)) * rounds_left(gs) / 4.0)
    # without the Fleet: wilds pair with reals at the end; other reals need a
    # matching win or a held icon Intrigue
    pairs_wild = min(wild, len(reals)) + max(0, wild - len(reals)) // 2
    rest = reals[min(wild, len(reals)):]
    without = pairs_wild + sum(1.0 if r in held else 0.35 * p_win for r in rest)
    with_fleet = len(icons) // 2 + (0.5 * p_win if len(icons) % 2 else 0.0)
    future = 0.25 * p_win * rounds_left(gs) / 2.0          # later wins pair more easily
    return (max(0.0, with_fleet - without) + future) * _vp(gs, pid)


def _p_win(gs, pid) -> float:
    """Chance this player wins a given round's Conflict."""
    from src.ai.agents import _combat_build_strength
    return max(0.1, min(0.55, 0.15 + 0.5 * _combat_build_strength(gs, pid)
                        + 0.03 * min(gs.players[pid].troops_garrison, 5)))


def _p_command(gs, pid) -> float:
    """Chance a Reveal reaches 6 persuasion (Command): deck's persuasion x 5 + seat."""
    p = gs.players[pid]
    cards = [c for z in (p.deck, p.discard, p.hand, p.in_play) for c in z]
    exp = 5.0 * sum((c.persuasion or 0) for c in cards) / max(1, len(cards)) \
        + (2 if p.has_councilor else 0)
    return max(0.1, min(0.9, (exp - 3.5) / 4.0))


def _sit_ada(gs, pid, d) -> float:
    """Advanced Data Analysis: an Intrigue every round (counted as the
    activation) for a Spy gone for good - cheap with spare Spies, dear when
    the Spies are your engine."""
    p = gs.players[pid]
    if sum(p.spies_on_board.values()) == 0:
        return -50.0                                     # needs a Spy on the board to trash
    return -(0.6 + 0.35 * rounds_left(gs) * 0.75)


def _sit_forbidden(gs, pid, d) -> float:
    """Every Reveal: 3 swords for an influence point (a friendship / alliance
    at 2 or 4 is expensive to lose), or lose all spice and trash it."""
    from src.ai.agents import _influence_loss_value, _RES_VALUE
    p = gs.players[pid]
    loss = min(_influence_loss_value(gs, pid, f) for f in p.influence)
    swords = 3 * SWORD * 2.0 * _p_fight(gs, pid)
    per = max(swords - loss, -_RES_VALUE["spice"] * p.spice)
    return 0.9 * rounds_left(gs) * per


def _sit_navigation(gs, pid, d) -> float:
    """Board spaces cost 1 spice / solari less: one saving per paid visit."""
    from src.ai.agents import _RES_VALUE
    agents_n = 3 if getattr(gs.players[pid], "has_swordmaster", False) else 2
    return 0.3 * agents_n * rounds_left(gs) * 0.5 * (_RES_VALUE["spice"] + _RES_VALUE["solari"])


def _sit_planetary(gs, pid, d) -> float:
    """A card per Conflict won (the base counts 30% a round) + the trash now."""
    from src.ai.agents import _RES_VALUE, _best_trash_value
    return (_p_win(gs, pid) - 0.3) * rounds_left(gs) * _RES_VALUE.get("draw", 1.1) \
        + 0.6 * _best_trash_value(gs, pid)


def _sit_plasteel(gs, pid, d) -> float:
    """An extra Sardaukar skill when you next recruit a commander."""
    p = gs.players[pid]
    row = getattr(gs.bl, "skill_row", []) or []
    best = max((skill_value(gs, pid, k) for k in row), default=0.0)
    p_rec = 0.75 if (p.skills or p.commanders_garrison or p.solari >= 2) else 0.45
    return p_rec * best * min(1.0, rounds_left(gs) / 3.0)


def _sit_dropships(gs, pid, d) -> float:
    """Once a round an Agent turn counts as a combat space: a deploy without
    spending a combat space, worth it to a player who fights."""
    p = gs.players[pid]
    return rounds_left(gs) * _p_fight(gs, pid) * 0.9 * min(1.0, 0.3 + p.troops_garrison / 3.0)


def _sit_high_command(gs, pid, d) -> float:
    """Every commander recruit 1 solari cheaper."""
    from src.ai.agents import _RES_VALUE
    p = gs.players[pid]
    per_round = 0.6 if (p.skills or _skills_on()) else 0.3
    return per_round * rounds_left(gs) * _RES_VALUE["solari"]


def _sit_servo(gs, pid, d) -> float:
    """The Signet Ring reaches every faction space: worth more with tracks to raise."""
    p = gs.players[pid]
    open_tracks = sum(1 for v in p.influence.values() if v < 4)
    return 0.6 * rounds_left(gs) * open_tracks / 4.0


def _sit_spy_drones(gs, pid, d) -> float:
    """A trash on any round you recall a Spy (flip)."""
    from src.ai.agents import _best_trash_value
    return 0.4 * rounds_left(gs) * 0.5 * _best_trash_value(gs, pid)


def _sit_suspensor(gs, pid, d) -> float:
    """A troop into the Conflict per Intrigue drawn on your turn."""
    p = gs.players[pid]
    rate = 0.5 + 0.15 * min(3, p.influence.get("bene_gesserit", 0)) + 0.1 * min(3, len(p.intrigue_cards))
    return rounds_left(gs) * rate * (0.4 + 1.2 * _p_fight(gs, pid))


def _sit_depot(gs, pid, d) -> float:
    """Command (6+ persuasion): 2 swords - re-priced by this deck's chance to Command."""
    return (_p_command(gs, pid) - P_COMMAND) * rounds_left(gs) * 2 * SWORD * 2.0 * _p_fight(gs, pid)


def _sit_glowglobes(gs, pid, d) -> float:
    return 0.4


def _sit_messages(gs, pid, d) -> float:
    """+1 persuasion each Reveal: worth more when it puts a TSMF / big card in reach."""
    from src.ai.agents import _best_buy_value
    P = _hand_persuasion(gs, pid)
    return 0.15 * rounds_left(gs) * max(0.0, _best_buy_value(gs, pid, P + 1) - _best_buy_value(gs, pid, P))


SITUATIONAL = {   # every Bloodlines tile (+ Spaceport)
    "Advanced Data Analysis": _sit_ada,
    "CHOAM Transports": _sit_choam,
    "Delivery Bay": _sit_delivery_bay,
    "Forbidden Weapons": _sit_forbidden,
    "Gene-Locked Vault": _sit_vault,
    "Glowglobes": _sit_glowglobes,
    "Navigation Chamber": _sit_navigation,
    "Ornithopter Fleet": _sit_fleet,
    "Panopticon": _sit_panopticon,
    "Planetary Array": _sit_planetary,
    "Plasteel Blades": _sit_plasteel,
    "Rapid Dropships": _sit_dropships,
    "Sardaukar High Command": _sit_high_command,
    "Self-Destroying Messages": _sit_messages,
    "Servo-Receivers": _sit_servo,
    "Spy Drones": _sit_spy_drones,
    "Suspensor Suits": _sit_suspensor,
    "Training Depot": _sit_depot,
    "Spaceport": _sit_spaceport,
}


def tech_demand(gs, pid: int, d) -> float:
    """tech_deny: tiles are unique - a tile a rival values highly and can
    (nearly) afford now may be gone before your next chance (rush / deny)."""
    best = 0.0
    for q in range(gs.num_players):
        if q == pid:
            continue
        price = gs.bl.tech_price(q, d)
        sp = gs.players[q].spice
        p_buy = 0.6 if sp >= price else 0.3 if sp >= price - 2 else 0.05
        best = max(best, p_buy * max(0.0, tech_buy_value(gs, q, d, demand=False)))
    return best


def tech_buy_value(gs, pid: int, d, discount: int = 0, demand: bool = True) -> float:
    from src.ai.agents import _RES_VALUE
    from src.ai import agents
    price = gs.bl.tech_price(pid, d, discount)
    love = float(agents._TUNE.get("tech_love", 1.0))
    v = 0.8 * love * tech_value(gs, pid, d) - _RES_VALUE["spice"] * price
    kl = float(agents._TUNE.get("tech_learned", 0.0))
    if kl:
        lv = learned_tech_gain(gs, pid, d)
        if lv is not None:
            # measured buy-vs-decline gain in win share (tech_study.py), 1% = 1 point
            w = min(1.0, kl)
            v = (1.0 - w) * v + w * 100.0 * lv * max(1.0, kl)
    k = float(agents._TUNE.get("tech_deny", 0.0))
    if k and demand:
        v += k * 0.5 * tech_demand(gs, pid, d)
    return v


_LEARNED = {"m": None, "loaded": False}


def learned_tech_gain(gs, pid: int, d):
    """tech_learned: the tile's measured gain (win share) in this situation."""
    if not _LEARNED["loaded"]:
        _LEARNED["loaded"] = True
        try:
            import json, os
            with open(os.path.join(os.path.dirname(__file__), "..", "..", "models_tech", "tech_values.json")) as fh:
                _LEARNED["m"] = json.load(fh)
        except (OSError, ValueError):
            _LEARNED["m"] = None
    m = _LEARNED["m"]
    if not m or d.name not in m["tiles"]:
        return None
    from src.selfplay.tech_study import feature_vector, _situation
    t = m["tiles"][d.name]
    x = feature_vector(_situation(gs, pid))
    return t["a"] + sum(a * b for a, b in zip(t["w"], x))


def best_tech_buy(gs, pid: int, discount: int = 0) -> float:
    bl = gs.bl
    best = 0.0
    for i in bl.buyable_stacks(pid, discount):
        d, extra = bl.tech_at(pid, i)
        if d is not None:
            best = max(best, tech_buy_value(gs, pid, d, discount + extra))
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
    # Loyal only works at 3+ Emperor influence (user: taking it with no
    # Emperor influence is a wasted pick)
    # (user, 2026-10-16: still overvalued - it was being taken without 3
    # Emperor influence). Below 3 it does nothing until you get there.
    e = gs.players[pid].influence.get("emperor", 0)
    return 1.0 if e >= 3 else 0.2 if e == 2 else 0.03 if e == 1 else 0.0


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
        "Loyal": 2 * sw * f * (0.9 if p.influence.get("emperor", 0) >= 3 else 0.0),   # off below 3
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
    v += _plasteel_follow_up(gs, pid, skill)
    return v


def _plasteel_follow_up(gs, pid: int, taken: str = "") -> float:
    """Any recruit (re-recruits too) lets Plasteel Blades be trashed for a row
    skill. On the Reveal turn with a commander in the Conflict, a Desperate
    taken that way can be trashed at once for 3 swords - the user's combo:
    trash Desperate, re-recruit, Plasteel -> the row's other Desperate, trash
    it too (only once the first is gone: no duplicate skills)."""
    bl, p = gs.bl, gs.players[pid]
    if not bl.has(pid, "Plasteel Blades"):
        return 0.0
    held = set(p.skills) | ({taken} if taken else set())
    row = [k for k in dict.fromkeys(getattr(bl, "skill_row", []) or []) if k not in held]
    best = 0.0
    for k in row:
        if (k == "Desperate" and gs.player_in_reveal_buy == pid
                and bl.commanders_in_conflict.get(pid, 0) > 0):
            from src.ai.agents import fight_ev, _my_strength
            mine, units = _my_strength(gs, pid)
            val = fight_ev(gs, pid, mine + 3, units) - fight_ev(gs, pid, mine, units)
        else:
            val = skill_value(gs, pid, k)
        best = max(best, val)
    return 0.9 * best


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
    p = gs.players[pid]
    if c is None:                       # a late buy / recruit later in the turn
        if choice.startswith("stack:"):
            d, extra = bl.tech_at(pid, int(choice.split(":")[1]))
            return tech_buy_value(gs, pid, d, extra) if d is not None else 0.0
        if choice.startswith("board:") or choice == "supply":
            return commander_value(gs, pid, choice)
        return 0.0
    if choice == "decline":
        return 0.2
    if c.kind in ("tech_buy", "tech_offer"):
        d, extra = bl.tech_at(pid, int(choice.split(":")[1]))
        return tech_buy_value(gs, pid, d, c.discount + extra) if d is not None else 0.0
    if c.kind == "commander":
        return commander_value(gs, pid, choice)
    if c.kind == "plasteel":              # trash Plasteel Blades for a row skill
        k = choice.split(":", 1)[1]
        if k == "Desperate" and gs.player_in_reveal_buy == pid \
                and bl.commanders_in_conflict.get(pid, 0) > 0:
            # trashed at once on this Reveal turn: what +3 swords do to this fight
            from src.ai.agents import fight_ev, _my_strength
            mine, units = _my_strength(gs, pid)
            return fight_ev(gs, pid, mine + 3, units) - fight_ev(gs, pid, mine, units)
        return skill_value(gs, pid, k)
    if c.kind == "desperate":
        # like a Combat Intrigue: what +3 swords change in this Conflict's
        # expected reward, against saving the tile for a later Reveal. With
        # Plasteel Blades and the row's other Desperate, the tile comes back
        # this turn (re-recruit -> Plasteel), so saving it is worth little.
        if choice != "use":
            return 0.0
        from src.ai.agents import fight_ev, _my_strength
        mine, units = _my_strength(gs, pid)
        gain = fight_ev(gs, pid, mine + 3, units) - fight_ev(gs, pid, mine, units)
        later = 3 * SWORD * 2.0 * _p_fight(gs, pid) * min(1.0, (rounds_left(gs) - 1) / 2.0)
        if bl.has(pid, "Plasteel Blades") and "Desperate" in (getattr(bl, "skill_row", []) or []):
            later *= 0.3
        return gain - 0.5 * max(0.0, later)
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
    if c.kind == "complete_contract":
        from src.ai.agents import _card_effect_value
        ct = next((x for x in p.contracts_active if x.name == choice.split(":", 1)[1]), None)
        return 0.5 + (_card_effect_value(gs, pid, ct.rewards) if ct is not None else 0.0)
    if c.kind == "vault":
        return 1.0 if choice == "intrigue" else 0.9
    if c.kind == "ada_spy":                # keep the Spy on the better post: trash any
        return 0.0
    if c.kind == "forbidden":
        if choice == "swords":
            return 3 * SWORD * min(2.0, _conflict_worth(gs, pid)) - 1.0
        return -0.5 * p.spice
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
