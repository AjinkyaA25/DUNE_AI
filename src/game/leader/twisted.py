"""Piter De Vries' 12 Twisted Intrigue cards (src/game/leader/LEADERS.md).

They are Intrigue cards (they can be stolen and given away); once played they
go to Piter's own Twisted discard pile, not the shared Intrigue discard
(GameState.discard_intrigue_card routes them by `card.twisted`).

Option requirements use internal keys checked by choices._option_possible:
`_need_troops` (troops anywhere), `_need_conflict_troops`, `_need_alliance`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.game.intrigue.intrigue import IntrigueCard, IntrigueTiming
from src.game.leader.leader_definitions import fx, _fx, _choose

_PLOT = IntrigueTiming.PLOT
_COMBAT = IntrigueTiming.COMBAT
_PC = (IntrigueTiming.PLOT, IntrigueTiming.COMBAT)
FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")


def _opt(label, effect):
    return {"label": label, "effect": effect}


def create_twisted_intrigues() -> List[IntrigueCard]:
    IC = IntrigueCard
    cards = [
        IC("Ambitious", _PLOT, [{"choose_one": {"prompt": "Ambitious", "may": False, "options": [
            _opt("Lose 3 troops -> +1 influence where an opponent has more than you",
                 {"_need_troops": 3, "_need_behind": 1,
                  "_leader_fx": ["lose_troops", 3, {"_leader_fx": ["influence_behind"]}],
                  "_v": {"troops": -3, "influence_any": 1}})]}}]),
        IC("Calculating", _PLOT, [{"_leader_fx": ["unit_types_solari"]}]),
        IC("Controlled", _PC, [{"choose_by_combat": {
            "combat": {"swords": 1},
            "else": {"_leader_fx": ["controlled_peek"]}}}]),
        IC("Devious", _PLOT, [{"choose_one": {"prompt": "Devious", "may": False, "options": [
            _opt("Trash a card from your hand", {"_leader_fx": ["trash_from_hand"], "_v": {"trash": 1}}),
            _opt("Deploy up to 2 troops from your garrison", {"deploy_up_to": 2})]}}]),
        IC("Discerning", _PLOT, [{"choose_one": {"prompt": "Discerning", "options": [
            _opt("Discard a card -> draw a card", {"discard_then": {"draw": 1}}),
            _opt("If you have an Alliance: draw a card", {"_need_alliance": 1, "draw": 1})]}}]),
        IC("Insidious", _PLOT, [{"_leader_fx": ["give_intrigue"]}]),
        # "The card you play this turn": only before placing an Agent
        IC("Resourceful", _PLOT, [{"if_turn_start": {"_leader_fx": ["resourceful"]}}]),
        IC("Sadistic", _PLOT, [{"choose_one": {"prompt": "Sadistic", "may": False, "options": [
            _opt("Lose 1 troop -> draw a card",
                 {"_need_troops": 1, "_leader_fx": ["lose_troops", 1, {"draw": 1}],
                  "_v": {"troops": -1, "draw": 1}})]}}]),
        IC("Shrewd", _COMBAT, [{"choose_one": {"prompt": "Shrewd", "may": False, "options": [
            _opt("Lose 1 troop in the Conflict -> 1 spice",
                 {"_need_conflict_troops": 1,
                  "_leader_fx": ["lose_conflict_troops", 1, {"spice": 1}], "_v": {"spice": 1}})]}}]),
        IC("Sinister", _COMBAT, [{"choose_one": {"prompt": "Sinister", "may": False, "options": [
            _opt("Lose 2 troops -> an Intrigue + 1 solari",
                 {"_need_troops": 2, "_leader_fx": ["lose_troops", 2, {"intrigue": 1, "solari": 1}],
                  "_v": {"troops": -2, "intrigue": 1, "solari": 1}})]}}]),
        IC("Unnatural", _PLOT, [{"_leader_fx": ["unnatural"]}]),
        # "At the start of your turn: pass your turn." Played before placing an
        # Agent: play moves on to the next player (stall to see their moves).
        IC("Withdrawn", _PLOT, [{"if_turn_start": {"_leader_fx": ["withdrawn"]}}]),
    ]
    for c in cards:
        c.twisted = True
    return cards


# ---------------------------------------------------------------------------
# effects
# ---------------------------------------------------------------------------

def _plain_garrison(gs, p) -> int:
    return p.troops_garrison - getattr(p, "commanders_garrison", 0)


def _plain_conflict(gs, p) -> int:
    bl = getattr(gs, "bl", None)
    cmd = bl.commanders_in_conflict.get(p.id, 0) if bl is not None else 0
    return gs.troops_in_conflict.get(p.id, 0) - cmd


def playable(gs, p, spec) -> bool:
    """Can this Twisted effect do anything now (else the card can't be played)?"""
    name = spec[0]
    if name == "unit_types_solari":
        return (gs.troops_in_conflict.get(p.id, 0) + gs.sandworms_in_conflict.get(p.id, 0)
                + bool(getattr(gs, "duncan_in_conflict", {}).get(p.id))) > 0
    if name in ("give_intrigue", "unnatural"):
        return len(p.intrigue_cards) >= 2          # another Intrigue besides this one
    return True


def _lose_from_conflict(gs, p, n: int) -> int:
    k = min(n, _plain_conflict(gs, p))
    if k > 0:
        gs.troops_in_conflict[p.id] -= k
        p.troops_supply += k
        if p.leader is not None:
            p.leader.trigger("on_units_left_conflict", p, gs, k)
        gs.update_combat_strength(p.id)
    return k


@fx("lose_troops")
def _fx_lose_troops(gs, p, n: int, reward: Optional[Dict] = None) -> None:
    """Lose n of your troops -> reward. The player picks how many come from
    the garrison and how many from the Conflict."""
    gar, conf = _plain_garrison(gs, p), _plain_conflict(gs, p)
    if gar + conf < n:
        return
    splits = list(range(max(0, n - conf), min(n, gar) + 1))[::-1]
    opts = [(f"split:{g}", f"Lose {g} from your garrison, {n - g} from the Conflict"
             if 0 < g < n else (f"Lose {n} from your garrison" if g == n else f"Lose {n} from the Conflict"),
             {"_leader_fx": ["lose_split", g, n - g, reward]}) for g in splits]
    _choose(p, gs, f"Lose {n} troop{'s' if n > 1 else ''}", opts, may=False)


@fx("lose_split")
def _fx_lose_split(gs, p, g: int, c: int, reward: Optional[Dict] = None) -> None:
    if _plain_garrison(gs, p) < g or _plain_conflict(gs, p) < c:
        return
    p.troops_garrison -= g
    p.troops_supply += g
    if g and getattr(p, "undeployable_this_turn", 0):     # Piter's locked troop can be the one lost
        p.undeployable_this_turn = max(0, p.undeployable_this_turn - g)
    _lose_from_conflict(gs, p, c)
    if reward:
        _fx(p, gs, reward)


@fx("lose_conflict_troops")
def _fx_lose_conflict_troops(gs, p, n: int, reward: Optional[Dict] = None) -> None:
    if _lose_from_conflict(gs, p, n) == n and reward:
        _fx(p, gs, reward)


def _behind(gs, p) -> List[str]:
    return [f for f in FACTIONS if p.influence[f] < 6
            and any(q.influence[f] > p.influence[f] for q in gs.players if q.id != p.id)]


@fx("influence_behind")
def _fx_influence_behind(gs, p) -> None:
    opts = [(f, f"+1 {f.replace('_', ' ').title()} influence", {"_gain_influences": [f]})
            for f in _behind(gs, p)]
    _choose(p, gs, "Ambitious: gain 1 influence", opts, may=False)


@fx("unit_types_solari")
def _fx_unit_types_solari(gs, p) -> None:
    types = 0
    bl = getattr(gs, "bl", None)
    cmd = bl.commanders_in_conflict.get(p.id, 0) if bl is not None else 0
    if gs.troops_in_conflict.get(p.id, 0) - cmd > 0:
        types += 1
    if cmd > 0:
        types += 1
    if gs.sandworms_in_conflict.get(p.id, 0) > 0:
        types += 1
    if getattr(gs, "duncan_in_conflict", {}).get(p.id):
        types += 1
    if types:
        p.gain_solari(types)


@fx("controlled_peek")
def _fx_controlled_peek(gs, p) -> None:
    if not p.deck and p.discard:
        nd = list(p.discard)
        gs.shuffle_pile(nd, 1, p.id)
        p.deck, p.discard = nd + p.deck, []
    if not p.deck:
        return
    top = p.deck[-1].name
    opts = [("back", f"Put {top} back", {}),
            ("discard", f"Discard {top}", {"_leader_fx": ["discard_top"]})]
    if p.solari >= 1:
        opts.append(("draw", f"Pay 1 solari: draw {top}", {"_pay": {"solari": 1}, "_draw_top": 1,
                                                          "_v": {"draw": 1}}))
    _choose(p, gs, f"Controlled: top card of your deck is {top}", opts, may=False)


@fx("discard_top")
def _fx_discard_top(gs, p) -> None:
    if p.deck:
        p.discard.append(p.deck.pop())


@fx("trash_from_hand")
def _fx_trash_from_hand(gs, p) -> None:
    from src.game.leader.leader_definitions import _trash_hand_options
    _choose(p, gs, "Devious: trash a card from your hand", _trash_hand_options(p), may=False)


@fx("give_intrigue")
def _fx_give_intrigue(gs, p) -> None:
    opts, seen = [], set()
    for ic in p.intrigue_cards:
        for q in gs.players:
            key = (ic.name, q.id)
            if q.id == p.id or key in seen:
                continue
            seen.add(key)
            bonus = 2 if not getattr(ic, "twisted", False) else 1
            opts.append((f"give:{ic.name}:{q.id}", f"Give {ic.name} to player {q.id + 1} -> {bonus} spice",
                         {"_leader_fx": ["give_to", ic.name, q.id], "_v": {"spice": bonus, "intrigue": -1}}))
    _choose(p, gs, "Insidious: give an opponent an Intrigue card", opts, may=False)


@fx("give_to")
def _fx_give_to(gs, p, name: str, qid: int) -> None:
    ic = next((c for c in p.intrigue_cards if c.name == name), None)
    if ic is None:
        return
    p.intrigue_cards.remove(ic)
    gs.players[qid].add_intrigue(ic)
    _fx(p, gs, {"spice": 1 if getattr(ic, "twisted", False) else 2})


@fx("resourceful")
def _fx_resourceful(gs, p) -> None:
    p.extra_icons_this_turn = {"landsraad", "city", "desert"}


@fx("unnatural")
def _fx_unnatural(gs, p) -> None:
    opts, seen = [], set()
    for ic in p.intrigue_cards:
        if ic.name in seen:
            continue
        seen.add(ic.name)
        extra = {} if getattr(ic, "twisted", False) else {"troops": 1}
        opts.append((f"trash:{ic.name}", f"Trash {ic.name} -> an Intrigue" + (" + 1 troop" if extra else ""),
                     {"_leader_fx": ["trash_intrigue", ic.name], "_v": dict(extra)}))
    _choose(p, gs, "Unnatural: trash an Intrigue card", opts, may=False)


@fx("trash_intrigue")
def _fx_trash_intrigue(gs, p, name: str) -> None:
    ic = next((c for c in p.intrigue_cards if c.name == name), None)
    if ic is None:
        return
    p.intrigue_cards.remove(ic)
    gs.draw_intrigue_for_player(p.id, 1)
    if not getattr(ic, "twisted", False):
        _fx(p, gs, {"troops": 1})


@fx("withdrawn")
def _fx_withdrawn(gs, p) -> None:
    gs._pass_requested = p.id          # GameState.step passes the turn
