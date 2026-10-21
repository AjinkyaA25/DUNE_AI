"""Steersman Y'rkoon's 10 Navigation cards (src/game/leader/LEADERS.md).

Played automatically, left to right, each time he reaches 2 influence with
a Faction. "OR" cards become a pending choice; slot-dependent cards check
the slot the card sat in.
"""
from __future__ import annotations

from src.game.leader.leader_definitions import fx, _fx, _choose

NAV_IDS = tuple(range(1, 11))
FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")


def play_navigation(p, gs, card: int, slot: int, faction: str) -> None:
    title = f"Navigation card {card}"
    # automatic, so say so in the game log (user: card 3's solari went unnoticed)
    gs.note(f"{p.leader.name}: plays Navigation card {card} (slot {slot})"
            + (" -> 2 solari" + (" + 1 persuasion every Reveal turn for the rest of the game"
                                 if slot == 4 else "") if card == 3 else ""))
    if card == 1:
        others = [f for f in FACTIONS if f != faction and p.influence[f] >= 2 and p.influence[f] < 6]
        opts = [("spice", "1 spice", {"spice": 1})]
        opts += [(f"inf:{f}", f"Pay 2 solari: +1 {f.replace('_', ' ').title()} influence",
                  {"_pay": {"solari": 2}, "_gain_influences": [f]}) for f in others]
        _choose(p, gs, title, opts, may=False)
    elif card == 2:
        opts = []
        if p.spies_available > 0:
            opts.append(("spy", "Place a Spy", {"spy": 1}))
        if p.spies_on_board:
            opts.append(("recall", "Recall a Spy -> an Intrigue + 2 spice",
                         {"_recall_spy": 1, "intrigue": 1, "spice": 2}))
        _choose(p, gs, title, opts, may=False)
    elif card == 3:
        _fx(p, gs, {"solari": 2})
        if slot == 4:
            p.leader.state["nav_reveal_persuasion"] = 1     # +1 persuasion every Reveal turn
    elif card == 4:
        opts = [("spice", "1 spice", {"spice": 1})]
        if slot == 1 and gs.reserve_spice_must_flow:
            opts.append(("tsmf", "Pay 1 water: acquire The Spice Must Flow",
                         {"_pay": {"water": 1}, "_leader_fx": ["acquire_tsmf"], "_v": {"vp": 1}}))
        _choose(p, gs, title, opts, may=False)
    elif card == 5:
        from src.game.leader.leader_definitions import _trash_any_options
        _choose(p, gs, title, _trash_any_options(p, {"spice": 2}, " (+2 spice)"), may=False)
    elif card == 6:
        _choose(p, gs, title, [("troop", "1 troop", {"troops": 1}),
                               ("troops3", "Pay 3 solari: 3 troops", {"_pay": {"solari": 3}, "troops": 3})],
                may=False)
    elif card == 7:
        _fx(p, gs, {"spice": 1})
        if any(p.alliances.values()):
            _fx(p, gs, {"intrigue": 1})
    elif card == 8:
        _fx(p, gs, {"water": 1})
        if faction == "spacing_guild":
            _fx(p, gs, {"spice": 1})
    elif card == 9:
        _choose(p, gs, title, [("draw", "Draw a card", {"draw": 1}),
                               ("vp", "Pay 5 spice: 1 Victory Point", {"_pay": {"spice": 5}, "vp": 1})],
                may=False)
    elif card == 10:
        # lose 1 influence with a Faction of your choice -> gain 1 influence
        # with a Faction of your choice (the same one is allowed: dropping
        # from 2 and climbing back reaches 2 again -> the next Navigation card)
        names = {f: f.replace("_", " ").title() for f in FACTIONS}
        opts = []
        for lose in FACTIONS:
            if p.influence[lose] <= 0:
                continue
            for gain in FACTIONS:
                if p.influence[gain] - (1 if gain == lose else 0) >= 6:
                    continue
                opts.append((f"swap:{lose}>{gain}", f"-1 {names[lose]}, +1 {names[gain]}",
                             {"_lose_influence": lose, "_gain_influences": [gain]}))
        _choose(p, gs, title, opts)


@fx("acquire_tsmf")
def _fx_acquire_tsmf(gs, p) -> None:
    gs.acquire_reserve_free(p.id, "spice_must_flow") if hasattr(gs, "acquire_reserve_free") \
        else _acquire_tsmf_fallback(gs, p)


def _acquire_tsmf_fallback(gs, p) -> None:
    if not gs.reserve_spice_must_flow:
        return
    card = gs.reserve_spice_must_flow.pop()
    p.discard.append(card)
    p.tsmf_acquired = getattr(p, "tsmf_acquired", 0) + 1
    gs._trigger_acquire_effects(p.id, card)


@fx("return_contract")
def _fx_return_contract(gs, p, name: str) -> None:
    c = next((x for x in p.contracts_active if x.name == name), None)
    if c is not None:
        p.contracts_active.remove(c)
        gs.contract_bank.append(c)
