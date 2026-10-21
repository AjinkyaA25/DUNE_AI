"""A purchase stays open while a Plot Intrigue can still pay for it (user,
2026-10-21: High Council with 3 spice, then a spice Intrigue -> Panopticon for 4)."""
from src.data.card_definitions import create_imperium_cards, setup_game
from src.game.bloodlines.techs import TECH_BY_NAME
from src.game.gameState import ActionType as T


def _setup(spice, intrigue="Desert Mouse"):
    gs = setup_game(4, seed=2, neutral_leaders=True, use_bloodlines=True)
    pid = gs.get_current_player_id()
    p = gs.players[pid]
    d = TECH_BY_NAME["Panopticon"]                      # 5 spice, 4 with the High Council seat
    for st in gs.bl.tech_stacks:
        if d in st:
            st.remove(d)
    for st in gs.bl.tech_stacks:                        # nothing else affordable
        st[:] = [x for x in st if x.cost >= 5]
    gs.bl.tech_stacks[0].insert(0, d)
    p.spice, p.solari = spice, 5
    p.intrigue_cards = [ic for ic in gs.intrigue_deck if ic.name == intrigue][:1]
    if not any("High Council" in gs.get_legal_agent_spaces(pid, c) for c in p.hand):
        card = next(c for c in create_imperium_cards() if "High Council" in gs.get_legal_agent_spaces(pid, c))
        p.hand.append(card)                             # a card that can go to High Council
    hc = next(a for a in gs.get_valid_actions(pid)
              if a.action_type == T.AGENT_TURN and a.space_name == "High Council")
    gs.step(hc)
    while True:                                         # settle the space's own choices
        v = gs.get_valid_actions(pid)
        if any(a.action_type in (T.RESOLVE_BL_CHOICE, T.PLAY_INTRIGUE) for a in v) or \
                not any(a.action_type in (T.RESOLVE_CHOICE, T.RESOLVE_INFLUENCE) for a in v):
            return gs, pid, p
        gs.step(next(a for a in v if a.action_type in (T.RESOLVE_CHOICE, T.RESOLVE_INFLUENCE)))


def _buys(gs, pid):
    return [a.choice for a in gs.get_valid_actions(pid)
            if a.action_type == T.RESOLVE_BL_CHOICE and a.choice.startswith("stack:")]


def test_spice_intrigue_mid_turn_unlocks_the_tech():
    gs, pid, p = _setup(spice=3)
    assert p.has_councilor and gs.bl.tech_price(pid, TECH_BY_NAME["Panopticon"]) == 4
    v = gs.get_valid_actions(pid)
    assert not _buys(gs, pid)                           # 3 spice: not yet
    play = [a for a in v if a.action_type == T.PLAY_INTRIGUE and a.intrigue_card_name == "Desert Mouse"]
    assert play, [repr(a) for a in v]                   # ... but the Intrigue can be played first
    gs.step(play[0])
    assert p.spice == 4
    assert "stack:0" in _buys(gs, pid)                  # now Panopticon is offered
    gs.step(next(a for a in gs.get_valid_actions(pid) if a.choice == "stack:0"))
    assert any(t.name == "Panopticon" for t in p.techs) and p.spice == 0


def test_no_offer_when_nothing_can_pay():
    gs, pid, p = _setup(spice=3, intrigue="Strategic Stockpiling")
    v = gs.get_valid_actions(pid)
    assert not any(a.action_type == T.RESOLVE_BL_CHOICE and a.choice.startswith("stack:") for a in v)


def test_ui_says_an_intrigue_can_pay():
    import sys
    sys.path.insert(0, "ui")
    from server import purchase_prompt
    gs, pid, p = _setup(spice=3)
    msg = purchase_prompt(gs, pid)
    assert msg and "Panopticon 4" in msg and "Plot Intrigue" in msg
