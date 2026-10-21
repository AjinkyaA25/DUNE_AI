"""AI plays Leaders' space-dependent passives (user, 2026-10-17)."""
from types import SimpleNamespace

from src.data.card_definitions import setup_game


def _game(leaders, seed=11):
    return setup_game(4, seed=seed, leaders=leaders, use_bloodlines=True)


def test_jessica_values_a_bene_gesserit_space_once_she_has_memories():
    from src.ai.agents import _leader_space_value
    gs = _game(["Lady Jessica", "Gurney Halleck", "Chani", "Duncan Idaho"])
    a = SimpleNamespace(space_name="Secrets", space_option=None)
    assert _leader_space_value(gs, 0, a) == 0.0
    gs.players[0].leader.state["memories"] = 2
    assert _leader_space_value(gs, 0, a) > 3.0


def test_reverend_mother_values_the_repeat():
    from src.ai.agents import _choice_option_value
    gs = _game(["Lady Jessica", "Gurney Halleck", "Chani", "Duncan Idaho"])
    p = gs.players[0]
    p.leader.state["flipped"] = True
    p.leader.name = "Reverend Mother Jessica"
    p.water = 2
    eff = {"_pay": {"water": 1}, "_leader_fx": ["rm_repeat", "Secrets", None]}
    assert _choice_option_value(gs, 0, eff) > 0.0


def test_shaddam_signet_on_a_combat_space_gets_no_conflict_value():
    from src.ai.agents import make_agent
    from src.game.gameState import ActionType
    gs = _game(["Shaddam Corrino IV", "Gurney Halleck", "Chani", "Duncan Idaho"])
    p = gs.players[0]
    from src.game.board.board import UPRISING_BOARD
    ag = make_agent("heuristic", seed=0)
    if not any(c.name == "Signet Ring" for c in p.hand):
        ring_card = next(c for c in p.deck if c.name == "Signet Ring")
        p.deck.remove(ring_card)
        p.hand.append(ring_card)
    ring = next(a for a in gs.get_valid_actions(0)
                if a.action_type == ActionType.AGENT_TURN and a.card_name == "Signet Ring"
                and UPRISING_BOARD[a.space_name].is_combat_space)
    s_ring = ag.score(gs, 0, ring)
    p.no_deploy_this_turn = True
    s_nodeploy = ag._score_core(gs, 0, ring)
    p.no_deploy_this_turn = False
    assert abs(s_ring - s_nodeploy) < 1e-6 or s_ring <= s_nodeploy + 1e-6
    assert not p.no_deploy_this_turn
