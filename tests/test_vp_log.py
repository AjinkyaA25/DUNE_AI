"""Every VP a player has is in their VP breakdown (Player.vp_log), labelled."""
import random

from src.ai.agents import make_agent
from src.data.card_definitions import setup_game
from src.game.leader.leader_definitions import LEADER_NAMES


def test_vp_log_adds_up_and_is_labelled():
    labels = set()
    for g in range(4):
        r = random.Random(g)
        gs = setup_game(num_players=4, seed=g, leaders=r.sample(list(LEADER_NAMES), 4), use_bloodlines=True)
        ag = [make_agent("heuristic", seed=g * 4 + s) for s in range(4)]
        n = 0
        while not gs.game_over and n < 3000:
            pid = gs.player_in_reveal_buy
            if pid is None:
                pid = gs.get_current_player_id()
            gs.step(ag[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
            n += 1
        for p in gs.players:
            assert sum(d for _, d, _ in p.vp_log) == p.victory_points
            assert p.vp_log[0] == ("Starting point", 1, 0)
            assert not p.vp_ctx
            labels |= {s for s, _, _ in p.vp_log}
    assert any(s.endswith(" friendship") for s in labels)
    assert all(s and s != "Other" for s in labels), labels
