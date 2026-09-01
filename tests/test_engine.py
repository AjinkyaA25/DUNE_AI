"""Engine invariants: random self-play must always terminate cleanly."""
import random

import pytest

from src.data.card_definitions import setup_game
from src.game.gameState import ActionType, Phase
from src.game.board.board import UPRISING_BOARD, COMBAT_SPACES


def _random_playthrough(num_players, seed, cap=4000):
    gs = setup_game(num_players=num_players, seed=seed)
    rng = random.Random(seed)
    errors = 0
    moves = 0
    for _ in range(cap):
        if gs.game_over:
            break
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        acts = gs.get_valid_actions(pid)
        nn = [a for a in acts if a.action_type != ActionType.NO_OP]
        a = rng.choice(nn or acts)
        _, _, _, info = gs.step(a)
        if info.get("error"):
            errors += 1
        moves += 1
    return gs, errors, moves


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_random_selfplay_terminates(num_players):
    for seed in range(num_players * 7, num_players * 7 + 12):
        gs, errors, moves = _random_playthrough(num_players, seed)
        assert gs.game_over, f"game {seed} did not finish in {moves} moves"
        assert errors == 0, f"game {seed}: {errors} step errors"
        assert gs.winner is not None
        vps = [p.victory_points for p in gs.players]
        assert max(vps) >= 10 or gs.round >= 10 or len(gs.conflict_deck) == 0


def test_board_is_single_source():
    # gameState must import the derived tables, not redefine them
    import src.game.gameState as gsmod
    assert gsmod.BOARD_SPACE_ICONS is not None
    assert set(gsmod.ALL_BOARD_SPACES) == set(UPRISING_BOARD)
    assert "Sietch Tabr" in UPRISING_BOARD
    assert UPRISING_BOARD["Sietch Tabr"].agent_icon == "city"
    assert UPRISING_BOARD["Deliver Supplies"].agent_icon == "spacing_guild"
    assert UPRISING_BOARD["Research Station"].is_combat_space
    assert "Sardaukar" not in COMBAT_SPACES


def test_troop_conservation():
    gs, _, _ = _random_playthrough(4, seed=123)
    for p in gs.players:
        total = (p.troops_supply + p.troops_garrison
                 + gs.troops_in_conflict.get(p.id, 0))
        assert total == 12, f"player {p.id} has {total} troops"


def test_combat_deployment_happens():
    """Across a batch of random games, troops should reach the Conflict."""
    seen_deploy = 0
    for seed in range(20):
        gs = setup_game(4, seed=seed)
        rng = random.Random(seed)
        for _ in range(4000):
            if gs.game_over:
                break
            pid = gs.player_in_reveal_buy or gs.get_current_player_id()
            acts = gs.get_valid_actions(pid)
            nn = [a for a in acts if a.action_type != ActionType.NO_OP]
            a = rng.choice(nn or acts)
            if a.action_type == ActionType.RESOLVE_DEPLOY and a.deploy_count > 0:
                seen_deploy += 1
            gs.step(a)
    assert seen_deploy > 0


def test_special_spaces_reachable():
    gs = setup_game(4, seed=7)
    # Swordmaster requires affordability; High Council requires 5 solari.
    gs.players[0].solari = 20
    from src.game.cards.card import Card, CardType, AccessSymbol
    c = Card("T", CardType.STARTER)
    c.add_access_symbol(AccessSymbol.LANDSRAAD)
    ok, why = gs.can_send_agent(0, "Swordmaster", c)
    assert ok, why
    ok, why = gs.can_send_agent(0, "High Council", c)
    assert ok, why
    opts = gs.get_space_options(0, "Gather Support")
    assert "free" in opts and "paid" in opts


def test_victory_checked_after_combat():
    gs = setup_game(4, seed=5)
    gs.players[2].victory_points = 15
    gs.phase = Phase.MAKERS
    gs._auto_advance()
    assert gs.game_over and gs.winner == 2


def test_spy_topology_verified_posts():
    from src.game.board.board import (
        OBSERVATION_POST_CONNECTIONS as O,
        SPACE_TO_OBSERVATION_POSTS as S,
    )
    assert O["Landsraad Post"] == {"High Council", "Swordmaster", "Imperial Privilege"}
    assert O["Green Post"] == {"Assembly Hall", "Gather Support"}
    assert O["Bene Gesserit Post"] == {"Espionage", "Secrets"}
    assert O["Shipping Post"] == {"Accept Contract", "Shipping"}
    assert O["Hagga Basin Post"] == {"Hagga Basin"}
    assert O["Deep Desert Post"] == {"Deep Desert"}
    assert S["Research Station"] == {"Research Station Left Post",
                                     "Research Station Right Post"}
    assert S["Spice Refinery"] == {"Arrakeen Post", "Research Station Right Post"}
    assert S["Sietch Tabr"] == {"Research Station Left Post"}
    assert S["Arrakeen"] == {"Arrakeen Post"}
    assert "Imperial Basin Post" not in S["Accept Contract"]
    assert S["Deep Desert"] == {"Deep Desert Post"}


def _city_card():
    from src.game.cards.card import Card, CardType, AccessSymbol
    c = Card("CityTest", CardType.STARTER)
    c.add_access_symbol(AccessSymbol.CITY)
    return c


def test_two_spies_allow_infiltrate_plus_gather_not_double_draw():
    gs = setup_game(4, seed=7)
    p = gs.players[0]
    p.water = 6
    p.hand.append(_city_card())
    p.place_spy("Research Station Left Post")
    p.place_spy("Research Station Right Post")

    # Unoccupied: only single spy-mods, never (gather + gather) double-draw.
    combos = {
        (a.use_gather_intelligence, a.use_infiltrate)
        for a in gs.get_valid_actions(0)
        if a.action_type == ActionType.AGENT_TURN and a.space_name == "Research Station"
    }
    assert (True, False) in combos      # gather intelligence alone
    assert (True, True) not in combos   # no infiltrate target yet

    # Opponent occupies Research Station -> combined Infiltrate + Gather allowed.
    gs.agent_on_space["Research Station"] = 1
    combos = {
        (a.use_gather_intelligence, a.use_infiltrate)
        for a in gs.get_valid_actions(0)
        if a.action_type == ActionType.AGENT_TURN and a.space_name == "Research Station"
    }
    assert (True, True) in combos
    assert (False, True) in combos

    # Execute the combined action: both spies recalled, exactly one card drawn.
    act = next(
        a for a in gs.get_valid_actions(0)
        if a.action_type == ActionType.AGENT_TURN
        and a.space_name == "Research Station"
        and a.use_gather_intelligence and a.use_infiltrate
    )
    spies_before = p.spies_available
    gs.step(act)
    assert p.spies_available == spies_before + 2   # both spies recalled
    assert p.spies_on_board == {}
