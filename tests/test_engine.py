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


def test_card_corrections_2026_09_01():
    """Spot-check the user-verified Imperium card fixes."""
    from src.game.effects import EffectResolver
    from src.data.card_definitions import create_imperium_cards
    from src.game.cards.card import AccessSymbol, CardTag

    cs = {c.name: c for c in create_imperium_cards()}

    # Ecological Testing Station: Fremen+City access, Fremen-bond -> water.
    e = cs["Ecological Testing Station"]
    assert e.access_symbols == {AccessSymbol.FREMEN, AccessSymbol.CITY}
    assert e.reveal_effects == [{"if_fremen_bond": {"water": 1}}]

    # Interstellar Trade: acquire a contract; reveal = 1 persuasion / contract.
    it = cs["Interstellar Trade"]
    assert it.acquire_effects == [{"contract": 1}]
    assert AccessSymbol.SPACING not in it.access_symbols

    # Junction HQ: no Spacing Guild access.
    assert AccessSymbol.SPACING not in cs["Junction Headquarters"].access_symbols

    gs = setup_game(2, seed=3)
    p = gs.players[0]

    # persuasion_per_contract scales with completed contracts.
    p.contracts_completed = [object(), object(), object()]
    gs.persuasion_pool[0] = 0
    EffectResolver.resolve_single_effect({"persuasion_per_contract": 1}, p, gs)
    assert gs.get_persuasion(0) == 3

    # choose_by_contracts: VP at 4+, else spice.
    vp0, sp0 = p.victory_points, p.spice
    EffectResolver.resolve_single_effect(
        {"choose_by_contracts": {"n": 4, "yes": {"vp": 1}, "no": {"spice": 1}}}, p, gs)
    assert p.spice == sp0 + 1 and p.victory_points == vp0
    p.contracts_completed = [object()] * 4
    EffectResolver.resolve_single_effect(
        {"choose_by_contracts": {"n": 4, "yes": {"vp": 1}, "no": {"spice": 1}}}, p, gs)
    assert p.victory_points == vp0 + 1

    # In High Places reveal: recall 2 spies -> +3 persuasion.
    p.place_spy("Emperor Post")
    p.place_spy("Fremen Post")
    gs.persuasion_pool[0] = 0
    EffectResolver.resolve_single_effect(
        {"recall_spies_persuasion": {"count": 2, "persuasion": 3}}, p, gs)
    assert gs.get_persuasion(0) == 3
    assert sum(p.spies_on_board.values()) == 0


def test_card_corrections_batch3():
    from src.game.effects import EffectResolver
    from src.data.card_definitions import create_imperium_cards
    from src.game.cards.card import AccessSymbol, CardTag

    cs = {c.name: c for c in create_imperium_cards()}
    assert cs["Leadership"].access_symbols == {AccessSymbol.DESERT, AccessSymbol.FREMEN}
    assert cs["Long Live the Fighters"].access_symbols == {AccessSymbol.FREMEN,
                                                           AccessSymbol.CITY}
    assert cs["Sardaukar Soldier"].trash_effects == [{"intrigue": 1}]
    assert cs["Maula Pistol"].agent_effects == [{"draw": 1}]

    gs = setup_game(2, seed=11)
    p = gs.players[0]

    # Sardaukar Soldier: trashing it grants an Intrigue.
    from src.game.cards.card import Card, CardType
    ss = next(c for c in create_imperium_cards() if c.name == "Sardaukar Soldier")
    p.hand.append(ss)
    gs.add_pending_trash(0, 1)
    n_intrigue = len(p.intrigue_cards)
    from src.game.gameState import GameAction, ActionType
    gs.step(GameAction(ActionType.RESOLVE_TRASH, 0, trash_card_name="Sardaukar Soldier"))
    assert len(p.intrigue_cards) == n_intrigue + 1
    assert ss in p.trash

    # Reliable Informant: spy restricted to the 3 faction posts.
    gs.add_pending_spy_placement(0, 1, allow_occupied=False,
                                 allowed_posts=["Emperor Post", "Bene Gesserit Post",
                                                "Fremen Post"])
    spy_actions = [a for a in gs.get_valid_actions(0)
                   if a.action_type == ActionType.RESOLVE_SPY]
    assert spy_actions and all(
        a.spy_post_name in {"Emperor Post", "Bene Gesserit Post", "Fremen Post"}
        for a in spy_actions)

    # Leadership reveal: +1 sword per OTHER card revealed this turn.
    p._revealed_this_turn = 4
    gs.swords_this_reveal[0] = 0
    EffectResolver.resolve_single_effect({"swords_per_other_revealed_card": 1}, p, gs)
    assert gs.swords_this_reveal[0] == 3

    # Maker Keeper agent: BG2 -> water, Fremen2 -> spice (no persuasion).
    p.influence["bene_gesserit"] = 2
    p.influence["fremen"] = 2
    w0, sp0 = p.water, p.spice
    EffectResolver.resolve_single_effect(
        {"if_influence_bene_gesserit_2": {"water": 1},
         "if_influence_fremen_2": {"spice": 1}}, p, gs)
    assert p.water == w0 + 1 and p.spice == sp0 + 1


def test_card_corrections_batch4():
    from src.game.effects import EffectResolver
    from src.data.card_definitions import create_imperium_cards
    from src.game.gameState import GameAction, ActionType

    cs = {c.name: c for c in create_imperium_cards()}
    assert cs["Shishakli"].agent_effects == [{"trash_then": {"draw": 1}}]
    assert cs["Smuggler's Haven"].agent_effects == [
        {"pay_then": {"cost": {"spice": 4}, "vp": 1}}]

    gs = setup_game(2, seed=9)
    p = gs.players[0]

    # Shishakli: trashing a card draws a card; declining does not.
    from src.game.cards.card import Card, CardType
    p.hand = [Card("Junk", CardType.STARTER) for _ in range(3)]
    p.deck = [Card(f"D{i}", CardType.STARTER) for i in range(5)]
    EffectResolver.resolve_single_effect({"trash_then": {"draw": 1}}, p, gs)
    hand_n = len(p.hand)
    gs.step(GameAction(ActionType.RESOLVE_TRASH, 0, trash_card_name="Junk"))
    assert len(p.hand) == hand_n           # -1 trashed, +1 drawn
    assert any(c.name == "Junk" for c in p.trash)

    EffectResolver.resolve_single_effect({"trash_then": {"draw": 1}}, p, gs)
    hand_n = len(p.hand)
    gs.step(GameAction(ActionType.RESOLVE_TRASH, 0, trash_card_name=None))  # decline
    assert len(p.hand) == hand_n           # no draw on decline

    # Smuggler's Haven reveal: +2 spice only when a Spy borders a Maker space.
    sp0 = p.spice
    EffectResolver.resolve_single_effect({"if_spy_at_maker_post": {"spice": 2}}, p, gs)
    assert p.spice == sp0                  # no maker-post spy yet
    p.place_spy("Hagga Basin Post")
    EffectResolver.resolve_single_effect({"if_spy_at_maker_post": {"spice": 2}}, p, gs)
    assert p.spice == sp0 + 2

    # Smuggler's Haven agent: 4 spice -> 1 VP is an ARROW = optional accept/decline.
    p.spice = 4
    vp0 = p.victory_points
    EffectResolver.resolve_single_effect(
        {"pay_then": {"cost": {"spice": 4}, "vp": 1}}, p, gs)
    assert len(gs.pending_optional_payments) == 1          # not auto-paid
    opts = {a.accept_optional for a in gs.get_valid_actions(0)
            if a.action_type == ActionType.RESOLVE_OPTIONAL}
    assert opts == {True, False}
    gs.step(GameAction(ActionType.RESOLVE_OPTIONAL, 0, accept_optional=False))  # decline
    assert p.spice == 4 and p.victory_points == vp0
    EffectResolver.resolve_single_effect(
        {"pay_then": {"cost": {"spice": 4}, "vp": 1}}, p, gs)
    gs.step(GameAction(ActionType.RESOLVE_OPTIONAL, 0, accept_optional=True))   # accept
    assert p.spice == 0 and p.victory_points == vp0 + 1


def test_arrow_effects_are_optional():
    """Every cost->reward arrow surfaces as an accept/decline choice."""
    from src.game.effects import EffectResolver
    from src.game.gameState import GameAction, ActionType
    from src.game.cards.card import Card, CardType

    gs = setup_game(2, seed=4)
    p = gs.players[0]
    p.hand = [Card(f"H{i}", CardType.STARTER) for i in range(4)]
    p.deck = [Card(f"D{i}", CardType.STARTER) for i in range(6)]

    # Captured Mentat agent: discard_then -> optional, decline = no discard.
    EffectResolver.resolve_single_effect({"discard_then": {"draw": 1, "intrigue": 1}}, p, gs)
    assert len(gs.pending_optional_payments) == 1
    hand0, disc0 = len(p.hand), len(p.discard)
    gs.step(GameAction(ActionType.RESOLVE_OPTIONAL, 0, accept_optional=False))
    assert len(p.hand) == hand0 and len(p.discard) == disc0
    assert not gs.pending_optional_payments

    # Accepting: discards 1, draws 1, gains an intrigue.
    ig0 = len(p.intrigue_cards)
    EffectResolver.resolve_single_effect({"discard_then": {"draw": 1, "intrigue": 1}}, p, gs)
    gs.step(GameAction(ActionType.RESOLVE_OPTIONAL, 0, accept_optional=True))
    assert len(p.intrigue_cards) == ig0 + 1

    # pay_then with an unaffordable cost is never queued.
    p.spice = 1
    EffectResolver.resolve_single_effect(
        {"pay_then": {"cost": {"spice": 3}, "trash": 1, "draw": 1}}, p, gs)
    assert not gs.pending_optional_payments


def test_sardaukar_coordination_deploy_recruited_uncapped():
    from src.game.effects import EffectResolver
    gs = setup_game(2, seed=6)
    p = gs.players[0]
    gs.current_conflict = gs.current_conflict or object()  # ensure a conflict ref
    p.troops_garrison = 5
    p.troops_recruited_this_turn = 4          # e.g. recruited at the Sardaukar space
    EffectResolver.resolve_single_effect({"deploy_recruited": 1}, p, gs)
    pend = next(d for d in gs.pending_deployments if d.player_id == 0)
    assert pend.max_deploy == 4               # all recruited troops, no 9/2 cap
    # garrison-bounded when fewer troops on hand
    gs.pending_deployments.clear()
    p.troops_garrison = 2
    EffectResolver.resolve_single_effect({"deploy_recruited": 1}, p, gs)
    assert next(d for d in gs.pending_deployments if d.player_id == 0).max_deploy == 2


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
