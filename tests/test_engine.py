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


def test_card_corrections_batch5():
    from src.game.effects import EffectResolver
    from src.data.card_definitions import create_imperium_cards
    from src.game.cards.card import Card, CardType, AccessSymbol, CardTag
    from src.game.gameState import GameAction, ActionType

    cs = {c.name: c for c in create_imperium_cards()}
    assert cs["Spy Network"].access_symbols == set()          # no access
    assert cs["Spy Network"].acquire_effects == [{"spy": 1}]
    assert cs["Strike Fleet"].acquire_effects == [{"spy": 1}]
    assert cs["Steersman"].acquire_effects == [{"influence_spacing_guild": 1}]
    assert cs["Steersman"].agent_effects == [{"draw": 1, "uplift": 1}]
    assert AccessSymbol.FREMEN in cs["Stilgar, the Devoted"].access_symbols

    gs = setup_game(2, seed=8)
    p = gs.players[0]

    # Space-Time Folding: discard a Guild card -> draw 2 (1 base + 1 bonus).
    guild = Card("GuildJunk", CardType.IMPERIUM); guild.add_tag(CardTag.SPACING_GUILD)
    p.hand = [guild, Card("X", CardType.STARTER)]
    p.deck = [Card(f"D{i}", CardType.STARTER) for i in range(6)]
    EffectResolver.resolve_single_effect(
        {"discard_then_sg": {"base": {"draw": 1}, "sg_bonus": {"draw": 1}}}, p, gs)
    hand_before = len(p.hand)
    gs.step(GameAction(ActionType.RESOLVE_OPTIONAL, 0, accept_optional=True))
    assert guild in p.discard
    assert len(p.hand) == hand_before - 1 + 2                 # -1 discard, +2 draw

    # recall_spy_then is now a MAY (optional) — declining keeps the spy.
    p.place_spy("Emperor Post")
    p.place_spy("Fremen Post")
    EffectResolver.resolve_single_effect({"recall_spy_then": {"intrigue": 1}}, p, gs)
    assert gs.pending_optional_payments
    ig0 = len(p.intrigue_cards)
    gs.step(GameAction(ActionType.RESOLVE_OPTIONAL, 0, accept_optional=False))
    assert sum(p.spies_on_board.values()) == 2 and len(p.intrigue_cards) == ig0
    EffectResolver.resolve_single_effect({"recall_spy_then": {"intrigue": 1}}, p, gs)
    gs.step(GameAction(ActionType.RESOLVE_OPTIONAL, 0, accept_optional=True))
    assert sum(p.spies_on_board.values()) == 1 and len(p.intrigue_cards) == ig0 + 1


def test_card_corrections_batch6():
    from src.game.effects import EffectResolver
    from src.data.card_definitions import create_imperium_cards
    from src.game.combat.conflict import Conflict, BattleIcon
    from src.game.gameState import ActionType

    cs = {c.name: c for c in create_imperium_cards()}
    assert cs["Subversive Advisor"].acquire_effects == [{"spy": 1}]
    sa = cs["Subversive Advisor"].agent_effects[0]
    assert sa["trash_self"] == 1
    assert sa["if_faction_agent"] == {"influence_faction_visited": 2}

    gs = setup_game(2, seed=2)
    p = gs.players[0]

    def _won(icon):
        c = Conflict(name="w", conflict_level=1, first_place_reward={},
                     second_place_reward={}, battle_icon=icon)
        gs.won_conflicts[0].append(c)

    # Only a Desert Mouse icon -> just 1 spice.
    _won(BattleIcon.DESERT_MOUSE)
    sp0 = p.spice
    EffectResolver.resolve_single_effect({"beast_spoils": 1}, p, gs)
    assert p.spice == sp0 + 1
    assert not any(t.player_id == 0 for t in gs.pending_trashes)

    # Add Crysknife + Ornithopter -> spice, a trash choice, and a troop.
    _won(BattleIcon.CRYSKNIFE)
    _won(BattleIcon.ORNITHOPTER)
    sp0, tr0 = p.spice, p.troops_garrison
    EffectResolver.resolve_single_effect({"beast_spoils": 1}, p, gs)
    assert p.spice == sp0 + 1
    assert p.troops_garrison == tr0 + 1
    assert any(t.player_id == 0 for t in gs.pending_trashes)


def test_battle_icons():
    from src.game.gameState import GameState
    from collections import Counter

    # Each player starts with exactly one token; pool = 2 mouse + 2 crysknife.
    gs = GameState(num_players=4, seed=0)
    assert all(len(p.battle_icons) == 1 for p in gs.players)
    pool = Counter(i for p in gs.players for i in p.battle_icons)
    assert pool == {"desert_mouse": 2, "crysknife": 2}

    p = gs.players[0]
    p.battle_icons = ["crysknife"]
    vp0 = p.victory_points

    # Winning a matching non-wild icon -> immediate VP, pair consumed.
    assert gs._award_battle_icon(0, "crysknife") is True
    assert p.victory_points == vp0 + 1
    assert p.battle_icons == []

    # A non-matching icon is just held.
    gs._award_battle_icon(0, "desert_mouse")
    assert p.battle_icons == ["desert_mouse"] and p.victory_points == vp0 + 1

    # Wild never matches immediately.
    gs._award_battle_icon(0, "wild")
    assert sorted(p.battle_icons) == ["desert_mouse", "wild"]
    assert p.victory_points == vp0 + 1

    # ...but pairs with a real icon at Endgame.
    gs._resolve_endgame_battle_icons()
    assert p.victory_points == vp0 + 2
    assert p.battle_icons == []


def test_ignore_gates_and_emperor_access():
    from src.game.effects import EffectResolver
    from src.game.cards.card import Card, CardType, AccessSymbol

    gs = setup_game(4, seed=1)
    p = gs.players[0]

    # Shipping needs 2 Spacing Guild influence; a bare Desert card can't go.
    desert = Card("D", CardType.STARTER); desert.add_access_symbol(AccessSymbol.DESERT)
    p.influence["spacing_guild"] = 0
    p.spice = 3                                    # Shipping also costs 3 spice
    ok, _ = gs.can_send_agent(0, "Shipping", desert)
    assert not ok
    # Insider Information / Undercover Asset flag lifts the requirement.
    EffectResolver.resolve_single_effect({"ignore_influence_gates": 1}, p, gs)
    assert p.ignore_influence_gates_this_turn
    ok, why = gs.can_send_agent(0, "Shipping", desert)
    assert ok, why

    # Emperor's Invitation: a non-Emperor card reaches Sardaukar (Emperor space).
    gs2 = setup_game(4, seed=2)
    q = gs2.players[0]
    q.hand = [Card("DDP", CardType.STARTER)]
    q.hand[0].add_access_symbol(AccessSymbol.DESERT)
    q.spice = 4                                    # Sardaukar costs 4 spice
    ok, _ = gs2.can_send_agent(0, "Sardaukar", q.hand[0])
    assert not ok
    EffectResolver.resolve_single_effect({"emperor_access_or_draw": 1}, q, gs2)
    assert q.grant_emperor_access_this_turn        # no Emperor card in hand -> took access
    ok, why = gs2.can_send_agent(0, "Sardaukar", q.hand[0])
    assert ok, why


def test_deploy_budget_does_not_stack():
    """2 (once) + every troop recruited this turn; a 2nd deploy icon adds no +2."""
    from src.game.effects import EffectResolver
    from src.game.gameState import GameAction, ActionType

    gs = setup_game(4, seed=1)
    p = gs.players[0]
    p.troops_garrison = 10

    # Simulate a Combat space that recruited 1 troop (like Desert Tactics).
    p.deploy_budget_this_turn = 0
    p.deployed_this_turn = 0
    p.troops_recruited_this_turn = 1
    p.deploy_budget_this_turn = max(p.deploy_budget_this_turn,
                                    2 + p.troops_recruited_this_turn)  # -> 3
    gs.pending_deployments.append(
        __import__("src.game.gameState", fromlist=["PendingDeployment"])
        .PendingDeployment(0, p.troops_garrison))
    hi = max(a.deploy_count for a in gs.get_valid_actions(0)
             if a.action_type == ActionType.RESOLVE_DEPLOY)
    assert hi == 3
    gs.step(GameAction(ActionType.RESOLVE_DEPLOY, 0, deploy_count=3))
    assert p.deployed_this_turn == 3

    # Adaptive Tactics: +1 troop, +grant_deploy — the +2 does NOT come again.
    p.troops_recruited_this_turn = 1        # unchanged; AT troop is "extra"
    EffectResolver.resolve_single_effect(
        {"troops": 1, "grant_deploy": 1}, p, gs)
    # budget = 2 + recruited(1) + extra(1) = 4; already deployed 3 -> only 1 left
    hi = max(a.deploy_count for a in gs.get_valid_actions(0)
             if a.action_type == ActionType.RESOLVE_DEPLOY)
    assert hi == 1
    gs.step(GameAction(ActionType.RESOLVE_DEPLOY, 0, deploy_count=1))
    assert p.deployed_this_turn == 4        # 2 icon + 1 desert-tactics + 1 AT


def test_reserve_stacks():
    from src.data.card_definitions import (create_imperium_cards,
                                           create_reserve_prepare_the_way,
                                           create_reserve_spice_must_flow)
    from src.game.gameState import GameAction, ActionType

    assert not any(c.name == "Prepare the Way" for c in create_imperium_cards())
    assert len(create_reserve_prepare_the_way()) == 8
    assert len(create_reserve_spice_must_flow()) == 10

    gs = setup_game(4, seed=2)
    assert len(gs.reserve_prepare_the_way) == 8
    assert len(gs.reserve_spice_must_flow) == 10

    # Trashing a reserve card returns it to its stack, not the trash pile.
    p = gs.players[0]
    tsmf = create_reserve_spice_must_flow(1)[0]
    p.discard.append(tsmf)
    n = len(gs.reserve_spice_must_flow)
    gs.add_pending_trash(0, 1)
    gs.step(GameAction(ActionType.RESOLVE_TRASH, 0,
                       trash_card_name="The Spice Must Flow"))
    assert len(gs.reserve_spice_must_flow) == n + 1
    assert tsmf not in p.trash

    ptw = create_reserve_prepare_the_way(1)[0]
    p.hand.append(ptw)
    n = len(gs.reserve_prepare_the_way)
    gs.add_pending_trash(0, 1)
    gs.step(GameAction(ActionType.RESOLVE_TRASH, 0, trash_card_name="Prepare the Way"))
    assert len(gs.reserve_prepare_the_way) == n + 1
    assert ptw not in p.trash


def test_intrigue_leverage_manipulate():
    from src.game.effects import EffectResolver
    from src.data.card_definitions import create_intrigue_deck
    from src.game.gameState import GameAction, ActionType

    d = {c.name: c for c in create_intrigue_deck()}

    # Leverage: unplayable until you have gained spice this turn.
    gs = setup_game(2, seed=7)
    p = gs.players[0]
    p.gained_spice_this_turn = False
    ok, _ = d["Leverage"].can_play(p, gs, is_agent_turn=True)
    assert not ok
    p.gain_spice(1)                              # sets the flag
    assert p.gained_spice_this_turn
    ok, _ = d["Leverage"].can_play(p, gs, is_agent_turn=True)
    assert ok
    sol0 = p.solari
    EffectResolver.resolve_single_effect({"solari": 1, "contract": 1}, p, gs)
    assert p.solari == sol0 + 1

    # Manipulate: reserve the best Row card at cost -1, only this player buys it.
    gs2 = setup_game(2, seed=8)
    q = gs2.players[0]
    row_before = list(gs2.imperium_row)
    EffectResolver.resolve_single_effect({"manipulate": 1}, q, gs2)
    assert q.reserved_card is not None and q.reserved_card not in gs2.imperium_row
    assert q.reserved_card in row_before
    assert len(gs2.imperium_row) == len(row_before)          # refilled
    assert q.reserved_discount == 1

    # It shows up as a buy option at the reduced price during the buy phase.
    gs2.player_in_reveal_buy = 0
    gs2.persuasion_pool[0] = max(0, q.reserved_card.cost - 1)
    buys = [a.acquire_card_name for a in gs2.get_valid_actions(0)
            if a.action_type == ActionType.ACQUIRE_CARD]
    assert q.reserved_card.name in buys
    name, price = q.reserved_card.name, max(0, q.reserved_card.cost - 1)
    gs2.step(GameAction(ActionType.ACQUIRE_CARD, 0, acquire_card_name=name))
    assert q.reserved_card is None
    assert gs2.persuasion_pool[0] == 0
    assert any(c.name == name for c in q.discard)


def test_intrigue_corrections_2026_09_04():
    from src.game.effects import EffectResolver
    from src.data.card_definitions import create_intrigue_deck
    from collections import Counter

    d = create_intrigue_deck()
    assert Counter(c.name for c in d)["Detonation"] == 2

    gs = setup_game(2, seed=3)
    p = gs.players[0]

    # Sietch Ritual: restricted influence choice (bene or fremen only).
    p.influence["emperor"] = 5
    EffectResolver.resolve_single_effect({"influence_bene_or_fremen": 1}, p, gs)
    assert p.influence["emperor"] == 5
    assert p.influence["bene_gesserit"] + p.influence["fremen"] == 1

    # Imperium Politics: emperor or spacing only.
    e0, s0 = p.influence["emperor"], p.influence["spacing_guild"]
    EffectResolver.resolve_single_effect({"influence_emperor_or_spacing": 1}, p, gs)
    assert (p.influence["emperor"] + p.influence["spacing_guild"]) == e0 + s0 + 1

    # Distraction: special spy ONLY with 3+ units in the Conflict.
    gs.troops_in_conflict[0] = 2
    n0 = len(gs.pending_spy_placements)
    EffectResolver.resolve_single_effect({"if_units_in_conflict_3": {"spy_special": 1}}, p, gs)
    assert len(gs.pending_spy_placements) == n0
    gs.troops_in_conflict[0] = 3
    EffectResolver.resolve_single_effect({"if_units_in_conflict_3": {"spy_special": 1}}, p, gs)
    assert len(gs.pending_spy_placements) == n0 + 1

    # Call to Arms: troops per card acquired this turn.
    p.cards_acquired_this_turn = 3
    g0 = p.troops_garrison
    EffectResolver.resolve_single_effect({"troops_per_card_acquired_this_turn": 1}, p, gs)
    assert p.troops_garrison == g0 + 3

    # Inspire Awe: acquired card -> hand only if a sandworm is in the Conflict.
    if gs.imperium_row:
        gs.sandworms_in_conflict[0] = 1
        h0 = len(p.hand)
        EffectResolver.resolve_single_effect(
            {"acquire_free": {"max_cost": 99, "to_hand_if_sandworm": True}}, p, gs)
        assert len(p.hand) == h0 + 1


def test_battle_for_arrakeen_recall_spies():
    from src.data.card_definitions import conflict_level_3_pool
    from src.game.gameState import GameState
    bfa = next(c for c in conflict_level_3_pool() if c.name == "Battle for Arrakeen")
    assert bfa.first_place_reward == {
        "vp": 1, "control": True, "may_recall_spies_for_vp": {"count": 2, "vp": 1}}
    assert bfa.second_place_reward == {"intrigue": 1, "spice": 1, "solari": 3}

    gs = GameState(num_players=4, seed=1)
    p = gs.players[0]
    p.place_spy("Emperor Post"); p.place_spy("Fremen Post"); p.place_spy("Green Post")
    vp0 = p.victory_points
    gs._apply_combat_reward(0, {"may_recall_spies_for_vp": {"count": 2, "vp": 1}}, bfa)
    assert p.victory_points == vp0 + 1
    assert sum(p.spies_on_board.values()) == 1        # 2 recalled

    # Worm doubles the VP but not the spy count.
    assert gs._scale_reward({"may_recall_spies_for_vp": {"count": 2, "vp": 1}}, True) == {
        "may_recall_spies_for_vp": {"count": 2, "vp": 2}}


def test_conflict_rewards_2026_09_03():
    from src.data.card_definitions import conflict_level_1_pool, conflict_level_2_pool
    l1 = {c.name: c for c in conflict_level_1_pool()}
    l2 = {c.name: c for c in conflict_level_2_pool()}

    assert l1["Skirmish (Crysknife)"].second_place_reward == {"spice": 1, "intrigue": 1}
    assert l1["Skirmish (Crysknife)"].third_place_reward == {"spice": 1}
    assert l1["Skirmish (Desert Mouse)"].first_place_reward == {"solari": 2}
    assert l2["CHOAM Security"].first_place_reward == {
        "contract": 1, "troops": 1, "influence_spacing_guild": 1}
    assert l2["Protect the Sietches"].first_place_reward == {
        "influence_fremen": 1, "water": 1, "troops": 1}
    assert l2["Shadow Contest"].first_place_reward == {
        "influence_bene_gesserit": 1, "intrigue": 1}
    sf = l2["Spice Freighters"]
    assert sf.first_place_reward == {
        "influence_any": 1, "may_pay_spice_for_vp": {"cost": 3, "vp": 1}}
    assert sf.second_place_reward == {"spice": 1, "water": 1, "troops": 1}
    assert sf.third_place_reward == {"spice": 1, "troops": 1}
    assert l2["Seize Spice Refinery"].first_place_reward == {
        "control": True, "spy": 1, "spice": 2, "troops": 1}
    assert l2["Storms in the South"].first_place_reward == {"spice": 2, "spy_special": 1}
    assert l2["Test of Loyalty"].first_place_reward == {
        "influence_emperor": 1, "spy": 1, "solari": 2}
    assert l2["Trade Dispute"].first_place_reward == {
        "contract": 1, "trash": 1, "water": 1}

    # A sandworm doubles the VP-conversion cost AND payout.
    from src.game.gameState import GameState
    gs = GameState(num_players=4, seed=1)
    scaled = gs._scale_reward(dict(sf.first_place_reward), True)
    assert scaled == {"influence_any": 2,
                      "may_pay_spice_for_vp": {"cost": 6, "vp": 2}}
    assert gs._scale_reward({"control": True, "vp": 1}, True) == {"control": True, "vp": 2}


def test_card_corrections_batch7():
    from src.game.effects import EffectResolver
    from src.data.card_definitions import create_imperium_cards
    from src.game.cards.card import Card, CardType, CardTag

    cs = {c.name: c for c in create_imperium_cards()}
    assert cs["Tread in Darkness"].agent_effects == [
        {"if_tag_other_bene": {"trash": 1, "draw": 1}}]
    assert cs["Wheels Within Wheels"].reveal_effects == [{"spy": 1}]
    ww_agent = cs["Wheels Within Wheels"].agent_effects[0]
    assert ww_agent["if_influence_emperor_2"] == {"solari": 2}
    assert ww_agent["if_influence_spacing_guild_2"] == {"spice": 1}

    gs = setup_game(2, seed=1)
    p = gs.players[0]

    # Nested return_self_to_hand (Weirding Woman) now fires.
    ww = Card("Weirding Woman", CardType.IMPERIUM); ww.add_tag(CardTag.BENE_GESSERIT)
    bene = Card("BeneCard", CardType.IMPERIUM); bene.add_tag(CardTag.BENE_GESSERIT)
    p.in_play = [ww, bene]
    gs._self_ref_pending = []
    EffectResolver.resolve_single_effect(
        {"if_tag_other_bene": {"return_self_to_hand": 1}}, p, gs)
    EffectResolver._resolve_self_referential(
        {"if_tag_other_bene": {"return_self_to_hand": 1}}, ww, p, gs)
    assert ww in p.hand and ww not in p.in_play

    # ...but NOT when the condition is false (no other Bene card).
    ww2 = Card("Weirding Woman", CardType.IMPERIUM); ww2.add_tag(CardTag.BENE_GESSERIT)
    p.in_play = [ww2]
    gs._self_ref_pending = []
    EffectResolver.resolve_single_effect(
        {"if_tag_other_bene": {"return_self_to_hand": 1}}, p, gs)
    EffectResolver._resolve_self_referential(
        {"if_tag_other_bene": {"return_self_to_hand": 1}}, ww2, p, gs)
    assert ww2 in p.in_play

    # Treacherous Maneuver: trashes itself + another Emperor card, gains influence.
    gs2 = setup_game(2, seed=5)
    q = gs2.players[0]
    tm = Card("Treacherous Maneuver", CardType.IMPERIUM); tm.add_tag(CardTag.EMPEROR)
    emp = Card("EmpCard", CardType.IMPERIUM); emp.add_tag(CardTag.EMPEROR)
    q.in_play = [tm, emp]
    gs2._current_agent_space = "Sardaukar"       # an Emperor faction space
    inf0 = q.influence["emperor"]
    EffectResolver.resolve_single_effect({"trash_pair_emperor_influence": 1}, q, gs2)
    assert tm in q.trash and emp in q.trash and not q.in_play
    assert q.influence["emperor"] == inf0 + 1


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
