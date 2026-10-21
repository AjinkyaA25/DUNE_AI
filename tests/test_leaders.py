"""Leaders: framework wiring and per-leader smoke."""
import random

from src.data.card_definitions import setup_game
from src.game.gameState import ActionType
from src.game.leader.leader_definitions import create_leaders, assign_leaders


def test_all_leaders_assignable_and_distinct():
    names = [ld.name for ld in create_leaders()]
    assert len(names) == len(set(names)) == 18        # 9 Uprising (Jessica is double-sided) + 9 Bloodlines
    gs = setup_game(4, seed=1, leaders=names[:4])
    assert [p.leader.name for p in gs.players] == names[:4]


def test_random_assignment_is_distinct():
    gs = setup_game(4, seed=2)
    got = [p.leader.name for p in gs.players]
    assert len(set(got)) == 4


def test_signet_fires_on_signet_ring():
    # Force a known leader and check the signet effect resolves.
    gs = setup_game(4, seed=3, leaders=["Shaddam Corrino IV", "Gurney Halleck",
                                        "Lady Jessica", "Muad'Dib"])
    p = gs.players[0]
    # give the player a Signet Ring in hand and a legal landsraad space
    from src.game.cards.card import Card, CardType, AccessSymbol
    ring = next((c for c in p.hand if c.name == "Signet Ring"), None)
    if ring is None:
        ring = Card("Signet Ring", CardType.STARTER)
        for s in (AccessSymbol.LANDSRAAD, AccessSymbol.CITY, AccessSymbol.DESERT):
            ring.add_access_symbol(s)
        p.hand.append(ring)
    p.solari = 5                                  # both options affordable
    solari_before, troops_before = p.solari, p.troops_garrison
    gs.apply_agent_turn(0, ring, "Assembly Hall")
    # Shaddam (verified card): no deploying this turn; 1 solari + 1 troop OR pay 3 -> contract
    assert p.no_deploy_this_turn
    pc = gs.pending_choice_for(0)
    assert pc is not None and {o[0] for o in pc.options} >= {"troop"}
    gs.step(next(a for a in gs.get_valid_actions(0)
                 if a.action_type == ActionType.RESOLVE_CHOICE and a.choice == "troop"))
    assert p.solari == solari_before + 1 and p.troops_garrison == troops_before + 1


def test_leader_games_complete():
    for seed in range(6):
        gs = setup_game(4, seed=seed)
        rng = random.Random(seed)
        for _ in range(5000):
            if gs.game_over:
                break
            pid = gs.player_in_reveal_buy or gs.get_current_player_id()
            acts = gs.get_valid_actions(pid)
            nn = [a for a in acts if a.action_type != ActionType.NO_OP]
            gs.step(rng.choice(nn or acts))
        assert gs.game_over
