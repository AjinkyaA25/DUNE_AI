"""Every Bloodlines tech tile does what its text says (user audit, 2026-10-15).

    python -m pytest tests/test_bloodlines_techs.py -q
"""
import random

import pytest

from src.data.card_definitions import (conflict_level_3_pool, create_imperium_cards,
                                       create_starter_cards, setup_game)
from src.game.bloodlines.techs import BLOODLINES_TECHS, TECH_BY_NAME, TechTile
from src.game.gameState import ActionType as T, Phase


def game(seed=2):
    gs = setup_game(4, seed=seed, neutral_leaders=True, use_bloodlines=True)
    return gs, gs.get_current_player_id()


def buy(gs, pid, name, spice=20):
    """Put `name` on top of stack 0 and buy it."""
    d = TECH_BY_NAME[name]
    for st in gs.bl.tech_stacks:
        if d in st:
            st.remove(d)
    gs.bl.tech_stacks[0].insert(0, d)
    gs.players[pid].spice = spice
    gs.bl.buy_tech(pid, 0)


def give(gs, pid, name):
    gs.players[pid].techs.append(TechTile(TECH_BY_NAME[name]))


def acts(gs, pid, t=None):
    v = gs.get_valid_actions(pid)
    return [a for a in v if t is None or a.action_type == t]


def bl_opts(gs, pid):
    return gs.bl.pending_options(pid)


def test_all_18_tiles_are_in_the_market():
    gs, _ = game()
    assert {d.name for st in gs.bl.tech_stacks for d in st} == BLOODLINES_TECHS
    assert len(BLOODLINES_TECHS) == 18


# ---------------------------------------------------------------- Advanced Data Analysis
def test_ada_needs_a_spy_on_the_board():
    gs, pid = game()
    p = gs.players[pid]
    d = TECH_BY_NAME["Advanced Data Analysis"]
    for st in gs.bl.tech_stacks:
        if d in st:
            st.remove(d)
    gs.bl.tech_stacks[0].insert(0, d)
    p.spice = 10
    assert 0 not in gs.bl.buyable_stacks(pid)          # no Spy on the board
    p.spies_on_board["Arrakeen Post"] = 1
    p.spies_available -= 1
    assert 0 in gs.bl.buyable_stacks(pid)


def test_ada_trashes_a_spy_for_good_and_you_pick_which():
    gs, pid = game()
    p = gs.players[pid]
    for post in ("Arrakeen Post", "Fremen Post"):
        p.spies_on_board[post] = 1
        p.spies_available -= 1
    buy(gs, pid, "Advanced Data Analysis")
    assert sorted(bl_opts(gs, pid)) == ["post:Arrakeen Post", "post:Fremen Post"]
    gs.bl.resolve_choice(pid, "post:Fremen Post")
    assert p.spies_on_board == {"Arrakeen Post": 1}
    assert p.spies_available + sum(p.spies_on_board.values()) == 2     # 2 Spies from now on


def test_ada_flip_once_per_round_agent_or_reveal_for_an_intrigue():
    gs, pid = game()
    p = gs.players[pid]
    give(gs, pid, "Advanced Data Analysis")
    assert "Advanced Data Analysis" in gs.bl.activatable(pid, agent_turn=True)
    assert "Advanced Data Analysis" in gs.bl.activatable(pid, agent_turn=False)   # Reveal turn too
    n = len(p.intrigue_cards)
    gs.bl.activate(pid, "Advanced Data Analysis")
    assert len(p.intrigue_cards) == n + 1
    assert "Advanced Data Analysis" not in gs.bl.activatable(pid, agent_turn=False)  # once per round
    gs.bl.on_round_start()
    assert "Advanced Data Analysis" in gs.bl.activatable(pid, agent_turn=True)


# ---------------------------------------------------------------- CHOAM Transports
def test_choam_transports_contract_draw_and_endgame_vp():
    gs, pid = game()
    p = gs.players[pid]
    buy(gs, pid, "CHOAM Transports")
    assert any(a.action_type == T.RESOLVE_CONTRACT for a in acts(gs, pid))   # take a face-up contract
    h = len(p.hand)
    gs.bl.on_complete_contract(pid)
    assert len(p.hand) == h + 1
    p.contracts_completed = [object()] * 4
    vp = p.victory_points
    gs.bl.endgame()
    assert p.victory_points == vp + 1


# ---------------------------------------------------------------- Delivery Bay / Training Depot (Command)
@pytest.mark.parametrize("name,check", [("Delivery Bay", "solari"), ("Training Depot", "swords")])
def test_command_techs(name, check):
    gs, pid = game()
    p = gs.players[pid]
    give(gs, pid, name)
    before = (p.solari, gs.swords_this_reveal.get(pid, 0))
    p.bl_reveal_persuasion, p.bl_command_fired = 6, False      # Reveal with 6+ persuasion
    gs.bl._check_command(pid)
    after = (p.solari, gs.swords_this_reveal.get(pid, 0))
    assert (after[0] - before[0], after[1] - before[1]) == ((2, 0) if check == "solari" else (0, 2))


def test_delivery_bay_acquire_draws():
    gs, pid = game()
    p = gs.players[pid]
    h = len(p.hand)
    buy(gs, pid, "Delivery Bay")
    assert len(p.hand) == h + 1


# ---------------------------------------------------------------- Forbidden Weapons
def test_forbidden_weapons_reveal_choice():
    gs, pid = game()
    p = gs.players[pid]
    give(gs, pid, "Forbidden Weapons")
    p.influence["fremen"] = 2
    p.spice = 4
    gs.bl.on_reveal(pid, [])
    assert sorted(bl_opts(gs, pid)) == ["spice", "swords"]
    gs.bl.resolve_choice(pid, "swords")
    assert gs.swords_this_reveal.get(pid, 0) == 3
    assert gs.pending_choice_for(pid) is not None                  # which influence to lose
    gs2, pid2 = game()
    p2 = gs2.players[pid2]
    give(gs2, pid2, "Forbidden Weapons")
    p2.spice = 4
    gs2.bl.on_reveal(pid2, [])
    # no influence to lose: the 3 swords are still a choice (no arrow, user)
    assert sorted(bl_opts(gs2, pid2)) == ["spice", "swords"]
    g3 = gs2.clone()
    g3.bl.resolve_choice(pid2, "swords")
    assert g3.swords_this_reveal.get(pid2, 0) == 3 and g3.pending_choice_for(pid2) is None
    gs2.bl.resolve_choice(pid2, "spice")
    assert p2.spice == 0 and not gs2.bl.has(pid2, "Forbidden Weapons")


def test_forbidden_weapons_acquire_troop_and_wall():
    gs, pid = game()
    p = gs.players[pid]
    gs.current_conflict = next(c for c in conflict_level_3_pool() if c.name == "Battle for Arrakeen")
    t = p.troops_garrison
    buy(gs, pid, "Forbidden Weapons")
    assert p.troops_garrison == t + 1 and not gs.shield_wall_intact


# ---------------------------------------------------------------- Gene-Locked Vault
def test_gene_locked_vault_choice_and_steal_protection():
    gs, pid = game()
    p = gs.players[pid]
    buy(gs, pid, "Gene-Locked Vault")
    assert sorted(bl_opts(gs, pid)) == ["draw", "intrigue"]
    h = len(p.hand)
    gs.bl.resolve_choice(pid, "draw")
    assert len(p.hand) == h + 1
    assert gs.bl.steal_protected(pid)                              # < 5 Intrigues


# ---------------------------------------------------------------- Glowglobes
def test_glowglobes_influence_and_top_card_shown():
    import sys
    sys.argv = ["x"]
    import ui.server as srv
    gs, pid = game()
    buy(gs, pid, "Glowglobes")
    assert gs.pending_choice_for(pid) is not None or gs.pending_influence_choices    # any-faction influence
    s = srv.Session(seat=pid, specs=["random"] * 3, seed=2, bloodlines=True)
    give(s.gs, pid, "Glowglobes")
    me = next(x for x in s.view(reveal=False)["players"] if x["id"] == pid)
    assert me["topCard"]["name"] == s.gs.players[pid].deck[-1].name


# ---------------------------------------------------------------- Navigation Chamber
def test_navigation_chamber_reduces_every_board_cost():
    gs, pid = game()
    p = gs.players[pid]
    give(gs, pid, "Navigation Chamber")
    assert gs._space_cost(pid, "High Council") == {"solari": 4}
    assert gs._space_cost(pid, "Heighliner") == {"spice": 4}
    assert gs._space_cost(pid, "Sardaukar") == {"spice": 3}
    p.spice = 0
    assert "pay_spice" in gs.get_space_options(pid, "Spice Refinery")      # 0 spice -> 4 solari
    p.solari = 1
    assert "paid" in gs.get_space_options(pid, "Gather Support")           # 1 solari instead of 2
    assert gs._swordmaster_cost(pid) == 7                                   # 8 -> 7


def test_navigation_chamber_spice_refinery_pays_nothing():
    from src.game.cards.card import AccessSymbol
    gs, pid = game()
    p = gs.players[pid]
    give(gs, pid, "Navigation Chamber")
    p.spice, p.solari = 0, 0
    city = next(c for c in create_imperium_cards()
                if any(getattr(s, "value", str(s)) == "city" for s in c.access_symbols))
    p.hand.append(city)
    a = next(x for x in acts(gs, pid, T.AGENT_TURN)
             if x.card_name == city.name and x.space_name == "Spice Refinery" and x.space_option == "pay_spice")
    gs.step(a)
    assert p.spice == 0 and p.solari == 4                    # 0 spice -> 4 solari


# ---------------------------------------------------------------- Ornithopter Fleet
def test_ornithopter_fleet_icons_pair():
    gs, pid = game()
    p = gs.players[pid]
    p.battle_icons = ["wild", "crysknife"]
    vp = p.victory_points
    buy(gs, pid, "Ornithopter Fleet")
    assert p.victory_points == vp + 1 and p.battle_icons == []


# ---------------------------------------------------------------- Panopticon
def test_panopticon_reveal_spy_troop_and_endgame_influence():
    gs, pid = game()
    p = gs.players[pid]
    give(gs, pid, "Panopticon")
    t = p.troops_garrison
    gs.bl.on_reveal(pid, [])
    assert p.troops_garrison == t + 1
    assert any(x.player_id == pid for x in gs.pending_spy_placements)
    p.influence.update({"emperor": 0, "spacing_guild": 1, "bene_gesserit": 2, "fremen": 3})
    gs.bl.endgame()
    assert p.influence == {"emperor": 1, "spacing_guild": 2, "bene_gesserit": 2, "fremen": 3}


# ---------------------------------------------------------------- Planetary Array
def test_planetary_array_trash_and_draw_on_win():
    gs, pid = game()
    p = gs.players[pid]
    buy(gs, pid, "Planetary Array")
    assert any(x.player_id == pid for x in gs.pending_trashes)
    h = len(p.hand)
    gs.bl.on_win_conflict(pid) if hasattr(gs.bl, "on_win_conflict") else gs.bl._resolve(
        pid, TECH_BY_NAME["Planetary Array"].on_win_conflict)
    assert len(p.hand) == h + 1


# ---------------------------------------------------------------- Plasteel Blades
def test_plasteel_blades_solari_and_extra_skill():
    gs, pid = game()
    p = gs.players[pid]
    s = p.solari
    buy(gs, pid, "Plasteel Blades")
    assert p.solari == s + 4
    p.solari = 2
    gs.bl.recruit(pid, "Deliver Supplies", "board:" + gs.bl.row_skills(pid)[0])
    assert any(c.kind == "plasteel" for c in gs.bl.pending if c.player_id == pid)


# ---------------------------------------------------------------- Rapid Dropships
def test_rapid_dropships_troops_and_flip_agent_only():
    gs, pid = game()
    p = gs.players[pid]
    t = p.troops_garrison
    buy(gs, pid, "Rapid Dropships")
    assert p.troops_garrison == t + 2
    assert "Rapid Dropships" in gs.bl.activatable(pid, agent_turn=True)
    assert "Rapid Dropships" not in gs.bl.activatable(pid, agent_turn=False)


# ---------------------------------------------------------------- Sardaukar High Command
def test_sardaukar_high_command():
    gs, pid = game()
    p = gs.players[pid]
    vp = p.victory_points
    buy(gs, pid, "Sardaukar High Command")
    assert p.victory_points == vp + 1 and gs.bl.commander_cost(pid) == 1


# ---------------------------------------------------------------- Self-Destroying Messages
def test_self_destroying_messages():
    gs, pid = game()
    p = gs.players[pid]
    n = len(p.intrigue_cards)
    buy(gs, pid, "Self-Destroying Messages")
    assert len(p.intrigue_cards) == n + 2
    pp = gs.persuasion_pool.get(pid, 0)
    gs.bl.on_reveal(pid, [])
    assert gs.persuasion_pool.get(pid, 0) == pp + 1


# ---------------------------------------------------------------- Servo-Receivers
def test_servo_receivers_signet_has_all_faction_icons():
    gs, pid = game()
    give(gs, pid, "Servo-Receivers")
    ring = next(c for c in create_starter_cards() if c.name == "Signet Ring")
    assert {"emperor", "spacing_guild", "bene_gesserit", "fremen"} <= gs.bl.extra_icons(pid, ring)


# ---------------------------------------------------------------- Spy Drones
def test_spy_drones_flip_rules():
    gs, pid = game()
    p = gs.players[pid]
    buy(gs, pid, "Spy Drones")
    assert sum(1 for x in gs.pending_spy_placements if x.player_id == pid) >= 1   # 2 Spies to place
    # flip without a recall: just 1 solari
    s = p.solari
    p.recalled_spy_this_turn = False
    gs.bl.activate(pid, "Spy Drones")
    assert p.solari == s + 1 and not any(x.player_id == pid for x in gs.pending_trashes)
    assert "Spy Drones" not in gs.bl.activatable(pid, agent_turn=True)            # once per round
    # recalled BEFORE flipping: 1 solari and a trash
    gs.bl.on_round_start()
    p.recalled_spy_this_turn = True
    gs.bl.activate(pid, "Spy Drones")
    assert any(x.player_id == pid for x in gs.pending_trashes)


# ---------------------------------------------------------------- Suspensor Suits
def test_suspensor_suits_only_for_intrigues_after_it_is_owned():
    gs, pid = game()
    p = gs.players[pid]
    p.hand.append(next(c for c in create_starter_cards() if c.name == "Dagger"))
    buy_first = gs.clone()
    for g, order in ((gs, "intrigue first"), (buy_first, "buy first")):
        q = g.players[pid]
        d = TECH_BY_NAME["Suspensor Suits"]
        for st in g.bl.tech_stacks:
            if d in st:
                st.remove(d)
        g.bl.tech_stacks[0].insert(0, d)
        q.spice = 5
        g.step(next(a for a in g.get_valid_actions(pid)
                    if a.action_type == T.AGENT_TURN and a.space_name == "Assembly Hall"))
        claim = lambda: g.step(next(a for a in g.get_valid_actions(pid) if (a.choice or "").startswith("claim:")))
        tech = lambda: g.step(next(a for a in g.get_valid_actions(pid) if a.choice == "stack:0"))
        (tech(), claim()) if order == "buy first" else (claim(), tech())
        assert g.troops_in_conflict.get(pid, 0) == (1 if order == "buy first" else 0)
