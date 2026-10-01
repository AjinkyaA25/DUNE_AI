"""Bloodlines rules: tech market, Command, Sardaukar Commanders + skills."""
from src.data.card_definitions import setup_game
from src.game.bloodlines.rules import COMMANDER_SPACES
from src.game.bloodlines.techs import TECH_BY_NAME, TechTile
from src.game.gameState import ActionType


def _game(seed=0):
    gs = setup_game(num_players=4, seed=seed, neutral_leaders=True,
                    use_bloodlines=True)
    return gs, gs.players[gs.get_current_player_id()]


def _card(p, name):
    return next(c for c in p.hand + p.deck + p.discard if c.name == name)


def _to_hand(p, name):
    card = _card(p, name)
    for pile in (p.deck, p.discard):
        if card in pile:
            pile.remove(card)
            p.hand.append(card)
    return card


def _give_tech(gs, p, name):
    p.techs.append(TechTile(TECH_BY_NAME[name]))


def _choose(gs, p, prefix):
    acts = [a for a in gs.get_valid_actions(p.id)
            if a.action_type == ActionType.RESOLVE_BL_CHOICE
            and a.choice.startswith(prefix)]
    assert acts, [a.choice for a in gs.get_valid_actions(p.id)]
    gs.step(acts[0])
    return acts[0].choice


def test_market_is_three_stacks_without_ix_only_tiles():
    gs, _ = _game()
    stacks = gs.bl.tech_stacks
    assert len(stacks) == 3
    names = {d.name for s in stacks for d in s}
    assert len(names) == 34
    assert "Detonation Devices" not in names and "Troop Transports" not in names


def test_green_space_offers_tech_and_council_seat_discounts_it():
    gs, p = _game()
    dagger = _to_hand(p, "Dagger")                      # landsraad icon
    top = gs.bl.tech_stacks[0][0]
    p.spice = top.cost - 1
    p.has_councilor = True                              # -1 spice
    gs.apply_agent_turn(p.id, dagger, "Assembly Hall")
    assert "stack:0" in gs.bl.pending_options(p.id)
    gs.bl.resolve_choice(p.id, "stack:0")
    assert any(t.name == top.name for t in p.techs)
    assert p.spice == 0
    assert gs.bl.tech_stacks[0] == [] or gs.bl.tech_stacks[0][0] is not top


def test_command_fires_once_at_six_persuasion():
    gs, p = _game()
    _give_tech(gs, p, "Delivery Bay")                   # Command: 2 solari
    p.solari = 0
    gs.persuasion_pool[p.id] = 4
    gs.bl.open_command_window(p.id)
    assert p.solari == 0
    gs.gain_persuasion(p.id, 2)                         # reaches 6
    assert p.solari == 2
    gs.gain_persuasion(p.id, 3)
    assert p.solari == 2                                # only once per reveal


def test_recruit_commander_gives_skill_and_empties_space():
    gs, p = _game()
    p.solari = 5
    gs.bl.skill_row = ["Canny", "Hardy", "Loyal", "Canny"]
    gs.bl.on_agent_placed(p.id, "Sardaukar")
    opts = gs.bl.pending_options(p.id)
    assert "board:Canny" in opts and "decline" in opts
    gs.bl.resolve_choice(p.id, "board:Canny")
    assert p.skills == {"Canny"}
    assert p.commanders_garrison == 1 and p.solari == 3
    assert gs.bl.commander_on_space["Sardaukar"] is False
    # the space stays empty for everyone; a second skill can't be taken there
    gs.bl.end_agent_turn(p.id)
    gs.bl.on_agent_placed(p.id, "Sardaukar")
    assert not any(o.startswith("board:") for o in gs.bl.pending_options(p.id))


def test_commander_deploys_first_and_returns_to_commander_supply():
    gs, p = _game()
    p.solari = 5
    gs.bl.skill_row = ["Hardy", "Driven", "Fierce", "Loyal"]
    gs.bl.recruit(p.id, "High Council", "board:Hardy")
    p.troops_garrison += 2                              # plus two plain troops
    gs._do_deploy(p.id, 1)
    assert gs.bl.commanders_in_conflict[p.id] == 1      # commander went first
    assert p.commanders_garrison == 0
    supply0 = p.troops_supply
    gs.resolve_combat()
    assert p.commanders_supply == 1                     # not the troop supply
    assert p.troops_supply == supply0
    # re-recruit from supply on a commander space: 2 solari, no new skill
    p.solari = 2
    gs.bl.end_agent_turn(p.id)
    gs.bl.on_agent_placed(p.id, "Gather Support")
    gs.bl.resolve_choice(p.id, "supply")
    assert p.commanders_garrison == 1 and p.commanders_supply == 0
    assert p.skills == {"Hardy"}


def test_sword_skills_need_a_commander_in_the_conflict():
    gs, p = _game()
    p.skills = {"Loyal", "Fierce"}
    p.influence["emperor"] = 3
    assert gs.bl.strength_bonus(p.id) == 0
    gs.bl.commanders_in_conflict[p.id] = 1
    gs.troops_in_conflict[p.id] = 1
    assert gs.bl.strength_bonus(p.id) == 3              # Loyal 2 + Fierce 1
    other = gs.players[(p.id + 1) % 4]
    gs.sandworms_in_conflict[other.id] = 1
    assert gs.bl.strength_bonus(p.id) == 4              # Fierce +1 vs worms
    # retreating the last commander loses them
    gs.troops_in_conflict[p.id] = 0
    gs.bl.sync()
    assert gs.bl.strength_bonus(p.id) == 0


def test_navigation_chamber_and_ornithopter_fleet():
    gs, p = _game()
    _give_tech(gs, p, "Navigation Chamber")
    assert gs._space_cost(p.id, "High Council") == {"solari": 4}
    _give_tech(gs, p, "Ornithopter Fleet")
    p.battle_icons = []
    gs._award_battle_icon(p.id, "crysknife")
    assert p.battle_icons == ["wild"]


def test_chaumurky_wins_ties():
    gs, p = _game()
    for q in gs.players:
        q.victory_points, q.spice, q.solari, q.water = 10, 0, 0, 0
        q.troops_garrison = 0
    gs.players[0].spice = 5                             # would win the tie
    _give_tech(gs, gs.players[3], "Chaumurky")
    assert gs._determine_winner() == 3


def test_commander_spaces_exist_on_board():
    from src.game.board.board import UPRISING_BOARD
    assert all(s in UPRISING_BOARD for s in COMMANDER_SPACES)


# ---------------------------------------------------------------------------
# Bloodlines Imperium / Intrigue cards
# ---------------------------------------------------------------------------
def _bl_card(name):
    from src.game.bloodlines.cards import create_bloodlines_imperium_cards
    return next(c for c in create_bloodlines_imperium_cards() if c.name == name)


def test_bloodlines_cards_join_the_decks_only_with_bloodlines():
    gs, _ = _game()
    names = {c.name for c in gs.imperium_deck + gs.imperium_row}
    assert "Imperial Throneship" in names
    assert "Honor Guard" in {c.name for c in gs.intrigue_deck}
    plain = setup_game(num_players=4, seed=0, neutral_leaders=True)
    assert "Imperial Throneship" not in {c.name for c in plain.imperium_deck + plain.imperium_row}


def test_card_command_fires_with_six_persuasion_and_trashes_itself():
    gs, p = _game()
    bombast = _bl_card("Bombast")
    p.in_play.append(bombast)
    p.solari = 0
    gs.persuasion_pool[p.id] = 0
    from src.game.effects import EffectResolver
    EffectResolver.resolve_reveal_effects(bombast, p, gs)   # registers Command
    gs.bl.open_command_window(p.id)                         # 1 persuasion: no
    assert p.solari == 0
    gs.gain_persuasion(p.id, 5)
    assert p.solari == 3
    assert bombast not in p.in_play and bombast in p.trash


def test_corrupt_bureaucrat_pays_when_discarded():
    gs, p = _game()
    cb = _bl_card("Corrupt Bureaucrat")
    p.hand = [cb]
    p.solari = 0
    from src.game.effects import EffectResolver
    EffectResolver._discard_worst(p)
    gs.bl.sync()
    assert p.solari == 3


def test_delivery_logistics_uses_contract_icons():
    from src.game.contract.contract import Contract, ContractType
    gs, p = _game()
    dl = _bl_card("Delivery Logistics")
    p.hand.append(dl)
    assert not gs.get_legal_agent_spaces(p.id, dl)
    p.contracts_active.append(Contract("x", ContractType.BOARD_SPACE, {},
                                       {"board_space": "Arrakeen"}))
    assert "Arrakeen" in gs.get_legal_agent_spaces(p.id, dl)       # city icon


def test_litany_against_fear_draws_and_passes_the_turn():
    gs, p = _game()
    lit = _bl_card("Litany Against Fear")
    p.hand.append(lit)
    acts = [a for a in gs.get_valid_actions(p.id)
            if a.action_type == ActionType.ACTIVATE_TECH]
    assert [a.choice for a in acts] == ["card:Litany Against Fear"]
    hand0, agents0 = len(p.hand), p.agents_available
    gs.step(acts[0])
    assert lit in p.in_play and len(p.hand) == hand0      # -1 litany, +1 draw
    assert p.agents_available == agents0                  # no agent placed
    assert gs.get_current_player_id() != p.id             # turn passed


def test_possible_futures_gives_both_with_another_bene_card():
    gs, p = _game()
    pf = _bl_card("Possible Futures")
    p.in_play += [pf, _bl_card("Urgent Shigawire")]       # second BG card
    troops0 = p.troops_garrison
    from src.game.effects import EffectResolver
    EffectResolver.resolve_agent_effects(pf, p, gs)
    assert p.troops_garrison == troops0 + 2
    assert any(c.player_id == p.id for c in gs.pending_influence_choices)


def test_urgent_shigawire_gives_next_bene_card_all_icons():
    gs, p = _game()
    p.bl_shigawire = True
    lit = _bl_card("Litany Against Fear")                  # BG, no icons
    assert "Arrakeen" in gs.get_legal_agent_spaces(p.id, lit)


def test_skill_row_four_of_fourteen_tiles_refilled_when_taken():
    gs, p = _game()
    bl = gs.bl
    assert len(bl.skill_row) == 4 and len(bl.skill_deck) == 10
    from collections import Counter
    assert Counter(bl.skill_row + bl.skill_deck) == {k: 2 for k in
                                                     ("Canny", "Charismatic", "Desperate", "Driven",
                                                      "Fierce", "Hardy", "Loyal")}
    bl.skill_row = ["Canny", "Canny", "Hardy", "Loyal"]
    bl.skill_deck = ["Driven"] + bl.skill_deck[1:]
    p.solari = 5
    bl.on_agent_placed(p.id, "Sardaukar")
    opts = bl.pending_options(p.id)
    # only skills in the row, each once, plus decline
    assert sorted(o for o in opts if o.startswith("board:")) == \
        ["board:Canny", "board:Hardy", "board:Loyal"]
    bl.resolve_choice(p.id, "board:Canny")
    assert p.skills == {"Canny"}
    assert sorted(bl.skill_row) == ["Canny", "Driven", "Hardy", "Loyal"]   # refilled
    assert len(bl.skill_deck) == 9


def test_plasteel_blades_takes_a_row_skill():
    gs, p = _game()
    bl = gs.bl
    _give_tech(gs, p, "Plasteel Blades")
    bl.skill_row = ["Canny", "Hardy", "Loyal", "Fierce"]
    p.solari = 5
    bl.on_agent_placed(p.id, "Sardaukar")
    bl.resolve_choice(p.id, "board:Canny")
    opts = bl.pending_options(p.id)                       # Plasteel's bonus choice
    assert "skill:Hardy" in opts and "skill:Canny" not in opts and "decline" in opts
    bl.resolve_choice(p.id, "skill:Loyal")
    assert p.skills == {"Canny", "Loyal"}
    assert not bl.has(p.id, "Plasteel Blades")
    assert "Loyal" not in bl.skill_row[:3] or bl.skill_row.count("Loyal") <= 1
    assert len(bl.skill_row) == 4 and len(bl.skill_deck) == 8
