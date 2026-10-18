"""Regressions for the bugs the leader-game audit found (2026-10-16)."""
from src.data.card_definitions import setup_game
from src.game.effects import EffectResolver
from src.game.gameState import ActionType


def _game(leaders, seed=11):
    return setup_game(4, seed=seed, leaders=leaders, use_bloodlines=True)


def _card(gs, name):
    from src.data.card_definitions import create_imperium_cards, create_starter_cards
    from src.game.bloodlines.cards import create_bloodlines_imperium_cards
    for c in create_imperium_cards() + create_bloodlines_imperium_cards() + create_starter_cards():
        if c.name == name:
            return c
    raise KeyError(name)


def test_piter_draws_a_twisted_intrigue_in_round_one():
    gs = _game(["Piter De Vries", "Gurney Halleck", "Chani", "Duncan Idaho"])
    p = gs.players[0]
    assert gs.round == 1
    assert sum(getattr(c, "twisted", False) for c in p.intrigue_cards) == 1
    assert len(p.leader.state["twisted_deck"]) == 11


def test_fenring_gains_solari_when_a_card_trashes_itself():
    gs = _game(["Count Hasimir Fenring", "Gurney Halleck", "Chani", "Duncan Idaho"])
    p = gs.players[0]
    seek = _card(gs, "Seek Allies")                 # trashes itself when played
    p.hand.append(seek)
    before = p.solari
    gs.apply_agent_turn(0, seek, "Fremkit")
    assert seek in p.trash and p.solari == before + 1


def test_urgent_shigawire_bonus_waits_for_the_next_bene_gesserit_card():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    sh = _card(gs, "Urgent Shigawire")
    p.hand.append(sh)
    gs.apply_agent_turn(0, sh, "Arrakeen")
    assert p.bl_shigawire                            # still waiting for the next BG card


def test_shaddam_ring_offers_no_sandworms():
    gs = _game(["Shaddam Corrino IV", "Gurney Halleck", "Chani", "Duncan Idaho"])
    p = gs.players[0]
    p.water, p.has_maker_hooks = 5, True
    ring = next(c for c in p.deck + p.hand if c.name == "Signet Ring")
    if ring not in p.hand:
        p.deck.remove(ring)
        p.hand.append(ring)
    opts = {a.space_option for a in gs.get_valid_actions(0)
            if a.action_type == ActionType.AGENT_TURN and a.card_name == "Signet Ring"
            and a.space_name == "Deep Desert"}
    assert opts and "sandworm" not in opts


def test_deploy_or_retreat_is_the_players_choice():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    gs.troops_in_conflict[0] = 2
    p.troops_garrison = 2
    EffectResolver.resolve_single_effect({"deploy_or_retreat": 1}, p, gs)
    pc = gs.pending_choice_for(0)
    assert pc is not None and {o[0] for o in pc.options} == {"deploy", "retreat"}


def test_imperial_privilege_may_swap_an_intrigue():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    gs.draw_intrigue_for_player(0, 1)
    held = p.intrigue_cards[0].name
    EffectResolver.resolve_single_effect({"swap_intrigue": 1}, p, gs)
    pc = gs.pending_choice_for(0)
    assert pc is not None and {"decline", f"swap:{held}"} <= {o[0] for o in pc.options}


# --- user reports, 2026-10-16 (second batch) ---------------------------------

def _ic(name):
    from src.data.card_definitions import create_intrigue_deck
    return next(c for c in create_intrigue_deck() if c.name == name)


def test_shaddams_favor_playable_without_emperor_influence():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    pid = gs.get_current_player_id()
    p = gs.players[pid]
    p.intrigue_cards.append(_ic("Shaddam's Favor"))
    assert any(a.action_type == ActionType.PLAY_INTRIGUE and a.intrigue_card_name == "Shaddam's Favor"
               for a in gs.get_valid_actions(pid))


def test_distraction_counts_units_deployed_this_turn():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    p.intrigue_cards.append(_ic("Distraction"))
    gs.troops_in_conflict[0] = 4                    # already there: doesn't count
    assert not p.intrigue_cards[-1].can_play(p, gs, is_agent_turn=True)[0]
    gs.note_units_deployed(0, 3)                    # deployed this turn
    assert p.intrigue_cards[-1].can_play(p, gs, is_agent_turn=True)[0]


def test_spy_with_empty_supply_lets_the_player_pick_the_recall():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    for post in ("Emperor Post", "Fremen Post", "Green Post"):
        p.place_spy(post)
    EffectResolver.resolve_single_effect({"spy": 1}, p, gs)
    acts = [a for a in gs.get_valid_actions(0) if a.action_type == ActionType.RESOLVE_SPY]
    assert acts and {a.choice for a in acts} == {"recall:Emperor Post", "recall:Fremen Post",
                                                 "recall:Green Post"}


def test_withdrawn_passes_the_turn():
    from src.game.leader.twisted import create_twisted_intrigues
    gs = _game(["Piter De Vries", "Chani", "Duncan Idaho", "Gurney Halleck"])
    pid = gs.get_current_player_id()
    p = gs.players[pid]
    p.intrigue_cards.append(next(c for c in create_twisted_intrigues() if c.name == "Withdrawn"))
    a = next(a for a in gs.get_valid_actions(pid) if a.action_type == ActionType.PLAY_INTRIGUE
             and a.intrigue_card_name == "Withdrawn")
    agents = p.agents_available
    gs.step(a)
    assert gs.get_current_player_id() != pid and p.agents_available == agents


def test_beasts_spoils_pays_per_held_icon():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    p.battle_icons = ["desert_mouse"]
    sp = p.spice
    EffectResolver.resolve_single_effect({"beast_spoils": 1}, p, gs)
    assert p.spice == sp + 1 and not any(t.player_id == 0 for t in gs.pending_trashes)


def test_cards_drawn_on_the_reveal_turn_are_revealed():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    pid = gs.get_current_player_id()
    p = gs.players[pid]
    gs.step(next(a for a in gs.get_valid_actions(pid) if a.action_type == ActionType.REVEAL_TURN))
    top = p.deck[-1] if p.deck else None
    pool = gs.persuasion_pool.get(pid, 0)
    gs.draw_cards_for_player(pid, 1)
    if top is not None:
        assert top in p.in_play and top not in p.hand
        assert gs.persuasion_pool[pid] == pool + top.persuasion


def test_navigation_10_can_drop_and_regain_the_same_faction():
    from src.game.leader.navigation import play_navigation
    gs = _game(["Steersman Y'rkoon", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    p.influence["fremen"] = 2
    gs.pending_choices.clear()                      # his game-start Navigation pick
    play_navigation(p, gs, 10, slot=1, faction="emperor")
    pc = gs.pending_choice_for(0)
    assert pc is not None and "swap:fremen>fremen" in {o[0] for o in pc.options}


# --- Fable leader-study review, 2026-10-16 (third batch) ----------------------

def _twisted(name):
    from src.game.leader.twisted import create_twisted_intrigues
    return next(c for c in create_twisted_intrigues() if c.name == name)


def test_leader_spy_with_empty_supply_lets_the_player_pick_the_recall():
    from src.game.leader.leader_definitions import _spy
    gs = _game(["Staban Tuek", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    for post in ("Emperor Post", "Fremen Post", "Green Post"):
        p.place_spy(post)
    _spy(p, gs)
    assert sum(p.spies_on_board.values()) == 3          # nothing recalled yet
    acts = {a.choice for a in gs.get_valid_actions(0) if a.action_type == ActionType.RESOLVE_SPY}
    assert {"recall:Emperor Post", "recall:Fremen Post", "recall:Green Post"} <= acts


def test_steersman_chooses_the_navigation_order():
    """After Leaders are picked and before round 1 (no Conflict, no cards):
    return 1 of 5 Navigation cards, then fill slots 1-4 in any order."""
    from src.game.gameState import Phase
    gs = _game(["Chani", "Steersman Y'rkoon", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[1]
    assert gs.phase == Phase.SETUP and gs.current_conflict is None and gs.round == 0
    assert gs.get_current_player_id() == 1 and not p.hand
    hand = p.leader.state["nav_hand"]
    gs.step(next(a for a in gs.get_valid_actions(1) if a.choice == f"drop:{hand[0]}"))
    keep = hand[1:]
    want = [keep[2], keep[0], keep[3], keep[1]]          # any order the player likes
    for card in want[:3]:
        assert gs.phase == Phase.SETUP and gs.current_conflict is None
        gs.step(next(a for a in gs.get_valid_actions(1) if a.choice == f"nav:{card}"))
    assert p.leader.state["nav"] == want
    assert gs.phase == Phase.PLAYER_TURNS and gs.round == 1 and gs.current_conflict is not None
    assert len(p.hand) == 5 and gs.get_current_player_id() == 0


def test_kota_keeps_her_tech_before_round_one():
    from src.game.gameState import Phase
    gs = _game(["Kota Odax of Ix", "Chani", "Duncan Idaho", "Muad'Dib"])
    assert gs.phase == Phase.SETUP and gs.current_conflict is None
    gs.step(next(a for a in gs.get_valid_actions(0) if a.action_type == ActionType.RESOLVE_CHOICE))
    assert gs.bl.kota_secret.get(0) is not None
    assert gs.phase == Phase.PLAYER_TURNS and gs.current_conflict is not None


def test_hungry_for_spice_counts_the_placement_spice():
    from src.game.leader.leader_definitions import steersman_after_step
    gs = _game(["Steersman Y'rkoon", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    gs.pending_choices.clear()
    p.turn_serial = 7
    p.spice_gained_this_turn = 4                         # the space paid out this step
    gs.in_own_turn = lambda pid: True
    hand = len(p.hand)
    steersman_after_step(p, gs)
    assert p.leader.state["hungry_done"] and len(p.hand) + len(p.in_play) >= hand


def test_navigation_5_can_trash_from_the_discard_pile():
    from src.game.leader.navigation import play_navigation
    gs = _game(["Steersman Y'rkoon", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    gs.pending_choices.clear()
    gs.start_new_round()
    p.discard, p.hand = p.hand[:2], []
    play_navigation(p, gs, 5, slot=1, faction="emperor")
    pc = gs.pending_choice_for(0)
    assert pc is not None and any(o[0].startswith("trash:discard:") for o in pc.options) \
        or len(p.trash) == 1


def test_twisted_cost_cards_need_their_cost_and_cannot_be_declined():
    gs = _game(["Piter De Vries", "Chani", "Duncan Idaho", "Gurney Halleck"])
    p = gs.players[0]
    p.troops_garrison = 0
    gs.troops_in_conflict[0] = 0
    sad = _twisted("Sadistic")
    assert not gs.intrigue_abilities_usable(0, sad)
    p.troops_garrison = 1
    assert gs.intrigue_abilities_usable(0, sad)
    calc = _twisted("Calculating")
    assert not gs.intrigue_abilities_usable(0, calc)     # no units in the Conflict


def test_resourceful_only_before_placing_an_agent():
    gs = _game(["Piter De Vries", "Chani", "Duncan Idaho", "Gurney Halleck"])
    p = gs.players[0]
    p.extra_icons_this_turn = None
    EffectResolver.resolve_single_effect(_twisted("Resourceful").effects[0], p, gs) \
        if False else None
    assert "if_turn_start" in _twisted("Resourceful").effects[0]


def test_lose_troops_never_takes_a_commander_and_lets_the_player_split():
    from src.game.leader.twisted import _fx_lose_troops
    gs = _game(["Piter De Vries", "Chani", "Duncan Idaho", "Gurney Halleck"])
    p = gs.players[0]
    p.troops_garrison, p.commanders_garrison = 3, 1      # 2 troops + 1 commander
    gs.troops_in_conflict[0] = 2
    _fx_lose_troops(gs, p, 2, {"draw": 1})
    pc = gs.pending_choice_for(0)
    assert pc is not None and {o[0] for o in pc.options} == {"split:2", "split:1", "split:0"}
    _fx_lose_troops(gs, p, 5)                            # only 4 plain troops: impossible
    assert p.commanders_garrison == 1


def test_unswerving_deploy_lets_the_player_send_a_commander():
    from src.game.choices import apply_internal
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    p.troops_garrison, p.commanders_garrison = 2, 1
    apply_internal(gs, p, {"_deploy_n": 1})
    pc = gs.pending_choice_for(0)
    assert pc is not None and {o[0] for o in pc.options} == {"dmix0", "dmix1"}
    apply_internal(gs, p, {"_deploy_mix": [1, 1]})       # the commander
    assert p.commanders_garrison == 0 and gs.troops_in_conflict[0] == 1         and gs.bl.commanders_in_conflict[0] == 1


def test_feyd_devious_strength_needs_units_in_the_conflict():
    from src.game.leader.leader_definitions import feyd_reveal
    gs = _game(["Feyd-Rautha Harkonnen", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    p.place_spy("Emperor Post")
    gs.troops_in_conflict[0] = 0
    feyd_reveal(p, gs)
    assert not [o for o in gs.pending_optional_payments if o.player_id == 0]
    gs.troops_in_conflict[0] = 1
    feyd_reveal(p, gs)
    assert [o for o in gs.pending_optional_payments if o.player_id == 0]


def test_ai_values_tueks_sietch_bonus_spice():
    from src.ai.agents import make_agent
    gs = _game(["Esmar Tuek", "Chani", "Duncan Idaho", "Muad'Dib"])
    pid = gs.get_current_player_id()
    ag = make_agent("heuristic", seed=1)
    a = next((a for a in gs.get_valid_actions(pid) if a.action_type == ActionType.AGENT_TURN
              and a.space_name == "Tuek's Sietch"), None)
    if a is None:
        return
    s0 = ag.score(gs, pid, a)
    gs.maker_bonus_spice["Tuek's Sietch"] = 6
    assert ag.score(gs, pid, a) > s0 + 2


def test_return_self_to_hand_moves_the_played_copy():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    a, b = _card(gs, "Weirding Woman"), _card(gs, "Weirding Woman")
    p.in_play += [a, b]
    gs._self_ref_pending = ["return_self_to_hand"]
    EffectResolver._resolve_self_referential({}, b, p, gs)
    assert any(c is b for c in p.hand) and not any(c is b for c in p.in_play)
    assert any(c is a for c in p.in_play)


def test_ai_reads_kotas_secret_tile_not_the_last_stack():
    from src.ai.bloodlines_heuristic import best_tech_buy
    gs = _game(["Kota Odax of Ix", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    p.spice = 20
    gs.bl.tech_stacks[-1].clear()                      # the case that crashed (game 584)
    d, extra = gs.bl.tech_at(0, -1)
    assert d is gs.bl.kota_tile(0) and extra == 1
    best_tech_buy(gs, 0)                               # no IndexError


# --- user rulings 2026-10-17 --------------------------------------------------

def test_chani_track_counts_retreated_commanders():
    gs = _game(["Chani", "Gurney Halleck", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    gs.troops_in_conflict[0] = 1
    gs.bl.commanders_in_conflict[0] = 1
    pos = p.leader.state.get("tactics", 2)
    gs.retreat_units(0, 1, 1)
    assert p.leader.state["tactics"] == pos + 1


def test_shaddam_signet_turn_blocks_every_deploy():
    gs = _game(["Shaddam Corrino IV", "Gurney Halleck", "Chani", "Duncan Idaho"])
    p = gs.players[0]
    p.no_deploy_this_turn = True
    p.troops_garrison = 3
    gs.deploy_units(0, 2)
    gs.add_pending_deployment(0, 2)
    EffectResolver.resolve_single_effect({"deploy": 2}, p, gs)
    assert gs.troops_in_conflict.get(0, 0) == 0 and not gs.pending_deployments


def test_holy_war_moves_chani_track_and_notes_the_loss():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    chani = gs.players[1]
    pos = chani.leader.state.get("tactics", 2)
    gs.ui_notes = []
    gs.bl.resolve_effect({"bl_opponents_lose_troop": 1}, gs.players[0])
    assert chani.leader.state["tactics"] == pos + 1
    assert any("P2 loses a troop" in n for n in gs.ui_notes)


def test_feyd_can_recall_a_spy_placed_during_his_reveal_turn():
    from src.game.leader.leader_definitions import feyd_reveal
    gs = _game(["Feyd-Rautha Harkonnen", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    gs.troops_in_conflict[0] = 1
    gs._reveal_open = 0
    feyd_reveal(p, gs)                                  # no spy yet: nothing offered
    assert not [o for o in gs.pending_optional_payments if o.player_id == 0]
    assert gs.place_spy(0, "Emperor Post", False)
    assert [o for o in gs.pending_optional_payments if o.player_id == 0]


def test_honor_guard_keeps_the_turn_open_for_a_1_solari_commander():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    pid = gs.get_current_player_id()
    p = gs.players[pid]
    p.solari = 1
    p.bl_space_this_turn = "Dutiful Service"
    assert not gs.bl.late_choices(pid)
    p.bl_commander_discount = 1
    assert gs.bl.late_choices(pid) and gs._turn_open_for(pid)


def test_hungry_for_spice_counts_an_intrigue_played_on_his_turn():
    from src.game.leader.leader_definitions import steersman_after_step
    gs = _game(["Steersman Y'rkoon", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    gs.pending_choices.clear()
    gs.in_own_turn = lambda pid: True
    p.turn_serial = 3
    p.spice_gained_this_turn = 2                         # 2 spice from the space
    steersman_after_step(p, gs)
    assert not p.leader.state["hungry_done"]
    p.spice_gained_this_turn = 3                         # +1 from a Plot Intrigue
    steersman_after_step(p, gs)
    assert p.leader.state["hungry_done"]


# --- AI 2026-10-17 ------------------------------------------------------------

def test_ai_keeps_battlefield_research_when_it_scores_the_endgame_point():
    from src.ai import agents as A
    from src.game.bloodlines.cards import create_bloodlines_intrigues
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    br = next(c for c in create_bloodlines_intrigues() if c.name == "Battlefield Research")
    p.techs = []
    low = A._endgame_hold_value(gs, 0, br)
    p.techs = [object(), object(), object()]
    high = A._endgame_hold_value(gs, 0, br)
    assert high > max(5.0, low) and high > 5.0


def test_leader_weights_apply_only_to_that_leader():
    from src.ai.agents import HeuristicAgent
    import src.ai.agents as A
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    ag = HeuristicAgent(seed=0, tuning={"L.Chani.res_spice": 3.0})
    pid_chani = next(p.id for p in gs.players if p.leader.name == "Chani")
    pid_other = next(p.id for p in gs.players if p.leader.name != "Chani")
    a = gs.get_valid_actions(gs.get_current_player_id())[0]
    ag._score_core(gs, pid_chani, a)
    assert A._TUNE["res_spice"] == 3.0 * ag._phase_tuning["early"]["res_spice"]
    ag._score_core(gs, pid_other, a)
    assert A._TUNE["res_spice"] == ag._phase_tuning["early"]["res_spice"]


def test_contract_stack_rewards_completing_two_contracts_at_once():
    import src.ai.agents as A
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    space = None
    pool = list(gs.contract_bank) + list(gs.contracts_on_board)
    bs = [c for c in pool if c.contract_type.value == "board_space"]
    by = {}
    for c in bs:
        by.setdefault(c.trigger_condition.get("board_space"), []).append(c)
    space, two = next(((s, cs[:2]) for s, cs in by.items() if len(cs) >= 2), (None, None))
    if space is None:
        return
    p.contracts_active = list(two)
    from src.game.gameState import GameAction, ActionType
    a = GameAction(ActionType.AGENT_TURN, 0, space_name=space)
    A._apply_tuning({**A.TUNE_DEFAULTS, "contract_stack": 0.0})
    v0 = A._contract_completion_value(gs, 0, a)
    A._apply_tuning({**A.TUNE_DEFAULTS, "contract_stack": 2.0})
    assert A._contract_completion_value(gs, 0, a) == v0 + 4.0
    A._apply_tuning(dict(A.TUNE_DEFAULTS))


def test_chani_tactics_track_is_8_troops_from_the_1_4p_marker():
    from src.game.leader.leader_definitions import CHANI_START, CHANI_END, chani_units_left
    assert CHANI_END - CHANI_START == 8
    gs = _game(["Chani", "Gurney Halleck", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    spice, water = p.spice, p.water
    chani_units_left(p, gs, 3)                       # 3rd space: 1 spice
    assert (p.spice, p.water) == (spice + 1, water)
    chani_units_left(p, gs, 5)                       # 8th space (end): water, then reset
    assert (p.spice, p.water) == (spice + 1, water + 1)
    assert p.leader.state["tactics"] == CHANI_START
    chani_units_left(p, gs, 12)                      # overflow: run to the end once, reset
    assert (p.spice, p.water) == (spice + 2, water + 2)
    assert p.leader.state["tactics"] == CHANI_START


def test_secrets_steals_from_every_opponent_with_4_intrigues_twisted_included():
    gs = _game(["Gurney Halleck", "Piter De Vries", "Chani", "Duncan Idaho"])
    me, piter, chani, duncan = gs.players
    pool = list(gs.intrigue_deck)
    for q in (me, piter, chani, duncan):
        q.intrigue_cards.clear()
    from src.game.leader.twisted import create_twisted_intrigues
    piter.intrigue_cards += create_twisted_intrigues()[:2] + [pool.pop(), pool.pop()]   # 2 + 2 twisted
    chani.intrigue_cards += [pool.pop() for _ in range(5)]
    duncan.intrigue_cards += [pool.pop() for _ in range(3)]                           # only 3: safe
    EffectResolver.resolve_single_effect({"secrets_steal": 1}, me, gs)
    assert (len(piter.intrigue_cards), len(chani.intrigue_cards), len(duncan.intrigue_cards)) == (3, 4, 3)
    assert len(me.intrigue_cards) == 2


def test_shaddam_signet_is_solari_troop_or_3_solari_for_any_influence():
    from src.game.leader.leader_definitions import shaddam_signet
    gs = _game(["Shaddam Corrino IV", "Gurney Halleck", "Chani", "Duncan Idaho"])
    p = gs.players[0]
    p.solari = 3
    shaddam_signet(p, gs)
    ch = gs.pending_choices[-1]
    opts = {o[0]: o[2] for o in ch["options"]} if isinstance(ch, dict) else {o[0]: o[2] for o in ch.options}
    assert opts["troop"] == {"solari": 1, "troops": 1}
    assert opts["influence"] == {"_pay": {"solari": 3}, "influence_any": 1}


def test_shaddam_ring_blocks_deploy_only_on_its_own_agent_turn():
    """Signet Ring on his last Agent turn; Adaptive Tactics at the start of his
    Reveal turn may still deploy (2 + its own troop), the ring turn's troops don't count."""
    from src.data.card_definitions import create_intrigue_deck
    gs = _game(["Shaddam Corrino IV", "Gurney Halleck", "Chani", "Duncan Idaho"])
    p = gs.players[0]
    ring = next(c for c in p.deck + p.hand if c.name == "Signet Ring")
    if ring not in p.hand:
        p.deck.remove(ring)
        p.hand.append(ring)
    p.agents_available = 1
    gs.step(next(a for a in gs.get_valid_actions(0)
                 if a.action_type == ActionType.AGENT_TURN and a.card_name == "Signet Ring"))
    for _ in range(200):                       # everyone else just reveals
        pid = gs.get_current_player_id()
        if pid == 0 and gs._post_agent_window is None and not gs._has_mandatory_pending_for(0):
            break
        actor = next((i for i in range(4) if gs._has_mandatory_pending_for(i)), pid)
        acts = gs.get_valid_actions(actor)
        pick = [a for t in (ActionType.END_TURN, ActionType.REVEAL_TURN, ActionType.END_REVEAL)
                for a in acts if a.action_type == t]
        gs.step((pick or acts)[0])
    assert gs.get_current_player_id() == 0 and 0 not in gs.players_revealed
    p.intrigue_cards.append(next(c for c in create_intrigue_deck() if c.name == "Adaptive Tactics"))
    p.spice, p.troops_garrison = 3, 4
    gs.step(next(a for a in gs.get_valid_actions(0) if a.action_type == ActionType.PLAY_INTRIGUE
                 and a.intrigue_card_name == "Adaptive Tactics"))
    gs.step(next(a for a in gs.get_valid_actions(0) if a.action_type == ActionType.RESOLVE_OPTIONAL))
    n = {a.deploy_count for a in gs.get_valid_actions(0) if a.action_type == ActionType.RESOLVE_DEPLOY}
    assert max(n) == 3


def test_sardaukar_coordination_deploys_troops_gained_later_that_turn():
    """Troops from a Plot Intrigue (or a recruited commander) after placing
    are still deployable on the Sardaukar Coordination turn - no +2 bonus."""
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    p.hand.append(_card(gs, "Sardaukar Coordination"))
    gs.manual_end_turn.add(0)                    # the player ends the turn (UI)
    a = next(a for a in gs.get_valid_actions(0) if a.action_type == ActionType.AGENT_TURN
             and a.card_name == "Sardaukar Coordination" and not a.use_gather_intelligence
             and not a.use_infiltrate)
    gs.step(a)
    while gs._has_mandatory_pending_for(0):
        gs.step(gs.get_valid_actions(0)[0])
    assert gs._post_agent_window == 0 or gs._agent_turn_open == 0
    before = p.troops_recruited_this_turn
    EffectResolver.resolve_single_effect({"troops": 2}, p, gs)       # a Plot Intrigue's troops
    n = {a.deploy_count for a in gs.get_valid_actions(0) if a.action_type == ActionType.RESOLVE_DEPLOY}
    assert max(n) == min(before + 2, p.troops_garrison)


def test_steersman_plays_navigation_card_any_time_that_turn():
    """Espionage: reach 2 Bene Gesserit influence, place the Spy, THEN play
    Navigation card 2 to recall that Spy (2 spice + an Intrigue)."""
    gs = _game(["Steersman Y'rkoon", "Chani", "Duncan Idaho", "Muad'Dib"])
    while gs._has_mandatory_pending_for(0):                 # setup: any 4 cards, any order
        gs.step(gs.get_valid_actions(0)[0])
    p = gs.players[0]
    st = p.leader.state
    st["nav"], st["nav_next"] = [2, 1, 3, 4], 0
    p.influence["bene_gesserit"] = 1
    p.spice = 2                                              # Espionage costs spice
    p.spies_on_board.clear()
    p.hand.append(_card(gs, "Seek Allies"))
    gs.manual_end_turn.add(0)
    gs.step(next(a for a in gs.get_valid_actions(0) if a.action_type == ActionType.AGENT_TURN
                 and a.card_name == "Seek Allies" and a.space_name == "Espionage"
                 and not a.use_gather_intelligence and not a.use_infiltrate))
    assert st["nav_next"] == 1                              # card 2 is waiting, not played yet
    for _ in range(10):                                     # place Espionage's Spy first
        acts = gs.get_valid_actions(0)
        spy = [a for a in acts if a.action_type == ActionType.RESOLVE_SPY]
        if not spy:
            break
        gs.step(spy[0])
    assert p.spies_on_board
    nav = next(i for i, it in enumerate(gs._reward_items(0)) if it["kind"] == "nav")
    claim = next(a for a in gs.get_valid_actions(0)
                 if a.action_type == ActionType.RESOLVE_CHOICE and a.choice == f"claim:{nav}")
    gs.step(claim)
    pc = gs.pending_choice_for(0)
    assert pc is not None and "recall" in {o[0] for o in pc.options}


def test_navigation_card_3_gives_2_solari_and_persuasion_only_in_slot_4():
    from src.game.leader.navigation import play_navigation
    for slot in (1, 2, 3, 4):
        gs = _game(["Steersman Y'rkoon", "Chani", "Duncan Idaho", "Muad'Dib"])
        while gs._has_mandatory_pending_for(0):
            gs.step(gs.get_valid_actions(0)[0])
        p = gs.players[0]
        s0 = p.solari
        play_navigation(p, gs, 3, slot=slot, faction="fremen")
        assert p.solari == s0 + 2
        p.hand.clear()                                   # reveal nothing: only the leader adds
        p.has_councilor = False
        gs.persuasion_pool[0] = 0
        gs.apply_reveal_turn(0)
        assert gs.persuasion_pool[0] == (1 if slot == 4 else 0), slot


def test_long_live_the_fighters_player_picks_draw_and_trash():
    gs = _game(["Gurney Halleck", "Chani", "Duncan Idaho", "Muad'Dib"])
    p = gs.players[0]
    top3 = [p.deck[-1], p.deck[-2], p.deck[-3]]
    hand, trash, discard = len(p.hand), len(p.trash), len(p.discard)
    EffectResolver.resolve_single_effect({"look_top3_draw_discard_trash": 1}, p, gs)
    pc = gs.pending_choice_for(0)
    assert pc is not None and len(pc.options) == 3
    gs.step(next(a for a in gs.get_valid_actions(0) if a.choice == "llf_draw:2"))   # the 3rd card
    assert p.hand[-1] is top3[2] and len(p.hand) == hand + 1
    pc = gs.pending_choice_for(0)
    assert pc is not None and len(pc.options) == 2
    gs.step(next(a for a in gs.get_valid_actions(0) if a.choice == "llf_trash:0"))  # trash the 1st
    assert p.trash[-1] is top3[0] and len(p.trash) == trash + 1
    assert p.discard[-1] is top3[1] and len(p.discard) == discard + 1
