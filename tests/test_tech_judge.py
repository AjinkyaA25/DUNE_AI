"""tech_judge: tech tiles valued by the player's situation (user, 2026-10-19)."""
import src.ai.agents as A
import src.ai.bloodlines_heuristic as B
from src.data.card_definitions import setup_game
from src.game.bloodlines.techs import TECH_BY_NAME


def _gs(round_=5):
    gs = setup_game(4, seed=3, neutral_leaders=True, use_bloodlines=True)
    gs.round = round_
    A._apply_tuning({**A.TUNE_DEFAULTS, "tech_judge": 1.0})
    return gs, 0


def _val(gs, pid, name):
    return B.tech_value(gs, pid, TECH_BY_NAME[name])


def test_panopticon_worthless_with_every_friendship_key_without():
    gs, pid = _gs()
    p = gs.players[pid]
    p.influence = {f: 2 for f in p.influence}            # every friendship already
    with_all = B._sit_panopticon(gs, pid, None)
    p.influence = {f: 1 for f in p.influence}            # parked at 1, not visiting
    parked = B._sit_panopticon(gs, pid, None)
    assert with_all == 0.0
    assert parked > 2 * A._effect_value(gs, pid, {"vp": 1})   # 2+ friendships at the end


def test_choam_transports_needs_contracts():
    gs, pid = _gs(round_=6)
    p = gs.players[pid]
    p.contracts_completed, p.contracts_active = [], []
    trap = B._sit_choam(gs, pid, None)
    p.contracts_completed = [object()] * 3
    good = B._sit_choam(gs, pid, None)
    assert good > trap + 0.5 * A._effect_value(gs, pid, {"vp": 1})


def test_draw_tiles_value_a_strong_row_card_coming_into_reach():
    gs, pid = _gs(round_=2)
    p = gs.players[pid]
    gs.persuasion_pool[pid] = 0
    row = sorted(gs.imperium_row, key=lambda c: c.cost or 0)
    for c in p.hand:
        c.persuasion = 0
    p.has_councilor = False
    p.deck = [c for c in p.deck]
    for c in p.deck + p.discard:                         # the card drawn is worth 2 persuasion
        c.persuasion = 2
    gs.persuasion_pool[pid] = (row[-1].cost or 0) - 2    # the top row card is 2 short
    assert B._draw_now_value(gs, pid) > 0.0
    gs.persuasion_pool[pid] = 50                         # anything affordable already
    assert B._draw_now_value(gs, pid) == 0.0


def test_ornithopter_fleet_pays_for_unmatched_icons():
    gs, pid = _gs(round_=7)
    p = gs.players[pid]
    p.battle_icons = []
    none = B._sit_fleet(gs, pid, None)
    p.battle_icons = ["crysknife", "desert_mouse"]       # two singles, no way to match
    p.intrigue_cards = []
    singles = B._sit_fleet(gs, pid, None)
    assert singles > none + 0.5 * A._effect_value(gs, pid, {"vp": 1})
    # holding the endgame Intrigues that match both makes the Fleet pointless
    from src.game.intrigue.intrigue import IntrigueCard, IntrigueTiming
    p.intrigue_cards = [IntrigueCard(n, IntrigueTiming.ENDGAME,
                                     [{"spice_or_match_icon": {"icon": i, "spice": 1}}])
                        for n, i in (("Crysknife", "crysknife"), ("Desert Mouse", "desert_mouse"))]
    matched = B._sit_fleet(gs, pid, None)
    assert matched < singles


def test_old_flat_values_when_off():
    gs, pid = _gs()
    A._apply_tuning(dict(A.TUNE_DEFAULTS))
    p = gs.players[pid]
    p.influence = {f: 2 for f in p.influence}
    off = _val(gs, pid, "Panopticon")
    A._apply_tuning({**A.TUNE_DEFAULTS, "tech_judge": 1.0})
    on = _val(gs, pid, "Panopticon")
    assert on <= off                                      # the judgement drops the dead endgame part
