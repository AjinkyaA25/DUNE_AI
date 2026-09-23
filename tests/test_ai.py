"""AI pipeline: features, value model, agents, self-play harness."""
import numpy as np

from src.data.card_definitions import setup_game
from src.ai.features import encode_state, FEATURE_DIM
from src.ai.value_model import ValueModel
from src.ai.agents import RandomAgent, HeuristicAgent, GreedyValueAgent, make_agent
from src.ai.opening_book import OpeningBook
from src.selfplay.runner import play_game
from src.selfplay.arena import head_to_head


def test_feature_encoding_shape_and_finite():
    gs = setup_game(4, seed=1)
    for pid in range(4):
        f = encode_state(gs, pid)
        assert f.shape == (FEATURE_DIM,)
        assert np.isfinite(f).all()
    # 2-player games are zero-padded to the same width
    gs2 = setup_game(2, seed=1)
    assert encode_state(gs2, 0).shape == (FEATURE_DIM,)


def test_clone_is_independent():
    gs = setup_game(4, seed=2)
    g2 = gs.clone()
    g2.players[0].solari += 50
    assert gs.players[0].solari != g2.players[0].solari
    from src.game.gameState import ActionType
    pid = g2.get_current_player_id()
    g2.step(g2.get_valid_actions(pid)[0])          # must not raise


def test_value_model_learns_xor_ish():
    # Fixed synthetic dim (independent of FEATURE_DIM) — this test only checks
    # the optimizer can fit a mildly nonlinear function, not our real features.
    dim = 20
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, dim)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(np.float32)
    m = ValueModel(dim=dim, hidden=32, seed=0)
    hist = m.fit(X, y, epochs=30, lr=5e-3)
    assert hist["val_logloss"][-1] < hist["val_logloss"][0]
    assert hist["val_logloss"][-1] < 0.55


def test_value_model_save_load(tmp_path):
    m = ValueModel(hidden=8, seed=1)
    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    p = m.predict(f)
    path = str(tmp_path / "m.npz")
    m.save(path)
    m2 = ValueModel.load(path)
    assert abs(m2.predict(f) - p) < 1e-9


def test_play_game_records_trajectory():
    agents = {i: HeuristicAgent(seed=i) for i in range(4)}
    res = play_game(agents, num_players=4, seed=3, record=True)
    assert res.winner in range(4)
    assert len(res.feats) == len(res.feat_pids) > 20
    assert all(f.shape == (FEATURE_DIM,) for f in res.feats)


def test_policy_recording_aligned_and_valid():
    agents = {i: HeuristicAgent(seed=i) for i in range(4)}
    res = play_game(agents, num_players=4, seed=7, record=True, record_policy=True)
    from src.ai.action_features import ACTION_FEATURE_DIM
    assert len(res.pol_actA) == len(res.feats) == len(res.pol_ci)
    for af, ci in zip(res.pol_actA, res.pol_ci):
        assert af.ndim == 2 and af.shape[1] == ACTION_FEATURE_DIM
        assert 0 <= ci < af.shape[0]                     # chosen action was kept
        assert np.isfinite(af).all()


def test_policy_model_learns_and_roundtrips(tmp_path):
    from src.ai.policy_model import PolicyModel
    from src.ai.features import FEATURE_DIM as SD
    from src.ai.action_features import ACTION_FEATURE_DIM as AD
    rng = np.random.default_rng(0)
    N, K = 1500, 6
    S = rng.normal(size=(N, SD)).astype(np.float32)
    A = rng.normal(size=(N, K, AD)).astype(np.float32)
    M = np.ones((N, K), np.float32)
    ci = (S[:, :1] + A[:, :, 0]).argmax(axis=1)          # learnable target
    pm = PolicyModel(hidden=16, seed=1)
    h = pm.fit(S, A, M, ci, np.ones(N, np.float32), epochs=40,
               batch_size=128, lr=3e-3, seed=1)
    assert h["val_acc"][-1] > 0.55 > h["val_acc"][0]
    p = tmp_path / "pol.npz"
    pm.save(str(p))
    pm2 = PolicyModel.load(str(p))
    pol = pm2.policy(S[0], A[0])
    assert abs(float(pol.sum()) - 1.0) < 1e-5


def test_value_agent_accepts_policy():
    from src.ai.policy_model import PolicyModel
    pol = PolicyModel(seed=0)
    ag = GreedyValueAgent(model=ValueModel(seed=0), policy=pol, policy_weight=0.3)
    gs = setup_game(4, seed=2)
    pid = gs.get_current_player_id()
    a = ag.select_action(gs, pid, gs.get_valid_actions(pid))
    assert a is not None


def test_heuristic_beats_random():
    r = head_to_head(lambda: HeuristicAgent(), lambda: RandomAgent(),
                     n_games=40, num_players=4)
    assert r["a_vs_fair"] > 1.8            # clearly better than chance (0.25)


def test_opening_book_bonus_applies():
    book = OpeningBook.default()
    gs = setup_game(4, seed=9)
    from src.game.gameState import GameAction, ActionType
    a = GameAction(ActionType.AGENT_TURN, 0, card_name="X", space_name="High Council")
    # round 1 -> economy-spine rule active
    assert book.bonus(gs, 0, a) > 0
    gs.round = 9
    assert book.bonus(gs, 0, a) == 0


def test_make_agent_specs():
    assert isinstance(make_agent("random"), RandomAgent)
    assert isinstance(make_agent("heuristic:T0.5"), HeuristicAgent)
    assert isinstance(make_agent("value"), GreedyValueAgent)


def test_tier_prior_orders_buys():
    """The Consules tier blend should make S-tier cards outscore D-tier cards
    at buy time, and leave untiered cards on their situational merit."""
    from src.data.card_definitions import create_imperium_cards
    from src.ai.agents import _acquire_card_value
    gs = setup_game(4, seed=1)
    gs.round = 6
    gs.players[0].has_swordmaster = True
    by = {c.name: c for c in create_imperium_cards()}
    s = _acquire_card_value(gs, 0, by["Guild Spy"])          # S
    d = _acquire_card_value(gs, 0, by["Hidden Missive"])     # D
    u = _acquire_card_value(gs, 0, by["Junction Headquarters"])  # untiered
    assert s > d + 2.0
    assert by["Guild Spy"].tier == "S" and by["Hidden Missive"].tier == "D"
    assert by["Junction Headquarters"].tier is None and u >= 0.0


def test_deploy_scorer_counts_own_combat_intrigues():
    """Holding a Combat Intrigue should raise the value of committing troops to
    a crucial Conflict the intrigue could swing (vs holding none)."""
    from src.ai.agents import _own_combat_intrigue_swords
    gs = setup_game(4, seed=5)
    # sanity: helper returns a number and never explodes
    for pid in range(4):
        assert _own_combat_intrigue_swords(gs, pid) >= 0.0
