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
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, FEATURE_DIM)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(np.float32)
    m = ValueModel(hidden=32, seed=0)
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
