"""Privileged critic: it sees hidden information, the policy never does."""
import numpy as np

from src.ai.features import encode_state
from src.ai.hidden_features import encode_hidden
from src.data.card_definitions import setup_game


def test_opponent_hand_is_hidden_from_the_policy_but_not_the_critic():
    gs = setup_game(4, seed=3, neutral_leaders=True, use_bloodlines=True)
    q = gs.players[1]
    s0, h0 = encode_state(gs, 0).copy(), encode_hidden(gs, 0).copy()
    # swap a card between the opponent's (hidden) hand and their (hidden) deck
    i = next(i for i, c in enumerate(q.deck) if c.name not in {x.name for x in q.hand})
    q.hand[0], q.deck[i] = q.deck[i], q.hand[0]
    assert np.array_equal(encode_state(gs, 0), s0)          # the policy can't tell
    assert not np.array_equal(encode_hidden(gs, 0), h0)     # the critic can


def test_next_conflict_is_seen_by_the_critic():
    gs = setup_game(4, seed=3, neutral_leaders=True, use_bloodlines=True)
    h0 = encode_hidden(gs, 0).copy()
    gs.conflict_deck[0], gs.conflict_deck[1] = gs.conflict_deck[1], gs.conflict_deck[0]
    if gs.conflict_deck[0].name != gs.conflict_deck[1].name:
        assert not np.array_equal(encode_hidden(gs, 0), h0)
