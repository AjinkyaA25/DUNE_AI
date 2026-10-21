"""Inputs for the AlphaZero net. Everything here is visible to the acting
player: the shared state features (src/ai/features.py, already built from
legal information only) plus the player's own hand and intrigue cards.

Moves are encoded as the hand-built action features plus hashed identity
features ("card=Dagger", "space=Arrakeen", "deploy=3", ...), so every kind
of decision the engine offers gets its own input without a fixed move list."""
from __future__ import annotations

import zlib
from typing import List, Tuple

import numpy as np

from src.ai.action_features import ACTION_FEATURE_DIM, encode_action
from src.ai.features import FEATURE_DIM, encode_state

HAND_DIM = 256
HASH_DIM = 2048
MAX_IDX = 8                                   # hashed features per move
STATE_DIM = FEATURE_DIM + 2 * HAND_DIM
DENSE_DIM = ACTION_FEATURE_DIM

_FIELDS = ("card_name", "space_name", "spy_post_name", "uplift_space_name",
           "intrigue_card_name", "acquire_card_name", "reserve_type",
           "space_option", "trash_card_name", "influence_faction", "choice")


def _h(s: str, d: int) -> int:
    return zlib.crc32(s.encode("utf-8")) % d


def encode_obs(gs, pid: int) -> np.ndarray:
    p = gs.players[pid]
    hand = np.zeros(HAND_DIM, np.float32)
    for c in p.hand:
        hand[_h(c.name, HAND_DIM)] += 0.5
    intr = np.zeros(HAND_DIM, np.float32)
    for c in p.intrigue_cards:
        intr[_h(c.name, HAND_DIM)] += 0.5
    return np.concatenate([encode_state(gs, pid), hand, intr])


def action_idx(a) -> List[int]:
    t = a.action_type.value
    out = [_h("type=" + t, HASH_DIM)]
    for f in _FIELDS:
        v = getattr(a, f)
        if v is not None:
            out.append(_h(f"{t}|{f}={v}", HASH_DIM))
    if a.deploy_count:
        out.append(_h(f"deploy={a.deploy_count}", HASH_DIM))
    if a.action_type.value == "resolve_contract":
        out.append(_h(f"contract={a.contract_index}", HASH_DIM))
    if a.accept_optional:
        out.append(_h(f"{t}|accept", HASH_DIM))
    if a.use_gather_intelligence:
        out.append(_h("gather", HASH_DIM))
    if a.use_infiltrate:
        out.append(_h("infiltrate", HASH_DIM))
    return out[:MAX_IDX]


def encode_actions(gs, pid: int, acts) -> Tuple[np.ndarray, np.ndarray]:
    """(dense [A, DENSE_DIM], idx [A, MAX_IDX] padded with HASH_DIM)."""
    dense = np.stack([encode_action(gs, pid, a, 0.0) for a in acts]).astype(np.float32)
    idx = np.full((len(acts), MAX_IDX), HASH_DIM, np.int32)
    for i, a in enumerate(acts):
        ii = action_idx(a)
        idx[i, :len(ii)] = ii
    return dense, idx
