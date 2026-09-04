"""
Headless game runner for self-play.

`play_game(agents, num_players, seed, ...)` plays one full game with no I/O and
returns a GameResult carrying the winner, final VP, and (optionally) the
per-decision feature trajectory used to build training data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.data.card_definitions import setup_game
from src.game.gameState import ActionType
from src.ai.features import encode_state

MOVE_CAP = 1500


@dataclass
class GameResult:
    winner: Optional[int]
    final_vp: List[int]
    num_moves: int
    num_players: int
    truncated: bool = False
    leaders: List[str] = field(default_factory=list)
    # trajectory: (perspective_pid, feature_vector) captured at each decision
    feats: List[np.ndarray] = field(default_factory=list)
    feat_pids: List[int] = field(default_factory=list)


def play_game(agents, num_players: int = 4, seed: Optional[int] = None,
              leaders=None, record: bool = True,
              use_choam: bool = True, neutral_leaders: bool = True) -> GameResult:
    """
    `neutral_leaders` defaults to True: Leader ability text is unverified, so
    self-play / eval / training runs ignore Leaders unless `leaders` names
    specific ones (which always takes priority over this flag).
    """
    gs = setup_game(num_players=num_players, seed=seed, use_choam=use_choam,
                    leaders=leaders,
                    neutral_leaders=neutral_leaders and leaders is None)
    feats: List[np.ndarray] = []
    feat_pids: List[int] = []

    moves = 0
    truncated = False
    while not gs.game_over:
        if moves >= MOVE_CAP:
            truncated = True
            break
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        non_noop = [a for a in valid if a.action_type != ActionType.NO_OP]
        if record and non_noop:
            feats.append(encode_state(gs, pid))
            feat_pids.append(pid)
        action = agents[pid].select_action(gs, pid, valid)
        gs.step(action)
        moves += 1

    if not gs.game_over:
        gs.check_victory_conditions()

    final_vp = [p.victory_points for p in gs.players]
    winner = gs.winner
    if winner is None:                       # truncated / unresolved
        winner = int(np.argmax(final_vp))

    return GameResult(
        winner=winner,
        final_vp=final_vp,
        num_moves=moves,
        num_players=num_players,
        truncated=truncated,
        leaders=[getattr(p.leader, "name", "?") for p in gs.players],
        feats=feats,
        feat_pids=feat_pids,
    )
