"""
LookaheadAgent: the heuristic plus a short look into the future.

For the decisions that shape a round (where an Agent goes, when to Reveal,
how many troops to deploy) the heuristic's top `k` moves are each tried on
`m` determinized copies of the game (opponents' hands / decks / intrigues
re-randomized, so nothing hidden is seen). After the move, play continues
for `depth` more player turns (Agent or Reveal turns by anyone, with every
pending choice, buy and combat in between resolved by the heuristic), and
the position reached is scored by the position evaluator
(src/az/evaluator.py): the searching player's chance to win.

The move played maximises
    heuristic score + look_w x 100 x (win chance after the move
                                      - win chance after the heuristic's pick)
so `look_w` (a tuning knob, mutable by src/selfplay/evolve.py) sets how many
heuristic points one percentage point of win chance is worth. A game that
ends inside the lookahead is scored by its real result.

spec: look[:D<depth>][:K<k>][:M<m>][:LW<look_w>][:ev=<evaluator.npz>][:tuned=<json>]
          [:E<round>][:EK<k>][:EM<m>]
      e.g. look:D3:K4:M2:tuned=config/evolve_look/champion.json

Endgame solver (E<round>, user 2026-10-19: "the endgame solver should
realistically start on round 6, the preparation round for round 7"): from
that round on, each candidate is played out to the END of the game on EM
determinized worlds (every seat by the heuristic) and scored by the real
result, instead of `depth` turns + the evaluator; EK candidates.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

from src.game.gameState import ActionType, GameAction, GameState

SEARCHED = (ActionType.AGENT_TURN, ActionType.REVEAL_TURN, ActionType.RESOLVE_DEPLOY)
TURNS = (ActionType.AGENT_TURN, ActionType.REVEAL_TURN)
DEFAULT_EVAL = os.path.join("models_eval_look", "evaluator.npz")
LOOK_W = 0.2                       # heuristic points per percentage point of win chance

_EVALS = {}


def load_evaluator(path: Optional[str] = None):
    from src.az.evaluator import Evaluator
    path = path or (DEFAULT_EVAL if os.path.exists(DEFAULT_EVAL)
                    else os.path.join("models_eval", "evaluator.npz"))
    if path not in _EVALS:
        _EVALS[path] = Evaluator(path)
    return _EVALS[path]


def _actor(g: GameState) -> int:
    p = g.player_in_reveal_buy
    return g.get_current_player_id() if p is None else p


class LookaheadAgent:
    name = "look"

    def __init__(self, depth: int = 3, k: int = 4, m: int = 2, look_w: Optional[float] = None,
                 evaluator: Optional[str] = None, tuning: Optional[dict] = None,
                 seed: Optional[int] = None, opening_book=None, step_cap: int = 200,
                 end_round: int = 0, end_k: int = 4, end_m: int = 4):
        from src.ai.agents import HeuristicAgent
        # the heuristic shortlists the candidates and plays every seat in the
        # lookahead; `tuning` (evolved weights) changes both
        self.h = HeuristicAgent(seed=seed, opening_book=opening_book, tuning=tuning)
        self.depth, self.k, self.m = depth, k, m
        self.look_w = look_w if look_w is not None else float((tuning or {}).get("look_w", LOOK_W))
        self.ev = load_evaluator(evaluator)
        self.rng = np.random.default_rng(seed)
        self.step_cap = step_cap
        self.end_round, self.end_k, self.end_m = end_round, end_k, end_m
        self.stats = {"searched": 0, "switched": 0, "seconds": 0.0}
        self.last = None

    def reset(self) -> None:
        pass

    def select_action(self, gs: GameState, pid: int, valid: List[GameAction]) -> GameAction:
        self.last = None
        acts = [a for a in valid if a.action_type != ActionType.NO_OP] or valid
        if (self.look_w <= 0 or len(acts) <= 1
                or not any(a.action_type in SEARCHED for a in acts)):
            return self.h.select_action(gs, pid, valid)
        t0 = time.time()
        endgame = bool(self.end_round) and gs.round >= self.end_round
        k, m = (self.end_k, self.end_m) if endgame else (self.k, self.m)
        scores = [self.h.score(gs, pid, a) for a in acts]
        order = sorted(range(len(acts)), key=lambda i: -scores[i])[: k]
        cands = [acts[i] for i in order]
        from src.ai.determinize import determinize
        # common random numbers: world j is the same for every candidate
        seeds = self.rng.integers(0, 2**31 - 1, size=m)
        win = np.full((len(cands), m), np.nan)
        leaves, where = [], []
        for i, c in enumerate(cands):
            key = repr(c)
            for j in range(m):
                g = determinize(gs, pid, np.random.default_rng(int(seeds[j])))
                ga = next((x for x in g.get_valid_actions(pid) if repr(x) == key), None)
                if ga is None:
                    continue
                g.step(ga)
                self._advance(g, to_end=endgame)
                if g.game_over:
                    win[i, j] = self._result(g, pid)
                    if endgame:
                        # a 0/1 result per world is noisy: break ties by the VP margin
                        vp = [q.victory_points for q in g.players]
                        lead = vp[pid] - max(v for q, v in enumerate(vp) if q != pid)
                        win[i, j] += 0.03 * max(-5, min(5, lead))
                else:
                    leaves.append(self.ev.encode(g, pid))
                    where.append((i, j))
        if leaves:
            for (i, j), (wp, _) in zip(where, self.ev.batch(leaves, pid, gs.num_players)):
                win[i, j] = wp[pid]
        with np.errstate(all="ignore"):
            mean = np.nanmean(win, axis=1)
        base = mean[0] if not np.isnan(mean[0]) else 0.0
        total = [scores[order[i]] + (0.0 if np.isnan(mean[i]) else
                                     self.look_w * 100.0 * (mean[i] - base))
                 for i in range(len(cands))]
        best = int(np.argmax(total))
        self.stats["searched"] += 1
        self.stats["switched"] += best != 0
        self.stats["seconds"] += time.time() - t0
        self.last = {"cands": cands, "win": np.where(np.isnan(mean), -1.0, mean).tolist(),
                     "total": [float(x) for x in total], "chosen": best}
        return cands[best]

    def _advance(self, g: GameState, to_end: bool = False) -> None:
        """Play on until `depth` more Agent / Reveal turns have been taken
        (to_end: until the game is over)."""
        turns = steps = 0
        depth, cap = (10 ** 9, 20 * self.step_cap) if to_end else (self.depth, self.step_cap)
        while not g.game_over and turns < depth and steps < cap:
            q = _actor(g)
            va = g.get_valid_actions(q)
            a = self.h.select_action(g, q, va)
            if a.action_type in TURNS:
                turns += 1
            g.step(a)
            steps += 1

    @staticmethod
    def _result(g: GameState, pid: int) -> float:
        if not g.game_over:
            g.check_victory_conditions()
        vp = [p.victory_points for p in g.players]
        if g.winner is not None:
            return float(g.winner == pid)
        top = max(vp)
        w = [i for i, v in enumerate(vp) if v == top]
        return 1.0 / len(w) if pid in w else 0.0
