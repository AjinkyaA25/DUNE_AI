"""
RoundSearchAgent: determinized Monte-Carlo search over the heuristic.

The 1-ply GreedyValueAgent and the old ISMCTSAgent both judged positions
mid-round with a weak value net, and ISMCTS spread 24 simulations over up
to 14 moves (a visit or two each), so it played worse than the heuristic it
was built on. This agent is a policy-improvement step instead:

  - Only the decisions that shape a round are searched: which card/space an
    Agent goes to, when to Reveal, how many troops to deploy. Everything
    else (buys, pending choices, intrigues) uses the heuristic directly.
  - Candidates = the heuristic's top `k` moves. Each is played out `m`
    times to the END OF THE ROUND (combat resolved) on a determinized copy
    of the game (opponents' hands / decks / intrigues and deck order
    re-randomized: the search can't see hidden information), with every
    player following the heuristic.
  - The round-end position is scored for the searching player relative to
    the opponents (VP, conflicts, influence, resources, board...).
  - The heuristic's own choice is kept unless another candidate beats it by
    `margin` on average, so noise can't make it worse than the heuristic.

spec: search[:K<k>][:M<m>][:MG<margin>][:round]   e.g. search:K5:M24
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from src.game.gameState import ActionType, GameAction, GameState

SEARCHED = (ActionType.AGENT_TURN, ActionType.REVEAL_TURN,
            ActionType.RESOLVE_DEPLOY)


def round_value(gs: GameState, pid: int) -> float:
    """Position score at a round boundary, relative to the best opponent
    (VP-equivalent units)."""
    from src.ai.agents import heuristic_state_value

    def raw(q: int) -> float:
        p = gs.players[q]
        v = heuristic_state_value(gs, q)
        return 10.0 * p.victory_points + 20.0 * v
    mine = raw(pid)
    best = max(raw(q.id) for q in gs.players if q.id != pid)
    if gs.game_over:
        return 100.0 if gs.winner == pid else -100.0
    return mine - best


def end_value(gs: GameState, pid: int) -> float:
    """Game result for pid: win share (ties split) + a small VP-margin term
    so near-misses still rank above blowouts."""
    if not gs.game_over:
        gs.check_victory_conditions()
    vp = [p.victory_points for p in gs.players]
    top = max(vp)
    winners = [i for i, v in enumerate(vp) if v == top]
    if gs.winner is not None:
        winners = [gs.winner]
    win = (1.0 / len(winners)) if pid in winners else 0.0
    margin = vp[pid] - max(v for i, v in enumerate(vp) if i != pid)
    return win + 0.02 * margin


class RoundSearchAgent:
    name = "search"

    def __init__(self, k: int = 4, m: int = 6, margin: float = 1.0,
                 seed: Optional[int] = None, opening_book=None,
                 move_cap: int = 3000, horizon: str = "end",
                 tuning: Optional[dict] = None):
        from src.ai.agents import HeuristicAgent
        # the heuristic both shortlists the candidates and plays every seat in
        # the playouts; `tuning` (e.g. the human-fitted knobs) changes both
        self.h = HeuristicAgent(seed=seed, opening_book=opening_book, tuning=tuning)
        self.k, self.m, self.margin = k, m, margin
        self.rng = np.random.default_rng(seed)
        self.move_cap = move_cap
        # "end": play every sample to the end of the game and score the real
        # result (games are ~0.05 s with the heuristic); "round": stop when
        # the round ends and score the position (cheaper, noisier)
        self.horizon = horizon
        self.stats = {"searched": 0, "switched": 0, "seconds": 0.0}

    def reset(self) -> None:
        pass

    def select_action(self, gs: GameState, pid: int,
                      valid: List[GameAction]) -> GameAction:
        self.last = None
        acts = [a for a in valid if a.action_type != ActionType.NO_OP] or valid
        if len(acts) <= 1 or not any(a.action_type in SEARCHED for a in acts):
            return self.h.select_action(gs, pid, valid)
        t0 = time.time()
        scores = [self.h.score(gs, pid, a) for a in acts]
        order = sorted(range(len(acts)), key=lambda i: -scores[i])[: self.k]
        cands = [acts[i] for i in order]
        from src.ai.determinize import determinize
        # common random numbers: sample j uses the same hidden-information
        # world for every candidate, so differences come from the move
        seeds = self.rng.integers(0, 2**31 - 1, size=self.m)
        vals = np.full((len(cands), self.m), np.nan)
        for i, a in enumerate(cands):
            for j, sd in enumerate(seeds):
                g = determinize(gs, pid, np.random.default_rng(int(sd)))
                ga = next((x for x in g.get_valid_actions(pid)
                           if repr(x) == repr(a)), None)
                if ga is None:
                    continue
                g.step(ga)
                vals[i, j] = self._playout(g, pid, gs.round)
        self.stats["searched"] += 1
        self.stats["seconds"] += time.time() - t0
        # paired comparison against the heuristic's choice (cands[0]):
        # switch only when the gain is clear of the sampling noise
        best, best_gain = 0, 0.0
        for i in range(1, len(cands)):
            d = vals[i] - vals[0]
            d = d[~np.isnan(d)]
            if len(d) < 3:
                continue
            gain = d.mean()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 1e9
            if gain > self.margin and gain > 1.5 * se and gain > best_gain:
                best, best_gain = i, gain
        if best:
            self.stats["switched"] += 1
        # what the search saw, for search-based self-play training
        self.last = {"cands": cands, "scores": np.nanmean(np.where(
            np.isnan(vals), -1.0, vals), axis=1).tolist(), "chosen": best}
        return cands[best]

    def _playout(self, g: GameState, pid: int, start_round: int) -> float:
        moves = 0
        to_end = self.horizon == "end"
        while not g.game_over and (to_end or g.round == start_round)                 and moves < self.move_cap:
            cur = g.player_in_reveal_buy
            if cur is None:
                cur = g.get_current_player_id()
            g.step(self.h.select_action(g, cur, g.get_valid_actions(cur)))
            moves += 1
        if to_end:
            return end_value(g, pid)
        return round_value(g, pid)
