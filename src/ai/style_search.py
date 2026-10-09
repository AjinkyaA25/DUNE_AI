"""
Style search: pick a playstyle each round by playing the game out.

At the start of each round the agent asks, for every playstyle in
config/styles/ (techs / swordmaster / high_council / commanders /
balanced): "if I play the rest of the game this way from HERE, how often do
I win?" It plays M games to the end from the current position for each
style (hidden information re-sampled per game; common random numbers, so
every style is judged on the same sampled worlds; opponents modelled as the
balanced player), and adopts the best style for this round. A style is only
dropped for another one that beats it by `margin`, so it doesn't flip on
noise. Moves are then made by that style's player: the heuristic, or round
search with the style's tuning (spec ...:search).

This is how the AI learns WHEN each style is right and mixes them: the
choice depends on its own resources, the combat results so far, the tech
market and skill row, and what the opponents hold - and it can change
from round to round.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from src.game.gameState import ActionType, GameAction, GameState

STYLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "config", "styles")


def load_styles() -> Dict[str, dict]:
    return {os.path.basename(p)[:-5]: json.load(open(p, encoding="utf-8"))
            for p in sorted(glob.glob(os.path.join(STYLE_DIR, "*.json")))}


class StyleSearchAgent:
    name = "stylesearch"

    def __init__(self, m: int = 16, margin: float = 0.05, seed: Optional[int] = None,
                 base: str = "heuristic", search_kw: Optional[dict] = None,
                 styles: Optional[List[str]] = None, opponent: str = "balanced"):
        from src.ai.agents import HeuristicAgent
        from src.ai.search import RoundSearchAgent
        allst = load_styles()
        self.styles = {k: v for k, v in allst.items() if styles is None or k in styles}
        self.m, self.margin = m, margin
        self.rng = np.random.default_rng(seed)
        self.roll = {k: HeuristicAgent(seed=seed, tuning=t) for k, t in self.styles.items()}
        self.opp = HeuristicAgent(seed=seed, tuning=allst.get(opponent))
        if base == "search":
            kw = dict(search_kw or {})
            self.play = {k: RoundSearchAgent(seed=seed, tuning=t, **kw)
                         for k, t in self.styles.items()}
        else:
            self.play = self.roll
        self.current: Optional[str] = None
        self.chosen_round = -1
        self.history: List[dict] = []        # per round: style + playout means

    def reset(self) -> None:
        self.current, self.chosen_round, self.history = None, -1, []

    def _playout(self, g: GameState, pid: int, mine) -> float:
        from src.ai.search import end_value
        n = 0
        while not g.game_over and n < 3000:
            cur = g.player_in_reveal_buy
            if cur is None:
                cur = g.get_current_player_id()
            ag = mine if cur == pid else self.opp
            g.step(ag.select_action(g, cur, g.get_valid_actions(cur)))
            n += 1
        return end_value(g, pid)

    def choose(self, gs: GameState, pid: int) -> str:
        from src.ai.determinize import determinize
        names = list(self.styles)
        seeds = self.rng.integers(0, 2**31 - 1, size=self.m)
        vals = np.zeros((len(names), self.m))
        for i, k in enumerate(names):
            for j in range(self.m):
                g = determinize(gs, pid, np.random.default_rng(int(seeds[j])))
                vals[i, j] = self._playout(g, pid, self.roll[k])
        means = vals.mean(axis=1)
        best = names[int(np.argmax(means))]
        if self.current in names:
            cur = means[names.index(self.current)]
            if means.max() - cur < self.margin:
                best = self.current          # not clearly better: keep the style
        self.history.append({"round": gs.round, "style": best,
                             "means": {k: round(float(v), 3) for k, v in zip(names, means)}})
        return best

    def select_action(self, gs: GameState, pid: int,
                      valid: List[GameAction]) -> GameAction:
        if gs.round != self.chosen_round and gs.phase.value == "player_turns" \
                and gs.player_in_reveal_buy is None:
            self.chosen_round = gs.round
            self.current = self.choose(gs, pid)
        style = self.current or "balanced"
        if style not in self.play:
            style = next(iter(self.play))
        return self.play[style].select_action(gs, pid, valid)
