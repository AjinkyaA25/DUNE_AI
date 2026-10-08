#!/usr/bin/env python
"""
Fit the playstyle objective model (src/ai/playstyle.py) on the video games.

Every recorded agent turn of every player in the games with exact Bloodlines
events (merged chat: commander recruits + skills, techs) is a sample:
  x = style_state(gs, pid) just before the turn
  y = what the player spent on in that turn: Swordmaster, the High Council
      seat, a commander recruit (board or supply), a tech, or nothing
Winners' turns weigh WINNER_W times as much, so the model leans toward what
the winning players did in each position.

Usage:
  python video_scrape/train_objectives.py      # -> config/objective_model.npz
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import replay as R  # noqa: E402
from clean import clean_game  # noqa: E402
from src.ai.playstyle import OBJECTIVES, ObjectiveModel, owned, style_state  # noqa: E402
from src.game.gameState import ActionType  # noqa: E402

WINNER_W = 2.0


class LabelReplay(R.Replay):
    """A replay that notes each player's position before every recorded
    agent turn and what they bought before their next one."""

    def __init__(self, game):
        super().__init__(game)
        self.open = {}           # pid -> (x, owned_before, recruits_before, techs_before)
        self.recruits_n = collections.Counter()
        self.techs_n = collections.Counter()
        self.rows = []           # (x, label, pid)

    def _close(self, pid):
        if pid not in self.open:
            return
        x, before, rec0, tech0 = self.open.pop(pid)
        now = owned(self.gs, pid)
        got = []
        if now["swordmaster"] > before["swordmaster"]:
            got.append("swordmaster")
        if now["high_council"] > before["high_council"]:
            got.append("high_council")
        if self.recruits_n[pid] > rec0:
            got.append("commanders")
        if self.techs_n[pid] > tech0:
            got.append("techs")
        for g in got or ["none"]:
            self.rows.append((x, OBJECTIVES.index(g), pid, 1.0 / max(1, len(got))))

    def _step(self, pid, action, record=False):
        at = action.action_type
        if record and at == ActionType.AGENT_TURN:
            self._close(pid)
            self.open[pid] = (style_state(self.gs, pid), owned(self.gs, pid),
                              self.recruits_n[pid], self.techs_n[pid])
        if at == ActionType.RESOLVE_BL_CHOICE and action.choice:
            if action.choice.startswith("board:") or action.choice == "supply":
                self.recruits_n[pid] += 1
            elif action.choice.startswith("stack:"):
                self.techs_n[pid] += 1
        return super()._step(pid, action, record)

    def run(self):
        super().run()
        for pid in list(self.open):
            self._close(pid)


def main() -> None:
    fac = R._card_factory()
    rep = json.load(open(os.path.join(ROOT, "data", "video_games", "replay_report.json")))
    X, y, w = [], [], []
    games = 0
    for r in rep:
        if r.get("skipped"):
            continue
        g = json.load(open(os.path.join(HERE, "games", r["game"] + ".json"), encoding="utf-8"))
        if not any(a["kind"] in ("tech", "sardaukar", "swordmaster") for a in g["actions"]):
            continue           # commander / tech choices unknown in this game
        rp = LabelReplay(clean_game(g, fac))
        rp.run()
        win = rp.winner()
        for x, lab, pid, frac in rp.rows:
            X.append(x)
            y.append(lab)
            w.append(frac * (WINNER_W if pid == win else 1.0))
        games += 1
        print(f"  {r['game']}: {len(rp.rows)} turns", flush=True)
    X, y, w = np.array(X), np.array(y), np.array(w)
    counts = collections.Counter(OBJECTIVES[i] for i in y)
    print(f"{games} games, {len(y)} agent turns: {dict(counts)}")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    nv = len(y) // 5
    va, tr = idx[:nv], idx[nv:]
    m = ObjectiveModel()
    fit = m.fit(X[tr], y[tr], w[tr])
    P = np.array([m.probs(x) for x in X[va]])
    acc = float((P.argmax(1) == y[va]).mean())
    base = float((y[va] == np.bincount(y[tr]).argmax()).mean())
    nll = float(-np.log(P[np.arange(len(va)), y[va]] + 1e-9).mean())
    prior = np.bincount(y[tr], minlength=len(OBJECTIVES)) / len(tr)
    nll0 = float(-np.log(prior[y[va]] + 1e-9).mean())
    print(f"held-out: accuracy {acc:.3f} (always-'{OBJECTIVES[np.bincount(y[tr]).argmax()]}' "
          f"{base:.3f}), log-loss {nll:.3f} (prior only {nll0:.3f})")
    m = ObjectiveModel()
    m.fit(X, y, w)
    m.save()
    print("saved config/objective_model.npz")


if __name__ == "__main__":
    main()
