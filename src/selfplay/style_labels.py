#!/usr/bin/env python
"""
Low-noise style labels: from a situation, how well does each playstyle do?

Plays style-league games (four fixated heuristic styles, seat order cycling)
and, at a few randomly chosen round starts, stops for one seat and plays the
rest of the game out M times with every style in config/styles/ (common
random numbers: style k and style j are judged on the same M re-sampled
worlds; opponents are the balanced heuristic, as in StyleSearchAgent). With
M=128 a style's mean is known to about +-0.04, against +-0.12 at the M=16 of
the style-search agent's own logs, so the best style in a situation is
actually knowable and a selector can be trained on it
(src/ai/style_select.py).

Each label row: game, round, seat, the seat's own style, the situation
features (situation / situation_rich) and the full net encoding, and the
mean / sd of every style's playout results.

  python -m src.selfplay.style_labels --games 500 --per-game 4 --m 128 \
      --out data/style_league/labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np


def label(gs, pid: int, m: int, rng, styles, opp) -> dict:
    from src.ai.determinize import determinize
    from src.ai.search import end_value
    names = list(styles)
    seeds = rng.integers(0, 2**31 - 1, size=m)
    vals = np.zeros((len(names), m))
    for i, k in enumerate(names):
        for j in range(m):
            g = determinize(gs, pid, np.random.default_rng(int(seeds[j])))
            n = 0
            while not g.game_over and n < 3000:
                cur = g.player_in_reveal_buy
                if cur is None:
                    cur = g.get_current_player_id()
                ag = styles[k] if cur == pid else opp
                g.step(ag.select_action(g, cur, g.get_valid_actions(cur)))
                n += 1
            vals[i, j] = end_value(g, pid)
    return {"means": {k: round(float(v), 4) for k, v in zip(names, vals.mean(1))},
            "sds": {k: round(float(v), 4) for k, v in zip(names, vals.std(1))}}


def play(args):
    g, seed0, per_game, m = args
    from src.ai.agents import HeuristicAgent, make_agent
    from src.ai.features import encode_state
    from src.ai.style_search import load_styles
    from src.ai.style_select import situation_rich
    from src.data.card_definitions import setup_game
    from src.selfplay.style_league import PERMS, STYLES, situation, spec_for
    rng = np.random.default_rng(seed0 + 7919 * g)
    perm = PERMS[g % 24]
    names = [STYLES[perm[s]] for s in range(4)]
    agents = [make_agent(spec_for("heuristic", n), seed=g * 4 + s) for s, n in enumerate(names)]
    allst = load_styles()
    roll = {k: HeuristicAgent(seed=g, tuning=t) for k, t in allst.items()}
    opp = HeuristicAgent(seed=g, tuning=allst["balanced"])
    # which (round, seat) to label: distinct rounds 1..9, random seats
    rounds = sorted(rng.choice(np.arange(1, 10), size=per_game, replace=False).tolist())
    todo = {r: int(rng.integers(4)) for r in rounds}
    gs = setup_game(num_players=4, seed=seed0 + g, neutral_leaders=True, use_bloodlines=True)
    rows, seen, n, t0 = [], 0, 0, time.time()
    while not gs.game_over and n < 3000:
        if gs.round != seen and gs.phase.value == "player_turns" and gs.player_in_reveal_buy is None:
            seen = gs.round
            if gs.round in todo:
                s = todo[gs.round]
                r = {"game": g, "round": gs.round, "seat": s, "style": names[s],
                     "x": situation(gs, s), "xr": situation_rich(gs, s),
                     "enc": [round(float(v), 4) for v in encode_state(gs, s)]}
                r.update(label(gs, s, m, rng, roll, opp))
                r["m"] = m
                rows.append(r)
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    vp = [p.victory_points for p in gs.players]
    win = gs.winner if gs.winner is not None else max(range(4), key=lambda s: vp[s])
    for r in rows:
        r.update({"final_vp": vp[r["seat"]], "won": int(win == r["seat"]),
                  "seconds": round(time.time() - t0, 1)})
    return g, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--per-game", type=int, default=4)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=600_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):                      # resume (per game)
        done = {json.loads(l)["game"] for l in open(a.out)}
    todo = [(g, a.seed, a.per_game, a.m) for g in range(a.games) if g not in done]
    t0 = time.time()
    with Pool(a.workers) as pool, open(a.out, "a") as f:
        for i, (g, rows) in enumerate(pool.imap_unordered(play, todo)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"[{i + 1}/{len(todo)}] games, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
