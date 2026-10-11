#!/usr/bin/env python
"""
Record what a strong search thinks, position by position, for fitting the
heuristic's (situational) weights to it - expert iteration with an
interpretable student.

Four RoundSearchAgents (default search:K5:M48:B - 48 playouts per candidate,
buys searched too) play full Bloodlines games. At every decision the search
evaluated we keep:
  gs      a clone of the position before the move
  pid     who was deciding
  cands   the search's candidate moves (the heuristic's top K)
  values  mean playout result per candidate (≈ win chance + 0.02 * VP margin)
  chosen  index of the move it played
plus the game's winner and final VP. One pickle per game in
data/search_positions/, so the run can be stopped and resumed any time.

  python -m src.selfplay.search_positions --games 120 --workers 24
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import time
from multiprocessing import Pool

OUT = os.path.join("data", "search_positions")


def play_one(args):
    seed, spec = args
    path = os.path.join(OUT, f"g{seed}.pkl")
    if os.path.exists(path):
        return path, 0, 0.0
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    t0 = time.time()
    agents = [make_agent(spec, seed=seed * 4 + s) for s in range(4)]
    gs = setup_game(num_players=4, seed=seed, neutral_leaders=True, use_bloodlines=True)
    rec, moves = [], 0
    while not gs.game_over and moves < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        before = gs.clone()
        a = agents[pid].select_action(gs, pid, gs.get_valid_actions(pid))
        info = getattr(agents[pid], "last", None)
        if info:
            rec.append({"gs": before, "pid": pid, "round": gs.round,
                        "cands": info["cands"], "values": info["scores"],
                        "chosen": info["chosen"]})
        gs.step(a)
        moves += 1
    if not gs.game_over:
        gs.check_victory_conditions()
    vp = [p.victory_points for p in gs.players]
    winner = gs.winner if gs.winner is not None else max(range(4), key=lambda i: vp[i])
    for r in rec:
        r["won"] = r["pid"] == winner
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"seed": seed, "spec": spec, "winner": winner, "vp": vp,
                     "rounds": gs.round, "positions": rec}, f)
    os.replace(tmp, path)
    return path, len(rec), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="search:K5:M48:B")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--base-seed", type=int, default=500_000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(args.base_seed + i, args.spec) for i in range(args.games)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (path, n, secs) in enumerate(pool.imap_unordered(play_one, jobs), 1):
            have = len(glob.glob(os.path.join(OUT, "*.pkl")))
            print(f"[{i}/{len(jobs)}] {os.path.basename(path)}: {n} positions, "
                  f"{secs / 60:.1f} min  ({have} games on disk, "
                  f"{(time.time() - t0) / 60:.0f} min elapsed)", flush=True)


if __name__ == "__main__":
    main()
