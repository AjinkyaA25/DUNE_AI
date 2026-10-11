#!/usr/bin/env python
"""
Train the value + policy nets on search self-play and the video games.

Data
  data/search_selfplay/*.npz   RoundSearchAgent self-play (Bloodlines):
                               the policy learns to pick the move the
                               search picked; the value net learns who won
  data/video_games/*.npz       replayed human games (Uprising+Bloodlines),
                               weighted up by --video-weight
Models
  warm-started from --init-dir (value_best / policy_best), written to
  --out-dir as value_<tag>.npz / policy_<tag>.npz; never overwrites the
  current best.
Test
  Bloodlines arena, the new agent in one rotating seat vs three of each
  opponent (current best trained agent, heuristic); parallel.

Usage:
  python train_search.py --tag s1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from src.ai.policy_model import PolicyModel
from src.ai.value_model import ValueModel
from src.selfplay.generate import load_policy_shards, load_shards


def _arena_game(args):
    g, cand, opp, base_seed = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    seat = g % 4
    agents = [make_agent(cand if s == seat else opp, seed=g * 4 + s) for s in range(4)]
    gs = setup_game(num_players=4, seed=base_seed + g, neutral_leaders=True,
                    use_bloodlines=True)
    n = 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    vp = [p.victory_points for p in gs.players]
    win = [gs.winner] if gs.winner is not None else \
        [i for i, v in enumerate(vp) if v == max(vp)]
    return (1.0 / len(win) if seat in win else 0.0), gs.round


def arena(cand: str, opp: str, n: int, workers: int, base_seed: int) -> dict:
    with Pool(workers) as pool:
        res = pool.map(_arena_game, [(g, cand, opp, base_seed) for g in range(n)],
                       chunksize=1)
    p = sum(r[0] for r in res) / n
    se = (p * (1 - p) / n) ** 0.5
    return {"vs": opp, "games": n, "win_share": round(p, 3),
            "ci95": [round(p - 1.96 * se, 3), round(p + 1.96 * se, 3)],
            "avg_rounds": round(sum(r[1] for r in res) / n, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--search-dir", default="data/search_selfplay")
    ap.add_argument("--video-dir", default="data/video_games")
    ap.add_argument("--video-weight", type=float, default=4.0)
    ap.add_argument("--init-dir", default="models_policy2")
    ap.add_argument("--out-dir", default="models_search")
    ap.add_argument("--value-epochs", type=int, default=20)
    ap.add_argument("--policy-epochs", type=int, default=12)
    ap.add_argument("--arena-games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    sX, sy, sw, sPA, sPM, sPCI = load_policy_shards(args.search_dir)
    vX, vy, vw, vPA, vPM, vPCI = load_policy_shards(args.video_dir)
    vw = vw * args.video_weight
    print(f"search self-play: {len(sy)} decisions   video games: {len(vy)} "
          f"decisions (x{args.video_weight} weight)", flush=True)
    X = np.concatenate([sX, vX]); y = np.concatenate([sy, vy])
    w = np.concatenate([sw, vw])
    PA = np.concatenate([sPA, vPA]); PM = np.concatenate([sPM, vPM])
    PCI = np.concatenate([sPCI, vPCI])

    value = ValueModel.load(os.path.join(args.init_dir, "value_best.npz"))
    hist = value.fit(X, y, sample_weight=w, epochs=args.value_epochs, lr=3e-3, seed=1)
    v_out = os.path.join(args.out_dir, f"value_{args.tag}.npz")
    value.save(v_out)
    print(f"value net  -> {v_out}  val_logloss={hist['val_logloss'][-1]:.4f}", flush=True)

    # policy: imitate the search's choice (and the humans' moves); plain
    # imitation weights - the search already did the credit assignment
    policy = PolicyModel.load(os.path.join(args.init_dir, "policy_best.npz"))
    ph = policy.fit(X, PA, PM, PCI, w, epochs=args.policy_epochs, seed=1)
    p_out = os.path.join(args.out_dir, f"policy_{args.tag}.npz")
    policy.save(p_out)
    print(f"policy net -> {p_out}  val_acc={ph.get('val_acc', ['?'])[-1]}", flush=True)

    cand = f"value:{v_out}:{p_out}"
    prev = (f"value:{os.path.join(args.init_dir, 'value_best.npz')}:"
            f"{os.path.join(args.init_dir, 'policy_best.npz')}")
    report = {"tag": args.tag, "search_decisions": int(len(sy)),
              "video_decisions": int(len(vy)), "arena": []}
    for opp in (prev, "heuristic"):
        r = arena(cand, opp, args.arena_games, args.workers, 90_000)
        report["arena"].append(r)
        print(f"arena vs {opp[:40]}: {r}", flush=True)
    report["minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(args.out_dir, f"report_{args.tag}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
