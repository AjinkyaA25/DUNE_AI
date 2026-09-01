#!/usr/bin/env python
"""
Self-play training loop for the Dune AI value network.

Each iteration:
  1. generate self-play games with the current best agent (+ exploration + book)
  2. refit the ValueModel on a replay buffer of recent shards
  3. arena: GreedyValueAgent(new) vs current best; promote if win-rate high enough
  4. checkpoint model + metrics

Usage:
  python train.py --iterations 20 --games-per-iter 400 --workers 8 --players 4
  python train.py --iterations 3 --games-per-iter 150 --workers 4   # quick smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import time

from src.ai.value_model import ValueModel
from src.ai.agents import GreedyValueAgent, HeuristicAgent, make_agent
from src.ai.opening_book import OpeningBook
from src.selfplay.generate import generate_selfplay, load_shards
from src.selfplay.arena import head_to_head

MODELS_DIR = "models"
DATA_DIR = "data/selfplay"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--games-per-iter", type=int, default=400)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--replay-k", type=int, default=4, help="shards kept in buffer")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--promote-winrate", type=float, default=0.0,
                    help="promote if a_vs_fair exceeds this (0 -> beat fair share)")
    ap.add_argument("--arena-games", type=int, default=160)
    ap.add_argument("--no-book", action="store_true")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    use_book = not args.no_book
    book = OpeningBook([]) if args.no_book else OpeningBook.default()

    metrics_path = os.path.join(MODELS_DIR, "metrics.csv")
    new_metrics = not os.path.exists(metrics_path)
    mf = open(metrics_path, "a", newline="")
    mw = csv.writer(mf)
    if new_metrics:
        mw.writerow(["iter", "gen_agent", "samples", "val_logloss",
                     "a_vs_fair", "promoted", "seconds"])

    best_spec = f"heuristic:T{args.temperature}"   # bootstrap generator
    best_model_path = None

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        print(f"\n=== iteration {it}/{args.iterations}  (gen agent: {best_spec}) ===")

        man = generate_selfplay(
            n_games=args.games_per_iter, agent_spec=best_spec,
            num_players=args.players, workers=args.workers, out_dir=DATA_DIR,
            base_seed=it * 100_000, use_book=use_book, shard_tag=f"it{it:02d}",
        )
        print(f"  generated {man['n_samples']} samples "
              f"({man['truncated_games']} truncated) in {man['seconds']}s "
              f"pos_rate={man['positive_rate']:.3f}")

        X, y, w = load_shards(DATA_DIR, last_k=args.replay_k)
        model = ValueModel(hidden=args.hidden, seed=it)
        hist = model.fit(X, y, sample_weight=w, epochs=40, lr=3e-3, seed=it)
        val_ll = hist["val_logloss"][-1]
        mpath = os.path.join(MODELS_DIR, f"value_v{it:02d}.npz")
        model.save(mpath)
        print(f"  trained model -> {mpath}  val_logloss={val_ll:.4f}")

        cand_model = model

        def make_cand():
            return GreedyValueAgent(model=cand_model, temperature=0.0,
                                    opening_book=book)

        if best_model_path is None:
            def make_prev():
                return HeuristicAgent(opening_book=book)
        else:
            prev_model = ValueModel.load(best_model_path)

            def make_prev():
                return GreedyValueAgent(model=prev_model, temperature=0.0,
                                        opening_book=book)

        res = head_to_head(make_cand, make_prev, n_games=args.arena_games,
                           num_players=args.players)
        a_vs_fair = res["a_vs_fair"]
        promote = a_vs_fair > (args.promote_winrate or 1.05)
        print(f"  arena vs {'heuristic' if best_model_path is None else 'prev model'}"
              f": winrate={res['a_winrate']:.3f}  a_vs_fair={a_vs_fair:.2f}  "
              f"-> {'PROMOTE' if promote else 'keep previous'}")

        if promote:
            best_model_path = mpath
            best_spec = f"value:{mpath}:T{args.temperature}"
            model.save(os.path.join(MODELS_DIR, "value_best.npz"))

        mw.writerow([it, best_spec, man["n_samples"], f"{val_ll:.4f}",
                     f"{a_vs_fair:.3f}", int(promote), round(time.time() - t0, 1)])
        mf.flush()

    mf.close()
    print(f"\nDone. Best model: {best_model_path or '(heuristic still best)'}")


if __name__ == "__main__":
    main()
