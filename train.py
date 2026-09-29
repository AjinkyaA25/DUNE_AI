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
from src.ai.policy_model import PolicyModel, awr_weights
from src.ai.agents import GreedyValueAgent, HeuristicAgent, make_agent
from src.ai.opening_book import OpeningBook
from src.selfplay.generate import (generate_selfplay, load_shards,
                                   load_policy_shards)
from src.selfplay.arena import head_to_head

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--data-dir", default="data/selfplay")
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
    ap.add_argument("--bloodlines", action="store_true",
                    help="self-play and arena games use the Bloodlines expansion")
    ap.add_argument("--rogue-spec", type=str, default=None,
                    help="seat a sparring-partner agent in each self-play game "
                         "(e.g. 'bully:T0.4' — the crucial-combat exploiter). "
                         "Its own decision points are excluded from training "
                         "data; it only shapes how the value seats are scored.")
    ap.add_argument("--rogue-seats", type=int, default=1,
                    help="how many seats the rogue occupies (rotated per game)")
    ap.add_argument("--no-policy", action="store_true",
                    help="disable the AWR policy head (value net only)")
    ap.add_argument("--policy-weight", type=float, default=0.25,
                    help="GreedyValueAgent weight on log pi(a|s) from the policy "
                         "head (the prior that can see compounding moves a "
                         "1-ply value-max cannot)")
    ap.add_argument("--awr-beta", type=float, default=0.4,
                    help="AWR temperature: lower -> sharper up-weighting of "
                         "above-expectation trajectories' actions")
    ap.add_argument("--policy-cap", type=int, default=220_000,
                    help="max decisions used to fit the policy head per iter")
    ap.add_argument("--heuristic-weight", type=float, default=0.35,
                    help="GreedyValueAgent's pull back toward the flat heuristic "
                         "prior (default 0.35) -- lower it to let the trained "
                         "value net's own predictions drive action selection "
                         "more directly, e.g. when the net was trained on a "
                         "signal (like a discounted/speed-aware target) the "
                         "heuristic itself has no notion of")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="disable warm-starting: re-init both nets from scratch "
                         "every iteration (the old behaviour). Default is warm "
                         "-- keep training the SAME value/policy objects (Adam "
                         "momentum included) across iterations instead of "
                         "discarding them, rolling back to the last PROMOTED "
                         "checkpoint after a rejected iteration so a bad "
                         "candidate can't compound into future training.")
    ap.add_argument("--value-epochs", type=int, default=40)
    ap.add_argument("--value-epochs-warm", type=int, default=20,
                    help="epochs for iterations 2+ once warm-started (needs "
                         "less gradient work than a from-scratch fit)")
    ap.add_argument("--policy-epochs", type=int, default=15)
    ap.add_argument("--policy-epochs-warm", type=int, default=10)
    args = ap.parse_args()
    MODELS_DIR = args.models_dir
    DATA_DIR = args.data_dir

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
    best_policy_path = None
    warm = not args.no_warm_start
    cur_model = None            # kept alive across iterations when warm=True
    cur_policy = None

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        print(f"\n=== iteration {it}/{args.iterations}  (gen agent: {best_spec}) ===")

        man = generate_selfplay(
            n_games=args.games_per_iter, agent_spec=best_spec,
            num_players=args.players, workers=args.workers, out_dir=DATA_DIR,
            base_seed=it * 100_000, use_book=use_book, shard_tag=f"it{it:02d}",
            rogue_spec=args.rogue_spec, rogue_seats=args.rogue_seats,
            record_policy=not args.no_policy,
            use_bloodlines=args.bloodlines,
        )
        print(f"  generated {man['n_samples']} samples "
              f"({man['truncated_games']} truncated) in {man['seconds']}s "
              f"pos_rate={man['positive_rate']:.3f}")

        X, y, w = load_shards(DATA_DIR, last_k=args.replay_k)
        if warm and cur_model is not None:
            model = cur_model                     # continue training the SAME net
            v_epochs = args.value_epochs_warm
        else:
            model = ValueModel(hidden=args.hidden, seed=it)
            v_epochs = args.value_epochs
        hist = model.fit(X, y, sample_weight=w, epochs=v_epochs, lr=3e-3, seed=it)
        val_ll = hist["val_logloss"][-1]
        mpath = os.path.join(MODELS_DIR, f"value_v{it:02d}.npz")
        model.save(mpath)
        print(f"  trained model -> {mpath}  val_logloss={val_ll:.4f}"
              f"{'  (warm)' if warm and cur_model is model else ''}")
        if warm:
            cur_model = model

        cand_model = model
        cand_policy = None
        if not args.no_policy:
            try:
                pX, pRet, pW, PA, PM, PCI = load_policy_shards(
                    DATA_DIR, last_k=args.replay_k)
                if len(pX) > args.policy_cap:                 # keep it a few min/iter
                    import numpy as _np
                    keep = _np.random.default_rng(it).choice(
                        len(pX), args.policy_cap, replace=False)
                    pX, pRet, pW = pX[keep], pRet[keep], pW[keep]
                    PA, PM, PCI = PA[keep], PM[keep], PCI[keep]
                aw, madv = awr_weights(model, pX, pRet, pW, beta=args.awr_beta)
                if warm and cur_policy is not None:
                    policy = cur_policy
                    p_epochs = args.policy_epochs_warm
                else:
                    policy = PolicyModel(hidden=args.hidden, seed=it)
                    p_epochs = args.policy_epochs
                ph = policy.fit(pX, PA, PM, PCI, aw, epochs=p_epochs, seed=it)
                ppath = os.path.join(MODELS_DIR, f"policy_v{it:02d}.npz")
                policy.save(ppath)
                cand_policy = policy
                if warm:
                    cur_policy = policy
                print(f"  trained policy -> {ppath}  "
                      f"val_logloss={ph['val_logloss'][-1]:.4f} "
                      f"acc={ph['val_acc'][-1]:.3f} mean_adv={madv:.3f}")
            except FileNotFoundError as e:
                print(f"  policy head skipped: {e}")

        def make_cand():
            return GreedyValueAgent(model=cand_model, temperature=0.0,
                                    opening_book=book,
                                    heuristic_weight=args.heuristic_weight,
                                    policy=cand_policy,
                                    policy_weight=args.policy_weight)

        if best_model_path is None:
            def make_prev():
                return HeuristicAgent(opening_book=book)
        else:
            prev_model = ValueModel.load(best_model_path)
            prev_policy = (PolicyModel.load(best_policy_path)
                           if best_policy_path else None)

            def make_prev():
                return GreedyValueAgent(model=prev_model, temperature=0.0,
                                        opening_book=book,
                                        heuristic_weight=args.heuristic_weight,
                                        policy=prev_policy,
                                        policy_weight=args.policy_weight)

        res = head_to_head(make_cand, make_prev, n_games=args.arena_games,
                           num_players=args.players,
                           use_bloodlines=args.bloodlines)
        a_vs_fair = res["a_vs_fair"]
        promote = a_vs_fair > (args.promote_winrate or 1.05)
        print(f"  arena vs {'heuristic' if best_model_path is None else 'prev model'}"
              f": winrate={res['a_winrate']:.3f}  a_vs_fair={a_vs_fair:.2f}  "
              f"-> {'PROMOTE' if promote else 'keep previous'}")

        if promote:
            best_model_path = mpath
            spec = f"value:{mpath}"
            if cand_policy is not None:
                best_policy_path = os.path.join(MODELS_DIR, f"policy_v{it:02d}.npz")
                spec += f":{best_policy_path}"
                cand_policy.save(os.path.join(MODELS_DIR, "policy_best.npz"))
            spec += (f":T{args.temperature}:HW{args.heuristic_weight}"
                     f":PW{args.policy_weight}")
            best_spec = spec
            model.save(os.path.join(MODELS_DIR, "value_best.npz"))
        elif warm:
            # Rejected candidate: roll the LIVE training objects back to the
            # last promoted checkpoint so next iteration's warm start doesn't
            # keep fine-tuning from a worse-than-best point (a rejected net's
            # weights + Adam momentum would otherwise carry forward anyway).
            if best_model_path is not None:
                cur_model = ValueModel.load(best_model_path)
            if best_policy_path is not None:
                cur_policy = PolicyModel.load(best_policy_path)

        mw.writerow([it, best_spec, man["n_samples"], f"{val_ll:.4f}",
                     f"{a_vs_fair:.3f}", int(promote), round(time.time() - t0, 1)])
        mf.flush()

    mf.close()
    print(f"\nDone. Best model: {best_model_path or '(heuristic still best)'}")


if __name__ == "__main__":
    main()
