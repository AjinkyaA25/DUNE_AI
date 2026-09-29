#!/usr/bin/env python
"""
Fine-tune the current best value/policy nets on games recorded from YOUR play.

`play_game.py` saves every decision you make (not the AI's) into
`data/human_games/*.npz`, in exactly the same shard schema self-play uses
(see src/selfplay/generate.py `_play_one` / `load_shards` / `load_policy_shards`).
This script warm-starts from the current best checkpoints and fine-tunes on
a mix of your games plus a slice of the self-play replay buffer (so a
handful of human decisions doesn't overfit the net into forgetting
everything it learned from self-play), then arena-tests the result against
the current best. It never overwrites value_best.npz/policy_best.npz itself
-- promotion is a manual `cp`, printed at the end, so a bad fine-tune can't
silently regress the agent you already have.

Usage:
  python play_game.py --players 4 --human 0      # play a game or several first
  python train_from_human.py                      # then fine-tune on them
  python train_from_human.py --human-weight 8 --arena-games 120
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.ai.value_model import ValueModel
from src.ai.policy_model import PolicyModel, awr_weights
from src.ai.agents import GreedyValueAgent
from src.ai.opening_book import OpeningBook
from src.selfplay.generate import load_shards, load_policy_shards
from src.selfplay.arena import head_to_head


def _subsample(rng, n_keep, *arrays):
    n = len(arrays[0])
    if n <= n_keep:
        return arrays
    idx = rng.choice(n, n_keep, replace=False)
    return tuple(a[idx] for a in arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-dir", default="data/human_games",
                    help="where play_game.py saved your decisions")
    ap.add_argument("--base-data-dir", default="data/selfplay_policy2",
                    help="self-play replay shards to blend in so the net "
                         "doesn't forget everything but your handful of "
                         "games")
    ap.add_argument("--base-replay-k", type=int, default=2,
                    help="self-play shards to draw the blend from")
    ap.add_argument("--base-multiplier", type=int, default=40,
                    help="cap the blended self-play sample count at this "
                         "many times your human sample count, so your games "
                         "are a real fraction of the fine-tune signal "
                         "instead of drowned out 1000-to-1 by the replay "
                         "buffer")
    ap.add_argument("--models-dir", default="models_policy2")
    ap.add_argument("--out-tag", default="human",
                    help="fine-tuned checkpoints save as value_<tag>.npz / "
                         "policy_<tag>.npz, next to (not over) the current "
                         "best")
    ap.add_argument("--human-weight", type=float, default=6.0,
                    help="extra multiplier on your games' sample weight, on "
                         "top of the multiplier implied by --base-multiplier")
    ap.add_argument("--value-epochs", type=int, default=12)
    ap.add_argument("--policy-epochs", type=int, default=8)
    ap.add_argument("--awr-beta", type=float, default=0.4)
    ap.add_argument("--heuristic-weight", type=float, default=0.35)
    ap.add_argument("--policy-weight", type=float, default=0.25)
    ap.add_argument("--arena-games", type=int, default=80)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    vbest = os.path.join(args.models_dir, "value_best.npz")
    pbest = os.path.join(args.models_dir, "policy_best.npz")
    if not os.path.exists(vbest):
        raise SystemExit(f"no base model at {vbest} -- run train.py first "
                         f"to get a starting checkpoint to fine-tune")

    try:
        hX, hy, hw = load_shards(args.human_dir)
    except FileNotFoundError:
        raise SystemExit(f"no games recorded in {args.human_dir} yet -- "
                         f"play some first: python play_game.py --players 4 --human 0")
    print(f"your games:   {len(hy)} decisions from {args.human_dir}")
    hw = hw * args.human_weight

    bX, by, bw = load_shards(args.base_data_dir, last_k=args.base_replay_k)
    bX, by, bw = _subsample(rng, len(hy) * args.base_multiplier, bX, by, bw)
    print(f"self-play mix: {len(by)} decisions from {args.base_data_dir} "
          f"(last {args.base_replay_k} shards, capped at {args.base_multiplier}x your count)")

    X = np.concatenate([hX, bX]); y = np.concatenate([hy, by]); w = np.concatenate([hw, bw])

    model = ValueModel.load(vbest)
    model.fit(X, y, sample_weight=w, epochs=args.value_epochs, lr=3e-3, seed=args.seed)
    v_out = os.path.join(args.models_dir, f"value_{args.out_tag}.npz")
    model.save(v_out)
    print(f"fine-tuned value net  -> {v_out}")

    cand_policy = None
    p_out = None
    if os.path.exists(pbest):
        try:
            hX2, hy2, hw2, hPA, hPM, hPCI = load_policy_shards(args.human_dir)
        except FileNotFoundError:
            hPA = None
        if hPA is None:
            print("no policy-head data in your human games -- skipping policy fine-tune")
        else:
            hw2 = hw2 * args.human_weight
            try:
                bX2, by2, bw2, bPA, bPM, bPCI = load_policy_shards(
                    args.base_data_dir, last_k=args.base_replay_k)
                bX2, by2, bw2, bPA, bPM, bPCI = _subsample(
                    rng, len(hy2) * args.base_multiplier,
                    bX2, by2, bw2, bPA, bPM, bPCI)
                pX = np.concatenate([hX2, bX2]); pRet = np.concatenate([hy2, by2])
                pW = np.concatenate([hw2, bw2])
                PA = np.concatenate([hPA, bPA]); PM = np.concatenate([hPM, bPM])
                PCI = np.concatenate([hPCI, bPCI])
            except FileNotFoundError:
                pX, pRet, pW, PA, PM, PCI = hX2, hy2, hw2, hPA, hPM, hPCI

            aw, madv = awr_weights(model, pX, pRet, pW, beta=args.awr_beta)
            policy = PolicyModel.load(pbest)
            policy.fit(pX, PA, PM, PCI, aw, epochs=args.policy_epochs, seed=args.seed)
            p_out = os.path.join(args.models_dir, f"policy_{args.out_tag}.npz")
            policy.save(p_out)
            cand_policy = policy
            print(f"fine-tuned policy head -> {p_out}  mean_adv={madv:.3f}")

    # ── arena: fine-tuned candidate vs. the current best (never auto-promotes) ──
    book = OpeningBook.default()
    prev_model = ValueModel.load(vbest)
    prev_policy = PolicyModel.load(pbest) if os.path.exists(pbest) else None

    def make_cand():
        return GreedyValueAgent(model=model, opening_book=book,
                                heuristic_weight=args.heuristic_weight,
                                policy=cand_policy, policy_weight=args.policy_weight)

    def make_prev():
        return GreedyValueAgent(model=prev_model, opening_book=book,
                                heuristic_weight=args.heuristic_weight,
                                policy=prev_policy, policy_weight=args.policy_weight)

    res = head_to_head(make_cand, make_prev, n_games=args.arena_games,
                       num_players=args.players)
    print(f"\narena: fine-tuned vs. current best over {args.arena_games} games")
    print(f"  winrate={res['a_winrate']:.3f}  a_vs_fair={res['a_vs_fair']:.2f}x")
    if res["a_vs_fair"] > 1.0:
        print("  -> looks stronger. To promote it to the live agent:")
    else:
        print("  -> did NOT beat the current best. Not promoting automatically.")
    print(f"       cp {v_out} {vbest}")
    if p_out:
        print(f"       cp {p_out} {pbest}")


if __name__ == "__main__":
    main()
