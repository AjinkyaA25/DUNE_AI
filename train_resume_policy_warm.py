#!/usr/bin/env python
"""
Resume the interrupted models_policy2 warm-start run from iteration 3/14.

Iterations 1-2 already completed and PROMOTED (see models_policy2/metrics.csv
and train_policy_warm_run1.log): value_v02.npz / policy_v02.npz are the
current-best checkpoints. This script re-creates train.py's main() loop with
the same default hyperparameters that produced that log (models-dir=
models_policy2, data-dir=data/selfplay_policy2, iterations=14, all other
args at their train.py defaults), but starts at iteration 3, seeding
best_model_path/best_policy_path/cur_model/cur_policy from the v02
checkpoints instead of from scratch.

Loading from the saved checkpoint (rather than resuming an in-memory Adam
optimizer state) matches exactly what train.py itself does on a rejected
iteration's rollback path -- Adam momentum is not persisted across a
ValueModel.load either way.
"""
from __future__ import annotations

import csv
import os
import time

from src.ai.value_model import ValueModel
from src.ai.policy_model import PolicyModel, awr_weights
from src.ai.agents import GreedyValueAgent, HeuristicAgent
from src.ai.opening_book import OpeningBook
from src.selfplay.generate import generate_selfplay, load_shards, load_policy_shards
from src.selfplay.arena import head_to_head

MODELS_DIR = "models_policy2"
DATA_DIR = "data/selfplay_policy2"
ITERATIONS = 14
START_ITER = 3
GAMES_PER_ITER = 400
PLAYERS = 4
REPLAY_K = 4
TEMPERATURE = 0.7
HIDDEN = 64
PROMOTE_WINRATE = 0.0
ARENA_GAMES = 160
POLICY_WEIGHT = 0.25
AWR_BETA = 0.4
POLICY_CAP = 220_000
HEURISTIC_WEIGHT = 0.35
VALUE_EPOCHS_WARM = 20
POLICY_EPOCHS_WARM = 10


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    book = OpeningBook.default()

    metrics_path = os.path.join(MODELS_DIR, "metrics.csv")
    new_metrics = not os.path.exists(metrics_path)
    mf = open(metrics_path, "a", newline="")
    mw = csv.writer(mf)
    if new_metrics:
        mw.writerow(["iter", "gen_agent", "samples", "val_logloss",
                     "a_vs_fair", "promoted", "seconds"])

    best_model_path = os.path.join(MODELS_DIR, "value_v02.npz")
    best_policy_path = os.path.join(MODELS_DIR, "policy_v02.npz")
    best_spec = (f"value:{best_model_path}:{best_policy_path}"
                 f":T{TEMPERATURE}:HW{HEURISTIC_WEIGHT}:PW{POLICY_WEIGHT}")
    cur_model = ValueModel.load(best_model_path)
    cur_policy = PolicyModel.load(best_policy_path)

    for it in range(START_ITER, ITERATIONS + 1):
        t0 = time.time()
        print(f"\n=== iteration {it}/{ITERATIONS}  (gen agent: {best_spec}) ===", flush=True)

        man = generate_selfplay(
            n_games=GAMES_PER_ITER, agent_spec=best_spec,
            num_players=PLAYERS, workers=max(1, (os.cpu_count() or 2) - 1),
            out_dir=DATA_DIR, base_seed=it * 100_000, use_book=True,
            shard_tag=f"it{it:02d}", rogue_spec=None, rogue_seats=1,
            record_policy=True,
        )
        print(f"  generated {man['n_samples']} samples "
              f"({man['truncated_games']} truncated) in {man['seconds']}s "
              f"pos_rate={man['positive_rate']:.3f}", flush=True)

        X, y, w = load_shards(DATA_DIR, last_k=REPLAY_K)
        model = cur_model
        hist = model.fit(X, y, sample_weight=w, epochs=VALUE_EPOCHS_WARM, lr=3e-3, seed=it)
        val_ll = hist["val_logloss"][-1]
        mpath = os.path.join(MODELS_DIR, f"value_v{it:02d}.npz")
        model.save(mpath)
        print(f"  trained model -> {mpath}  val_logloss={val_ll:.4f}  (warm)", flush=True)
        cur_model = model

        cand_model = model
        cand_policy = None
        try:
            pX, pRet, pW, PA, PM, PCI = load_policy_shards(DATA_DIR, last_k=REPLAY_K)
            if len(pX) > POLICY_CAP:
                import numpy as _np
                keep = _np.random.default_rng(it).choice(len(pX), POLICY_CAP, replace=False)
                pX, pRet, pW = pX[keep], pRet[keep], pW[keep]
                PA, PM, PCI = PA[keep], PM[keep], PCI[keep]
            aw, madv = awr_weights(model, pX, pRet, pW, beta=AWR_BETA)
            policy = cur_policy
            ph = policy.fit(pX, PA, PM, PCI, aw, epochs=POLICY_EPOCHS_WARM, seed=it)
            ppath = os.path.join(MODELS_DIR, f"policy_v{it:02d}.npz")
            policy.save(ppath)
            cand_policy = policy
            cur_policy = policy
            print(f"  trained policy -> {ppath}  "
                  f"val_logloss={ph['val_logloss'][-1]:.4f} "
                  f"acc={ph['val_acc'][-1]:.3f} mean_adv={madv:.3f}", flush=True)
        except FileNotFoundError as e:
            print(f"  policy head skipped: {e}", flush=True)

        def make_cand():
            return GreedyValueAgent(model=cand_model, temperature=0.0,
                                    opening_book=book,
                                    heuristic_weight=HEURISTIC_WEIGHT,
                                    policy=cand_policy,
                                    policy_weight=POLICY_WEIGHT)

        prev_model = ValueModel.load(best_model_path)
        prev_policy = PolicyModel.load(best_policy_path) if best_policy_path else None

        def make_prev():
            return GreedyValueAgent(model=prev_model, temperature=0.0,
                                    opening_book=book,
                                    heuristic_weight=HEURISTIC_WEIGHT,
                                    policy=prev_policy,
                                    policy_weight=POLICY_WEIGHT)

        res = head_to_head(make_cand, make_prev, n_games=ARENA_GAMES, num_players=PLAYERS)
        a_vs_fair = res["a_vs_fair"]
        promote = a_vs_fair > (PROMOTE_WINRATE or 1.05)
        print(f"  arena vs prev model: winrate={res['a_winrate']:.3f}  "
              f"a_vs_fair={a_vs_fair:.2f}  -> {'PROMOTE' if promote else 'keep previous'}", flush=True)

        if promote:
            best_model_path = mpath
            spec = f"value:{mpath}"
            if cand_policy is not None:
                best_policy_path = os.path.join(MODELS_DIR, f"policy_v{it:02d}.npz")
                spec += f":{best_policy_path}"
                cand_policy.save(os.path.join(MODELS_DIR, "policy_best.npz"))
            spec += f":T{TEMPERATURE}:HW{HEURISTIC_WEIGHT}:PW{POLICY_WEIGHT}"
            best_spec = spec
            model.save(os.path.join(MODELS_DIR, "value_best.npz"))
        else:
            if best_model_path is not None:
                cur_model = ValueModel.load(best_model_path)
            if best_policy_path is not None:
                cur_policy = PolicyModel.load(best_policy_path)

        mw.writerow([it, best_spec, man["n_samples"], f"{val_ll:.4f}",
                     f"{a_vs_fair:.3f}", int(promote), round(time.time() - t0, 1)])
        mf.flush()

    mf.close()
    print(f"\nDone. Best model: {best_model_path}", flush=True)


if __name__ == "__main__":
    main()
