#!/usr/bin/env python
"""
Search-based self-play ("expert iteration" data generation).

Four RoundSearchAgents play full Bloodlines games (blank leaders). At every
decision the search actually evaluated (agent placement, reveal, deploy) we
record:
  X    state features of the deciding player
  PA   features of the search's candidate moves (up to 5), PM mask
  PCI  index of the move the search chose     -> policy imitation target
  PS   the search's mean playout score per candidate (for soft targets)
  y    discounted win target of that player    -> value target (same as
       self-play: GAMMA ** rounds-until-the-win, 0 for losers)
  w    sample weight
Shards land in data/search_selfplay/ in the self-play shard format, so
train.py / train_from_human.py loaders read them directly.

Runs until stopped, writing one shard per `--games-per-shard` games.

Usage:
  python -m src.selfplay.search_selfplay --workers 12 --spec search:K5:M8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

GAMMA = 0.90
KMAX = 20     # PA padded to the policy head's candidate width


def play_one(args):
    game_idx, spec, base_seed = args
    from src.ai.action_features import encode_action
    from src.ai.agents import make_agent
    from src.ai.features import encode_state
    from src.data.card_definitions import setup_game

    seed = base_seed + game_idx
    agents = [make_agent(spec, seed=seed * 4 + s) for s in range(4)]
    gs = setup_game(num_players=4, seed=seed, neutral_leaders=True,
                    use_bloodlines=True)
    rec = []
    moves = 0
    while not gs.game_over and moves < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        ag = agents[pid]
        feats = None
        if any(a.action_type.value in ("agent_turn", "reveal_turn", "resolve_deploy")
               for a in valid):
            feats = encode_state(gs, pid)
        a = ag.select_action(gs, pid, valid)
        info = getattr(ag, "last", None)
        if info and feats is not None:
            sc = info["scores"]
            lo, hi = min(sc), max(sc)
            span = (hi - lo) or 1.0
            PA = np.stack([encode_action(gs, pid, c, (v - lo) / span)
                           for c, v in zip(info["cands"], sc)])
            rec.append((feats, pid, gs.round, PA, info["chosen"], sc))
        gs.step(a)
        moves += 1
    if not gs.game_over:
        gs.check_victory_conditions()
    vp = [p.victory_points for p in gs.players]
    winner = gs.winner if gs.winner is not None else int(np.argmax(vp))
    return rec, winner, gs.round, vp


def write_shard(results, out_dir: str, tag: str) -> dict:
    from src.ai.action_features import ACTION_FEATURE_DIM
    X, y, w, PA, PM, PCI, PS = [], [], [], [], [], [], []
    rounds = []
    for rec, winner, rnd, vp in results:
        rounds.append(rnd)
        speed_w = 1.0 + max(0, 8 - rnd) * 0.3
        for feats, pid, r, pa, ci, sc in rec:
            X.append(feats)
            y.append(GAMMA ** max(0, rnd - r) if pid == winner else 0.0)
            w.append(speed_w)
            k = min(len(pa), KMAX)
            pad = np.zeros((KMAX, ACTION_FEATURE_DIM), np.float32)
            pad[:k] = pa[:k]
            m = np.zeros(KMAX, np.float32)
            m[:k] = 1.0
            s = np.full(KMAX, np.nan, np.float32)
            s[:k] = sc[:k]
            PA.append(pad); PM.append(m); PCI.append(ci); PS.append(s)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.npz")
    np.savez_compressed(path, X=np.stack(X).astype(np.float32),
                        y=np.array(y, np.float32), w=np.array(w, np.float32),
                        PA=np.stack(PA), PM=np.stack(PM),
                        PCI=np.array(PCI, np.int32), PS=np.stack(PS))
    man = {"shard": os.path.basename(path), "games": len(results),
           "samples": len(y), "avg_rounds": round(float(np.mean(rounds)), 2),
           "switch_rate": round(float(np.mean([c != 0 for c in PCI])), 3)}
    with open(path + ".json", "w") as f:
        json.dump(man, f)
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="search:K5:M8")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--games-per-shard", type=int, default=48)
    ap.add_argument("--shards", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--out", default="data/search_selfplay")
    ap.add_argument("--base-seed", type=int, default=0,
                    help="0 = continue after the shards already on disk")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    done = len([f for f in os.listdir(args.out) if f.endswith(".npz")])
    base = args.base_seed or 1_000_000 + done * 10_000
    n = 0
    with Pool(args.workers) as pool:
        while args.shards == 0 or n < args.shards:
            t0 = time.time()
            jobs = [(i, args.spec, base) for i in range(args.games_per_shard)]
            res = pool.map(play_one, jobs, chunksize=1)
            man = write_shard(res, args.out, f"s{done + n:04d}_{base}")
            man["minutes"] = round((time.time() - t0) / 60, 1)
            print(json.dumps(man), flush=True)
            n += 1
            base += 10_000


if __name__ == "__main__":
    main()
