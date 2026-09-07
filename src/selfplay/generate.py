"""
Parallel self-play data generation.

Plays N games with a given agent spec and writes training shards
(features X, win-label y, sample-weight w) as .npz to out_dir.
"""
from __future__ import annotations

import json
import os
import time
from typing import List

import numpy as np

from src.ai.agents import make_agent
from src.ai.opening_book import OpeningBook
from src.selfplay.runner import play_game

_WORKER_CFG: dict = {}


def _worker_init(cfg: dict) -> None:
    _WORKER_CFG.clear()
    _WORKER_CFG.update(cfg)
    OpeningBook.clear_cache()


def _play_one(game_idx: int):
    cfg = _WORKER_CFG
    n = cfg["num_players"]
    book = OpeningBook.default() if cfg["use_book"] else OpeningBook([])
    seed = cfg["base_seed"] + game_idx
    agents = {s: make_agent(cfg["agent_spec"], seed=seed * 4 + s,
                            opening_book=book)
              for s in range(n)}
    res = play_game(agents, num_players=n, seed=seed, record=True,
                    use_choam=cfg["use_choam"])
    if not res.feats:
        return None
    X = np.stack(res.feats).astype(np.float32)
    # A flat 1.0/0.0 win label carries no notion of WHEN the win happens —
    # a 1-ply value-maximizer has no reason to prefer a state that wins
    # soon over one that wins eventually, since both regress to the same
    # target. Discount the winner's own samples by how many rounds still
    # separate them from the actual win (GAMMA=0.90): a state one round
    # from winning scores ~0.90, five rounds out ~0.59, nine rounds out
    # (an R1 state that wins at R10) ~0.39. This makes "close to a fast
    # win" and "close to a slow win" genuinely different targets, so the
    # learned value directly rewards ending the game sooner, not just
    # eventually. Losing samples are unaffected (label stays 0).
    GAMMA = 0.90
    y = np.array([
        GAMMA ** max(0, res.rounds_played - rnd) if pid == res.winner else 0.0
        for pid, rnd in zip(res.feat_pids, res.feat_rounds)
    ], dtype=np.float32)
    base_w = 0.4 if res.truncated else 1.0
    # On top of the discounted target itself, still up-weight a fast (R7/8)
    # game's samples overall — it has far fewer decision points than a
    # 9-10 round grind, so without this it's diluted in the replay buffer
    # purely by sample count even though it's the trajectory worth learning.
    speed_w = 1.0 + max(0, 8 - res.rounds_played) * 0.3
    w = np.full(len(y), base_w * speed_w, dtype=np.float32)
    return X, y, w, res.winner, res.final_vp, res.truncated, res.rounds_played


def generate_selfplay(n_games: int, agent_spec: str = "heuristic:T0.7",
                      num_players: int = 4, workers: int = 4,
                      out_dir: str = "data/selfplay", base_seed: int = 0,
                      use_book: bool = True, use_choam: bool = True,
                      shard_tag: str = "s") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(num_players=num_players, agent_spec=agent_spec,
               base_seed=base_seed, use_book=use_book, use_choam=use_choam)
    t0 = time.time()

    results = []
    if workers <= 1:
        _worker_init(cfg)
        for i in range(n_games):
            results.append(_play_one(i))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_worker_init, initargs=(cfg,)) as pool:
            results = pool.map(_play_one, range(n_games))

    Xs, ys, ws = [], [], []
    winners, truncs, rounds_played = [], 0, []
    for r in results:
        if r is None:
            continue
        X, y, w, win, vp, trunc, rp = r
        Xs.append(X); ys.append(y); ws.append(w)
        winners.append(win)
        truncs += int(trunc)
        rounds_played.append(rp)

    X = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    shard = os.path.join(out_dir, f"{shard_tag}_{base_seed}_{n_games}.npz")
    np.savez_compressed(shard, X=X, y=y, w=w)

    manifest = {
        "shard": os.path.basename(shard),
        "n_games": n_games, "n_samples": int(len(y)),
        "agent_spec": agent_spec, "num_players": num_players,
        "use_book": use_book, "truncated_games": truncs,
        "positive_rate": float(y.mean()),
        "avg_rounds_played": round(sum(rounds_played) / max(1, len(rounds_played)), 2),
        "fast_games_r7_r8": sum(1 for rp in rounds_played if rp in (7, 8)),
        "seconds": round(time.time() - t0, 1),
    }
    with open(shard + ".json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_shards(out_dir: str, last_k: int = 0):
    shards = sorted(p for p in os.listdir(out_dir) if p.endswith(".npz"))
    if last_k > 0:
        shards = shards[-last_k:]
    Xs, ys, ws = [], [], []
    for s in shards:
        z = np.load(os.path.join(out_dir, s))
        Xs.append(z["X"]); ys.append(z["y"]); ws.append(z["w"])
    if not Xs:
        raise FileNotFoundError(f"no shards in {out_dir}")
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws))
