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
    y = np.array([1.0 if pid == res.winner else 0.0
                  for pid in res.feat_pids], dtype=np.float32)
    w = np.full(len(y), 0.4 if res.truncated else 1.0, dtype=np.float32)
    return X, y, w, res.winner, res.final_vp, res.truncated


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
    winners, truncs = [], 0
    for r in results:
        if r is None:
            continue
        X, y, w, win, vp, trunc = r
        Xs.append(X); ys.append(y); ws.append(w)
        winners.append(win)
        truncs += int(trunc)

    X = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    shard = os.path.join(out_dir, f"{shard_tag}_{base_seed}_{n_games}.npz")
    np.savez_compressed(shard, X=X, y=y, w=w)

    manifest = {
        "shard": os.path.basename(shard),
        "n_games": n_games, "n_samples": int(len(y)),
        "agent_spec": agent_spec, "num_players": num_players,
        "use_book": use_book, "truncated_games": truncs,
        "positive_rate": float(y.mean()),
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
