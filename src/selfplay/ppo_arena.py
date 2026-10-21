"""Matches between any 4 agent specs (user, 2026-10-20: "test the PPO against the
E6 and see how it performs and play it against itself to show strength").

Seats rotate through all 24 orders, random real Leaders, same game seeds for
every run with the same --base (so two runs that differ in one spec are paired).

    python -m src.selfplay.ppo_arena --spec A --spec B --spec C --spec D --games 400 --out reports/x.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool

import src.selfplay.evolve as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True, help="4 specs, one per player")
    ap.add_argument("--name", action="append", default=None, help="label per spec")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--base", type=int, default=91_000_000)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    assert len(A.spec) == 4
    names = A.name or A.spec
    t0 = time.time()
    per_game = []
    with Pool(A.workers, maxtasksperchild=50) as pool:
        for g, rnd, out in pool.imap_unordered(_one, [(A.base + g, A.spec) for g in range(A.games)]):
            per_game.append({"g": g, "rounds": rnd, "win": [out[i]["win"] for i in range(4)],
                             "vp": [out[i]["vp"] for i in range(4)]})
            if len(per_game) % max(1, A.games // 10) == 0:
                print(f"{len(per_game)}/{A.games} games, {time.time() - t0:.0f}s", flush=True)
    n = len(per_game)
    rounds = sum(r["rounds"] for r in per_game) / n
    by_r8 = sum(r["rounds"] <= 8 for r in per_game) / n
    res = []
    for i in range(4):
        w = sum(r["win"][i] for r in per_game) / n
        res.append({"name": names[i], "spec": A.spec[i], "share": round(w, 4),
                    "se": round(math.sqrt(w * (1 - w) / n), 4),
                    "vp": round(sum(r["vp"][i] for r in per_game) / n, 3)})
        print(f"{names[i]:<28} win {w:6.1%} (se {res[-1]['se']:.1%})  avg VP {res[-1]['vp']:.2f}")
    print(f"games {n} | avg rounds {rounds:.2f} | ended by R8 {by_r8:.1%} | {time.time() - t0:.0f}s")
    if A.out:
        json.dump({"games": n, "base": A.base, "rounds": round(rounds, 3), "ended_by_r8": round(by_r8, 4),
                   "seconds": round(time.time() - t0), "players": res, "per_game": per_game},
                  open(A.out, "w"), indent=1)


def _one(args):
    g, specs = args
    _, rnd, out = E.play((g, specs, True))
    return g, rnd, out


if __name__ == "__main__":
    main()
