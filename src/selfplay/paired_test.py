"""How much do paired seeds save? (user, 2026-10-19: "use the paired seeds to
see how much more efficient it can be made").

Plays every config in 1 seat vs 3 copies of --opp on the SAME game seeds
(Leaders, seats, shuffles, agent tie-breaks) and compares the noise of
"config A minus config B" measured on shared seeds against measuring them on
separate seeds. efficiency = var(unpaired difference) / var(paired difference)
= how many times fewer games the paired comparison needs for the same error.

    python -m src.selfplay.paired_test --opp config/best_ai.json A.json B.json [...] --games 4000
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from multiprocessing import Pool

import numpy as np

import src.selfplay.evolve as E


def wins_on_seeds(pool, cand, opp, n, base):
    specs = [cand if ":" in cand.split("/")[0] else f"heuristic:tuned={cand}"] + [f"heuristic:tuned={opp}"] * 3
    out, rounds = np.zeros(n), np.zeros(n)
    res = pool.map(E.play, [(base + g, specs, True) for g in range(n)], chunksize=8)
    for g, (_, rnd, o) in enumerate(res):
        out[g] = o[0]["win"]
        rounds[g] = rnd
    ROUNDS[cand] = float(rounds.mean())
    return out


ROUNDS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--opp", default="config/best_ai.json")
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--base", type=int, default=77_000_000)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    with Pool(A.workers, maxtasksperchild=200) as pool:
        W = {c: wins_on_seeds(pool, c, A.opp, A.games, A.base) for c in A.configs}
    rows = []
    for c, w in W.items():
        print(f"{c}: {w.mean():.2%} | avg rounds {ROUNDS[c]:.2f}")
    for a, b in itertools.combinations(A.configs, 2):
        d = W[a] - W[b]
        var_p = d.var()
        var_u = W[a].var() + W[b].var()
        same = float(np.mean(W[a] == W[b]))
        rho = float(np.corrcoef(W[a], W[b])[0, 1]) if W[a].std() and W[b].std() else 1.0
        eff = var_u / var_p if var_p > 0 else float("inf")
        n = len(d)
        rec = {"a": a, "b": b, "diff": round(float(d.mean()), 4),
               "se_paired": round(math.sqrt(var_p / n), 4), "se_unpaired": round(math.sqrt(var_u / n), 4),
               "corr": round(rho, 3), "same_result": round(same, 3), "efficiency": round(eff, 2)}
        rows.append(rec)
        print(json.dumps(rec))
    if A.out:
        json.dump({"games": A.games, "opp": A.opp, "share": {c: float(w.mean()) for c, w in W.items()},
                   "rounds": ROUNDS,
                   "pairs": rows}, open(A.out, "w"), indent=1)


if __name__ == "__main__":
    main()
