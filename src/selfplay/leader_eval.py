"""Per-Leader strength: a candidate and a baseline each play the same seeds in
one seat against 3 x --opp (user, 2026-10-23: "fine tune the AI for each and
every leader ... see if the AI is getting better").

Each game of the candidate is paired with the baseline's game on the same seed,
seat and Leader, so the per-Leader difference has far less luck in it than
the two win shares alone.

    python -m src.selfplay.leader_eval --cand SPEC --base SPEC --opp SPEC --games 18000 --out reports/x.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from multiprocessing import Pool

from src.selfplay.ppo_train import eval_game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--games", type=int, default=18000)
    ap.add_argument("--first", type=int, default=20_000, help="first eval seed (away from the trainer's)")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    gs = range(A.first, A.first + A.games)
    with Pool(A.workers, maxtasksperchild=50) as pool:
        c = pool.map(eval_game, [(g, A.cand, A.opp) for g in gs], chunksize=8)
        b = pool.map(eval_game, [(g, A.base, A.opp) for g in gs], chunksize=8)
    by = defaultdict(lambda: [0.0, 0.0, 0.0, 0])           # cand wins, base wins, sum d^2, n
    for (wc, _, ld), (wb, _, _) in zip(c, b):
        for k in (ld, "ALL"):
            e = by[k]
            e[0] += wc; e[1] += wb; e[2] += (wc - wb) ** 2; e[3] += 1
    res = {"cand": A.cand, "base": A.base, "opp": A.opp, "games": A.games,
           "rounds_cand": sum(x[1] for x in c) / len(c), "rounds_base": sum(x[1] for x in b) / len(b),
           "by_leader": {}}
    print(f"{'Leader':24s}  cand   base   diff   (se)    n")
    for k, (wc, wb, d2, n) in sorted(by.items(), key=lambda kv: (kv[0] == "ALL", -(kv[1][0] - kv[1][1]) / kv[1][3])):
        d = (wc - wb) / n
        se = math.sqrt(max(0.0, d2 / n - d * d) / n)
        res["by_leader"][k] = {"cand": round(wc / n, 4), "base": round(wb / n, 4), "diff": round(d, 4),
                               "se": round(se, 4), "n": n}
        print(f"{k:24s} {wc / n:5.1%}  {wb / n:5.1%}  {d:+5.1%}  ({se:4.1%})  {n}")
    print(f"avg rounds: cand {res['rounds_cand']:.2f}, base {res['rounds_base']:.2f}")
    if A.out:
        json.dump(res, open(A.out, "w"), indent=1)


if __name__ == "__main__":
    main()
