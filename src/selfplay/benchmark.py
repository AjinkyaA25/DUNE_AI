#!/usr/bin/env python
"""
Progress benchmark: every AI version against the same fixed opponent.

Each version plays N Bloodlines games (blank leaders) in one rotating seat
against three copies of the ORIGINAL hand-written heuristic, on the same
game seeds. A fair share is 25%; above that = stronger than the original.
Results -> reports/benchmark.json (appended runs keep history).

Usage:
  python -m src.selfplay.benchmark            # default version list
  python -m src.selfplay.benchmark --games 300
"""
from __future__ import annotations

import argparse
import json
import os
import time

VERSIONS = [
    ("random", "random",
     "picks a random legal move"),
    ("heuristic, Bloodlines-blind", "heuristic:nobl",
     "hand-written rules, ignores techs/commanders (the AI before Bloodlines support)"),
    ("heuristic (original)", "heuristic",
     "hand-written rules incl. Bloodlines scoring - the benchmark opponent itself"),
    ("value net v1 (models/)", "value:models/value_best.npz",
     "1-ply lookahead + value net, self-play training run 1 (base Uprising)"),
    ("value+policy v2 (models_policy/)",
     "value:models_policy/value_best.npz:models_policy/policy_best.npz",
     "adds the AWR policy head (base Uprising self-play)"),
    ("value+policy v3 (models_policy2/)",
     "value:models_policy2/value_best.npz:models_policy2/policy_best.npz",
     "policy head, warm-started run - previous 'current best'"),
    ("search-trained nets s1 (models_search/)",
     "value:models_search/value_s1.npz:models_search/policy_s1.npz",
     "nets trained on 192 search self-play games + 16 human video games"),
    ("heuristic, tuned to human games", "heuristic:tuned=config/heuristic_tuned.json",
     "11 heuristic weights fitted to 1,617 human decisions"),
]


def main() -> None:
    from train_search import arena
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=130_000)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)
    out = []
    for name, spec, what in VERSIONS:
        t0 = time.time()
        r = arena(spec, "heuristic", args.games, args.workers, args.seed)
        r.update(name=name, spec=spec, what=what,
                 minutes=round((time.time() - t0) / 60, 1))
        out.append(r)
        print(json.dumps(r), flush=True)
    rec = {"date": time.strftime("%Y-%m-%d %H:%M"), "games_per_version": args.games,
           "opponent": "3x original heuristic", "bloodlines": True, "results": out}
    path = os.path.join("reports", "benchmark.json")
    hist = json.load(open(path)) if os.path.exists(path) else []
    hist.append(rec)
    json.dump(hist, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
