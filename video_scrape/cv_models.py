#!/usr/bin/env python
"""
5-fold cross-validation of the heuristic fits (by game): for each fold, fit
on the other games' decisions of one group (all / winners / me) and score
the held-out games' decisions of every group. One fold per process:

  python video_scrape/cv_models.py --subset me --fold 0      # -> reports/cv/me_0.json
  python video_scrape/cv_models.py --summary
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import tune_heuristic as T  # noqa: E402
from src.ai.agents import TUNE_DEFAULTS  # noqa: E402

K = 5
OUT = os.path.join(ROOT, "reports", "cv")


def fit(pos: list) -> dict:
    """tune_heuristic's coordinate search, from the defaults."""
    cur = dict(TUNE_DEFAULTS)
    best, _ = T.agreement(pos, cur)
    for _ in range(3):
        improved = False
        for k, values in T.GRID.items():
            for v in values:
                if v == cur[k]:
                    continue
                trial = {**cur, k: v}
                sc, _ = T.agreement(pos, trial)
                if sc > best + 0.002:
                    best, cur, improved = sc, trial, True
        if not improved:
            break
    return cur


def run(subset: str, fold: int) -> None:
    pos = pickle.load(open(T.CACHE, "rb"))
    games = sorted({x["game"] for x in pos})
    random.Random(11).shuffle(games)
    hold = set(games[fold::K])
    keep = T.SUBSETS[subset]
    tuning = fit([x for x in pos if x["game"] not in hold and keep(x)])
    res = {"subset": subset, "fold": fold,
           "knobs": {k: v for k, v in tuning.items() if v != TUNE_DEFAULTS[k]}}
    for g, f in T.SUBSETS.items():
        val = [x for x in pos if x["game"] in hold and f(x)]
        res[g] = {"n": len(val), "fitted": T.agreement(val, tuning)[0],
                  "default": T.agreement(val, TUNE_DEFAULTS)[0]}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{subset}_{fold}.json"), "w"), indent=1)
    print(json.dumps(res))


def summary() -> None:
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    groups = list(T.SUBSETS)
    print("cross-validated agreement with each group's held-out moves "
          f"(mean over {K} folds, decision-weighted)")
    print(f"{'model fitted on':16s}" + "".join(f"{g:>12s}" for g in groups))

    def wmean(rs, g, key):
        n = sum(r[g]["n"] for r in rs)
        return sum(r[g][key] * r[g]["n"] for r in rs) / max(1, n)

    any_rows = rows[:K]
    print(f"{'(default)':16s}" + "".join(f"{wmean(any_rows, g, 'default'):12.3f}" for g in groups))
    for s in groups:
        rs = [r for r in rows if r["subset"] == s]
        if rs:
            print(f"{s:16s}" + "".join(f"{wmean(rs, g, 'fitted'):12.3f}" for g in groups))
    print("\nknobs chosen per fold (stable knobs = real preferences):")
    for s in groups:
        rs = [r for r in rows if r["subset"] == s]
        keys = sorted({k for r in rs for k in r["knobs"]})
        print(f"  {s}:")
        for k in keys:
            vals = [r["knobs"].get(k, TUNE_DEFAULTS[k]) for r in rs]
            print(f"    {k:14s} {vals}  (default {TUNE_DEFAULTS[k]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=sorted(T.SUBSETS))
    ap.add_argument("--fold", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run(a.subset, a.fold)
