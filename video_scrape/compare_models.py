#!/usr/bin/env python
"""
Compare heuristic weight-fits (tune_heuristic.py --subset ...) on the video
positions: the knobs, how well each predicts each group's moves on held-out
games, and where two of them disagree on the same positions.

  python video_scrape/compare_models.py
  python video_scrape/compare_models.py --a config/heuristic_me.json \
      --b config/heuristic_winners.json --on me
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import tune_heuristic as T  # noqa: E402
from src.ai.agents import HeuristicAgent, TUNE_DEFAULTS  # noqa: E402

MODELS = {
    "default": None,
    "all players": "config/heuristic_tuned_all88.json",
    "winners": "config/heuristic_winners.json",
    "you": "config/heuristic_me.json",
}


def load(path):
    return dict(TUNE_DEFAULTS) if path is None else {**TUNE_DEFAULTS, **json.load(open(path))}


def picks(pos, tuning):
    h = HeuristicAgent(seed=0, tuning=tuning)
    return [T.label(max(x["cands"], key=lambda a: h.score(x["gs"], x["pid"], a))) for x in pos]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="you")
    ap.add_argument("--b", default="winners")
    ap.add_argument("--on", default="me", choices=sorted(T.SUBSETS),
                    help="whose positions to compare the two models on")
    args = ap.parse_args()
    pos = pickle.load(open(T.CACHE, "rb"))
    games = sorted({x["game"] for x in pos})
    random.Random(7).shuffle(games)
    hold = set(games[:max(4, len(games) // 5)])
    tunings = {k: load(v) for k, v in MODELS.items() if v is None or os.path.exists(v)}

    # 1. knobs
    print("KNOBS (default -> fitted)")
    print(f"{'':16s}" + "".join(f"{k:>13s}" for k in tunings))
    for knob in TUNE_DEFAULTS:
        vals = [t[knob] for t in tunings.values()]
        if len(set(vals)) > 1:
            print(f"{knob:16s}" + "".join(f"{v:>13g}" for v in vals))

    # 2. held-out agreement: rows = model, cols = whose moves
    print("\nHELD-OUT AGREEMENT (share of decisions where the model picks the human's move)")
    cols = {k: [x for x in pos if x["game"] in hold and f(x)] for k, f in T.SUBSETS.items()}
    print(f"{'model':14s}" + "".join(f"{k + ' (' + str(len(v)) + ')':>16s}" for k, v in cols.items()))
    for name, t in tunings.items():
        print(f"{name:14s}" + "".join(f"{T.agreement(v, t)[0]:>16.3f}" for v in cols.values()))

    # 3. where A and B disagree, on every position of the chosen group
    on = [x for x in pos if T.SUBSETS[args.on](x)]
    pa, pb = picks(on, tunings[args.a]), picks(on, tunings[args.b])
    by_type = collections.defaultdict(lambda: [0, 0])
    swaps = collections.Counter()
    ca, cb = collections.Counter(), collections.Counter()
    for x, a, b in zip(on, pa, pb):
        by_type[x["type"]][0] += a != b
        by_type[x["type"]][1] += 1
        ca[a] += 1
        cb[b] += 1
        if a != b:
            swaps[(a, b)] += 1
    n = len(on)
    print(f"\n'{args.a}' vs '{args.b}' on {n} decisions by '{args.on}':")
    for t, (d, m) in sorted(by_type.items()):
        print(f"  {t:12s} disagree {d / m:5.1%}  ({d}/{m})")
    print(f"\n  most common disagreements ({args.a} would play -> {args.b} would play):")
    for (a, b), k in swaps.most_common(15):
        print(f"    {k:4d}  {a:38s} -> {b}")
    print(f"\n  what each model reaches for MORE often (per 100 decisions):")
    keys = set(ca) | set(cb)
    diff = sorted(keys, key=lambda k: (ca[k] - cb[k]))
    more_b = [(k, ca[k], cb[k]) for k in diff if cb[k] > ca[k]][:10]
    more_a = [(k, ca[k], cb[k]) for k in reversed(diff) if ca[k] > cb[k]][:10]
    print(f"    {args.a + ' more':44s} | {args.b} more")
    for i in range(max(len(more_a), len(more_b))):
        l = more_a[i] if i < len(more_a) else None
        r = more_b[i] if i < len(more_b) else None
        ls = f"{l[0][:30]:30s} {100 * l[1] / n:5.1f} vs {100 * l[2] / n:4.1f}" if l else " " * 44
        rs = f"{r[0][:30]:30s} {100 * r[2] / n:5.1f} vs {100 * r[1] / n:4.1f}" if r else ""
        print(f"    {ls} | {rs}")


if __name__ == "__main__":
    main()
