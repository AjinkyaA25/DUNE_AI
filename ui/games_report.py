#!/usr/bin/env python
"""
Summary of the games you played in the browser (game_logs/ui_*.json):
record against each opponent line-up, and how often your moves matched what
the AI would have played, with your most common differences.

  python ui/games_report.py
"""
from __future__ import annotations

import collections
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    games = []
    for p in sorted(glob.glob(os.path.join(ROOT, "game_logs", "ui_*.json"))):
        try:
            games.append(json.load(open(p, encoding="utf-8")))
        except (OSError, ValueError):
            continue
    if not games:
        print("No browser games yet (play at http://localhost:8765).")
        return
    done = [g for g in games if g.get("finished")]
    print(f"{len(games)} games started, {len(done)} finished")
    if done:
        wins = sum(bool(g.get("you_won")) for g in done)
        print(f"you won {wins}/{len(done)} = {wins / len(done):.0%} (fair share 25%)")
        by_opp = collections.defaultdict(lambda: [0, 0])
        for g in done:
            key = " + ".join(sorted(g.get("opponents", {}).values()))
            by_opp[key][0] += bool(g.get("you_won"))
            by_opp[key][1] += 1
        print("\nby opponents:")
        for k, (w, n) in sorted(by_opp.items(), key=lambda kv: -kv[1][1]):
            print(f"  {w}/{n}  {k}")

    mine = [e for g in games for e in g.get("log", [])
            if e.get("pid") == g.get("human_id") and "agree" in e]
    if mine:
        print(f"\nyour {len(mine)} decisions: the AI would have made the same move "
              f"{sum(e['agree'] for e in mine) / len(mine):.0%} of the time")
        by_type = collections.defaultdict(list)
        for e in mine:
            by_type[e["type"]].append(e["agree"])
        for t, v in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            print(f"  {t:22s} n={len(v):4d}  same as AI {sum(v) / len(v):.0%}")
        diffs = collections.Counter((e["label"], e["ai_pick"]) for e in mine if not e["agree"])
        print("\nmost common differences (you played -> AI would have played):")
        for (you, ai), n in diffs.most_common(12):
            print(f"  {n:3d}  {you[:44]:44s} -> {ai}")


if __name__ == "__main__":
    main()
