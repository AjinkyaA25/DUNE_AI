#!/usr/bin/env python
"""
Four-way arena: four different agents at one table, every game.

Seat order cycles through all 24 permutations (game g uses permutation
g % 24), so with a multiple of 24 games each agent sits in each seat
equally often. Bloodlines, blank leaders. Fair share = 25%.

Results -> reports/four_way.json (appended runs keep history).

Usage:
  python -m src.selfplay.four_way --games 96
  python -m src.selfplay.four_way --agents random search:K5:M8 search:K5:M24 \
      heuristic:tuned=config/heuristic_tuned.json --games 48
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

DEFAULT_AGENTS = [
    ("random", "random"),
    ("round search, 8 playouts", "search:K5:M8"),
    ("round search, 24 playouts", "search:K5:M24"),
    ("heuristic, tuned to human games", "heuristic:tuned=config/heuristic_tuned.json"),
]
PERMS = list(itertools.permutations(range(4)))


def _game(args):
    g, specs, base_seed = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    perm = PERMS[g % 24]                       # perm[seat] = agent index
    agents = [make_agent(specs[perm[s]], seed=g * 4 + s) for s in range(4)]
    gs = setup_game(num_players=4, seed=base_seed + g, neutral_leaders=True,
                    use_bloodlines=True)
    t0 = time.time()
    n = 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    vp = [p.victory_points for p in gs.players]
    win = [gs.winner] if gs.winner is not None else \
        [i for i, v in enumerate(vp) if v == max(vp)]
    # rank by VP (1 = best; ties share the better rank), winner forced to 1
    ranks = [1 + sum(1 for w in vp if w > v) for v in vp]
    for s in win:
        ranks[s] = 1
    out = {"game": g, "rounds": gs.round, "seconds": round(time.time() - t0, 1),
           "by_agent": {}}
    for s in range(4):
        out["by_agent"][perm[s]] = {"seat": s, "vp": vp[s], "rank": ranks[s],
                                    "win": (1.0 / len(win)) if s in win else 0.0}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs=4, default=None,
                    help="four agent specs (default: random, search M8, search M24, tuned heuristic)")
    ap.add_argument("--games", type=int, default=96)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--seed", type=int, default=150_000)
    args = ap.parse_args()
    named = DEFAULT_AGENTS if args.agents is None else [(s, s) for s in args.agents]
    specs = [s for _, s in named]

    t = time.time()
    res = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_game, [(g, specs, args.seed) for g in range(args.games)]):
            res.append(r)
            wins = [sum(x["by_agent"][i]["win"] for x in res) for i in range(4)]
            print(f"[{len(res)}/{args.games}] {(time.time() - t) / 60:.1f} min  "
                  f"wins so far: " + "  ".join(f"{named[i][0]}={wins[i]:.1f}" for i in range(4)),
                  flush=True)

    n = len(res)
    summary = []
    for i, (name, spec) in enumerate(named):
        rows = [x["by_agent"][i] for x in res]
        p = sum(r["win"] for r in rows) / n
        se = (p * (1 - p) / n) ** 0.5
        summary.append({
            "name": name, "spec": spec,
            "win_share": round(p, 3), "ci95": [round(max(0, p - 1.96 * se), 3),
                                               round(min(1, p + 1.96 * se), 3)],
            "avg_vp": round(sum(r["vp"] for r in rows) / n, 2),
            "avg_rank": round(sum(r["rank"] for r in rows) / n, 2),
            "rank_counts": {k: sum(1 for r in rows if r["rank"] == k) for k in (1, 2, 3, 4)},
        })
    rec = {"date": time.strftime("%Y-%m-%d %H:%M"), "games": n, "bloodlines": True,
           "avg_rounds": round(sum(x["rounds"] for x in res) / n, 2),
           "sec_per_game": round(sum(x["seconds"] for x in res) / n, 1),
           "wall_minutes": round((time.time() - t) / 60, 1),
           "results": summary, "games_detail": sorted(res, key=lambda x: x["game"])}
    print(json.dumps({k: v for k, v in rec.items() if k != "games_detail"}, indent=1))
    os.makedirs("reports", exist_ok=True)
    path = os.path.join("reports", "four_way.json")
    hist = json.load(open(path)) if os.path.exists(path) else []
    hist.append(rec)
    json.dump(hist, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
