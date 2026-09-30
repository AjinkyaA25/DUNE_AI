#!/usr/bin/env python
"""
Fit the heuristic's tunable knobs (agents.TUNE_DEFAULTS) to human play.

1. capture: replay every cleaned video game; at each human decision store
   the position (engine copy), the legal moves and the human's choice.
2. fit: coordinate search over the knobs, maximising how often the
   heuristic's top move matches the human's (same board space / same card /
   reveal), averaged over decision types so buys and reveals count as much
   as agent turns. Fit on most games, report agreement on held-out games.
3. writes config/heuristic_tuned.json - use it with `heuristic:tuned=<path>`
   and check it in an arena before trusting it.

Usage:
  python video_scrape/tune_heuristic.py            # capture (cached) + fit
"""
from __future__ import annotations

import collections
import glob
import json
import os
import pickle
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from src.ai.agents import HeuristicAgent, TUNE_DEFAULTS  # noqa: E402
from src.game.gameState import ActionType  # noqa: E402

CACHE = os.path.join(ROOT, "data", "tune_positions.pkl")
OUT = os.path.join(ROOT, "config", "heuristic_tuned.json")
GRID = {
    "res_contract": [0.5, 1, 2, 3, 5], "res_spy": [0.5, 1, 2, 3], "res_intrigue": [0.5, 1, 2, 3],
    "res_draw": [0.5, 1, 1.5, 2], "res_solari": [0.5, 0.75, 1, 1.5], "res_spice": [0.5, 1, 1.5],
    "res_water": [0.5, 1, 1.5, 2], "res_troops": [0.5, 1, 1.5],
    "sm_solari": [0, 0.5, 1, 2, 3], "combat": [0.25, 0.5, 0.75, 1, 1.5],
    "influence": [0.5, 0.75, 1, 1.5, 2], "reveal_bias": [-2, -1, 0, 1, 2, 3],
    "ptw_tax": [4, 8, 15, 30], "tier_blend": [0.3, 0.5, 0.7, 0.85],
    "faction_space": [0, 1, 2, 4, 6],
}


def label(a) -> str:
    t = a.action_type
    if t == ActionType.AGENT_TURN:
        return f"agent {a.space_name}"
    if t == ActionType.ACQUIRE_CARD:
        return f"buy {a.acquire_card_name}"
    if t == ActionType.ACQUIRE_RESERVE:
        return f"buy {a.reserve_type}"
    return t.value


def capture() -> list:
    import replay as R
    from clean import clean_game
    fac = R._card_factory()
    pos = []
    for p in sorted(glob.glob(os.path.join(HERE, "games", "*.json"))):
        g = json.load(open(p, encoding="utf-8"))
        if g.get("source") != "vision" or g.get("duplicate_of") or \
                len({x["color"] for x in g["players"]}) != 4:
            continue
        g = clean_game(g, fac)
        rp = R.Replay(g)

        def rec(pid, chosen, valid, _rp=rp, _vid=g["video_id"]):
            cands = [a for a in valid if a.action_type != ActionType.NO_OP]
            if len(cands) < 2 or chosen.action_type not in (
                    ActionType.AGENT_TURN, ActionType.ACQUIRE_CARD,
                    ActionType.ACQUIRE_RESERVE, ActionType.REVEAL_TURN):
                return
            gs2 = _rp.gs.clone()
            gs2.__dict__.pop("check_victory_conditions", None)   # replay override
            pos.append({"game": _vid, "gs": gs2, "pid": pid,
                        "cands": cands, "human": label(chosen),
                        "type": "buy" if "acquire" in chosen.action_type.value
                        else chosen.action_type.value})
        rp._record = rec
        rp.run()
        print(f"  {g['video_id']}: {len(pos)} positions so far", flush=True)
    return pos


def agreement(pos: list, tuning: dict) -> float:
    h = HeuristicAgent(seed=0, tuning=tuning)
    by = collections.defaultdict(list)
    for x in pos:
        best = max(x["cands"], key=lambda a: h.score(x["gs"], x["pid"], a))
        by[x["type"]].append(label(best) == x["human"])
    # every decision counts once (agent placements are half the data):
    # averaging per type let the fit trade agent placements for reveals
    return sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values()), \
        {k: round(sum(v) / len(v), 3) for k, v in by.items()}


def main() -> None:
    if os.path.exists(CACHE):
        pos = pickle.load(open(CACHE, "rb"))
    else:
        pos = capture()
        pickle.dump(pos, open(CACHE, "wb"))
    games = sorted({x["game"] for x in pos})
    random.Random(7).shuffle(games)
    hold = set(games[:4])
    fit = [x for x in pos if x["game"] not in hold]
    val = [x for x in pos if x["game"] in hold]
    print(f"{len(pos)} positions: fit {len(fit)} ({len(games) - 4} games), "
          f"held-out {len(val)} ({sorted(hold)})", flush=True)

    cur = dict(TUNE_DEFAULTS)
    base_fit, _ = agreement(fit, cur)
    base_val, base_by = agreement(val, cur)
    print(f"default heuristic: fit {base_fit:.3f}  held-out {base_val:.3f} {base_by}", flush=True)
    best = base_fit
    t0 = time.time()
    for sweep in range(3):
        improved = False
        for k, values in GRID.items():
            for v in values:
                if v == cur[k]:
                    continue
                trial = {**cur, k: v}
                sc, _ = agreement(fit, trial)
                if sc > best + 0.002:
                    best, cur, improved = sc, trial, True
                    print(f"  sweep {sweep}: {k}={v} -> fit {sc:.3f}  "
                          f"({time.time() - t0:.0f}s)", flush=True)
        if not improved:
            break
    val_sc, val_by = agreement(val, cur)
    _, fit_by = agreement(fit, cur)
    print(f"tuned: fit {best:.3f} {fit_by}\n       held-out {val_sc:.3f} {val_by} "
          f"(default {base_val:.3f} {base_by})", flush=True)
    changed = {k: v for k, v in cur.items() if v != TUNE_DEFAULTS[k]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(changed, open(OUT, "w"), indent=1)
    print(f"changed knobs -> {OUT}: {changed}")


if __name__ == "__main__":
    main()
