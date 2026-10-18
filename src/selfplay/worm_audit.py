#!/usr/bin/env python
"""
Worm audit: when a sandworm summon is legal, is summoning better than the
best other move?  The coaching question: "does summoning a worm in this
conflict give me more value than my alternative (Diplomacy for a faction
bump, drawing cards toward The Spice Must Flow, ...)?"

Heuristic games (one tuning at every seat). At every agent turn where a
maker-space "sandworm" option is legal, the best-scored worm move and the
best-scored non-worm move are each played out M times from re-sampled
worlds (common random numbers, the same heuristic at every seat), so the
answer is measured, not assumed. Each row also records the situation
(conflict, strengths, worms, round, ...) and the heuristic's own scores, so
the playout answer can be compared with what the heuristic would have done.

  python -m src.selfplay.worm_audit --games 200 --m 32 --spec heuristic:tuned=config/gap/base.json \
      --out data/worm_audit.jsonl --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np


def _situation(gs, pid, worm_a):
    from src.ai.agents import _my_strength, _opp_fight, _conflict_worth, _game_urgency
    from src.game.board.board import MAKER_SPACE_OPTIONS
    cc = gs.current_conflict
    p = gs.players[pid]
    mine, units = _my_strength(gs, pid)
    opp = [_opp_fight(gs, q) for q in range(gs.num_players) if q != pid]
    r = cc.first_place_reward
    vp = gs.players[pid].victory_points
    return {
        "round": gs.round, "conflict": cc.name, "level": cc.conflict_level,
        "c_vp": r.get("vp", 0),
        "c_conv": int(any(k.startswith("may_pay") for k in r)),
        "worms": MAKER_SPACE_OPTIONS[worm_a.space_name][1],
        "my_str": mine, "my_units": units,
        "my_troops_in": gs.troops_in_conflict.get(pid, 0),
        "garrison": p.troops_garrison,
        "opp_str": [round(float(o[0]), 2) if isinstance(o, tuple) else o for o in opp],
        "my_worms_in": gs.sandworms_in_conflict.get(pid, 0),
        "rival_worms_in": sum(v for q, v in gs.sandworms_in_conflict.items() if q != pid),
        "rival_hooks": sum(1 for q in gs.players if q.id != pid and q.has_maker_hooks),
        "agents_left": p.agents_available, "hand": len(p.hand),
        "intrigues": len(p.intrigue_cards), "spice": p.spice, "solari": p.solari,
        "vp": vp, "vp_gap": vp - max(q.victory_points for q in gs.players if q.id != pid),
        "worth": round(_conflict_worth(gs, pid), 3),
        "urgency": round(_game_urgency(gs, pid), 3),
    }


def _short(a):
    return {"space": a.space_name, "card": a.card_name, "opt": a.space_option,
            "type": a.action_type.name}


def label(gs, pid, cands, m, rng, h):
    from src.ai.determinize import determinize
    from src.ai.search import end_value
    seeds = rng.integers(0, 2**31 - 1, size=m)
    vals = np.full((len(cands), m), np.nan)
    for i, c in enumerate(cands):
        for j in range(m):
            g = determinize(gs, pid, np.random.default_rng(int(seeds[j])))
            ga = next((x for x in g.get_valid_actions(pid) if repr(x) == repr(c)), None)
            if ga is None:
                continue
            g.step(ga)
            n = 0
            while not g.game_over and n < 3000:
                cur = g.player_in_reveal_buy
                if cur is None:
                    cur = g.get_current_player_id()
                g.step(h.select_action(g, cur, g.get_valid_actions(cur)))
                n += 1
            vals[i, j] = end_value(g, pid)
    return vals


def play(args):
    g, seed0, m, spec, cap = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.gameState import ActionType
    rng = np.random.default_rng(seed0 + 7919 * g)
    agents = [make_agent(spec, seed=g * 4 + s) for s in range(4)]
    h = make_agent(spec, seed=g)
    gs = setup_game(num_players=4, seed=seed0 + g, neutral_leaders=True, use_bloodlines=True)
    rows, n, t0 = [], 0, time.time()
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        worm = [a for a in valid if a.action_type == ActionType.AGENT_TURN
                and a.space_option == "sandworm"]
        if worm and len(rows) < cap:
            sc = {repr(a): agents[pid].score(gs, pid, a) for a in valid}
            other = [a for a in valid if a.space_option != "sandworm"
                     and a.action_type != ActionType.NO_OP]
            wa = max(worm, key=lambda a: sc[repr(a)])
            if other:
                oa = max(other, key=lambda a: sc[repr(a)])
                vals = label(gs, pid, [wa, oa], m, rng, h)
                d = vals[0] - vals[1]
                d = d[~np.isnan(d)]
                r = {"game": g, "seat": pid, **_situation(gs, pid, wa),
                     "worm_act": _short(wa), "alt_act": _short(oa),
                     "h_worm": round(sc[repr(wa)], 3), "h_alt": round(sc[repr(oa)], 3),
                     "v_worm": round(float(np.nanmean(vals[0])), 4),
                     "v_alt": round(float(np.nanmean(vals[1])), 4),
                     "d": round(float(d.mean()), 4) if len(d) else None,
                     "d_se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4) if len(d) > 1 else None,
                     "m": int(len(d))}
                rows.append(r)
        a = agents[pid].select_action(gs, pid, valid)
        if rows and rows[-1].get("game") == g and "chose_worm" not in rows[-1] and worm:
            rows[-1]["chose_worm"] = int(a.space_option == "sandworm")
        gs.step(a)
        n += 1
    vp = [p.victory_points for p in gs.players]
    win = gs.winner if gs.winner is not None else max(range(4), key=lambda s: vp[s])
    for r in rows:
        r.update({"won": int(win == r["seat"]), "final_vp": vp[r["seat"]],
                  "seconds": round(time.time() - t0, 1)})
    return g, rows


def summarize(path):
    R = [json.loads(l) for l in open(path)]
    R = [r for r in R if r.get("d") is not None]
    if not R:
        print("no rows"); return
    d = np.array([r["d"] for r in R])
    se = np.array([r["d_se"] or 1.0 for r in R])
    hw = np.array([r["chose_worm"] for r in R])
    better = d > 0
    print(f"{len(R)} worm decisions from {len({r['game'] for r in R})} games")
    print(f"playouts say worm better: {better.mean():.0%} (clearly, d > 2se: {(d > 2 * se).mean():.0%}; "
          f"clearly worse: {(d < -2 * se).mean():.0%})")
    print(f"heuristic summoned: {hw.mean():.0%}; agrees with playouts {np.mean(hw == better):.0%}")
    print(f"  summoned but worse (over-summon): {np.mean(hw & ~better):.0%}   "
          f"skipped but better (missed worm): {np.mean(~hw.astype(bool) & better):.0%}")
    print(f"mean value of the playout-best choice over the heuristic's: "
          f"{np.mean(np.where(better, d, -d) - np.where(hw == 1, d, -d)):.4f} win share per decision")

    def by(key, f=lambda v: v):
        groups = {}
        for r, di, h in zip(R, d, hw):
            groups.setdefault(f(r[key]), []).append((di, h))
        print(f"-- by {key}")
        for k in sorted(groups):
            x = np.array(groups[k])
            print(f"  {str(k):>12s} n={len(x):4d}  worm better {np.mean(x[:, 0] > 0):.0%}  "
                  f"mean d {x[:, 0].mean():+.3f}  heuristic summons {x[:, 1].mean():.0%}")
    by("level"); by("c_vp"); by("c_conv"); by("rival_worms_in", lambda v: min(v, 2))
    by("round"); by("worms")
    by("vp_gap", lambda v: max(-3, min(3, v)))
    by("alt_act", lambda a: a["space"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--cap", type=int, default=12, help="max audited decisions per game")
    ap.add_argument("--spec", default="heuristic:tuned=config/gap/base.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=910_000)
    ap.add_argument("--out", default="")
    ap.add_argument("--summary", default="")
    a = ap.parse_args()
    if a.summary:
        summarize(a.summary); return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["game"] for l in open(a.out)}
    todo = [(g, a.seed, a.m, a.spec, a.cap) for g in range(a.games) if g not in done]
    t0 = time.time()
    with Pool(a.workers) as pool, open(a.out, "a") as f:
        for i, (g, rows) in enumerate(pool.imap_unordered(play, todo)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            if not rows:
                f.write(json.dumps({"game": g, "d": None}) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(todo)}] games, {time.time() - t0:.0f}s", flush=True)
    summarize(a.out)


if __name__ == "__main__":
    main()
