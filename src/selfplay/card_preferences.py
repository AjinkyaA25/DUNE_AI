#!/usr/bin/env python
"""
What does the AI want to buy? Search self-play where every buy decision is
settled by playouts over ALL affordable cards (search ...:B:KB0), with no
tier list anywhere. Each buy decision is logged with the playout value of
every option, so cards can be ranked by how much buying them helped.

Per decision -> one JSON line in reports/card_prefs/games.jsonl:
  game, round, pid, persuasion, options: [{card, value}], chosen, winner

value = mean playout result for that option (win share, ties split, plus
0.02 x VP margin), same hidden-information samples for every option.

Usage:
  python -m src.selfplay.card_preferences --games 112 --workers 28
  python -m src.selfplay.card_preferences --summarize
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool

OUT_DIR = os.path.join("reports", "card_prefs")
LOG = os.path.join(OUT_DIR, "games.jsonl")
SPEC = "search:K5:M24:B:KB0:tuned=config/heuristic_phased_final.json"
PASS = "(stop buying)"


def _option_name(a) -> str:
    from src.game.gameState import ActionType
    if a.action_type == ActionType.ACQUIRE_CARD:
        return a.acquire_card_name
    if a.action_type == ActionType.ACQUIRE_RESERVE:
        return {"prepare_the_way": "Prepare the Way",
                "spice_must_flow": "The Spice Must Flow"}[a.reserve_type]
    if a.action_type == ActionType.END_REVEAL:
        return PASS
    return a.action_type.value


def play_one(args):
    g, spec, base_seed = args
    from src.ai.agents import make_agent
    from src.ai.search import BUY_TYPES
    from src.data.card_definitions import setup_game
    agents = [make_agent(spec, seed=(base_seed + g) * 4 + s) for s in range(4)]
    gs = setup_game(num_players=4, seed=base_seed + g, neutral_leaders=True,
                    use_bloodlines=True)
    rows, n, t0 = [], 0, time.time()
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        rnd, pers = gs.round, gs.persuasion_pool[pid]
        a = agents[pid].select_action(gs, pid, valid)
        last = getattr(agents[pid], "last", None)
        if last and any(c.action_type in BUY_TYPES for c in last["cands"]):
            rows.append({
                "game": g, "round": rnd, "pid": pid, "persuasion": pers,
                "options": [{"card": _option_name(c), "value": round(v, 4)}
                            for c, v in zip(last["cands"], last["scores"])],
                "chosen": _option_name(a),
            })
        gs.step(a)
        n += 1
    vp = [p.victory_points for p in gs.players]
    winner = gs.winner if gs.winner is not None else vp.index(max(vp))
    for r in rows:
        r["winner"] = winner
        r["won"] = r["pid"] == winner
    return {"game": g, "rounds": gs.round, "seconds": round(time.time() - t0),
            "rows": rows}


def summarize() -> dict:
    """Rank cards by the playout advantage of buying them over stopping."""
    rows = [json.loads(l) for l in open(LOG, encoding="utf-8")]
    stat = defaultdict(lambda: {"offered": 0, "bought": 0, "adv": [],
                                "by_phase": defaultdict(list)})
    for r in rows:
        vals = {o["card"]: o["value"] for o in r["options"]}
        if PASS not in vals:
            continue                       # need the stop-buying baseline
        base = vals[PASS]
        ph = "early" if r["round"] <= 3 else "mid" if r["round"] <= 5 else "late"
        for card, v in vals.items():
            if card == PASS or card in _ACTION_NAMES:
                continue                   # an intrigue / tech, not a card
            s = stat[card]
            s["offered"] += 1
            s["bought"] += card == r["chosen"]
            s["adv"].append(v - base)
            s["by_phase"][ph].append(v - base)
    out = []
    for card, s in stat.items():
        a = s["adv"]
        m = sum(a) / len(a)
        sd = (sum((x - m) ** 2 for x in a) / max(1, len(a) - 1)) ** 0.5
        out.append({
            "card": card, "offered": s["offered"], "bought": s["bought"],
            "pick_rate": round(s["bought"] / s["offered"], 3),
            "adv_pts": round(100 * m, 2),                 # win-% points vs stopping
            "se_pts": round(100 * sd / len(a) ** 0.5, 2),
            "by_phase": {ph: [round(100 * sum(v) / len(v), 2), len(v)]
                         for ph, v in s["by_phase"].items()},
        })
    out.sort(key=lambda x: -x["adv_pts"])
    games = len({r["game"] for r in rows})
    res = {"games": games, "decisions": len(rows), "cards": out}
    json.dump(res, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=1)
    return res


def _action_names():
    from src.game.gameState import ActionType
    return {t.value for t in ActionType}


_ACTION_NAMES = _action_names()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=112)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--seed", type=int, default=310_000)
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.summarize:
        res = summarize()
        print(f"{res['games']} games, {res['decisions']} buy decisions")
        for c in res["cards"]:
            print(f"{c['card']:32s} {c['adv_pts']:+6.2f} +-{c['se_pts']:.2f}  "
                  f"picked {c['bought']:4d}/{c['offered']:<5d} ({c['pick_rate']:.0%})")
        return
    done = set()
    if os.path.exists(LOG):
        done = {json.loads(l)["game"] for l in open(LOG, encoding="utf-8")}
    todo = [(g, args.spec, args.seed) for g in range(args.games) if g not in done]
    t = time.time()
    with Pool(args.workers) as pool, open(LOG, "a", encoding="utf-8") as f:
        for i, r in enumerate(pool.imap_unordered(play_one, todo), 1):
            for row in r["rows"]:
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] game {r['game']}: {r['rounds']} rounds, "
                  f"{r['seconds']}s, {len(r['rows'])} buy decisions, "
                  f"{(time.time() - t) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
