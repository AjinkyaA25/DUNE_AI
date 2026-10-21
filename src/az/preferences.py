"""What the AlphaZero net prefers: cards, techs, contracts, skills, spaces.

Each listed net plays games against itself (all four seats, policy only,
best move) and every decision with a choice is tallied: for each option,
how often it was available and how often the net took it.

    python -m src.az.preferences [--nets 10,20,30,latest] [--games 120]

Writes reports/az_preferences.json and prints the top picks per category.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np


def _item(gs, a):
    """(category, name) for the decisions worth tallying, else None."""
    from src.game.gameState import ActionType as T
    t = a.action_type
    if t == T.ACQUIRE_CARD:
        return "card", a.acquire_card_name
    if t == T.ACQUIRE_RESERVE:
        return "card", {"prepare_the_way": "Prepare the Way",
                        "spice_must_flow": "The Spice Must Flow"}.get(a.reserve_type, a.reserve_type)
    if t == T.RESOLVE_CONTRACT:
        try:
            return "contract", gs.contracts_on_board[a.contract_index].name
        except (IndexError, AttributeError, TypeError):
            return None
    if t == T.RESOLVE_BL_CHOICE and gs.bl and a.choice:
        c = a.choice
        if c.startswith("stack:"):
            _d = gs.bl.tech_at(a.player_id, int(c.split(":")[1]))[0]
            st = [_d] if _d is not None else []
            return ("tech", st[0].name) if st else None
        for pre in ("skill:", "board:"):         # Plasteel Blades / board commander
            if c.startswith(pre) and c[len(pre):]:
                return "skill", c[len(pre):]
    if t == T.AGENT_TURN and a.space_name:
        return "space", a.space_name
    return None


def play(args):
    net_path, g = args
    from src.az.mcts import active, evaluate, legal
    from src.az.net import AZNet
    from src.data.card_definitions import setup_game
    net = AZNet.load(net_path)
    gs = setup_game(4, seed=7_000_000 + g, neutral_leaders=True, use_bloodlines=True)
    offered, chosen, prior = (collections.Counter() for _ in range(3))
    steps = 0
    while not gs.game_over and steps < 4000:
        pid = active(gs)
        acts = legal(gs, pid)
        if len(acts) == 1:
            gs.step(acts[0]); steps += 1
            continue
        pr, _ = evaluate(net, gs, pid, acts)
        k = int(np.argmax(pr))
        items = [_item(gs, a) for a in acts]
        # a space counts once per decision, however many cards could go there
        seen = set()
        for it, p in zip(items, pr):
            if it is None:
                continue
            prior[it] += float(p)
            if it not in seen:
                seen.add(it); offered[it] += 1
        if items[k] is not None:
            chosen[items[k]] += 1
        gs.step(acts[k]); steps += 1
    vp = [p.victory_points for p in gs.players]
    return ({"|".join(k): v for k, v in offered.items()},
            {"|".join(k): v for k, v in chosen.items()}, gs.round, vp)


def main():
    from multiprocessing import Pool
    from src.az.agent import latest_net
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="10,20,30,latest")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    A = ap.parse_args()
    nets = []
    for n in A.nets.split(","):
        path = latest_net() if n == "latest" else os.path.join("models_az", f"net_{n}.npz")
        if path and os.path.exists(path):
            nets.append((os.path.basename(path)[4:-4], path))
    out = {}
    with Pool(A.workers, maxtasksperchild=20) as pool:
        for gen, path in nets:
            res = pool.map(play, [(path, g) for g in range(A.games)])
            off, cho = collections.Counter(), collections.Counter()
            for o, c, _, _ in res:
                off.update(o); cho.update(c)
            out[gen] = {"games": len(res),
                        "rounds": float(np.mean([r[2] for r in res])),
                        "vp_max": float(np.mean([max(r[3]) for r in res])),
                        "items": {k: [off[k], cho.get(k, 0)] for k in off}}
            print(f"gen {gen}: {len(res)} games, {out[gen]['rounds']:.2f} rounds, "
                  f"winner {out[gen]['vp_max']:.2f} VP", flush=True)
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "az_preferences.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    last = list(out)[-1]
    for cat, min_off in (("card", 15), ("tech", 8), ("contract", 8), ("skill", 6), ("space", 30)):
        rows = []
        for k, (o, c) in out[last]["items"].items():
            cc, name = k.split("|", 1)
            if cc == cat and o >= min_off:
                hist = [out[g]["items"].get(k, [0, 0]) for g in out]
                rows.append((c / o, name, o, [h[1] / h[0] if h[0] else None for h in hist]))
        rows.sort(reverse=True)
        print(f"\n{cat}: pick rate when available (gen {' / '.join(out)})")
        fmt = lambda x: "  -  " if x is None else f"{x:5.0%}"
        for r, name, o, hist in rows[:12]:
            print(f"  {name:32s} {'  '.join(fmt(x) for x in hist)}   (offered {o})")
        if len(rows) > 12:
            print("  ... lowest:")
            for r, name, o, hist in rows[-5:]:
                print(f"  {name:32s} {'  '.join(fmt(x) for x in hist)}   (offered {o})")


if __name__ == "__main__":
    main()
