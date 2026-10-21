"""Speed-up safety net: record games (every move + every heuristic score) and
check that an optimised engine / heuristic reproduces them exactly.

    python -m src.selfplay.parity record --out reports/parity_ref.json [--games 12]
    python -m src.selfplay.parity check  --ref reports/parity_ref.json
    python -m src.selfplay.parity time   [--games 12]          # seconds per game, one process

Games: PPO (the UI net, all its decisions scored by the heuristic) in seats
0-1 and Best AI in 2-3, real Leaders, Bloodlines - the training-game mix.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

SPEC_PPO = "ppo:net=models_ui/ppo.npz:tuned=config/ppo/base.json:greedy"
SPEC_H = "heuristic:tuned=config/ppo/base.json"


def play(g: int, trace: bool = True):
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.leader.leader_definitions import LEADER_NAMES
    r = random.Random(g)
    gs = setup_game(num_players=4, seed=g, leaders=r.sample(list(LEADER_NAMES), 4), use_bloodlines=True)
    agents = [make_agent(SPEC_PPO if s < 2 else SPEC_H, seed=g * 4 + s) for s in range(4)]
    out, n = [], 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        if trace and len(valid) > 1:
            h = agents[pid].h if hasattr(agents[pid], "h") else agents[pid]
            sc = [round(float(h.score(gs, pid, a)), 6) for a in valid]
        else:
            sc = None
        a = agents[pid].select_action(gs, pid, valid)
        if trace:
            out.append([pid, repr(a), sc])
        gs.step(a)
        n += 1
    return {"moves": out, "vp": [p.victory_points for p in gs.players], "rounds": gs.round}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "check", "time"])
    ap.add_argument("--out", default="reports/parity_ref.json")
    ap.add_argument("--ref", default="reports/parity_ref.json")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--base", type=int, default=777000)
    A = ap.parse_args()
    if A.cmd == "record":
        games = {str(A.base + g): play(A.base + g) for g in range(A.games)}
        json.dump(games, open(A.out, "w"))
        print(f"recorded {len(games)} games, {sum(len(x['moves']) for x in games.values())} moves -> {A.out}")
    elif A.cmd == "check":
        ref = json.load(open(A.ref))
        bad = 0
        for g, want in ref.items():
            got = play(int(g))
            for i, (w, x) in enumerate(zip(want["moves"], got["moves"])):
                if w != x:
                    print(f"game {g} move {i}: expected {w[:2]} scores {w[2]}\n   got {x[:2]} scores {x[2]}")
                    bad += 1
                    break
            else:
                if len(want["moves"]) != len(got["moves"]) or want["vp"] != got["vp"]:
                    print(f"game {g}: length / VP differ")
                    bad += 1
        print("PARITY OK" if not bad else f"PARITY FAILED in {bad} games")
        sys.exit(1 if bad else 0)
    else:
        t = time.time()
        rounds = [play(A.base + g, trace=False)["rounds"] for g in range(A.games)]
        dt = time.time() - t
        print(f"{A.games} games, {dt / A.games:.2f} s/game, avg rounds {sum(rounds) / len(rounds):.2f}")


if __name__ == "__main__":
    main()
