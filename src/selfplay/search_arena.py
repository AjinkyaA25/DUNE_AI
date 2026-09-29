"""Arena: one `spec` seat (rotating) vs three heuristic players, in parallel.

  python -m src.selfplay.search_arena search:K5:M24 200 [--base]

Results are appended to reports/search_arena.jsonl.
"""
import json
import sys
import time
from multiprocessing import Pool

from src.ai.agents import make_agent
from src.data.card_definitions import setup_game

SPEC = sys.argv[1]
N = int(sys.argv[2])
BLOODLINES = "--base" not in sys.argv
OPP = "heuristic"


def play(g):
    seat = g % 4
    agents = [make_agent(SPEC if s == seat else OPP, seed=g * 4 + s) for s in range(4)]
    gs = setup_game(num_players=4, seed=70_000 + g, neutral_leaders=True,
                    use_bloodlines=BLOODLINES)
    t0 = time.time()
    n = 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    vp = [p.victory_points for p in gs.players]
    winners = [gs.winner] if gs.winner is not None else \
        [i for i, v in enumerate(vp) if v == max(vp)]
    st = getattr(agents[seat], "stats", {})
    return (1.0 / len(winners) if seat in winners else 0.0, vp[seat],
            gs.round, time.time() - t0, st.get("searched", 0), st.get("switched", 0))


if __name__ == "__main__":
    t = time.time()
    with Pool(28) as pool:
        res = pool.map(play, range(N), chunksize=1)
    wins = sum(r[0] for r in res)
    p = wins / N
    se = (p * (1 - p) / N) ** 0.5
    out = {"spec": SPEC, "bloodlines": BLOODLINES, "games": N,
           "win_share": round(p, 3), "ci95": [round(p - 1.96 * se, 3), round(p + 1.96 * se, 3)],
           "fair": 0.25, "avg_vp": round(sum(r[1] for r in res) / N, 2),
           "avg_rounds": round(sum(r[2] for r in res) / N, 2),
           "sec_per_game": round(sum(r[3] for r in res) / N, 1),
           "searched": sum(r[4] for r in res), "switched": sum(r[5] for r in res),
           "wall_seconds": round(time.time() - t)}
    print(json.dumps(out))
    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/search_arena.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
