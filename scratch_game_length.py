"""Game length when all four seats are the same AI: the round the game ends,
how often it ends by someone reaching 10 VP vs running out of rounds, and
the winner's VP. Humans (video games) end on round 7.7 on average."""
import collections
import json
import sys
from multiprocessing import Pool

AGENTS = [
    ("original heuristic (old fight scoring)", "heuristic:tuned=config/fight_old.json"),
    ("heuristic + fight model", "heuristic"),
    ("your style + fight model", "heuristic:tuned=config/heuristic_me.json"),
    ("winners' style + fight model", "heuristic:tuned=config/heuristic_winners.json"),
    ("value+policy v2 (old training)", "value:models_policy/value_best.npz:models_policy/policy_best.npz"),
    ("RL retrained on fixed rules (iter 2)", "value:models_rl_fixed/value_v02.npz:models_rl_fixed/policy_v02.npz"),
]


def play(args):
    g, spec = args
    from src.data.card_definitions import setup_game
    from src.ai.agents import make_agent
    gs = setup_game(num_players=4, seed=60_000 + g, neutral_leaders=True, use_bloodlines=True)
    ag = [make_agent(spec, seed=g * 4 + i) for i in range(4)]
    n = 0
    while not gs.game_over and n < 4000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(ag[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    vp = [p.victory_points for p in gs.players]
    return gs.round, max(vp), max(vp) >= 10


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = []
    with Pool(24) as pool:
        for name, spec in AGENTS:
            res = pool.map(play, [(g, spec) for g in range(games)], chunksize=4)
            rounds = collections.Counter(r for r, _, _ in res)
            row = {"name": name, "games": games,
                   "avg_round": round(sum(r for r, _, _ in res) / games, 2),
                   "by_10vp": round(sum(1 for *_, t in res if t) / games, 3),
                   "winner_vp": round(sum(v for _, v, _ in res) / games, 2),
                   "rounds": dict(sorted(rounds.items()))}
            out.append(row)
            print(json.dumps(row), flush=True)
    json.dump(out, open("reports/game_length.json", "w"), indent=1)
