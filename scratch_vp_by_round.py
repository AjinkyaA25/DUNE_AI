"""Leader VP / average VP at the end of each round in AI self-play, to
compare with the human video games (leader reaches ~9.2 VP after round 7)."""
import collections
import sys
from multiprocessing import Pool

SPEC = sys.argv[1] if len(sys.argv) > 1 else "heuristic:tuned=config/heuristic_me.json"


def run(g):
    from src.data.card_definitions import setup_game
    from src.ai.agents import make_agent
    gs = setup_game(num_players=4, seed=90_000 + g, neutral_leaders=True, use_bloodlines=True)
    ag = [make_agent(SPEC, seed=g * 4 + i) for i in range(4)]
    out, n, r0 = {}, 0, gs.round
    while not gs.game_over and n < 4000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(ag[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
        if gs.round != r0:
            vp = [p.victory_points for p in gs.players]
            out[r0] = (max(vp), sum(vp) / 4)
            r0 = gs.round
    vp = [p.victory_points for p in gs.players]
    out[r0] = (max(vp), sum(vp) / 4)
    return out


if __name__ == "__main__":
    with Pool(8) as pool:
        res = pool.map(run, range(int(sys.argv[2]) if len(sys.argv) > 2 else 120))
    lead, avg = collections.defaultdict(list), collections.defaultdict(list)
    for o in res:
        for r, (l, a) in o.items():
            lead[r].append(l)
            avg[r].append(a)
    print(f"AI {SPEC} self-play - round: leader VP / avg VP (games still running)")
    print("  " + "  ".join(f"r{r}:{sum(lead[r]) / len(lead[r]):.1f}/{sum(avg[r]) / len(avg[r]):.1f}({len(lead[r])})"
                           for r in sorted(lead)))
