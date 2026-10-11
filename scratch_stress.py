"""Stress test: many full games with every agent type; reports crashes and
games that don't finish (per-game move cap and wall-clock cap)."""
import collections
import sys
import time
import traceback
from multiprocessing import Pool


def run(args):
    g, specs = args
    from src.data.card_definitions import setup_game
    from src.ai.agents import make_agent
    gs = setup_game(num_players=4, seed=40_000 + g, neutral_leaders=True, use_bloodlines=True)
    ag = [make_agent(specs[(g + i) % len(specs)], seed=g * 4 + i) for i in range(4)]
    kinds = collections.Counter()
    n, t0 = 0, time.time()
    last = None
    try:
        while not gs.game_over and n < 4000 and time.time() - t0 < 240:
            pid = gs.player_in_reveal_buy
            if pid is None:
                pid = gs.get_current_player_id()
            pc = gs.pending_choice_for(pid)
            if pc:
                kinds[pc.kind] += 1
            valid = gs.get_valid_actions(pid)
            a = ag[pid].select_action(gs, pid, valid)
            last = (gs.round, gs.phase.value, pid, a.action_type.value, len(valid))
            gs.step(a)
            n += 1
        if gs.game_over:
            return None, kinds, gs.round
        return f"game {g} {specs}: not finished after {n} moves / {time.time() - t0:.0f}s; last {last}", kinds, gs.round
    except Exception:
        return f"game {g} {specs}: " + traceback.format_exc()[-700:], kinds, gs.round


if __name__ == "__main__":
    t = time.time()
    jobs = [(g, ["random", "heuristic", "heuristic:tuned=config/heuristic_me.json"]) for g in range(240)]
    jobs += [(1000 + g, ["search:K3:M2", "heuristic", "random"]) for g in range(12)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as pool:
        res = list(pool.imap_unordered(run, jobs))
    errs = [r[0] for r in res if r[0]]
    K = sum((r[1] for r in res), collections.Counter())
    print(f"{len(res)} games in {time.time() - t:.0f}s, problems: {len(errs)}")
    for e in errs[:4]:
        print(e)
    print("choices offered:", dict(K.most_common()))
    print("avg rounds:", round(sum(r[2] for r in res) / len(res), 2))
