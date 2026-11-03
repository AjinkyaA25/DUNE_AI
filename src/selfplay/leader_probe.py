"""What does a spec actually do with a Leader's signature decisions? (user,
2026-10-25: "steersman is not learning which navigation cards belong in which
slots ... opening theory for liet is changed").

Plays games with the Leader in a rotating seat (the spec in every seat) and
tallies:
  * Steersman: the Navigation card returned to the box and the card laid in
    each slot (card 4 wants slot 1, card 3 wants slot 4);
  * any Leader: its round-1 and round-2 Agent turns (space + card played).

    python -m src.selfplay.leader_probe --leader "Liet Kynes" --spec SPEC [--spec SPEC2] --games 400
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from multiprocessing import Pool


def play(args):
    g, spec, leader = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.gameState import ActionType
    from src.game.leader.leader_definitions import LEADER_NAMES
    r = random.Random(9_000_000 + g)
    seat = g % 4
    others = r.sample([n for n in LEADER_NAMES if n != leader], 3)
    lineup = others[:seat] + [leader] + others[seat:]
    gs = setup_game(num_players=4, seed=9_000_000 + g, leaders=lineup, use_bloodlines=True)
    agents = [make_agent(spec, seed=g * 4 + s) for s in range(4)]
    turns = []
    n = 0
    while not gs.game_over and gs.round <= 2 and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        a = agents[pid].select_action(gs, pid, gs.get_valid_actions(pid))
        if pid == seat and a.action_type == ActionType.AGENT_TURN:
            turns.append((gs.round, a.space_name, a.card_name))
        gs.step(a)
        n += 1
    st = gs.players[seat].leader.state
    drop = next((i for i in st.get("nav_hand", []) if i not in st.get("nav_keep", st.get("nav", []))), None)
    return {"turns": turns, "nav": list(st.get("nav", [])), "drop": drop, "hand": list(st.get("nav_hand", []))}


def report(res, leader):
    n = len(res)
    if leader.startswith("Steersman"):
        drops = Counter(x["drop"] for x in res)
        print("  returned to the box:", ", ".join(f"#{k} {v / n:.0%}" for k, v in drops.most_common()))
        for slot in range(5):
            c = Counter(x["nav"][slot] for x in res if len(x["nav"]) > slot)
            print(f"  slot {slot + 1}:", ", ".join(f"#{k} {v / n:.0%}" for k, v in c.most_common(5)))
        has4 = [x for x in res if 4 in x["nav"]]
        has3 = [x for x in res if 3 in x["nav"]]
        print(f"  card 4 in slot 1: {sum(x['nav'][0] == 4 for x in has4) / max(1, len(has4)):.0%} of {len(has4)} games"
              f" | card 3 in slot 4: {sum(len(x['nav']) > 3 and x['nav'][3] == 3 for x in has3) / max(1, len(has3)):.0%}"
              f" of {len(has3)}")
    for rnd in (1, 2):
        c = Counter((t[1], t[2]) for x in res for t in x["turns"] if t[0] == rnd)
        first = Counter((x["turns"][0][1], x["turns"][0][2]) for x in res
                        if x["turns"] and x["turns"][0][0] == rnd) if rnd == 1 else None
        if first:
            print(f"  round 1 FIRST Agent:", "; ".join(f"{s} <- {cd} {v / n:.0%}" for (s, cd), v in first.most_common(6)))
        print(f"  round {rnd} Agent turns:", "; ".join(f"{s} <- {cd} {v / n:.0%}" for (s, cd), v in c.most_common(8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leader", required=True)
    ap.add_argument("--spec", action="append", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=24)
    A = ap.parse_args()
    with Pool(A.workers, maxtasksperchild=40) as pool:
        for spec in A.spec:
            res = pool.map(play, [(g, spec, A.leader) for g in range(A.games)], chunksize=4)
            print(f"{A.leader} | {spec} | {len(res)} games")
            report(res, A.leader)


if __name__ == "__main__":
    main()
