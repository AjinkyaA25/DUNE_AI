"""How the heuristic AI deploys troops and uses intrigues (new engine rules)."""
import collections
import time
from multiprocessing import Pool


def run(g):
    from src.data.card_definitions import setup_game
    from src.ai.agents import make_agent
    from src.game.gameState import ActionType
    from src.game.intrigue.intrigue import IntrigueTiming
    gs = setup_game(num_players=4, seed=20_000 + g, neutral_leaders=True, use_bloodlines=True)
    ag = [make_agent("heuristic", seed=g * 4 + i) for i in range(4)]
    dep = []            # (chosen, max possible, round, garrison after)
    ci = collections.Counter()   # combat intrigue outcomes
    plays = collections.Counter()
    phases = collections.Counter()
    n = 0
    while not gs.game_over and n < 4000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        a = ag[pid].select_action(gs, pid, valid)
        if a.action_type == ActionType.RESOLVE_DEPLOY:
            mx = max(x.deploy_count for x in valid if x.action_type == ActionType.RESOLVE_DEPLOY)
            if mx > 0:
                dep.append((a.deploy_count, mx, gs.round, gs.players[pid].troops_garrison - a.deploy_count))
        if a.action_type == ActionType.PLAY_INTRIGUE:
            ic = next(c for c in gs.players[pid].intrigue_cards if c.name == a.intrigue_card_name)
            ts = ic.timing if isinstance(ic.timing, (set, tuple, list, frozenset)) else (ic.timing,)
            plays[a.intrigue_card_name] += 1
            phases["combat" if gs.phase.name == "COMBAT" else "plot"] += 1
            if gs.phase.name == "COMBAT":
                # did it change this player's placing?
                def place(q):
                    return sorted((gs.combat_strength.get(i, 0) for i in range(gs.num_players)), reverse=True).index(gs.combat_strength.get(q, 0))
                before = place(pid)
                gs.step(a)
                n += 1
                after = place(pid)
                ci["improved placing" if after < before else "no change"] += 1
                continue
        gs.step(a)
        n += 1
    held = sum(len(p.intrigue_cards) for p in gs.players)
    return dep, ci, plays, phases, held, gs.round


if __name__ == "__main__":
    t = time.time()
    with Pool(16) as pool:
        res = pool.map(run, range(200))
    dep = [d for r in res for d in r[0]]
    ci = sum((r[1] for r in res), collections.Counter())
    plays = sum((r[2] for r in res), collections.Counter())
    phases = sum((r[3] for r in res), collections.Counter())
    held = sum(r[4] for r in res)
    print(f"{len(res)} games, {time.time() - t:.0f}s")
    maxed = sum(1 for c, m, *_ in dep if c == m)
    print(f"deploy decisions (with 1+ troop available): {len(dep)}; deployed the MAX: {maxed / len(dep):.0%}; "
          f"deployed 0: {sum(1 for c, *_ in dep if c == 0) / len(dep):.0%}")
    for lo, hi in ((1, 3), (4, 5), (6, 10)):
        sub = [d for d in dep if lo <= d[2] <= hi]
        if sub:
            print(f"  rounds {lo}-{hi}: max {sum(1 for c, m, *_ in sub if c == m) / len(sub):.0%}, "
                  f"avg deployed {sum(c for c, *_ in sub) / len(sub):.1f} of avg {sum(m for _, m, *_ in sub) / len(sub):.1f} possible, "
                  f"garrison left 0: {sum(1 for d in sub if d[3] == 0) / len(sub):.0%}")
    print(f"intrigues played per game: {sum(plays.values()) / len(res):.1f} ({dict(phases)}); held at game end: {held / len(res):.1f} per game")
    print("combat intrigues in combat:", dict(ci))
    print("most played:", plays.most_common(12))
