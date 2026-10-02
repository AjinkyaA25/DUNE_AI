"""Rounds 1-3: where agents go and what gets bought (cost), AI self-play vs
the human video games."""
import collections
import sys
from multiprocessing import Pool

SPEC = sys.argv[1] if len(sys.argv) > 1 else "heuristic:tuned=config/heuristic_phased_snapshot.json"


def run(g):
    from src.data.card_definitions import setup_game
    from src.ai.agents import make_agent
    from src.game.gameState import ActionType
    gs = setup_game(num_players=4, seed=97_000 + g, neutral_leaders=True, use_bloodlines=True)
    ag = [make_agent(SPEC, seed=g * 4 + i) for i in range(4)]
    spaces, costs, trashed = collections.Counter(), [], collections.Counter()
    n = 0
    while not gs.game_over and n < 4000 and gs.round <= 3:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        a = ag[pid].select_action(gs, pid, gs.get_valid_actions(pid))
        if a.action_type == ActionType.AGENT_TURN:
            spaces[a.space_name] += 1
        if a.action_type == ActionType.ACQUIRE_CARD:
            c = next((c for c in gs.imperium_row if c.name == a.acquire_card_name), None)
            costs.append(getattr(c, "cost", 0))
        if a.action_type == ActionType.ACQUIRE_RESERVE:
            costs.append(2 if a.reserve_type == "prepare_the_way" else 9)
        if a.action_type == ActionType.RESOLVE_TRASH and a.trash_card_name:
            trashed[a.trash_card_name] += 1
        gs.step(a)
        n += 1
    return spaces, costs, trashed


def humans():
    sys.path[:0] = ["video_scrape", "."]
    import tune_heuristic as T
    from src.data.card_definitions import create_imperium_cards
    cost = {c.name: c.cost for c in create_imperium_cards()}
    cost.update({"prepare_the_way": 2, "spice_must_flow": 9})
    pos = T.load_positions(include_ui=False)
    early = [x for x in pos if x.get("round", x["gs"].round) <= 3]
    games = len({x["game"] for x in pos})
    sp = collections.Counter(x["human"][6:] for x in early if x["type"] == "agent_turn")
    bc = [cost.get(x["human"][4:]) for x in early if x["type"] == "buy"]
    bc = [c for c in bc if c is not None]
    return sp, bc, games


if __name__ == "__main__":
    with Pool(12) as pool:
        res = pool.map(run, range(200))
    sp = sum((r[0] for r in res), collections.Counter())
    costs = [c for r in res for c in r[1]]
    tr = sum((r[2] for r in res), collections.Counter())
    hsp, hcost, hg = humans()
    tot_ai, tot_h = sum(sp.values()), sum(hsp.values())
    print(f"ROUNDS 1-3   AI = {SPEC}")
    print(f"{'space':22s} {'AI %':>6s} {'human %':>8s}")
    for k in sorted(set(sp) | set(hsp), key=lambda k: -(sp[k] / tot_ai + hsp[k] / tot_h)):
        print(f"{k[:22]:22s} {100 * sp[k] / tot_ai:6.1f} {100 * hsp[k] / tot_h:8.1f}")
    for name, cs in (("AI", costs), ("humans", hcost)):
        print(f"{name}: buys rounds 1-3 avg cost {sum(cs) / len(cs):.2f}; cost 4+: "
              f"{sum(1 for c in cs if c >= 4) / len(cs):.0%}; cost <=2: {sum(1 for c in cs if c <= 2) / len(cs):.0%}")
    print("AI trashes in rounds 1-3 (per player-game):", [(k, round(v / 800, 2)) for k, v in tr.most_common(6)])
