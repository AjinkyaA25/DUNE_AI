"""What the AI buys (and how much it dilutes its deck) and where it puts
spies, vs the human video games."""
import collections
import sys
from multiprocessing import Pool

SPEC = sys.argv[1] if len(sys.argv) > 1 else "heuristic"


def play(g):
    from src.data.card_definitions import setup_game
    from src.ai.agents import make_agent
    from src.game.gameState import ActionType
    gs = setup_game(num_players=4, seed=70_000 + g, neutral_leaders=True, use_bloodlines=True)
    ag = [make_agent(SPEC, seed=g * 4 + i) for i in range(4)]
    buys, costs, spies = collections.Counter(), [], collections.Counter()
    reveal_buys = collections.defaultdict(int)
    reveals = 0
    spy_sends = infiltrate = gather = 0
    n = 0
    while not gs.game_over and n < 4000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        a = ag[pid].select_action(gs, pid, gs.get_valid_actions(pid))
        t = a.action_type
        if t == ActionType.REVEAL_TURN:
            reveals += 1
        if t == ActionType.ACQUIRE_CARD:
            card = next((c for c in gs.imperium_row if c.name == a.acquire_card_name), None)
            buys[a.acquire_card_name] += 1
            costs.append(getattr(card, "cost", 0) or 0)
            reveal_buys[(pid, gs.round)] += 1
        if t == ActionType.ACQUIRE_RESERVE:
            nm = "Prepare the Way" if a.reserve_type == "prepare_the_way" else "The Spice Must Flow"
            buys[nm] += 1
            costs.append(2 if nm == "Prepare the Way" else 9)
            reveal_buys[(pid, gs.round)] += 1
        if t == ActionType.RESOLVE_SPY:
            spies[a.spy_post_name] += 1
        if t == ActionType.AGENT_TURN:
            infiltrate += bool(a.use_infiltrate)
            gather += bool(a.use_gather_intelligence)
        gs.step(a)
        n += 1
    decks = [len(p.deck) + len(p.hand) + len(p.discard) + len(p.in_play) for p in gs.players]
    return buys, costs, spies, reveals, len(reveal_buys), sum(decks) / 4, infiltrate, gather, gs.round


if __name__ == "__main__":
    with Pool(24) as pool:
        res = pool.map(play, range(200))
    G = len(res)
    buys = sum((r[0] for r in res), collections.Counter())
    costs = [c for r in res for c in r[1]]
    spies = sum((r[2] for r in res), collections.Counter())
    reveals = sum(r[3] for r in res)
    rev_with_buy = sum(r[4] for r in res)
    print(f"AI = {SPEC}, {G} games, ends round {sum(r[8] for r in res) / G:.2f}")
    print(f"cards bought per player per game: {sum(buys.values()) / (4 * G):.1f}; "
          f"avg cost {sum(costs) / len(costs):.2f}; reveal turns with no buy: "
          f"{1 - rev_with_buy / reveals:.0%}; final deck size {sum(r[5] for r in res) / G:.1f}")
    print(f"share of buys costing <=2: {sum(1 for c in costs if c <= 2) / len(costs):.0%}; "
          f">=5: {sum(1 for c in costs if c >= 5) / len(costs):.0%}")
    print("most bought:", [(k, round(v / (4 * G), 2)) for k, v in buys.most_common(12)], "(per player-game)")
    print(f"spies placed per player per game: {sum(spies.values()) / (4 * G):.1f}; "
          f"infiltrate {sum(r[6] for r in res) / (4 * G):.2f}, gather intel {sum(r[7] for r in res) / (4 * G):.2f} per player-game")
    print("spy posts:", [(k, round(100 * v / sum(spies.values()))) for k, v in spies.most_common(12)], "(% of placements)")
