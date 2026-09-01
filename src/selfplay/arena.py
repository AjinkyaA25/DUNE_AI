"""
Head-to-head evaluation between two agent specs, with seat rotation so each
agent occupies every seat equally.  Returns win-rate of agent A + a 95% CI.
"""
from __future__ import annotations

import math
from typing import Callable

from src.selfplay.runner import play_game


def _wilson(wins: float, n: int) -> tuple:
    if n == 0:
        return (0.0, 0.0, 1.0)
    z = 1.96
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def head_to_head(make_a: Callable, make_b: Callable, n_games: int = 200,
                 num_players: int = 4, base_seed: int = 10_000,
                 verbose: bool = False) -> dict:
    """
    make_a / make_b: zero-arg factories returning an Agent (called once per game
    so stateful agents get a clean instance).
    A occupies seat (game_idx % num_players); B fills the rest.
    """
    a_wins = 0.0
    b_wins = 0.0
    a_points = 0.0
    for g in range(n_games):
        a_seat = g % num_players
        agents = {}
        for s in range(num_players):
            agents[s] = make_a() if s == a_seat else make_b()
        res = play_game(agents, num_players=num_players, seed=base_seed + g,
                        record=False)
        # split ties equally
        winners = [i for i, v in enumerate(res.final_vp)
                   if v == max(res.final_vp)]
        share = 1.0 / len(winners)
        if a_seat in winners:
            a_wins += share
            a_points += share
        b_share = sum(share for w in winners if w != a_seat)
        b_wins += b_share
        if verbose and (g + 1) % 25 == 0:
            print(f"  {g+1}/{n_games}  A winrate ~ {a_wins/(g+1):.3f}")

    wr, lo, hi = _wilson(a_wins, n_games)
    # normalise vs a fair share (1/num_players)
    return {
        "n": n_games,
        "a_winrate": wr,
        "a_winrate_ci": (lo, hi),
        "fair_share": 1.0 / num_players,
        "a_vs_fair": wr / (1.0 / num_players),
    }


if __name__ == "__main__":
    import argparse
    from src.ai.agents import make_agent

    ap = argparse.ArgumentParser(description="Head-to-head agent evaluation")
    ap.add_argument("--a", default="heuristic", help="agent spec for player A")
    ap.add_argument("--b", default="random", help="agent spec for the other seats")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--players", type=int, default=4)
    args = ap.parse_args()

    r = head_to_head(lambda: make_agent(args.a), lambda: make_agent(args.b),
                     n_games=args.games, num_players=args.players, verbose=True)
    print(f"\nA = {args.a!r}   vs   B = {args.b!r}   ({args.games} games, "
          f"{args.players}p)")
    print(f"  A win-rate : {r['a_winrate']:.3f}  "
          f"(95% CI {r['a_winrate_ci'][0]:.3f}-{r['a_winrate_ci'][1]:.3f})")
    print(f"  fair share : {r['fair_share']:.3f}")
    print(f"  A vs fair  : {r['a_vs_fair']:.2f}x")
