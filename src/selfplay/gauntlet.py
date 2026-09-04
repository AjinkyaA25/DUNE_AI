"""
Multi-agent evaluation:

  * mixed-seat mode  -- pit several agent specs against each other, rotating
    which seat holds which spec so every spec plays every seat equally.
    (e.g. 1 random + 3 trained, or v10 vs 3x v05).

      python -m src.selfplay.gauntlet --seats random,value:models/value_best.npz,\
value:models/value_best.npz,value:models/value_best.npz --games 240

  * ladder mode -- round-robin a set of model checkpoints so you can watch
    win-rate climb with version number ("progress metric").  Each pairing plays
    `challenger` in one seat and `defender` in the other three.

      python -m src.selfplay.gauntlet --ladder "models/value_v*.npz" --games 120
      python -m src.selfplay.gauntlet --ladder models/value_v03.npz,models/value_v08.npz
"""
from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from typing import Callable, Dict, List

from src.ai.agents import make_agent
from src.selfplay.runner import play_game


def _winners(final_vp: List[int]) -> List[int]:
    top = max(final_vp)
    return [i for i, v in enumerate(final_vp) if v == top]


def mixed_seat(specs: List[str], n_games: int, base_seed: int = 30_000,
               num_players: int = 4) -> Dict[str, float]:
    """Every spec plays every seat equally; returns win-rate per spec."""
    wins: Dict[str, float] = defaultdict(float)
    games_as: Dict[str, int] = defaultdict(int)
    rot = list(range(num_players))
    for g in range(n_games):
        # rotate the spec->seat assignment
        shift = g % num_players
        seat_spec = {s: specs[(s + shift) % num_players] for s in range(num_players)}
        agents = {s: make_agent(seat_spec[s], seed=base_seed + g * 10 + s)
                  for s in range(num_players)}
        res = play_game(agents, num_players=num_players, seed=base_seed + g,
                        record=False)
        w = _winners(res.final_vp)
        share = 1.0 / len(w)
        for s in range(num_players):
            games_as[seat_spec[s]] += 1
            if s in w:
                wins[seat_spec[s]] += share
    return {sp: wins[sp] / max(1, games_as[sp]) for sp in set(specs)}


def duel(challenger: str, defender: str, n_games: int, base_seed: int,
         num_players: int = 4) -> float:
    """challenger in one (rotating) seat vs `defender` in the rest. Win-rate of challenger."""
    cw = 0.0
    for g in range(n_games):
        cseat = g % num_players
        agents = {s: make_agent(challenger if s == cseat else defender,
                                seed=base_seed + g * 10 + s)
                  for s in range(num_players)}
        res = play_game(agents, num_players=num_players, seed=base_seed + g,
                        record=False)
        w = _winners(res.final_vp)
        if cseat in w:
            cw += 1.0 / len(w)
    return cw / n_games


def ladder(paths: List[str], n_games: int, num_players: int = 4) -> None:
    specs = [f"value:{p}" for p in paths]
    names = [p.split("/")[-1].split("\\")[-1] for p in paths]
    print(f"Ladder round-robin ({n_games} games/pairing, {num_players}p, "
          f"challenger vs 3x defender):\n")
    header = "challenger \\ defender".ljust(22) + "".join(n[:12].rjust(13) for n in names)
    print(header)
    for i, (ci, cn) in enumerate(zip(specs, names)):
        row = cn[:20].ljust(22)
        for j, dj in enumerate(specs):
            if i == j:
                row += "     -   ".rjust(13)
            else:
                wr = duel(ci, dj, n_games, base_seed=50_000 + i * 1000 + j * 7,
                          num_players=num_players)
                fair = 1.0 / num_players
                row += f"{wr:.2f} ({wr/fair:.1f}x)".rjust(13)
        print(row)
    print(f"\n(fair share = {1.0/num_players:.2f}; >1.0x means the challenger "
          f"beats that defender.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-agent gauntlet")
    ap.add_argument("--seats", type=str,
                    help="comma-separated agent specs, one per seat (mixed-seat mode)")
    ap.add_argument("--ladder", type=str,
                    help="comma-separated model paths OR a glob like 'models/value_v*.npz'")
    ap.add_argument("--games", type=int, default=160)
    ap.add_argument("--players", type=int, default=4)
    args = ap.parse_args()

    if args.ladder:
        if any(ch in args.ladder for ch in "*?["):
            paths = sorted(glob.glob(args.ladder))
        else:
            paths = [p.strip() for p in args.ladder.split(",") if p.strip()]
        if len(paths) < 2:
            ap.error("--ladder needs at least 2 model files")
        ladder(paths, args.games, args.players)
        return

    if not args.seats:
        ap.error("give --seats or --ladder")
    specs = [s.strip() for s in args.seats.split(",")]
    if len(specs) != args.players:
        ap.error(f"--seats needs exactly {args.players} entries")
    wr = mixed_seat(specs, args.games, num_players=args.players)
    fair = 1.0 / args.players
    print(f"Mixed-seat gauntlet ({args.games} games, {args.players}p, "
          f"seat-rotated):\n")
    for sp in sorted(wr, key=lambda s: -wr[s]):
        print(f"  {sp:44.44}  win-rate {wr[sp]:.3f}   ({wr[sp]/fair:.2f}x fair)")


if __name__ == "__main__":
    main()
