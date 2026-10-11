#!/usr/bin/env python
"""
Fit the heuristic's (phase-specific) weights to WINNING, not to imitation.

Coordinate ascent: for each knob in the grid, play the candidate weights in
one rotating seat against a fixed field and keep a change only if it raises
the win share by more than MIN_GAIN - and still does on a fresh set of
games (guards against keeping lucky draws). Games use fixed seeds, so every
candidate sees the same deals (paired comparison).

Phases (agents.phase_of): early = rounds 1-3, mid = 4-5, late = 6+.

  python -m src.selfplay.tune_by_wins --start config/heuristic_me.json \
      --out config/heuristic_phased.json --games 2000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from multiprocessing import Pool

KNOBS = {
    "res_solari": [0.5, 0.75, 1.0, 1.5], "res_spice": [0.5, 0.75, 1.0, 1.5],
    "res_water": [0.5, 0.75, 1.0, 1.5], "res_troops": [0.5, 1.0, 1.5],
    "res_draw": [0.5, 1.0, 1.5], "res_intrigue": [0.5, 1.0, 2.0],
    "res_spy": [1.0, 2.0, 3.0], "influence": [1.0, 1.5, 2.0, 3.0],
    "combat": [0.25, 0.5, 1.0, 1.5], "reveal_bias": [-2.0, 0.0, 2.0],
}
# whole-game knobs tried before the phase-specific ones
GLOBAL = {"troop_hold": [0.5, 1.0, 1.5, 2.0], "ci_hold": [0.5, 1.0, 1.5, 2.5],
          "plot_hold": [0.0, 0.6, 1.2, 2.0]}
FIELD = ["heuristic", "heuristic:tuned=config/heuristic_winners.json",
         "heuristic:tuned=config/heuristic_me.json"]
MIN_GAIN = 0.012


def _game(args):
    g, cand_path, base = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    seat = g % 4
    rng = random.Random(base + g)
    opp = [FIELD[(g // 4 + i) % len(FIELD)] for i in range(3)]
    rng.shuffle(opp)
    specs = opp[:seat] + [f"heuristic:tuned={cand_path}"] + opp[seat:]
    agents = [make_agent(s, seed=(base + g) * 4 + i) for i, s in enumerate(specs)]
    gs = setup_game(num_players=4, seed=base + g, neutral_leaders=True, use_bloodlines=True)
    n = 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    vp = [p.victory_points for p in gs.players]
    win = [gs.winner] if gs.winner is not None else [i for i, v in enumerate(vp) if v == max(vp)]
    return 1.0 / len(win) if seat in win else 0.0


class Evaluator:
    def __init__(self, games: int, workers: int):
        self.games, self.pool = games, Pool(workers)
        # relative path: agent specs are ':'-separated, so no drive letter
        os.makedirs(os.path.join("reports", "tbw_tmp"), exist_ok=True)
        self.dir = os.path.relpath(tempfile.mkdtemp(prefix="run_", dir=os.path.join("reports", "tbw_tmp")))
        self.n = 0

    def win_share(self, tuning: dict, base: int) -> float:
        self.n += 1
        path = os.path.join(self.dir, f"c{self.n}.json")
        json.dump(tuning, open(path, "w"))
        r = self.pool.map(_game, [(g, path, base) for g in range(self.games)], chunksize=8)
        return sum(r) / len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="weights to start from (json)")
    ap.add_argument("--out", default="config/heuristic_phased.json")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--phases", default="early,mid,late")
    args = ap.parse_args()
    cur = json.load(open(args.start)) if args.start else {}
    ev = Evaluator(args.games, args.workers)
    BASE, CHECK = 900_000, 950_000          # tuning deals / confirmation deals
    best = ev.win_share(cur, BASE)
    log = [{"start": args.start, "win": best}]
    print(f"start {args.start}: {best:.3f} (fair 0.25, field {FIELD})", flush=True)
    t0 = time.time()
    for sweep in range(args.sweeps):
        improved = False
        plan = [(k, v) for k, v in GLOBAL.items()] + [
            (f"{ph}.{knob}", values) for ph in args.phases.split(",")
            for knob, values in KNOBS.items()]
        for key, values in plan:
            if True:
                knob = key.split(".", 1)[-1]
                now = cur.get(key, cur.get(knob))
                for v in values:
                    if now is not None and v == now:
                        continue
                    trial = {**cur, key: v}
                    sc = ev.win_share(trial, BASE)
                    if sc > best + MIN_GAIN:
                        # confirm on unseen deals against the current weights
                        a, b = ev.win_share(trial, CHECK), ev.win_share(cur, CHECK)
                        ok = a > b + MIN_GAIN / 2
                        print(f"  {key}={v}: {sc:.3f} vs {best:.3f}; check {a:.3f} vs {b:.3f}"
                              f" -> {'KEEP' if ok else 'reject'}  ({(time.time() - t0) / 60:.0f} min)",
                              flush=True)
                        if ok:
                            cur, best, improved = trial, sc, True
                            log.append({"set": key, "value": v, "win": sc, "check": [a, b]})
                            json.dump(cur, open(args.out, "w"), indent=1)
        if not improved:
            break
    json.dump(cur, open(args.out, "w"), indent=1)
    json.dump(log, open(args.out.replace(".json", "_log.json"), "w"), indent=1)
    print(f"done: {best:.3f} -> {args.out}: {cur}", flush=True)


if __name__ == "__main__":
    main()
