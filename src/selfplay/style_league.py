#!/usr/bin/env python
"""
Style league: four fixated playstyles at one table, every game.

  techs         buys techs and plays through them (config/styles/techs.json)
  swordmaster   early Swordmaster, then combat + faction influence
  high_council  council seat, solari -> strong cards + The Spice Must Flow
  commanders    Sardaukar commanders, skills, combat

Seat order cycles through all 24 permutations. At the start of every round
each player's situation is logged (style_state from src/ai/playstyle.py plus
VP, conflicts won, influence, deck size) with their style, and the final VP
/ winner, so a model can learn WHICH style wins in WHICH situation.

Output: one JSON line per game in --out.

Usage:
  python -m src.selfplay.style_league --base heuristic --games 2400 --out data/style_league/heuristic.jsonl
  python -m src.selfplay.style_league --base search:K5:M24 --games 192 --out data/style_league/search.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

STYLES = ("techs", "swordmaster", "high_council", "commanders")
PERMS = list(itertools.permutations(range(4)))


def situation(gs, pid: int) -> list:
    from src.ai.playstyle import style_state
    p = gs.players[pid]
    x = [float(v) for v in style_state(gs, pid)]
    x += [len(gs.won_conflicts.get(pid, [])) / 4.0,
          sum(p.influence.values()) / 12.0,
          (len(p.deck) + len(p.discard) + len(p.hand) + len(p.in_play)) / 20.0,
          max(q.victory_points for q in gs.players if q.id != pid) / 10.0]
    return x


def spec_for(base: str, style: str) -> str:
    return f"{base}:tuned=config/styles/{style}.json"


def play(args):
    g, base, seed0, adaptive = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    perm = PERMS[g % 24]                    # perm[seat] = style index
    names = [STYLES[perm[s]] for s in range(4)]
    if adaptive:
        # the adaptive agent replaces one fixed style (cycling), so it meets
        # every style equally often
        out = STYLES[(g // 24) % 4]
        names = ["adaptive" if n == out else n for n in names]
    agents = [make_agent(adaptive if n == "adaptive" else spec_for(base, n),
                         seed=g * 4 + s) for s, n in enumerate(names)]
    gs = setup_game(num_players=4, seed=seed0 + g, neutral_leaders=True,
                    use_bloodlines=True)
    snaps = []
    seen_round = 0
    n = 0
    t0 = time.time()
    while not gs.game_over and n < 3000:
        if gs.round != seen_round and gs.phase.value == "player_turns":
            seen_round = gs.round
            snaps.append({"round": gs.round,
                          "x": [situation(gs, s) for s in range(4)]})
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    vp = [p.victory_points for p in gs.players]
    win = gs.winner if gs.winner is not None else max(range(4), key=lambda s: vp[s])
    from src.ai.playstyle import owned
    return {"game": g, "seed": seed0 + g, "base": base, "rounds": gs.round,
            "seconds": round(time.time() - t0, 1),
            "style": names,
            "adaptive_history": next((getattr(a, "history", None) for a in agents
                                      if getattr(a, "name", "") in ("stylesearch", "styleselect")), None),
            "vp": vp, "winner": win,
            "owned": [owned(gs, s) for s in range(4)],
            "skills": [sorted(gs.players[s].skills) for s in range(4)],
            "snaps": snaps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="heuristic")
    ap.add_argument("--games", type=int, default=240)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--seed", type=int, default=400_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adaptive", default="", help="agent spec that replaces one style per game")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):                      # resume
        done = {json.loads(l)["game"] for l in open(a.out)}
    todo = [(g, a.base, a.seed, a.adaptive) for g in range(a.games) if g not in done]
    wins = {s: 0 for s in STYLES + ("adaptive",)}
    with Pool(a.workers) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(play, todo)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            wins[r["style"][r["winner"]]] += 1
            if (i + 1) % 24 == 0 or i + 1 == len(todo):
                print(f"[{i + 1}/{len(todo)}] wins {wins}", flush=True)


if __name__ == "__main__":
    main()
