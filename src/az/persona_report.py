"""Before / after a training run: how each persona seat plays and scores.

Each game seats the four personas (rotated) as AlphaZero agents with the
coach on, all using one network; VP is attributed to its source (as in
src.selfplay.gap_arena) and tracked by round.

    python -m src.az.persona_report --nets models_az_coached/net_0.npz,models_az_coached/net_4.npz
        [--personas smf,sword,commanders,faction] [--games 120] [--sims 0]

Writes reports/persona_report.json and prints a summary per net.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

from src.selfplay.gap_arena import cat


def game(args):
    g, net, styles, sims, coach = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.player.player import Player
    order = [styles[(g + k) % 4] for k in range(4)]
    ag = [make_agent(f"az:S{sims}:net={net}:coach={coach}:style={s}", seed=g * 4 + i)
          for i, s in enumerate(order)]
    log, orig = [], Player.gain_vp

    def gv(self, amount, source=None):
        if id(self) in real:
            f, names = sys._getframe(1), []
            while f and len(names) < 3:
                names.append(f.f_code.co_name); f = f.f_back
            log.append((self.id, amount, cat(names), gs.round))
        return orig(self, amount, source)

    Player.gain_vp = gv
    gs = setup_game(4, seed=12_000_000 + g, neutral_leaders=True, use_bloodlines=True)
    real = {id(p) for p in gs.players}
    sm_round, n = {}, 0
    while not gs.game_over and n < 5000:
        p = gs.player_in_reveal_buy
        pid = gs.get_current_player_id() if p is None else p
        gs.step(ag[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
        for q in gs.players:
            if q.has_swordmaster and q.id not in sm_round:
                sm_round[q.id] = gs.round
    Player.gain_vp = orig
    vp = [p.victory_points for p in gs.players]
    w = gs.winner if gs.winner is not None else int(np.argmax(vp))
    seats = []
    for pid, style in enumerate(order):
        src = collections.Counter()
        by_round = collections.Counter()
        for q, amt, c, r in log:
            if q == pid:
                src[c] += amt
                by_round[r] += amt
        p = gs.players[pid]
        cards = p.hand + p.deck + p.discard + p.in_play
        seats.append({"style": style, "win": int(w == pid), "vp": vp[pid], "src": dict(src),
                      "vp_by_r6": 1 + sum(v for r, v in by_round.items() if r <= 6),
                      "sm_by6": int(sm_round.get(pid, 99) <= 6), "hc": int(p.has_councilor),
                      "tsmf": sum(c.name == "The Spice Must Flow" for c in cards),
                      "daggers": sum(c.name == "Dagger" for c in cards),
                      "techs": len(getattr(p, "techs", [])),
                      "commanders": getattr(p, "commanders_garrison", 0) + getattr(p, "commanders_supply", 0),
                      "influence": sum(p.influence.values()),
                      "contracts": len(getattr(p, "contracts_completed", []))})
    return {"rounds": gs.round, "seats": seats}


def summarize(res):
    by = collections.defaultdict(list)
    for r in res:
        for s in r["seats"]:
            by[s["style"]].append(s)
    out = {"games": len(res), "rounds": float(np.mean([r["rounds"] for r in res])),
           "ended_by_r8": float(np.mean([r["rounds"] <= 8 for r in res])), "styles": {}}
    for style, ss in sorted(by.items()):
        src = collections.Counter()
        for s in ss:
            src.update(s["src"])
        d = {k: float(np.mean([s[k] for s in ss])) for k in
             ("win", "vp", "vp_by_r6", "sm_by6", "hc", "tsmf", "daggers", "techs", "commanders",
              "influence", "contracts")}
        d["src"] = {k: v / len(ss) for k, v in src.most_common()}
        out["styles"][style] = d
    return out


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", required=True)
    ap.add_argument("--personas", default="smf,sword,commanders,faction")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--sims", type=int, default=0)
    ap.add_argument("--coach", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=10)
    A = ap.parse_args()
    styles = A.personas.split(",")
    report = {}
    with Pool(A.workers, maxtasksperchild=10) as pool:
        for net in A.nets.split(","):
            res = pool.map(game, [(g, net, styles, A.sims, A.coach) for g in range(A.games)])
            report[net] = s = summarize(res)
            print(f"\n== {net}: {s['games']} games, avg rounds {s['rounds']:.2f}, over by R8 {s['ended_by_r8']:.0%}")
            for style, d in s["styles"].items():
                print(f"  {style:10s} win {d['win']:.0%}  VP {d['vp']:.2f} (by R6 {d['vp_by_r6']:.2f})  "
                      f"SM<=R6 {d['sm_by6']:.0%}  HC {d['hc']:.0%}  TSMF {d['tsmf']:.2f}  daggers {d['daggers']:.2f}  "
                      f"techs {d['techs']:.2f}  cmdrs {d['commanders']:.2f}  infl {d['influence']:.1f}  "
                      f"contracts {d['contracts']:.2f}")
                print("             VP from: " + ", ".join(f"{k} {v:.2f}" for k, v in d["src"].items()))
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "persona_report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
