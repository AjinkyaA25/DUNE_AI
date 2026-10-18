"""How an AI fights: VP sources, Conflict strengths, wasted troops, and
Combat Intrigue use, from 4-player games of one agent spec against itself.

    python -m src.selfplay.combat_report SPEC [--games 200] [--workers 10]

e.g. SPEC = heuristic:tuned=config/gap/tsmf5_dp1_wb10_tm1_op1.json | az:S0
Writes reports/combat_<name>.json and prints a summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

from src.selfplay.gap_arena import cat


def _combat_timed(ic) -> bool:
    from src.game.intrigue.intrigue import IntrigueTiming as T
    ts = ic.timing if isinstance(ic.timing, (set, frozenset, tuple, list)) else (ic.timing,)
    return T.COMBAT in ts


def _places(strength: dict) -> dict:
    """pid -> placement (1 = best; ties share) among players with strength > 0."""
    order = sorted({s for s in strength.values() if s > 0}, reverse=True)
    return {pid: order.index(s) + 1 for pid, s in strength.items() if s > 0}


def game(args):
    g, spec = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.gameState import ActionType, Phase
    from src.game.player.player import Player

    vp_log = []
    orig = Player.gain_vp

    def gv(self, amount):
        if id(self) in real:
            f, names = sys._getframe(1), []
            while f and len(names) < 3:
                names.append(f.f_code.co_name); f = f.f_back
            vp_log.append((self.id, amount, cat(names)))
        return orig(self, amount)

    Player.gain_vp = gv
    agents = [make_agent(spec, seed=g * 4 + s) for s in range(4)]
    gs = setup_game(4, seed=600_000 + g, neutral_leaders=True, use_bloodlines=True)
    real = {id(p) for p in gs.players}
    conflicts, plays = [], []
    pay = {"offered": 0, "taken": 0}
    sm_round = {}
    pre = {"round": None, "strength": None}
    orig_resolve = gs.resolve_combat

    def resolve():
        c = gs.current_conflict
        rec = None
        if c is not None:
            parts = gs.get_combat_participants()
            st = {pid: gs.combat_strength.get(pid, 0) for pid in parts}
            rec = {"round": gs.round, "level": c.conflict_level, "name": c.name,
                   "strength": st,
                   "pre": dict(pre["strength"] or st) if pre["round"] == gs.round else dict(st),
                   "troops": {pid: gs.troops_in_conflict.get(pid, 0) for pid in parts},
                   "worms": {pid: gs.sandworms_in_conflict.get(pid, 0) for pid in parts},
                   "held_combat_intr": {p.id: sum(_combat_timed(ic) for ic in p.intrigue_cards)
                                        for p in gs.players}}
        res = orig_resolve()
        if rec is not None and res is not None:
            rec["rewarded"] = sorted(res.rewards_awarded)
            rec["winner"] = res.winner
            conflicts.append(rec)
        return res

    gs.resolve_combat = resolve
    n = 0
    while not gs.game_over and n < 6000:
        p = gs.player_in_reveal_buy
        pid = gs.get_current_player_id() if p is None else p
        if gs.phase == Phase.COMBAT and pre["round"] != gs.round and gs.current_conflict:
            pre["round"] = gs.round                       # strengths before any Combat Intrigue
            pre["strength"] = {q: gs.combat_strength.get(q, 0) for q in gs.get_combat_participants()}
        a = agents[pid].select_action(gs, pid, gs.get_valid_actions(pid))
        if a.action_type == ActionType.PLAY_INTRIGUE and gs.phase == Phase.COMBAT:
            plays.append({"round": gs.round, "pid": pid, "card": a.intrigue_card_name})
        if a.action_type == ActionType.RESOLVE_OPTIONAL and any(
                getattr(op, "label", "") == "conflict_vp" for op in gs.pending_optional_payments):
            pay["offered"] += 1
            pay["taken"] += int(a.accept_optional)
        gs.step(a)
        n += 1
        for q in gs.players:
            if q.has_swordmaster and q.id not in sm_round:
                sm_round[q.id] = gs.round
    Player.gain_vp = orig
    vp = [p.victory_points for p in gs.players]
    src = collections.defaultdict(collections.Counter)
    for pid, amt, c in vp_log:
        src[pid][c] += amt
    w = gs.winner if gs.winner is not None else int(np.argmax(vp))
    return {"rounds": gs.round, "vp": vp, "winner": w,
            "src": {k: dict(v) for k, v in src.items()},
            "conflicts": conflicts, "plays": plays, "pay": pay,
            "players": [{"sm": p.has_swordmaster, "sm_round": sm_round.get(p.id),
                         "hc": p.has_councilor, "infl": sum(p.influence.values()),
                         "techs": [t.name for t in getattr(p, "techs", [])]} for p in gs.players]}


def summarize(res, name):
    N = len(res)
    out = {"games": N, "rounds": float(np.mean([r["rounds"] for r in res])),
           "winner_vp": float(np.mean([max(r["vp"]) for r in res])),
           "avg_vp": float(np.mean([np.mean(r["vp"]) for r in res]))}
    # VP sources (per player, and for winners)
    tot, win = collections.Counter(), collections.Counter()
    for r in res:
        for pid, s in r["src"].items():
            tot.update(s)
            if int(pid) == r["winner"]:
                win.update(s)
    out["vp_sources_per_player"] = {k: v / (4 * N) for k, v in tot.most_common()}
    out["vp_sources_winner"] = {k: v / N for k, v in win.most_common()}
    # Conflicts
    by_level = collections.defaultdict(list)
    dep = wasted = won = 0
    troops_dep = []
    surplus = []
    n_conf = 0
    for r in res:
        for c in r["conflicts"]:
            n_conf += 1
            s = sorted(c["strength"].values(), reverse=True) + [0, 0]
            by_level[c["level"]].append((s[0], s[1], len([v for v in c["strength"].values() if v > 0])))
            if s[0] > 0:
                surplus.append(s[0] - s[1])
            for pid, st in c["strength"].items():
                if c["troops"].get(pid, 0) > 0 or st > 0:
                    dep += 1
                    troops_dep.append(c["troops"].get(pid, 0))
                    if pid not in c["rewarded"]:
                        wasted += 1
                    if pid == c["winner"]:
                        won += 1
    out["conflicts"] = n_conf
    out["deployments_per_conflict"] = dep / max(1, n_conf)
    out["troops_per_deployment"] = float(np.mean(troops_dep)) if troops_dep else 0
    out["share_deployments_no_reward"] = wasted / max(1, dep)
    out["share_deployments_won"] = won / max(1, dep)
    out["winning_margin_median"] = float(np.median(surplus)) if surplus else 0
    out["winning_margin_ge5"] = float(np.mean(np.array(surplus) >= 5)) if surplus else 0
    out["by_level"] = {lv: {"n": len(v), "first": float(np.mean([a for a, _, _ in v])),
                            "second": float(np.mean([b for _, b, _ in v])),
                            "fighters": float(np.mean([f for _, _, f in v]))}
                       for lv, v in sorted(by_level.items())}
    # Combat Intrigues
    n_plays = sum(len(r["plays"]) for r in res)
    changed = useless = 0
    held_unplayed_close = held_total = 0
    for r in res:
        for c in r["conflicts"]:
            played = {p["pid"] for p in r["plays"] if p["round"] == c["round"]}
            pre_pl, post_pl = _places(c["pre"]), _places(c["strength"])
            for pid in played:
                if post_pl.get(pid, 9) < pre_pl.get(pid, 9):
                    changed += 1
                else:
                    useless += 1
            # held a Combat Intrigue, fought, didn't play, and missed 1st by <= 3
            top = max(c["strength"].values()) if c["strength"] else 0
            for pid, st in c["strength"].items():
                if st > 0 and c["held_combat_intr"].get(pid, 0):
                    held_total += 1
                    if pid not in played and pid != c["winner"] and top - st <= 3:
                        held_unplayed_close += 1
    pl = [q for r in res for q in r["players"]]
    out["swordmaster_share"] = float(np.mean([q["sm"] for q in pl]))
    out["swordmaster_by_r4"] = float(np.mean([bool(q["sm_round"]) and q["sm_round"] <= 4 for q in pl]))
    out["high_council_share"] = float(np.mean([q["hc"] for q in pl]))
    out["influence_total"] = float(np.mean([q["infl"] for q in pl]))
    out["panopticon_share"] = float(np.mean(["Panopticon" in q["techs"] for q in pl]))
    off = sum(r["pay"]["offered"] for r in res)
    out["conflict_vp_payments_taken"] = sum(r["pay"]["taken"] for r in res) / max(1, off)
    out["conflict_vp_payment_offers_per_game"] = off / N
    out["combat_intrigue_plays_per_player_game"] = n_plays / (4 * N)
    out["plays_that_improved_placement"] = changed / max(1, changed + useless)
    out["fighters_holding_combat_intrigue"] = held_total
    out["held_unplayed_and_lost_1st_by_le3"] = held_unplayed_close / max(1, held_total)
    return out


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--name", default=None)
    A = ap.parse_args()
    name = A.name or A.spec.split("/")[-1].replace(":", "_").replace(".json", "")
    with Pool(A.workers, maxtasksperchild=10) as pool:
        res = pool.map(game, [(g, A.spec) for g in range(A.games)])
    out = summarize(res, name)
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", f"combat_{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
