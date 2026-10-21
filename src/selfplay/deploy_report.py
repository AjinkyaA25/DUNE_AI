"""Why the AI deploys: every troop deploy decision, what the AI believed, and
what actually happened (user, 2026-10-18: "it deploys even when it cannot
place in the combat").

For each RESOLVE_DEPLOY the AI chose it records the AI's own view - its
strength estimate, each rival's forecast (visible + expected additions), the
chance it gave itself of finishing 1st / 2nd / 3rd / out of the rewards, and
the score split into "expected reward gained" vs "worth of keeping the
troops" - then, when the Conflict resolves, the real placement. Troops sent
into a fight finished out of the rewards are WASTED; troops beyond what a win
needed are OVERKILL.

    python -m src.selfplay.deploy_report SPEC [--games 300] [--workers 6]

SPEC e.g. heuristic:tuned=config/best_ai.json
Writes reports/deploy_<name>.json and prints a summary + worst examples.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from multiprocessing import Pool

PLACES = ("1st", "2nd", "3rd", "out")


def _places(strength: dict) -> dict:
    """pid -> placement (1 = best; ties share) among players with strength > 0."""
    order = sorted({s for s in strength.values() if s > 0}, reverse=True)
    return {pid: order.index(s) + 1 for pid, s in strength.items() if s > 0}


def _belief(gs, pid, n, cmd):
    """The AI's own view of deploying n units now (mirrors fight_ev)."""
    import src.ai.agents as A
    cc = gs.current_conflict
    mine0, u0 = A._my_strength(gs, pid)
    mine, u = A._my_strength(gs, pid, add_troops=n, add_cmd=cmd)
    ci = A._own_combat_intrigue_swords(gs, pid) * 0.6
    k2 = 2.0 if A._worm_doubled(gs, pid, None) else 1.0
    rewards = [A._conflict_worth(gs, pid, None),
               k2 * A._reward_worth(gs, pid, cc.second_place_reward),
               k2 * A._reward_worth(gs, pid, cc.third_place_reward) if gs.num_players >= 4 else 0.0]
    rivals = []
    for q in range(gs.num_players):
        if q == pid:
            continue
        mu, sd, can = A._opp_fight(gs, q, pid)
        vis = 2 * gs.troops_in_conflict.get(q, 0) + 3 * gs.sandworms_in_conflict.get(q, 0) \
            + gs.swords_this_reveal.get(q, 0)
        rivals.append({"pid": q, "revealed": q in gs.players_revealed, "visible": round(vis, 1), "forecast": round(mu, 1),
                       "spread": round(sd, 1), "can_fight": float(can) > 0.5, "p_in": float(can),
                       "agents_left": gs.players[q].agents_available})

    def dist(strength, units):
        if units <= 0:
            return [0.0, 0.0, 0.0, 1.0]
        d = [1.0]
        for r in rivals:
            pa = A._p_ahead(r["forecast"], r["spread"], r["p_in"], strength, r["revealed"])
            nd = [0.0] * (len(d) + 1)
            for k, pk in enumerate(d):
                nd[k] += pk * (1 - pa)
                nd[k + 1] += pk * pa
            d = nd
        return [d[0], d[1], d[2], sum(d[3:])]

    d1, d0 = dist(mine + ci, u), dist(mine0 + ci, u0)
    ev1 = A.fight_ev(gs, pid, mine + ci, u)
    ev0 = A.fight_ev(gs, pid, mine0 + ci, u0)
    hold = n * A._troop_hold_value(gs)
    dhold = n * A._TUNE.get("deploy_hold", 0.4) * min(1.0, A._rounds_left(gs) / 2.0) if n else 0.0
    return {
        "my_strength_now": round(mine0, 1), "my_strength_after": round(mine, 1),
        "my_hand_swords_counted": round(0.7 * sum(getattr(c, "swords", 0) or 0
                                                   for c in gs.players[pid].hand), 1)
        if gs.phase.name != "COMBAT" and pid not in gs.players_revealed else 0.0,
        "combat_intrigue_counted": round(ci, 1),
        "rivals": rivals,
        "p_place_after": [round(x, 3) for x in d1], "p_place_before": [round(x, 3) for x in d0],
        "reward_worth": [round(x, 2) for x in rewards],
        "ev_gain": round(ev1 - ev0, 2), "hold_cost": round(hold, 2), "deploy_hold_cost": round(dhold, 2),
    }


def game(args):
    g, spec = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.gameState import ActionType, GameState
    from src.game.leader.leader_definitions import LEADER_NAMES

    agents = [make_agent(spec, seed=g * 4 + s) for s in range(4)]
    lineup = random.Random(700_000 + g).sample(list(LEADER_NAMES), 4)
    gs = setup_game(num_players=4, seed=700_000 + g, leaders=lineup, use_bloodlines=True)
    decisions, fights = [], []
    open_dec = collections.defaultdict(list)          # round -> decisions this Conflict

    orig = GameState.resolve_combat

    def resolve(self):
        if self is gs and self.current_conflict is not None:
            st = {q: float(self.combat_strength.get(q, 0) or 0) for q in range(4)}
            units = {q: self.troops_in_conflict.get(q, 0) + self.sandworms_in_conflict.get(q, 0)
                     for q in range(4)}
            pl = _places(st)
            fights.append({"round": self.round, "level": self.current_conflict.conflict_level,
                           "strength": st, "units": units, "place": pl})
            for d in open_dec.pop(self.round, []):
                q = d["pid"]
                place = pl.get(q)
                d["result"] = PLACES[min(place, 4) - 1] if place else "out"
                d["final_strength"] = st[q]
                d["final_rivals"] = sorted((st[r] for r in range(4) if r != q), reverse=True)
                for r in d["rivals"]:
                    r["final"] = st[r["pid"]]
                d["wasted"] = d["result"] == "out" or (place is not None and place > 3)
                if place == 1:
                    second = max((st[r] for r in range(4) if r != q), default=0)
                    d["overkill_strength"] = st[q] - second
        return orig(self)

    GameState.resolve_combat = resolve
    try:
        n = 0
        while not gs.game_over and n < 3000:
            pid = gs.player_in_reveal_buy
            if pid is None:
                pid = gs.get_current_player_id()
            valid = gs.get_valid_actions(pid)
            a = agents[pid].select_action(gs, pid, valid)
            deploys = [v for v in valid if v.action_type == ActionType.RESOLVE_DEPLOY]
            if deploys and gs.current_conflict is not None:
                ag = agents[pid]
                scores = {}
                if hasattr(ag, "score"):
                    for v in deploys:
                        try:
                            scores[(v.deploy_count, v.deploy_cmd)] = round(float(ag.score(gs, pid, v)), 2)
                        except Exception:
                            pass
                chosen = a.deploy_count if a.action_type == ActionType.RESOLVE_DEPLOY else 0
                if chosen > 0:
                    if hasattr(ag, "score"):
                        ag.score(gs, pid, a)              # apply this agent's tuning for the breakdown
                    d = {"game": g, "pid": pid, "leader": lineup[pid], "round": gs.round,
                         "level": gs.current_conflict.conflict_level, "conflict": gs.current_conflict.name,
                         "deployed": chosen, "garrison_before": gs.players[pid].troops_garrison,
                         "in_conflict_before": gs.troops_in_conflict.get(pid, 0),
                         "my_agents_left": gs.players[pid].agents_available,
                         "phase": gs.phase.name, "options": {f"{k[0]}": v for k, v in scores.items()},
                         **_belief(gs, pid, chosen, a.deploy_cmd)}
                    decisions.append(d)
                    open_dec[gs.round].append(d)
            gs.step(a)
            n += 1
    finally:
        GameState.resolve_combat = orig
    return decisions, fights, gs.round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--examples", type=int, default=8)
    A = ap.parse_args()
    decs, fights, rounds = [], [], []
    with Pool(A.workers) as pool:
        for d, f, r in pool.imap_unordered(game, [(g, A.spec) for g in range(A.games)], chunksize=2):
            decs += d; fights += f; rounds.append(r)
    done = [d for d in decs if "result" in d]
    troops = sum(d["deployed"] for d in done)
    wasted = [d for d in done if d["wasted"]]
    # overkill once per fight: troops deployed into a win beyond what winning needed
    won = collections.defaultdict(lambda: [0, 0.0])
    for d in done:
        if d["result"] == "1st":
            k = (d["game"], d["round"], d["pid"])
            won[k][0] += d["deployed"]
            won[k][1] = d["overkill_strength"]
    over = sum(min(dep, max(0, int((margin - 1) // 2))) for dep, margin in won.values())
    # how good is the rival forecast? (rivals that could still fight)
    fc = [(r["forecast"], r["final"]) for d in done for r in d["rivals"] if r["can_fight"] and "final" in r]
    # predicted chance of finishing out of the rewards vs what happened, in bins
    bins = collections.defaultdict(lambda: [0, 0.0, 0])
    for d in done:
        b = min(4, int(d["p_place_after"][3] * 5))
        bins[b][0] += 1; bins[b][1] += d["p_place_after"][3]; bins[b][2] += d["wasted"]
    pred_out = sum(d["p_place_after"][3] for d in done) / max(1, len(done))
    by_place = collections.Counter(d["result"] for d in done)
    # out-of-rewards units across ALL fights (any route: deploys, card effects, recruits)
    lost_units = sum(f["units"][q] for f in fights for q in range(4)
                     if f["units"][q] and (f["place"].get(q) is None or f["place"][q] > 3))
    all_units = sum(f["units"][q] for f in fights for q in range(4))
    summary = {
        "spec": A.spec, "games": A.games, "avg_rounds": round(sum(rounds) / len(rounds), 2),
        "deploy_decisions": len(done), "per_game": round(len(done) / A.games, 2),
        "troops_deployed": troops,
        "result_of_deploys": {k: round(by_place[k] / max(1, len(done)), 3) for k in PLACES},
        "wasted_decisions": len(wasted), "wasted_troops": sum(d["deployed"] for d in wasted),
        "wasted_troops_per_game": round(sum(d["deployed"] for d in wasted) / A.games, 2),
        "ai_predicted_out_of_rewards": round(pred_out, 3),
        "overkill_troops_per_game": round(over / A.games, 2),
        "all_fights_units_out_of_rewards_share": round(lost_units / max(1, all_units), 3),
        "rival_forecast_mean": round(sum(f for f, _ in fc) / max(1, len(fc)), 2),
        "rival_actual_final_mean": round(sum(a for _, a in fc) / max(1, len(fc)), 2),
        "p_out_calibration": {f"{b * 20}-{b * 20 + 20}%": {"n": v[0], "predicted": round(v[1] / v[0], 3),
                                                           "actual": round(v[2] / v[0], 3)}
                              for b, v in sorted(bins.items())},
        "first_troop_entries": sum(1 for d in done if d["in_conflict_before"] == 0),
        "first_troop_entries_wasted": sum(1 for d in done if d["in_conflict_before"] == 0 and d["wasted"]),
    }
    # where the AI's belief was wrong: thought it would place, finished out
    surprise = sorted(wasted, key=lambda d: d["p_place_after"][3])
    os.makedirs("reports", exist_ok=True)
    name = A.spec.replace(":", "_").replace("/", "_").replace("=", "_")[:60]
    with open(f"reports/deploy_{name}.json", "w") as f:
        json.dump({"summary": summary, "decisions": done}, f)
    print(json.dumps(summary, indent=1))
    print(f"\nWorst surprises ({A.examples}): deployed, finished out of the rewards")
    for d in surprise[:A.examples]:
        rv = ", ".join(f"P{r['pid'] + 1} sees {r['visible']} -> forecast {r['forecast']}+/-{r['spread']} (ended {r.get('final')})"
                       f"{'' if r['can_fight'] else ' (can\'t fight)'}" for r in d["rivals"])
        print(f"\n- game {d['game']} R{d['round']} L{d['level']} {d['leader']}: deployed {d['deployed']} "
              f"(garrison {d['garrison_before']}, already in {d['in_conflict_before']}, agents left {d['my_agents_left']})")
        print(f"  AI strength estimate {d['my_strength_now']} -> {d['my_strength_after']} "
              f"(incl. {d['my_hand_swords_counted']} hand swords, {d['combat_intrigue_counted']} intrigue)")
        print(f"  rivals: {rv}")
        print(f"  AI's odds after deploy 1st/2nd/3rd/out = {d['p_place_after']} (before {d['p_place_before']})")
        print(f"  score: reward gained {d['ev_gain']} - keep-troops {d['hold_cost']} - deploy-hold "
              f"{d['deploy_hold_cost']} | options {d['options']}")
        print(f"  ACTUAL: {d['result']} with {d['final_strength']} vs rivals {d['final_rivals']}")


if __name__ == "__main__":
    main()
