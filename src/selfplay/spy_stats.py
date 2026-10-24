"""How useful is a Spy on each observation post? (user, 2026-10-23: "some spies
are better placed than other spies due to the frequency of visiting,
importance of draw and such").

Plays games (Best AI = the UI's PPO in every seat, random Leaders) and
measures, from what actually happens:

  * per post: placements, how each Spy left the post (Gather Intelligence =
    a card draw, Infiltrate = into an occupied space, other recall) and the
    rounds it sat there -> a per-round USE hazard (uses / spy-rounds);
  * the same per Leader (shrunk towards the overall rate);
  * per board space: Agent visits per player-round (overall and per Leader)
    and how often it is already taken by an opponent when a player places
    an Agent (the "block" rate that makes Infiltrate worth something).

    python -m src.selfplay.spy_stats --games 600 --workers 28
        -> models_spy/spy_stats.json (read by agents._spy_post_value, knob spy_freq)

Plan check (2026-10-23, "spies should go where the AI is planning on going"):
every Spy placement also records whether a post the player's remaining hand
can reach this round was on offer and whether the chosen one was such a post
(action_features.spy_plan_features), and the report shows how soon those
Spies were used. --report-only prints that for a spec without writing the
stats file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool

SPEC = "ppo:net=models_ui/ppo.npz:tuned=config/ppo_ui.json:greedy"
OUT = os.path.join("models_spy", "spy_stats.json")
SHRINK = 30.0          # spy-rounds of prior weight on the overall rate for a Leader's own rate


def play(args):
    g, spec = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.gameState import ActionType
    from src.game.leader.leader_definitions import LEADER_NAMES
    r = random.Random(7_000_000 + g)
    lineup = r.sample(list(LEADER_NAMES), 4)
    gs = setup_game(num_players=4, seed=7_000_000 + g, leaders=lineup, use_bloodlines=True)
    agents = [make_agent(spec, seed=g * 4 + s) for s in range(4)]
    from src.ai.action_features import spy_plan_features
    live = {}                      # (pid, post, k) -> (round placed, plan)
    spies = []                     # (leader, post, round placed, round left, how)
    visits = []                    # (leader, space, round)
    blocked = defaultdict(int)     # space -> agent placements that found it taken by an opponent
    placements = 0
    n = 0

    def board(pid):
        return dict(gs.players[pid].spies_on_board)

    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        a = agents[pid].select_action(gs, pid, gs.get_valid_actions(pid))
        before = [board(q) for q in range(4)]
        plan = None
        if a.action_type == ActionType.RESOLVE_SPY and a.spy_post_name:
            opts = [b for b in gs.get_valid_actions(pid) if b.action_type == ActionType.RESOLVE_SPY]
            f = spy_plan_features(gs, pid, a)
            plan = (bool(f[0] > 0), any(spy_plan_features(gs, pid, b)[0] > 0 for b in opts), bool(f[2] > 0))
        how = "recall"
        if a.action_type == ActionType.AGENT_TURN:
            placements += 1
            for sp, occ in gs.agent_on_space.items():
                if occ is not None and occ != pid:
                    blocked[sp] += 1
            visits.append((lineup[pid], a.space_name, gs.round))
            how = "gather" if a.use_gather_intelligence else ("infiltrate" if a.use_infiltrate else "recall")
        rnd = gs.round
        gs.step(a)
        n += 1
        for q in range(4):
            after = board(q)
            for post in set(before[q]) | set(after):
                b, c = before[q].get(post, 0), after.get(post, 0)
                for k in range(c, b):                   # left the post
                    r0, pl = live.pop((q, post, k), (rnd, None))
                    spies.append((lineup[q], post, r0, rnd, how if q == pid else "recall", pl))
                for k in range(b, c):                   # placed
                    live[(q, post, k)] = (rnd, plan if q == pid else None)
    end = gs.round
    for (q, post, k), (r0, pl) in live.items():
        spies.append((lineup[q], post, r0, end, "never", pl))
    return {"lineup": lineup, "rounds": end, "spies": spies, "visits": visits,
            "blocked": dict(blocked), "placements": placements}


def fit(games: list) -> dict:
    from src.game.board.board import OBSERVATION_POST_CONNECTIONS
    posts = sorted(OBSERVATION_POST_CONNECTIONS)
    tot = defaultdict(lambda: defaultdict(float))
    led = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for G in games:
        for leader, post, r0, r1, how, _ in G["spies"]:
            for d in (tot[post], led[leader][post]):
                d["placed"] += 1
                d[how] += 1
                d["spy_rounds"] += max(0.5, r1 - r0 + 0.5)    # placed mid-round on average
    out = {"games": len(games), "rounds": sum(G["rounds"] for G in games) / max(1, len(games)), "posts": {},
           "by_leader": {}, "spaces": {}, "by_leader_spaces": {}}

    def rates(d, prior=None):
        sr = d["spy_rounds"]
        res = {k: round(d[k], 1) for k in ("placed", "gather", "infiltrate", "recall", "never", "spy_rounds")}
        for k in ("gather", "infiltrate"):
            if prior is None:
                res[k + "_h"] = round(d[k] / sr, 4) if sr else 0.0
            else:
                res[k + "_h"] = round((d[k] + SHRINK * prior[k + "_h"]) / (sr + SHRINK), 4)
        return res

    for post in posts:
        out["posts"][post] = rates(tot[post])
    for leader, dd in sorted(led.items()):
        out["by_leader"][leader] = {post: rates(dd[post], out["posts"][post]) for post in posts}
    # space visit rates per player-round, block rate per Agent placement
    pr = sum(4 * G["rounds"] for G in games)
    pl = sum(G["placements"] for G in games)
    vis = defaultdict(float)
    lvis = defaultdict(lambda: defaultdict(float))
    lrounds = defaultdict(float)
    blk = defaultdict(float)
    for G in games:
        for leader in G["lineup"]:
            lrounds[leader] += G["rounds"]
        for leader, sp, _ in G["visits"]:
            vis[sp] += 1
            lvis[leader][sp] += 1
        for sp, c in G["blocked"].items():
            blk[sp] += c
    for sp in sorted(set(vis) | set(blk)):
        out["spaces"][sp] = {"visit": round(vis[sp] / pr, 4), "block": round(blk[sp] / max(1, pl), 4)}
    for leader in sorted(lvis):
        out["by_leader_spaces"][leader] = {
            sp: round((lvis[leader][sp] + 20 * out["spaces"].get(sp, {}).get("visit", 0)) / (lrounds[leader] + 20), 4)
            for sp in out["spaces"]}
    return out


def plan_report(games: list) -> dict:
    """Spy placements made when a hand-reachable post was on offer: how often
    the AI took one, and how soon each kind of Spy paid off."""
    rows = [(pl, r1 - r0, how) for G in games for _, _, r0, r1, how, pl in G["spies"] if pl is not None]
    offer = [r for r in rows if r[0][1]]
    out = {"placements": len(rows), "with_reachable_offer": len(offer),
           "took_reachable": round(sum(r[0][0] for r in offer) / max(1, len(offer)), 4),
           "took_rival_space": round(sum(r[0][2] for r in rows) / max(1, len(rows)), 4)}
    for name, sel in (("all", rows), ("reachable", [r for r in rows if r[0][0]]),
                      ("not_reachable", [r for r in rows if not r[0][0]])):
        n = max(1, len(sel))
        out[name] = {"n": len(sel),
                     "used_same_round": round(sum(1 for _, d, h in sel if h in ("gather", "infiltrate") and d == 0) / n, 4),
                     "used_ever": round(sum(1 for _, _, h in sel if h in ("gather", "infiltrate")) / n, 4),
                     "never_used": round(sum(1 for _, _, h in sel if h == "never") / n, 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--report-only", default="", help="write only the plan check here (json)")
    A = ap.parse_args()
    with Pool(A.workers, maxtasksperchild=20) as pool:
        games = pool.map(play, [(g, A.spec) for g in range(A.games)], chunksize=2)
    pr = plan_report(games)
    print(f"avg rounds {sum(G['rounds'] for G in games) / len(games):.2f}")
    print(json.dumps(pr, indent=1))
    if A.report_only:
        pr["spec"], pr["games"] = A.spec, len(games)
        pr["rounds"] = sum(G["rounds"] for G in games) / len(games)
        json.dump(pr, open(A.report_only, "w"), indent=1)
        return
    res = fit(games)
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(res, open(A.out, "w"), indent=1)
    print(f"{res['games']} games, {res['rounds']:.2f} avg rounds")
    print(f"{'post':28s} placed  gather/rd  infil/rd  never  recall")
    for post, d in sorted(res["posts"].items(), key=lambda kv: -(kv[1]["gather_h"] + kv[1]["infiltrate_h"])):
        print(f"{post:28s} {d['placed']:6.0f}  {d['gather_h']:8.3f}  {d['infiltrate_h']:8.3f}  "
              f"{d['never']:5.0f}  {d['recall']:6.0f}")


if __name__ == "__main__":
    main()
