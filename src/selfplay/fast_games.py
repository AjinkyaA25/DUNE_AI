"""Learn from the games that end quickly (8 rounds or fewer).

    python -m src.selfplay.fast_games play    --games 600 --spec look:D3:K3:M1:tuned=<json> [--workers 28]
    python -m src.selfplay.fast_games analyse
    python -m src.selfplay.fast_games fit     --start <json> --out <json> [--workers 28]

play     records every game (setup seed + index of each move among
         get_valid_actions, so it replays exactly) with per-round snapshots
         and VP sources to data/fast_games/games.jsonl.gz.
analyse  compares the winners of fast games (<= --fast rounds) with the
         winners of slow games: VP by round, where the VP came from, when
         they got Swordmaster / High Council, TSMF, conflicts, influence,
         techs. Writes data/fast_games/analysis.json.
fit      replays the fast games, captures every decision the WINNER made,
         and moves the heuristic weights (one at a time, x0.7 / x0.85 /
         x1.2 / x1.4) to agree with those decisions more often; 20% of
         the games are held out to check the fit generalises.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

# FAST_DIR / FAST_LEADERS=1 (environment): a separate data folder, games with Leaders
DIR = os.environ.get("FAST_DIR") or os.path.join("data", "fast_games")
GAMES = os.path.join(DIR, "games.jsonl.gz")
FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")


# ------------------------------------------------------------------ play

def _snap(gs):
    out = []
    for p in gs.players:
        out.append({"vp": p.victory_points, "sm": bool(p.has_swordmaster), "hc": bool(p.has_councilor),
                    "tsmf": p.tsmf_acquired, "inf": [p.influence[f] for f in FACTIONS],
                    "all": sum(p.alliances.values()), "conf": len(gs.won_conflicts.get(p.id, [])),
                    "techs": len(getattr(p, "techs", []) or []),
                    "deck": len(p.deck) + len(p.hand) + len(p.discard) + len(p.in_play),
                    "solari": p.solari, "spice": p.spice, "water": p.water})
    return out


def _setup(seed, leaders=None):
    from src.data.card_definitions import setup_game
    if leaders:
        return setup_game(num_players=4, seed=seed, leaders=leaders, use_bloodlines=True)
    return setup_game(num_players=4, seed=seed, neutral_leaders=True, use_bloodlines=True)


def play_one(args):
    g, specs, seed0 = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.player.player import Player
    from src.selfplay.gap_arena import cat
    seed = seed0 + g
    agents = [make_agent(specs[s], seed=seed * 4 + s) for s in range(4)]
    leaders = None
    if os.environ.get("FAST_LEADERS"):
        from src.game.leader.leader_definitions import LEADER_NAMES
        leaders = random.Random(seed).sample(list(LEADER_NAMES), 4)
    gs = _setup(seed, leaders)
    real = {id(p) for p in gs.players}
    src = [collections.Counter() for _ in range(4)]
    orig = Player.gain_vp

    def gv(self, amount, source=None):
        if id(self) in real:
            f = sys._getframe(1); names = []
            while f and len(names) < 3:
                names.append(f.f_code.co_name); f = f.f_back
            src[self.id][f"r{gs.round}:" + cat(names)] += amount
        return orig(self, amount, source)
    Player.gain_vp = gv
    moves, rounds, last_round, n = [], [], gs.round, 0
    try:
        while not gs.game_over and n < 4000:
            if gs.round != last_round:
                rounds.append(_snap(gs)); last_round = gs.round
            pid = gs.player_in_reveal_buy
            if pid is None:
                pid = gs.get_current_player_id()
            valid = gs.get_valid_actions(pid)
            a = agents[pid].select_action(gs, pid, valid)
            k = valid.index(a) if a in valid else next(i for i, v in enumerate(valid) if repr(v) == repr(a))
            moves.append(k)
            gs.step(a); n += 1
    finally:
        Player.gain_vp = orig
    if not gs.game_over:
        return None
    rounds.append(_snap(gs))
    vp = [p.victory_points for p in gs.players]
    win = gs.winner if gs.winner is not None else int(np.argmax(vp))
    return {"g": g, "seed": seed, "specs": list(specs), "leaders": leaders, "moves": moves, "rounds": gs.round,
            "vp": vp, "winner": win, "snaps": rounds, "src": [dict(c) for c in src]}


def play(n, specs, workers, seed0, tag):
    os.makedirs(DIR, exist_ok=True)
    done = set()
    if os.path.exists(GAMES):
        with gzip.open(GAMES, "rt") as f:
            done = {(r.get("tag"), r["g"]) for r in map(json.loads, f)}
    jobs = [(g, specs, seed0) for g in range(n) if (tag, g) not in done]
    t0, k = time.time(), 0
    with Pool(workers, maxtasksperchild=20) as pool, gzip.open(GAMES, "at") as f:
        for r in pool.imap_unordered(play_one, jobs):
            if r is None:
                continue
            r["tag"] = tag
            f.write(json.dumps(r) + "\n"); f.flush()
            k += 1
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)} games, {time.time() - t0:.0f}s", flush=True)


def load(path=GAMES):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------ analyse

def analyse(fast_max=8):
    games = load()
    fast = [g for g in games if g["rounds"] <= fast_max]
    slow = [g for g in games if g["rounds"] > fast_max]
    out = {"games": len(games), "fast": len(fast), "fast_share": len(fast) / max(1, len(games)),
           "avg_rounds": float(np.mean([g["rounds"] for g in games]))}
    if any(g.get("leaders") for g in games):           # which Leaders win the fast games
        fw = collections.Counter(g["leaders"][g["winner"]] for g in fast if g.get("leaders"))
        aw = collections.Counter(g["leaders"][g["winner"]] for g in games if g.get("leaders"))
        out["fast_winner_leaders"] = {l: [fw[l], aw[l]] for l in sorted(aw, key=lambda x: -fw[x])}

    def first_round(g, key):
        w = g["winner"]
        for r, s in enumerate(g["snaps"], 1):
            if s[w][key]:
                return r
        return None

    def summary(gs_):
        res = {}
        for r in (2, 4, 6):                  # snaps[r - 1] = state after round r
            vals = [g["snaps"][r - 1][g["winner"]]["vp"] for g in gs_ if len(g["snaps"]) >= r]
            others = [np.mean([g["snaps"][r - 1][q]["vp"] for q in range(4) if q != g["winner"]])
                      for g in gs_ if len(g["snaps"]) >= r]
            res[f"winner_vp_after_r{r}"] = round(float(np.mean(vals)), 2)
            res[f"others_vp_after_r{r}"] = round(float(np.mean(others)), 2)
        for key, name in (("sm", "swordmaster"), ("hc", "high_council")):
            fr = [first_round(g, key) for g in gs_]
            got = [x for x in fr if x]
            res[f"{name}_share"] = round(len(got) / max(1, len(gs_)), 3)
            res[f"{name}_round"] = round(float(np.mean(got)), 2) if got else None
        fin = [g["snaps"][-1][g["winner"]] for g in gs_]
        res["final_vp"] = round(float(np.mean([f["vp"] for f in fin])), 2)
        res["tsmf"] = round(float(np.mean([f["tsmf"] for f in fin])), 2)
        res["conflicts_won"] = round(float(np.mean([f["conf"] for f in fin])), 2)
        res["alliances"] = round(float(np.mean([f["all"] for f in fin])), 2)
        res["influence_total"] = round(float(np.mean([sum(f["inf"]) for f in fin])), 2)
        res["techs"] = round(float(np.mean([f["techs"] for f in fin])), 2)
        res["deck"] = round(float(np.mean([f["deck"] for f in fin])), 2)
        srcs = collections.Counter()
        for g in gs_:
            for k, v in g["src"][g["winner"]].items():
                srcs[k.split(":", 1)[1]] += v
        res["vp_sources"] = {k: round(v / max(1, len(gs_)), 2) for k, v in srcs.most_common()}
        return res

    out["fast_winners"] = summary(fast) if fast else {}
    out["slow_winners"] = summary(slow) if slow else {}
    json.dump(out, open(os.path.join(DIR, "analysis.json"), "w"), indent=1)
    print(f"{len(games)} games, avg {out['avg_rounds']:.2f} rounds, "
          f"{len(fast)} ended by round {fast_max} ({out['fast_share']:.1%})")
    keys = list(out["fast_winners"])
    print(f"{'':28} {'fast winners':>14} {'slow winners':>14}")
    for k in keys:
        a, b = out["fast_winners"].get(k), out["slow_winners"].get(k)
        if isinstance(a, dict):
            print(f"{k}:")
            for kk in sorted(set(a) | set(b or {}), key=lambda x: -(a.get(x, 0))):
                print(f"  {kk:26} {a.get(kk, 0):>14} {(b or {}).get(kk, 0):>14}")
        else:
            print(f"{k:28} {str(a):>14} {str(b):>14}")
    return out


# ------------------------------------------------------------------ fit

def label(a) -> str:
    from src.game.gameState import ActionType
    t = a.action_type
    if t == ActionType.AGENT_TURN:
        return f"agent {a.space_name} {getattr(a, 'card_name', '')}"
    if t == ActionType.ACQUIRE_CARD:
        return f"buy {a.acquire_card_name}"
    if t == ActionType.ACQUIRE_RESERVE:
        return f"buy {a.reserve_type}"
    return repr(a)


def capture(games):
    """Replay -> [{game, gs, pid, cands, chosen}] for every winner decision
    with 2+ options of a type the heuristic shapes."""
    from src.data.card_definitions import setup_game
    from src.game.gameState import ActionType
    keep = (ActionType.AGENT_TURN, ActionType.ACQUIRE_CARD, ActionType.ACQUIRE_RESERVE,
            ActionType.REVEAL_TURN, ActionType.RESOLVE_DEPLOY, ActionType.END_REVEAL)
    pos = []
    for g in games:
        gs = _setup(g["seed"], g.get("leaders"))
        for k in g["moves"]:
            pid = gs.player_in_reveal_buy
            if pid is None:
                pid = gs.get_current_player_id()
            valid = gs.get_valid_actions(pid)
            a = valid[k]
            cands = [x for x in valid if x.action_type.value != "no_op"]
            if pid == g["winner"] and len(cands) > 1 and a.action_type in keep:
                pos.append({"game": g["g"], "gs": gs.clone(), "pid": pid, "cands": cands,
                            "chosen": label(a)})
            gs.step(a)
        if [p.victory_points for p in gs.players] != g["vp"]:
            print(f"  warning: game {g['g']} did not replay exactly", flush=True)
    return pos


_POS = None


def _init(path):
    global _POS
    import pickle
    _POS = pickle.load(open(path, "rb"))


def _agree(args):
    tuning, which = args
    from src.ai.agents import HeuristicAgent
    h = HeuristicAgent(seed=0, tuning=tuning)
    hit = 0
    for x in _POS[which]:
        best = max(x["cands"], key=lambda a: h.score(x["gs"], x["pid"], a))
        hit += label(best) == x["chosen"]
    return hit / max(1, len(_POS[which]))


def fit(start, out, workers, fast_max=8, sweeps=2):
    import pickle
    from src.selfplay.evolve import MUTABLE, effective
    games = [g for g in load() if g["rounds"] <= fast_max]
    ids = sorted({g["g"] for g in games})
    random.Random(7).shuffle(ids)
    hold = set(ids[: max(5, len(ids) // 5)])
    print(f"{len(games)} fast games ({len(hold)} held out); capturing winners' decisions", flush=True)
    t0 = time.time()
    pos = capture(games)
    data = {"fit": [x for x in pos if x["game"] not in hold], "val": [x for x in pos if x["game"] in hold]}
    pk = os.path.join(DIR, "positions.pkl")
    pickle.dump(data, open(pk, "wb"))
    print(f"  {len(data['fit'])} fit / {len(data['val'])} held-out decisions, {time.time() - t0:.0f}s",
          flush=True)
    cur = json.load(open(start))
    eff = effective(cur)
    keys = [k for k in MUTABLE if k != "look_w" and isinstance(eff.get(k), (int, float)) and eff[k] > 0]
    with Pool(workers, initializer=_init, initargs=(pk,)) as pool:
        base_fit, base_val = pool.map(_agree, [(cur, "fit"), (cur, "val")])
        print(f"start weights agree with the fast winners: fit {base_fit:.3f}  held-out {base_val:.3f}",
              flush=True)
        best = base_fit
        for sweep in range(sweeps):
            improved = False
            trials = [(k, m) for k in keys for m in (0.7, 0.85, 1.2, 1.4)]
            tun = [{**cur, k: round(effective(cur)[k] * m, 6)} for k, m in trials]
            scores = pool.map(_agree, [(t, "fit") for t in tun])
            # take the best change per weight, greedily, re-checking each against the running fit
            by_key = {}
            for (k, m), t, sc in zip(trials, tun, scores):
                if sc > by_key.get(k, (None, best + 0.002))[1]:
                    by_key[k] = (t[k], sc)
            for k, (v, sc) in sorted(by_key.items(), key=lambda kv: -kv[1][1]):
                trial = {**cur, k: v}
                s2 = pool.map(_agree, [(trial, "fit")])[0]
                if s2 > best + 0.002:
                    best, cur, improved = s2, trial, True
                    print(f"  sweep {sweep}: {k} -> {v:g}  fit {s2:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            if not improved:
                break
        val = pool.map(_agree, [(cur, "val")])[0]
    print(f"fitted: fit {best:.3f} (start {base_fit:.3f}), held-out {val:.3f} (start {base_val:.3f})")
    json.dump(cur, open(out, "w"), indent=1)
    print("wrote", out)
    return base_val, val


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("play")
    a.add_argument("--games", type=int, default=600)
    a.add_argument("--spec", required=True, help="agent spec for all 4 seats")
    a.add_argument("--workers", type=int, default=28)
    a.add_argument("--seed", type=int, default=8_000_000)
    a.add_argument("--tag", default="look")
    b = sub.add_parser("analyse")
    b.add_argument("--fast", type=int, default=8)
    c = sub.add_parser("fit")
    c.add_argument("--start", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=28)
    c.add_argument("--fast", type=int, default=8)
    A = ap.parse_args()
    if A.cmd == "play":
        play(A.games, [A.spec] * 4, A.workers, A.seed, A.tag)
    elif A.cmd == "analyse":
        analyse(A.fast)
    else:
        fit(A.start, A.out, A.workers, A.fast)


if __name__ == "__main__":
    main()
