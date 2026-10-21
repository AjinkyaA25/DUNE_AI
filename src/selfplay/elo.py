"""Elo ratings for PPO snapshots (user, 2026-10-20: "implement an elo system to see if
the AI is getting better over time").

Every rated game is a 4-player game between named players (PPO snapshots, frozen
pool nets, Best AI). Each game is split into its 6 pairwise results by finishing
order (winner first, then VP; equal VP = half a win each; two seats of the same
player are skipped), and ALL games stored so far are refit together (Bradley-Terry
maximum likelihood, Elo scale: P(i beats j) = 1 / (1 + 10^((Rj - Ri) / 400))), with
Best AI fixed at 1500. Refitting everything means the order games came in doesn't
matter, unlike incremental Elo updates, and old snapshots' ratings keep sharpening as
newer ones play them.

ppo_train --elo-games N calls rate_snapshot() at every eval, so the newest snapshot plays N
games vs the rated field. Ratings go to <out>/elo.json, games to <out>/elo_games.jsonl.

    python -m src.selfplay.elo models_ppo_self          # table of ratings by iteration
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np

ANCHOR, ANCHOR_RATING = "best_ai", 1500.0
SCALE = 400 / math.log(10)                   # natural-log strength -> Elo points


def pairwise(players, vp, winner):
    """One 4-player game -> [(i, j, score of i vs j)] by finishing order."""
    key = [(1 if s == winner else 0, vp[s]) for s in range(len(players))]
    out = []
    for a in range(len(players)):
        for b in range(a + 1, len(players)):
            if players[a] == players[b]:
                continue
            sc = 1.0 if key[a] > key[b] else 0.0 if key[a] < key[b] else 0.5
            out.append((players[a], players[b], sc))
    return out


def fit(games, anchor=ANCHOR, anchor_rating=ANCHOR_RATING, prior_sd=800.0, iters=50):
    """games: [{"players": [4 names], "vp": [4], "winner": seat}] -> {name: (rating, se, games)}.
    A weak Gaussian prior (prior_sd Elo around the anchor) keeps players with only wins
    or only losses finite."""
    names = sorted({p for g in games for p in g["players"]} | {anchor})
    ix = {n: i for i, n in enumerate(names)}
    n = len(names)
    S = np.zeros((n, n))                     # S[i, j] = score of i vs j
    N = np.zeros((n, n))
    played = np.zeros(n)
    for g in games:
        for p in set(g["players"]):
            played[ix[p]] += 1
        for a, b, sc in pairwise(g["players"], g["vp"], g["winner"]):
            i, j = ix[a], ix[b]
            S[i, j] += sc; S[j, i] += 1 - sc
            N[i, j] += 1; N[j, i] += 1
    k = ix[anchor]
    free = [i for i in range(n) if i != k]
    th = np.zeros(n)                         # strength in natural-log units, anchor at 0
    lam = (SCALE / prior_sd) ** 2
    H = np.eye(len(free)) * lam
    for _ in range(iters):
        P = 1 / (1 + np.exp(th[None, :] - th[:, None]))      # P[i, j] = P(i beats j)
        grad = (S - N * P).sum(1) - lam * th
        W = N * P * (1 - P)
        Hfull = np.diag(W.sum(1) + lam) - W
        H = Hfull[np.ix_(free, free)]
        step = np.linalg.solve(H, grad[free])
        th[free] += step
        if np.abs(step).max() < 1e-7:
            break
    se = np.zeros(n)
    se[free] = np.sqrt(np.diag(np.linalg.inv(H)))
    return {nm: (round(float(anchor_rating + SCALE * th[ix[nm]]), 1), round(float(SCALE * se[ix[nm]]), 1),
                 int(played[ix[nm]])) for nm in names}


def play(args):
    """One rated game: (g, [4 names], {name: spec}) -> {"players", "vp", "winner", "g"}."""
    g, players, specs = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.leader.leader_definitions import LEADER_NAMES
    r = random.Random(7_000_000 + g)
    lineup = r.sample(list(LEADER_NAMES), 4)
    gs = setup_game(num_players=4, seed=7_000_000 + g, leaders=lineup, use_bloodlines=True)
    agents = [make_agent(specs[p], seed=g * 4 + s) for s, p in enumerate(players)]
    n = 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    vp = [q.victory_points for q in gs.players]
    win = gs.winner if gs.winner is not None else max(range(4), key=lambda s: vp[s])
    return {"g": g, "players": players, "vp": vp, "winner": win}


def _load(out):
    path = os.path.join(out, "elo.json")
    st = json.load(open(path)) if os.path.exists(path) else {"specs": {}, "ratings": {}, "next_game": 0}
    gpath = os.path.join(out, "elo_games.jsonl")
    games = [json.loads(l) for l in open(gpath)] if os.path.exists(gpath) else []
    return st, games


def rate_snapshot(pool, out, name, spec, n_games, field, anchor_spec, anchor_frac=0.25):
    """`name` (spec) plays n_games: one seat (rotating), the other three drawn from the
    rated field (Best AI with prob anchor_frac, else any earlier rated player, and the
    `field` {name: spec} of frozen nets) -> refit, save, return the ratings.
    With name=None it only plays the field among itself (bootstrap before training)."""
    st, games = _load(out)
    specs = st["specs"]
    specs[ANCHOR] = anchor_spec
    specs.update(field)
    if name:
        specs[name] = spec
    others = [p for p in specs if p not in (ANCHOR, name) and os.path.exists(_net_of(specs[p]))]
    r = random.Random(st["next_game"])
    jobs = []
    for g in range(st["next_game"], st["next_game"] + n_games):
        seat = g % 4
        pl = [ANCHOR if not others or r.random() < anchor_frac else r.choice(others) for _ in range(4)]
        if name:
            pl[seat] = name
        jobs.append((g, pl, specs))
    new = pool.map(play, jobs, chunksize=4)
    with open(os.path.join(out, "elo_games.jsonl"), "a") as f:
        for x in new:
            f.write(json.dumps(x) + "\n")
    games += new
    st["next_game"] += n_games
    st["ratings"] = fit(games)
    tmp = os.path.join(out, "elo.json.tmp")
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(out, "elo.json"))
    return st["ratings"]


def _net_of(spec):
    """Path of the .npz a ppo:net=... spec plays (heuristics: always available)."""
    for part in spec.split(":"):
        if part.startswith("net="):
            return part[4:]
    return os.devnull


def report(out):
    st, games = _load(out)
    rt = st["ratings"]
    order = sorted(rt, key=lambda p: (p != ANCHOR, p))
    print(f"{len(games)} rated games; Elo with {ANCHOR} fixed at {ANCHOR_RATING:.0f} (95% CI = +-2 se)")
    print(f"{'player':<16}{'Elo':>8}{'se':>7}{'games':>8}")
    for p in order:
        r, se, n = rt[p]
        print(f"{p:<16}{r:>8.0f}{se:>7.0f}{n:>8}")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "models_ppo_self")
