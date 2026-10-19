"""Training data + fit for the learned Conflict forecast (src/ai/fight_forecast.py).

Plays Best AI self-play games; during Player Turns it snapshots every
player's public (and own private) features now and then, and labels each
snapshot with how that round's Conflict actually ended for that player.

    python -m src.selfplay.fight_data [--games 800] [--workers 8] [--spec SPEC] [--refit]

Fits (ridge least squares / logistic) and writes models_fight/forecast.json:
  enter_w_*     P(no units now -> units in the fight at the end)     PUBLIC
  rival_w_*     final strength - visible now, given in the fight     PUBLIC
  rival_sd_w_*  spread of that (fit on |residual| * sqrt(pi/2))      PUBLIC
                (* = rev / unrev: the rival has / has not revealed yet)
  own_w       final swords (not troops/worms/skills) - swords now    PUBLIC + PRIVATE
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from multiprocessing import Pool

import numpy as np

SNAP_P = 0.15


def game(args):
    g, spec = args
    import src.ai.agents as A
    import src.ai.fight_forecast as F
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.gameState import GameState
    from src.game.leader.leader_definitions import LEADER_NAMES

    rng = random.Random(900_000 + g)
    agents = [make_agent(spec, seed=g * 4 + s) for s in range(4)]
    lineup = rng.sample(list(LEADER_NAMES), 4)
    gs = setup_game(num_players=4, seed=900_000 + g, leaders=lineup, use_bloodlines=True)
    rows, pending = [], {}                     # round -> rows waiting for the result

    orig = GameState.resolve_combat

    def resolve(self):
        if self is gs and self.current_conflict is not None:
            bl = getattr(self, "bl", None)
            for r in pending.pop(self.round, []):
                q = r["q"]
                st = float(self.combat_strength.get(q, 0) or 0)
                t, w = self.troops_in_conflict.get(q, 0), self.sandworms_in_conflict.get(q, 0)
                skill = float(bl.strength_bonus(q)) if bl is not None and bl.commanders_in_conflict.get(q, 0) else 0.0
                r["final"] = st
                r["in_final"] = float(t + w > 0)
                r["swords_final"] = (st - 2 * t - 3 * w - skill) if t + w > 0 else 0.0
                rows.append(r)
        return orig(self)

    GameState.resolve_combat = resolve
    try:
        n = 0
        while not gs.game_over and n < 3000:
            pid = gs.player_in_reveal_buy
            if pid is None:
                pid = gs.get_current_player_id()
            if gs.phase.name == "PLAYER_TURNS" and gs.current_conflict is not None and rng.random() < SNAP_P:
                for q in range(gs.num_players):
                    f = F.private_features(gs, q)
                    f["q"] = q
                    f["swords_now"] = float(gs.swords_this_reveal.get(q, 0))
                    mu, sd, can = A._opp_fight(gs, q)      # the hand-made rule, for comparison
                    f["old_mu"], f["old_sd"], f["old_can"] = float(mu), float(sd), float(can)
                    pending.setdefault(gs.round, []).append(f)
            valid = gs.get_valid_actions(pid)
            gs.step(agents[pid].select_action(gs, pid, valid))
            n += 1
    finally:
        GameState.resolve_combat = orig
    return rows


def _ridge(X, y, lam=1.0, w=None):
    if w is not None:
        X, y = X * np.sqrt(w)[:, None], y * np.sqrt(w)
    reg = lam * np.eye(X.shape[1])
    reg[0, 0] = 0.0                            # no shrinkage on the bias
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def _logistic(X, y, lam=1.0, iters=30):
    w = np.zeros(X.shape[1])
    for _ in range(iters):                     # Newton / IRLS
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + lam * w
        H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


def fit(rows):
    """Rival models fitted separately for rivals who have / have not revealed
    yet (user, 2026-10-18): after the Reveal only Combat Intrigues and
    intrigue deploys can still add strength."""
    import src.ai.fight_forecast as F
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    cut = int(0.8 * len(rows))
    tr, te = idx[:cut], idx[cut:]
    out, rep = {}, {}

    def mat(sel, names):
        return np.array([F.vec(rows[i], names) for i in sel]).reshape(len(sel), len(names))

    def ll(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)

    for tag, rv in (("unrev", 0.0), ("rev", 1.0)):
        side = lambda sel: [i for i in sel if rows[i]["revealed"] == rv]
        # enter: no units now -> units at the end
        e_tr = [i for i in side(tr) if rows[i]["units"] == 0]
        e_te = [i for i in side(te) if rows[i]["units"] == 0]
        ye = np.array([rows[i]["in_final"] for i in e_tr])
        out["enter_w_" + tag] = _logistic(mat(e_tr, F.PUBLIC), ye)
        yt = np.array([rows[i]["in_final"] for i in e_te])
        pe = 1 / (1 + np.exp(-mat(e_te, F.PUBLIC) @ out["enter_w_" + tag]))
        rep["enter_" + tag] = {"n": len(e_tr), "rate": round(float(ye.mean()), 3), "test_logloss": ll(yt, pe),
                               "test_logloss_const": ll(yt, np.full(len(yt), ye.mean())),
                               "test_logloss_old_rule": ll(yt, np.array([rows[i]["old_can"] for i in e_te]) * 0.98 + 0.01)}
        # strength added, given in the fight at the end
        r_tr = [i for i in side(tr) if rows[i]["in_final"]]
        r_te = [i for i in side(te) if rows[i]["in_final"]]
        Xr = mat(r_tr, F.PUBLIC)
        yr = np.array([rows[i]["final"] - rows[i]["visible"] for i in r_tr])
        w = _ridge(Xr, yr)
        out["rival_w_" + tag] = w
        out["rival_sd_w_" + tag] = _ridge(Xr, np.abs(yr - Xr @ w) * math.sqrt(math.pi / 2))
        Xt = mat(r_te, F.PUBLIC)
        yt = np.array([rows[i]["final"] - rows[i]["visible"] for i in r_te])
        old = np.array([rows[i]["old_mu"] - rows[i]["visible"] for i in r_te])
        pred = np.maximum(0, Xt @ w)
        rep["rival_" + tag] = {"n": len(r_tr), "mean_add": round(float(yr.mean()), 2),
                               "test_mae": round(float(np.mean(np.abs(yt - pred))), 2),
                               "test_bias": round(float(np.mean(pred - yt)), 2),
                               "test_mae_old_rule": round(float(np.mean(np.abs(yt - old))), 2),
                               "test_bias_old_rule": round(float(np.mean(old - yt)), 2)}
    # own: swords still to come (I see my hand), given in the fight at the end
    r_tr = [i for i in tr if rows[i]["in_final"]]
    r_te = [i for i in te if rows[i]["in_final"]]
    yo = np.array([rows[i]["swords_final"] - rows[i]["swords_now"] for i in r_tr])
    out["own_w"] = _ridge(mat(r_tr, F.PUBLIC + F.PRIVATE), yo)
    yot = np.array([rows[i]["swords_final"] - rows[i]["swords_now"] for i in r_te])
    old = np.array([0.0 if rows[i]["revealed"] else 0.7 * rows[i]["hand_swords"] for i in r_te]) \
        + 0.6 * np.array([rows[i]["combat_intrigue_swords"] for i in r_te])
    rep["own"] = {"n": len(r_tr), "mean_swords_to_come": round(float(yo.mean()), 2),
                  "test_mae": round(float(np.mean(np.abs(yot - np.maximum(0, mat(r_te, F.PUBLIC + F.PRIVATE) @ out["own_w"])))), 2),
                  "test_mae_old_rule": round(float(np.mean(np.abs(yot - old))), 2)}
    return out, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--spec", default="heuristic:tuned=config/best_ai.json")
    ap.add_argument("--rows", default="models_fight/rows.jsonl", help="saved snapshots (reused with --refit)")
    ap.add_argument("--refit", action="store_true", help="fit on the saved rows, play no games")
    A = ap.parse_args()
    rows = []
    if A.refit:
        with open(A.rows) as fh:
            rows = [json.loads(line) for line in fh]
    else:
        with Pool(A.workers) as pool:
            for r in pool.imap_unordered(game, [(g, A.spec) for g in range(A.games)], chunksize=2):
                rows += r
        os.makedirs(os.path.dirname(A.rows), exist_ok=True)
        with open(A.rows, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    import src.ai.fight_forecast as F
    w, rep = fit(rows)
    os.makedirs(os.path.dirname(F.MODEL_PATH), exist_ok=True)
    with open(F.MODEL_PATH, "w") as fh:
        json.dump({**{k: [round(float(x), 5) for x in v] for k, v in w.items()},
                   "public": list(F.PUBLIC), "private": list(F.PRIVATE),
                   "games": A.games, "rows": len(rows), "report": rep}, fh, indent=1)
    print(json.dumps({"rows": len(rows), **rep}, indent=1))


if __name__ == "__main__":
    main()
