"""Self-play worker: plays games with the current net and writes a shard.

Playout cap randomization (KataGo): with probability `p_full` a decision gets
a full search (`sims` simulations) and becomes a policy training target;
every other decision is played straight from the net's policy (sampled), so
a game costs a fraction of searching every move. Every recorded position
also gets a value target: the game's final result for each seat.

`eps` (exploration): that share of the unsearched decisions is a uniformly
random legal move instead, so games wander into lines the net would never
pick; the ones that end in wins get rewarded through the value targets and
the searches that follow.

`credit` (exploration credit): a random move is also judged directly. Its
advantage = half (the player's value at their next decision - value before
the move) + half (final result - value before); the policy target for that
position is the net's prior shifted toward the move by credit x advantage
(away from it when negative). `credit_frac` of random moves are recorded
this way, so they don't swamp the search targets.

`coach` > 0: the user's gameplay advice (src/az/coach.py) tilts every seat
the same way, credited by outcome like the personas.

`persona` > 0: each seat gets a playstyle (src/az/personas.py) that tilts the
move it plays by persona x bias. Search targets stay unbiased; a persona
move that differs from the net's own top choice is credited like a random
move, so the styles that win teach the policy.
"""
from __future__ import annotations

import os
import time

import numpy as np

from src.az.encode import encode_actions, encode_obs
from src.az.mcts import active, evaluate, legal, search, terminal_values
from src.az.net import AZNet

MAX_STEPS = 4000


def _rel(z: np.ndarray, pid: int) -> np.ndarray:
    n = len(z)
    return np.array([z[(pid + k) % n] for k in range(n)], np.float32)


def play_game(net: AZNet, seed: int, sims: int, p_full: float, p_value: float,
              rng: np.random.Generator, eps: float = 0.0, credit: float = 0.0,
              credit_frac: float = 0.15, persona: float = 0.0, coach: float = 0.0):
    from src.data.card_definitions import setup_game
    from src.az import personas as P
    gs = setup_game(num_players=4, seed=seed, neutral_leaders=True, use_bloodlines=True)
    seat_style = P.assign(rng) if persona else None
    moves: list = []                  # index of each move among get_valid_actions (replays)

    def record(pid, a):
        v = gs.get_valid_actions(pid)
        key = repr(a)
        moves.append(next((i for i, x in enumerate(v) if repr(x) == key), -1))

    def lean(pid, acts, p):           # play distribution tilted toward style / coach hints
        if seat_style is None and not coach:
            return p, None
        b = np.zeros(len(acts), np.float32)
        if seat_style is not None:
            b += persona * P.bias(gs, pid, acts, seat_style[pid])
        if coach:
            from src.az.coach import bias as coach_bias
            b += coach * coach_bias(gs, pid, acts)
        if not b.any():
            return p, b
        q = np.asarray(p, np.float64) * np.exp(b)
        return q / q.sum(), b
    pol, val = [], []                 # policy records / value-only records
    explore, waiting = [], {}         # credited random moves; pid -> one awaiting its next value
    steps = n_full = 0

    def reached(pid, v_pid):          # pid's next decision: close its pending random move
        i = waiting.pop(pid, None)
        if i is not None:
            explore[i][7] = v_pid
    vp_r6 = None                      # VP when round 6 is over (the scoring-pace check)
    while not gs.game_over and steps < MAX_STEPS:
        if vp_r6 is None and gs.round > 6:
            vp_r6 = [p.victory_points for p in gs.players]
        pid = active(gs)
        acts = legal(gs, pid)
        if len(acts) == 1:
            record(pid, acts[0])
            gs.step(acts[0])
            steps += 1
            continue
        if pid in waiting and not (eps and credit):
            waiting.pop(pid)
        if rng.random() < p_full:
            v_now = None
            if pid in waiting or (credit and (seat_style or coach)):
                v_now = float(evaluate(net, gs, pid, acts)[1][pid])
                reached(pid, v_now)
            visits = search(net, gs, pid, acts, sims, rng)
            n_full += 1
            if visits.sum() == 0:
                visits[:] = 1.0
            pi = visits / visits.sum()
            dense, idx = encode_actions(gs, pid, acts)
            obs = encode_obs(gs, pid)
            pol.append((pid, obs, dense, idx, pi))
            q, b = lean(pid, acts, pi)
            k = int(rng.choice(len(acts), p=q))
            a = acts[k]
            # the style / coach overruled search: judge that move by what follows
            if (credit and b is not None and b[k] > 0 and k != int(np.argmax(pi))
                    and pid not in waiting and v_now is not None and rng.random() < credit_frac):
                explore.append([pid, obs, dense, idx, pi, k, v_now, None])
                waiting[pid] = len(explore) - 1
        elif eps and rng.random() < eps:
            obs = encode_obs(gs, pid)
            if rng.random() < p_value:
                val.append((pid, obs))
            k = int(rng.integers(len(acts)))
            a = acts[k]
            if credit:
                pr, v = evaluate(net, gs, pid, acts)
                reached(pid, float(v[pid]))
                if rng.random() < credit_frac:
                    dense, idx = encode_actions(gs, pid, acts)
                    explore.append([pid, obs, dense, idx, pr, k, float(v[pid]), None])
                    waiting[pid] = len(explore) - 1
        else:
            pr, v = evaluate(net, gs, pid, acts)
            reached(pid, float(v[pid]))
            obs = encode_obs(gs, pid) if (seat_style or coach or rng.random() < p_value) else None
            if obs is not None and rng.random() < p_value:
                val.append((pid, obs))
            q, b = lean(pid, acts, pr)
            k = int(rng.choice(len(acts), p=q))
            a = acts[k]
            # a style move the net wouldn't have picked: judge it by what follows
            if (credit and b is not None and b[k] > 0 and k != int(np.argmax(pr))
                    and pid not in waiting and rng.random() < credit_frac):
                dense, idx = encode_actions(gs, pid, acts)
                explore.append([pid, obs, dense, idx, pr, k, float(v[pid]), None])
                waiting[pid] = len(explore) - 1
        record(pid, a)
        gs.step(a)
        steps += 1
    if not gs.game_over:
        return None
    z = terminal_values(gs)
    vp = [p.victory_points for p in gs.players]
    n_cred = 0
    for pid, obs, dense, idx, pr, k, v0, v1 in explore:
        v1 = float(z[pid]) if v1 is None else v1
        adv = 0.5 * (v1 - v0) + 0.5 * (float(z[pid]) - v0)
        s = float(np.clip(credit * adv, -0.9, 0.9))
        pi = np.asarray(pr, np.float64).copy()
        if s > 0:
            pi *= 1 - s
            pi[k] += s
        else:
            pi[k] *= 1 + s
            pi /= pi.sum()
        pol.append((pid, obs, dense, idx, pi.astype(np.float32)))
        n_cred += 1
    w = gs.winner if gs.winner is not None else int(np.argmax(vp))
    if vp_r6 is None:
        vp_r6 = vp                    # finished in round 6 or earlier
    return pol, val, z, {"rounds": gs.round, "vp": vp, "vp_r6": vp_r6, "steps": steps, "full": n_full,
                         "seed": seed, "moves": moves,
                         "credited": n_cred, "styles": seat_style, "winner": w}


def _save_games(shard_path: str, recs: list) -> None:
    """Self-play games, replayable in the UI: <run>/games/<shard>.jsonl.gz,
    one game per line (seed + index of each move among get_valid_actions)."""
    import gzip
    import json
    if not recs:
        return
    run = os.path.dirname(os.path.dirname(os.path.abspath(shard_path)))
    d = os.path.join(run, "games")
    os.makedirs(d, exist_ok=True)
    name = os.path.splitext(os.path.basename(shard_path))[0] + ".jsonl.gz"
    with gzip.open(os.path.join(d, name), "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + chr(10))


def worker(args):
    """Play `n_games` and write one shard. Returns (path, stats)."""
    net_path, out_path, seed0, n_games, sims, p_full, p_value = args[:7]
    eps = args[7] if len(args) > 7 else 0.0
    credit = args[8] if len(args) > 8 else 0.0
    persona = args[9] if len(args) > 9 else 0.0
    coach = args[10] if len(args) > 10 else 0.0
    net = AZNet.load(net_path)
    rng = np.random.default_rng(seed0)
    S, Z, POS, ACT_OFF, DENSE, IDX, PI = [], [], [], [0], [], [], []
    VS, VZ = [], []
    FW, VFW = [], []        # 1 = the winner's position in a game that ended by round 8
    stats = []
    t0 = time.time()
    game_recs = []
    for g in range(n_games):
        r = play_game(net, seed0 * 1000 + g, sims, p_full, p_value, rng, eps, credit,
                      persona=persona, coach=coach)
        if r is None:
            continue
        pol, val, z, st = r
        game_recs.append({"seed": st["seed"], "moves": st.pop("moves"), "rounds": st["rounds"],
                          "vp": st["vp"], "winner": st.get("winner"), "styles": st.get("styles"),
                          "net": net_path, "setup": {"players": 4, "neutral_leaders": True,
                                                     "use_bloodlines": True}})
        stats.append(st)
        fast_w = st.get("winner") if st["rounds"] <= 8 else None
        for pid, obs, dense, idx, pi in pol:
            FW.append(pid == fast_w)
            S.append(obs.astype(np.float16))
            Z.append(_rel(z, pid))
            DENSE.append(dense.astype(np.float16))
            IDX.append(idx.astype(np.int16))
            PI.append(pi.astype(np.float32))
            ACT_OFF.append(ACT_OFF[-1] + len(pi))
        for pid, obs in val:
            VFW.append(pid == fast_w)
            VS.append(obs.astype(np.float16))
            VZ.append(_rel(z, pid))
    _save_games(out_path, game_recs)
    np.savez(out_path,
             s=np.stack(S) if S else np.zeros((0, 1), np.float16),
             z=np.stack(Z) if Z else np.zeros((0, 4), np.float32),
             off=np.asarray(ACT_OFF, np.int64),
             dense=np.concatenate(DENSE) if DENSE else np.zeros((0, 1), np.float16),
             idx=np.concatenate(IDX) if IDX else np.zeros((0, 1), np.int16),
             pi=np.concatenate(PI) if PI else np.zeros(0, np.float32),
             vs=np.stack(VS) if VS else np.zeros((0, 1), np.float16),
             vz=np.stack(VZ) if VZ else np.zeros((0, 4), np.float32),
             fw=np.asarray(FW, np.bool_), vfw=np.asarray(VFW, np.bool_))
    return out_path, stats, time.time() - t0
