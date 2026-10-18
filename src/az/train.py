"""AlphaZero with hidden information: self-play -> train -> repeat.

    python -m src.az.train --games 1000000 --per-gen 10000 --workers 28

Each generation the workers play `per-gen` games with the current net
(src/az/selfplay.py), the net is trained on the last `window` generations
of positions on the GPU, and every `eval-every` generations the new net
plays the heuristic AI. Everything lands in --out (default models_az/):
net_<gen>.npz, log.jsonl (one line per generation), shards/. Rerunning the
same command resumes from the last saved generation.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 1 BLAS thread per worker (spawned workers inherit)
import argparse
import glob
import json
import time
from multiprocessing import Pool

import numpy as np

from src.az.encode import DENSE_DIM, HASH_DIM, MAX_IDX, STATE_DIM
from src.az.net import SHAPES, init_params

A_CAP = 24                     # moves kept per policy target (highest visit share)
HEUR = "heuristic:tuned=config/gap/tsmf5_dp1_wb10_tm1.json"


# ---------------------------------------------------------------- evaluation

def eval_game(args):
    """One 4-player game: the net in seat g % 4, the heuristic elsewhere."""
    net_path, g, mode, sims = args
    from src.ai.agents import make_agent
    from src.az.mcts import active, evaluate, legal, search, terminal_values
    from src.az.net import AZNet
    from src.data.card_definitions import setup_game
    net = AZNet.load(net_path)
    rng = np.random.default_rng(10_000_000 + g)
    seat = g % 4
    gs = setup_game(num_players=4, seed=5_000_000 + g, neutral_leaders=True, use_bloodlines=True)
    heur = {i: make_agent(HEUR, seed=g * 4 + i) for i in range(4) if i != seat}
    steps = 0
    vp_r6 = None
    while not gs.game_over and steps < 4000:
        if vp_r6 is None and gs.round > 6:
            vp_r6 = gs.players[seat].victory_points
        pid = active(gs)
        acts = legal(gs, pid)
        if pid != seat:
            a = heur[pid].select_action(gs, pid, gs.get_valid_actions(pid))
        elif len(acts) == 1:
            a = acts[0]
        elif mode == "search":
            a = acts[int(np.argmax(search(net, gs, pid, acts, sims, rng, noise=0.0)))]
        else:
            pr, _ = evaluate(net, gs, pid, acts)
            a = acts[int(np.argmax(pr))]
        gs.step(a)
        steps += 1
    if not gs.game_over:
        return None
    vp = [p.victory_points for p in gs.players]
    top = max(vp)
    winners = [gs.winner] if gs.winner is not None else [i for i, v in enumerate(vp) if v == top]
    win = 1.0 / len(winners) if seat in winners else 0.0
    return {"win": win, "vp": vp[seat], "opp_vp": float(np.mean([v for j, v in enumerate(vp) if j != seat])),
            "rounds": gs.round, "vp_r6": vp[seat] if vp_r6 is None else vp_r6}


def run_eval(pool, net_path, n, mode, sims=32):
    res = [r for r in pool.map(eval_game, [(net_path, g, mode, sims) for g in range(n)]) if r]
    if not res:
        return {}
    return {"games": len(res), "win_share": float(np.mean([r["win"] for r in res])),
            "vp": float(np.mean([r["vp"] for r in res])),
            "opp_vp": float(np.mean([r["opp_vp"] for r in res])),
            "rounds": float(np.mean([r["rounds"] for r in res])),
            "vp_r6": float(np.mean([r["vp_r6"] for r in res]))}


# ---------------------------------------------------------------- training

def load_buffer(paths, fast_repeat: int = 0):
    """fast_repeat > 0: the winner's positions from games that ended by round 8
    (shard flags fw / vfw) appear 1 + fast_repeat times, so the net learns
    more from fast wins."""
    S, Z, D, I, P, VS, VZ = [], [], [], [], [], [], []
    for p in paths:
        with np.load(p) as z:
            if len(z["z"]):
                off = z["off"]
                s, dense, idx, pi = z["s"], z["dense"], z["idx"], z["pi"]
                n = len(off) - 1
                d = np.zeros((n, A_CAP, DENSE_DIM), np.float16)
                ii = np.full((n, A_CAP, MAX_IDX), HASH_DIM, np.int16)
                pp = np.zeros((n, A_CAP), np.float32)
                for k in range(n):
                    lo, hi = off[k], off[k + 1]
                    sel = np.arange(lo, hi)
                    if hi - lo > A_CAP:
                        sel = lo + np.argsort(-pi[lo:hi], kind="stable")[:A_CAP]
                    m = len(sel)
                    d[k, :m], ii[k, :m], pp[k, :m] = dense[sel], idx[sel], pi[sel]
                    pp[k, :m] /= max(pp[k, :m].sum(), 1e-8)
                    if m < A_CAP:
                        pp[k, m:] = -1.0            # padding marker
                rows = np.arange(n)
                if fast_repeat and "fw" in z.files and z["fw"].any():
                    rows = np.concatenate([rows] + [np.flatnonzero(z["fw"])] * fast_repeat)
                S.append(s[rows]); Z.append(z["z"][rows]); D.append(d[rows]); I.append(ii[rows])
                P.append(pp[rows])
            if len(z["vz"]):
                rows = np.arange(len(z["vz"]))
                if fast_repeat and "vfw" in z.files and z["vfw"].any():
                    rows = np.concatenate([rows] + [np.flatnonzero(z["vfw"])] * fast_repeat)
                VS.append(z["vs"][rows]); VZ.append(z["vz"][rows])
    cat = lambda L, shape, dt: np.concatenate(L) if L else np.zeros(shape, dt)
    return (cat(S, (0, STATE_DIM), np.float16), cat(Z, (0, 4), np.float32),
            cat(D, (0, A_CAP, DENSE_DIM), np.float16), cat(I, (0, A_CAP, MAX_IDX), np.int16),
            cat(P, (0, A_CAP), np.float32), cat(VS, (0, STATE_DIM), np.float16),
            cat(VZ, (0, 4), np.float32))


def train(params, buf, steps, batch, lr, device):
    import torch
    import torch.nn.functional as F
    t = {k: torch.tensor(v, device=device, requires_grad=True) for k, v in params.items()}
    opt = torch.optim.Adam(list(t.values()), lr=lr, weight_decay=1e-5)
    S, Z, D, I, P, VS, VZ = buf
    gpu = lambda a, dt=None: torch.as_tensor(a, device=device, dtype=dt)
    S, Z, D, I, P = gpu(S), gpu(Z), gpu(D), gpu(I, torch.long), gpu(P)
    VS, VZ = gpu(VS), gpu(VZ)
    n, nv = len(S), len(VS)

    def trunk(x):
        h = F.relu(x.float() @ t["W1"] + t["b1"])
        return F.relu(h @ t["W2"] + t["b2"])

    def value(h):
        return F.relu(h @ t["V1"] + t["vb1"]) @ t["V2"] + t["vb2"]

    logs = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "acc": 0.0}
    for step in range(steps):
        bi = torch.randint(0, n, (batch,), device=device)
        h = trunk(S[bi])
        u = (h @ t["P"])[:, None, :] + D[bi].float() @ t["D"] + t["E"][I[bi]].sum(2) + t["pb"]
        lg = F.relu(u) @ t["pw"] + t["pc"]
        pi = P[bi]
        mask = pi >= 0
        lg = lg.masked_fill(~mask, -1e9)
        logp = F.log_softmax(lg, dim=1)
        target = pi.clamp(min=0)
        pl = -(target * logp).sum(1).mean()
        vpred = value(h)
        zb = Z[bi]
        ok = torch.isfinite(zb).all(1)            # human moves from unfinished games: no value target
        vl = F.mse_loss(vpred[ok], zb[ok]) if ok.any() else vpred.sum() * 0.0
        if nv:
            vi = torch.randint(0, nv, (batch,), device=device)
            vl = 0.5 * vl + 0.5 * F.mse_loss(value(trunk(VS[vi])), VZ[vi])
        loss = pl + 2.0 * vl
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            t["E"].grad[HASH_DIM] = 0
        opt.step()
        if step >= steps - 50:
            with torch.no_grad():
                logs["pl"] += pl.item() / 50
                logs["vl"] += vl.item() / 50
                logs["ent"] += -(target * torch.log(target + 1e-9)).sum(1).mean().item() / 50
                logs["acc"] += (lg.argmax(1) == target.argmax(1)).float().mean().item() / 50
    out = {k: v.detach().cpu().numpy() for k, v in t.items()}
    return out, logs


# ---------------------------------------------------------------- driver

def _style_stats(stats):
    """Per persona: win share, average VP, and average rounds of its games."""
    out = {}
    for st in stats:
        styles = st.get("styles")
        if not styles:
            continue
        for seat, s in enumerate(styles):
            d = out.setdefault(s, {"n": 0, "win": 0.0, "vp": 0.0})
            d["n"] += 1
            d["win"] += float(st.get("winner") == seat)
            d["vp"] += st["vp"][seat]
    if not out:
        return {}
    return {"styles": {s: {"win": round(d["win"] / d["n"], 4), "vp": round(d["vp"] / d["n"], 3)}
                       for s, d in sorted(out.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1_000_000)
    ap.add_argument("--per-gen", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--task-games", type=int, default=50)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--p-full", type=float, default=0.03)
    ap.add_argument("--p-value", type=float, default=0.03)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--search-eval-every", type=int, default=20)
    ap.add_argument("--search-eval-games", type=int, default=100)
    ap.add_argument("--out", default="models_az")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="share of unsearched self-play moves made uniformly at random")
    ap.add_argument("--eps-min", type=float, default=0.0)
    ap.add_argument("--eps-decay", type=int, default=50,
                    help="generations over which eps falls linearly to eps-min")
    ap.add_argument("--margin", type=float, default=0.02, help="value per VP of winning margin")
    ap.add_argument("--credit", type=float, default=0.0,
                    help="policy credit per unit of advantage for random moves (0 = off)")
    ap.add_argument("--persona", type=float, default=0.0,
                    help="playstyle lean per seat (sword / deck / resource / faction); 0 = off")
    ap.add_argument("--personas", default="sword,deck,resource,faction",
                    help="the four seat personas (src/az/personas.py)")
    ap.add_argument("--init", default=None,
                    help="start from this net (.npz) instead of a blank one (fine-tuning)")
    ap.add_argument("--human-dir", default=os.path.join("data", "human_az"),
                    help="your UI games' moves (written by ui/server.py)")
    ap.add_argument("--human-repeat", type=int, default=0,
                    help="add each of your recorded moves this many times per generation (0 = off)")
    ap.add_argument("--coach", type=float, default=0.0,
                    help="strength of the coach hints (src/az/coach.py) for every seat; 0 = off")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="every seat's reward + pace x (VP / rounds played): score fast")
    ap.add_argument("--fast-repeat", type=int, default=0,
                    help="extra copies of the winner's positions from games that ended by round 8")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="winner bonus per round the game ended before round 10")
    A = ap.parse_args()
    os.environ["AZ_MARGIN"], os.environ["AZ_SPEED"] = str(A.margin), str(A.speed)
    os.environ["AZ_PACE"] = str(A.pace)
    os.environ["AZ_PERSONAS"] = A.personas

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(A.out, "shards"), exist_ok=True)
    log_path = os.path.join(A.out, "log.jsonl")
    with open(os.path.join(A.out, "config.json"), "w") as f:
        json.dump(vars(A), f, indent=1)
    nets = sorted(glob.glob(os.path.join(A.out, "net_*.npz")),
                  key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if nets:
        gen = int(nets[-1].rsplit("_", 1)[1].split(".")[0])
        with np.load(nets[-1]) as z:
            params = {k: z[k] for k in z.files}
        print(f"resuming at generation {gen}", flush=True)
    elif A.init:
        with np.load(A.init) as z:
            params = {k: z[k] for k in z.files}
        gen = 0
        np.savez(os.path.join(A.out, "net_0.npz"), **params)
        print(f"fine-tuning from {A.init}", flush=True)
    else:
        gen, params = 0, init_params(0)
        np.savez(os.path.join(A.out, "net_0.npz"), **params)
    n_gens = A.games // A.per_gen
    print(f"device {device}; {n_gens} generations x {A.per_gen} games; state {STATE_DIM}, "
          f"params {sum(np.prod(s) for s in SHAPES.values()):,}", flush=True)

    with Pool(A.workers, maxtasksperchild=40) as pool:
        if gen == 0 and not os.path.exists(log_path):
            ev = run_eval(pool, os.path.join(A.out, "net_0.npz"), A.eval_games, "policy")
            with open(log_path, "a") as f:
                f.write(json.dumps({"gen": 0, "games": 0, "eval_policy": ev}) + "\n")
            print("gen 0 eval", ev, flush=True)
        while gen < n_gens:
            t0 = time.time()
            net_path = os.path.join(A.out, f"net_{gen}.npz")
            eps = max(A.eps_min, A.eps * (1 - gen / max(1, A.eps_decay))) if A.eps else 0.0
            tasks = [(net_path, os.path.join(A.out, "shards", f"g{gen + 1}_{i}.npz"),
                      (gen + 1) * 100_000 + i, A.task_games, A.sims, A.p_full, A.p_value, eps, A.credit,
                      A.persona, A.coach)
                     for i in range(A.per_gen // A.task_games)]
            stats = []
            for _, st, _ in pool.imap_unordered(__import__("src.az.selfplay", fromlist=["worker"]).worker, tasks):
                stats += st
            t_play = time.time() - t0
            gen += 1
            for old in glob.glob(os.path.join(A.out, "shards", "g*_*.npz")):
                if int(os.path.basename(old)[1:].split("_")[0]) <= gen - A.window:
                    os.remove(old)
            paths = [p for p in glob.glob(os.path.join(A.out, "shards", "g*_*.npz"))]
            if A.human_repeat:                      # learn from the user's UI games too
                human = sorted(glob.glob(os.path.join(A.human_dir, "*.npz")))
                paths += human * A.human_repeat
            t1 = time.time()
            buf = load_buffer(paths, A.fast_repeat)
            params, tl = train(params, buf, A.train_steps, A.batch, A.lr, device)
            del buf
            np.savez(os.path.join(A.out, f"net_{gen}.npz"), **params)
            for old in glob.glob(os.path.join(A.out, "net_*.npz")):
                g_old = int(old.rsplit("_", 1)[1].split(".")[0])
                if g_old < gen - 1 and g_old % 10:
                    os.remove(old)              # keep every 10th generation
            rec = {"gen": gen, "games": gen * A.per_gen, "play_s": round(t_play), "eps": round(eps, 3),
                   "credited": float(np.mean([s.get("credited", 0) for s in stats])),
                   **_style_stats(stats),
                   "train_s": round(time.time() - t1),
                   "rounds": float(np.mean([s["rounds"] for s in stats])),
                   "steps": float(np.mean([s["steps"] for s in stats])),
                   "full_per_game": float(np.mean([s["full"] for s in stats])),
                   "vp_mean": float(np.mean([np.mean(s["vp"]) for s in stats])),
                   "vp_max": float(np.mean([max(s["vp"]) for s in stats])),
                   "vp_r6_mean": float(np.mean([np.mean(s["vp_r6"]) for s in stats if s.get("vp_r6")])),
                   "vp_r6_max": float(np.mean([max(s["vp_r6"]) for s in stats if s.get("vp_r6")])),
                   "ended_by_r7": float(np.mean([s["rounds"] <= 7 for s in stats])),
                   "ended_by_r8": float(np.mean([s["rounds"] <= 8 for s in stats])),
                   **{k: round(v, 4) for k, v in tl.items()}}
            if gen % A.eval_every == 0:
                rec["eval_policy"] = run_eval(pool, os.path.join(A.out, f"net_{gen}.npz"),
                                              A.eval_games, "policy")
            if gen % A.search_eval_every == 0:
                rec["eval_search"] = run_eval(pool, os.path.join(A.out, f"net_{gen}.npz"),
                                              A.search_eval_games, "search", A.sims)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
