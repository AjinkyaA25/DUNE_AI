"""Position evaluator: each player's chance to win and expected final VP.

Starts from the AlphaZero net's trunk (src/az/net.py) and fine-tunes it with
two new heads on positions whose outcome is known:
  * win head  - softmax over the 4 seats (calibrated win probabilities; the
                AZ value head predicts win share + a VP-margin bonus and is
                overconfident on games unlike its own self-play)
  * VP head   - each seat's final VP
Inputs are encode_obs(gs, pid): only what `pid` can see, so the UI can show
a human their evaluation without leaking opponents' hidden cards.

Data: games among the UI's heuristic opponents (several specs, for variety)
plus AlphaZero self-play positions (models_az/shards, winner only).

    python -m src.az.evaluator gen   --games 15000 --workers 8
    python -m src.az.evaluator train [--net models_az/net_<g>.npz]
    python -m src.az.evaluator test  (calibration on held-out games)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Optional

import numpy as np

from src.az.encode import encode_obs

OUT = "models_eval"
SPECS = ("heuristic:tuned=config/gap/tsmf5_dp1_wb10_tm1_op1.json",
         "heuristic:tuned=config/gap/tsmf5.json",
         "heuristic:tuned=config/heuristic_phased_final.json",
         "heuristic:tuned=config/heuristic_tuned.json",
         "heuristic")
P_SAMPLE = 0.02                    # share of decisions recorded (x4 perspectives)
TEST_SEED0 = 3_000_000             # held-out games (gen uses 4_000_000+)


# ----------------------------------------------------------- Leader inputs
LEADERS_ON = bool(os.environ.get("EVAL_LEADERS"))     # gen: games with random real Leaders
_FEYD = {"start": 0, "A": 1, "B": 1, "C": 2, "D": 3, "E": 3, "F": 4, "end": 5}
PER_SEAT = 5


def _leader_names():
    from src.game.leader.leader_definitions import LEADER_NAMES
    return list(LEADER_NAMES)


def leader_dim() -> int:
    return 4 * (len(_leader_names()) + PER_SEAT) + 2


def leader_feats(gs, pid: int) -> np.ndarray:
    """Per seat (relative to pid): Leader one-hot + Leader track state; plus
    pid's own endgame VP banked in held Intrigues (only pid's hand is seen)."""
    names = _leader_names()
    k = len(names) + PER_SEAT
    out = np.zeros(leader_dim(), np.float32)
    n = gs.num_players
    for r in range(n):
        p = gs.players[(pid + r) % n]
        ld = getattr(p, "leader", None)
        if ld is None or ld.name not in names:
            continue
        b = r * k
        out[b + names.index(ld.name)] = 1.0
        st = ld.state
        s = b + len(names)
        out[s] = st.get("tactics", 2) / 10.0 if ld.name == "Chani" else 0.0
        out[s + 1] = _FEYD.get(st.get("feyd", "start"), 0) / 5.0 if ld.name.startswith("Feyd") else 0.0
        out[s + 2] = st.get("memories", 0) / 4.0 + (0.5 if st.get("flipped") else 0.0)
        out[s + 3] = (sum(getattr(c, "twisted", False) for c in p.intrigue_cards) / 3.0
                      if "twisted_deck" in st else 0.0)
        out[s + 4] = (st.get("nav_next", 0) / 4.0) if "nav" in st else 0.0
    me = gs.players[pid]
    try:
        from src.ai.agents import _endgame_chance
        ev = 0.0
        for ic in me.intrigue_cards:
            for e in ic.effects:
                if isinstance(e, dict):
                    for key, sub in e.items():
                        if key.startswith("if_bl_endgame_") and isinstance(sub, dict):
                            ev += _endgame_chance(gs, pid, key[len("if_bl_endgame_"):]) * sub.get("vp", 0)
        out[-2] = ev
    except Exception:
        pass
    out[-1] = len(me.techs) / 4.0 if getattr(me, "techs", None) else 0.0
    return out


def encode_eval(gs, pid: int, leaders: bool) -> np.ndarray:
    x = encode_obs(gs, pid)
    return np.concatenate([x, leader_feats(gs, pid)]) if leaders else x


def _rel(a, pid):
    n = len(a)
    return np.array([a[(pid + k) % n] for k in range(n)], np.float32)


# ------------------------------------------------------------------ data

def play_record(g: int, seed0: int, p_sample: float = P_SAMPLE, specs=SPECS):
    """One heuristic game -> (obs, winner one-hot, final VP, round), relative
    to each recorded perspective."""
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    rng = np.random.default_rng(seed0 + g)
    specs = [specs[int(rng.integers(len(specs)))] for _ in range(4)]
    ag = [make_agent(s, seed=g * 4 + i) for i, s in enumerate(specs)]
    if LEADERS_ON:
        import random as _r
        lineup = _r.Random(seed0 + g).sample(_leader_names(), 4)
        gs = setup_game(4, seed=seed0 + g, leaders=lineup, use_bloodlines=True)
    else:
        gs = setup_game(4, seed=seed0 + g, neutral_leaders=True, use_bloodlines=True)
    obs, meta = [], []
    n = 0
    while not gs.game_over and n < 5000:
        p = gs.player_in_reveal_buy
        pid = gs.get_current_player_id() if p is None else p
        if rng.random() < p_sample:
            for q in range(4):
                obs.append(encode_eval(gs, q, LEADERS_ON).astype(np.float16))
                meta.append((q, gs.round))
        gs.step(ag[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    if not gs.game_over:
        return None
    vp = [pl.victory_points for pl in gs.players]
    w = gs.winner if gs.winner is not None else int(np.argmax(vp))
    win = np.eye(4, dtype=np.float32)[w]
    Y = np.stack([_rel(win, q) for q, _ in meta]) if meta else np.zeros((0, 4), np.float32)
    V = np.stack([_rel(vp, q) for q, _ in meta]) if meta else np.zeros((0, 4), np.float32)
    R = np.array([r for _, r in meta], np.int8)
    return (np.stack(obs) if obs else np.zeros((0, 1), np.float16)), Y, V, R


def _gen_worker(args):
    games, seed0, path, specs = args
    parts = [r for g in games if (r := play_record(g, seed0, specs=specs)) is not None and len(r[1])]
    S = np.concatenate([p[0] for p in parts]); Y = np.concatenate([p[1] for p in parts])
    V = np.concatenate([p[2] for p in parts]); R = np.concatenate([p[3] for p in parts])
    np.savez(path, s=S, y=Y, vp=V, r=R)
    return path, len(S)


def gen(n_games: int, workers: int, seed0: int, tag: str, specs=SPECS):
    from multiprocessing import Pool
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    chunk = 25
    jobs = [(list(range(i, min(i + chunk, n_games))), seed0,
             os.path.join(OUT, "data", f"{tag}_{i:06d}.npz"), tuple(specs))
            for i in range(0, n_games, chunk)]
    jobs = [j for j in jobs if not os.path.exists(j[2])]       # resumable
    t0, tot = time.time(), 0
    with Pool(workers, maxtasksperchild=20) as pool:
        for k, (path, n) in enumerate(pool.imap_unordered(_gen_worker, jobs), 1):
            tot += n
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} chunks, {tot} positions, {time.time() - t0:.0f}s", flush=True)


def _load(tag: str):
    S, Y, V, R = [], [], [], []
    for p in sorted(glob.glob(os.path.join(OUT, "data", f"{tag}_*.npz"))):
        with np.load(p) as z:
            S.append(z["s"]); Y.append(z["y"]); V.append(z["vp"]); R.append(z["r"])
    return (np.concatenate(S), np.concatenate(Y), np.concatenate(V), np.concatenate(R))


def _az_positions(max_n: int, rng):
    """Winner targets from AlphaZero self-play shards (z: win share + margin
    bonus, relative to the recorded seat; the winner has the largest z)."""
    S, Y = [], []
    paths = sorted(glob.glob(os.path.join("models_az", "shards", "*.npz")))
    rng.shuffle(paths)
    n = 0
    for p in paths:
        try:
            with np.load(p) as z:
                for sk, zk in (("s", "z"), ("vs", "vz")):
                    s, zz = z[sk], z[zk]
                    if len(zz) and s.ndim == 2 and s.shape[1] > 1:
                        k = min(len(zz), 400)
                        idx = rng.choice(len(zz), k, replace=False)
                        S.append(s[idx]); Y.append(np.eye(4, dtype=np.float32)[zz[idx].argmax(1)])
                        n += k
        except (OSError, ValueError):            # shard being written
            continue
        if n >= max_n:
            break
    if not S:
        return np.zeros((0, 1), np.float16), np.zeros((0, 4), np.float32)
    return np.concatenate(S), np.concatenate(Y)


# ------------------------------------------------------------------ model

def init_from(net_path: str, seed: int = 0, extra: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    with np.load(net_path) as z:
        p = {k: z[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2")}
    if extra:                       # Leader inputs: new first-layer rows, start at 0
        p["W1L"] = np.zeros((extra, p["W1"].shape[1]), np.float32)
    H = p["W2"].shape[1]
    for name, out in (("w", 4), ("v", 4)):
        p[f"{name}1"] = rng.normal(0, 1 / np.sqrt(H), (H, 64)).astype(np.float32)
        p[f"{name}b1"] = np.zeros(64, np.float32)
        p[f"{name}2"] = rng.normal(0, 1 / 8, (64, out)).astype(np.float32)
        p[f"{name}b2"] = np.zeros(out, np.float32)
    p["vb2"][:] = 7.0
    p["temp"] = np.ones(1, np.float32)
    return p


class Evaluator:
    """win_probs / expected VP for every seat, from `pid`'s point of view."""

    def __init__(self, path: Optional[str] = None):
        path = path or os.path.join(OUT, "evaluator.npz")
        with np.load(path) as z:
            self.p = {k: z[k] for k in z.files if k != "meta"}
            self.meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        self.path = path
        self.leaders = "W1L" in self.p          # trained with Leader inputs

    def encode(self, gs, pid: int) -> np.ndarray:
        return encode_eval(gs, pid, self.leaders)

    def __call__(self, gs, pid: int):
        """-> (win probability per absolute seat, expected final VP per seat)."""
        return self.batch([self.encode(gs, pid)], pid, gs.num_players)[0]

    def batch(self, obs, pid: int, n: int = 4):
        p = self.p
        x = np.asarray(obs, np.float32)
        if self.leaders:
            d = p["W1"].shape[0]
            h = np.maximum(x[:, :d] @ p["W1"] + x[:, d:] @ p["W1L"] + p["b1"], 0)
        else:
            h = np.maximum(x @ p["W1"] + p["b1"], 0)
        h = np.maximum(h @ p["W2"] + p["b2"], 0)
        lg = (np.maximum(h @ p["w1"] + p["wb1"], 0) @ p["w2"] + p["wb2"]) / p["temp"][0]
        lg -= lg.max(1, keepdims=True)
        pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
        vp = np.maximum(h @ p["v1"] + p["vb1"], 0) @ p["v2"] + p["vb2"]
        out = []
        for a, b in zip(pr, vp):
            wa, va = np.zeros(n), np.zeros(n)
            for k in range(n):
                wa[(pid + k) % n], va[(pid + k) % n] = a[k], b[k]
            out.append((wa, va))
        return out


def train(net_path: str, steps: int, batch: int, lr: float, az_frac: float):
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    S, Y, V, R = _load("train")
    n_h = len(S)
    AS, AY = _az_positions(int(n_h * az_frac / max(1e-9, 1 - az_frac)), rng) if az_frac else (None, None)
    print(f"heuristic-game positions {n_h}, AlphaZero self-play positions {0 if AS is None else len(AS)}")
    TS, TY, TV, TR = _load("test")
    with np.load(net_path) as z:
        d_base = z["W1"].shape[0]
    extra = S.shape[1] - d_base
    if extra:
        az_frac, AS, AY = 0.0, None, None        # AlphaZero shards have no Leader inputs
        print(f"Leader inputs: {extra} extra features")
    p = init_from(net_path, extra=extra)
    t = {k: torch.tensor(v, device=dev, requires_grad=True) for k, v in p.items() if k != "temp"}
    trunk_keys = {"W1", "b1", "W2", "b2"}             # W1L (Leader rows) learns at full rate
    opt = torch.optim.Adam([{"params": [t[k] for k in t if k in trunk_keys], "lr": lr * 0.1},
                            {"params": [t[k] for k in t if k not in trunk_keys], "lr": lr}],
                           weight_decay=1e-4)
    g = lambda a, dt=torch.float32: torch.as_tensor(a, device=dev, dtype=dt)
    S, Y, V = g(S, torch.float16), g(Y), g(V)
    if AS is not None and len(AS):
        AS, AY = g(AS, torch.float16), g(AY)

    def heads(x):
        x = x.float()
        if extra:
            h = F.relu(x[:, :d_base] @ t["W1"] + x[:, d_base:] @ t["W1L"] + t["b1"])
        else:
            h = F.relu(x @ t["W1"] + t["b1"])
        h = F.relu(h @ t["W2"] + t["b2"])
        lg = F.relu(h @ t["w1"] + t["wb1"]) @ t["w2"] + t["wb2"]
        vp = F.relu(h @ t["v1"] + t["vb1"]) @ t["v2"] + t["vb2"]
        return lg, vp

    def test_metrics(temp=1.0):
        with torch.no_grad():
            out_l, out_v = [], []
            for i in range(0, len(TS), 8192):
                lg, vp = heads(g(TS[i:i + 8192], torch.float16))
                out_l.append(lg); out_v.append(vp)
            lg, vp = torch.cat(out_l), torch.cat(out_v)
            pr = F.softmax(lg / temp, 1)
            y = g(TY)
            return {"brier": float(((pr - y) ** 2).sum(1).mean()),
                    "nll": float(-(y * torch.log(pr + 1e-9)).sum(1).mean()),
                    "acc": float((pr.argmax(1) == y.argmax(1)).float().mean()),
                    "vp_mae": float((vp - g(TV)).abs().mean())}, lg

    print("before:", test_metrics()[0])
    t0 = time.time()
    # Positions from one game share its outcome, so the model soon starts
    # memorising games: keep the checkpoint with the best held-out NLL.
    best_nll, best_t, best_step = float("inf"), None, 0
    for step in range(1, steps + 1):
        bi = torch.randint(0, len(S), (batch,), device=dev)
        lg, vp = heads(S[bi])
        loss = F.cross_entropy(lg, Y[bi]) + 0.05 * F.smooth_l1_loss(vp, V[bi])
        if AS is not None and len(AS) and az_frac:
            ai = torch.randint(0, len(AS), (batch // 2,), device=dev)
            alg, _ = heads(AS[ai])
            loss = loss + az_frac * F.cross_entropy(alg, AY[ai])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps:
            m, _ = test_metrics()
            if m["nll"] < best_nll:
                best_nll, best_step = m["nll"], step
                best_t = {k: v.detach().clone() for k, v in t.items()}
            if step % 2000 == 0 or step == steps:
                print(f"step {step}: {m}  best step {best_step}  {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        for k in t:
            t[k].copy_(best_t[k])
    # temperature scaling on the held-out games
    _, lg = test_metrics()
    best = min(np.arange(0.6, 3.01, 0.05), key=lambda T: test_metrics(T)[0]["nll"])
    final, _ = test_metrics(best)
    out = {k: v.detach().cpu().numpy() for k, v in t.items()}
    out["temp"] = np.array([best], np.float32)
    meta = {"net": net_path, "steps": best_step, "train_positions": int(n_h), "leader_inputs": int(extra),
            "test": final, "temp": float(best),
            "trained": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "evaluator.npz"), meta=json.dumps(meta), **out)
    print("saved", os.path.join(OUT, "evaluator.npz"), meta)


def test(path: Optional[str] = None):
    S, Y, V, R = _load("test")
    ev = Evaluator(path)
    res = ev.batch(S.astype(np.float32), 0)
    P = np.stack([a for a, _ in res]); VPp = np.stack([b for _, b in res])
    print(f"held-out positions {len(S)}; temp {ev.meta.get('temp')}")
    print("round   n   brier  (uniform 0.750)  top-pick acc   VP error")
    for r in sorted(set(R.tolist())):
        m = R == r
        print(f"{r:>5} {m.sum():>5}  {((P[m] - Y[m]) ** 2).sum(1).mean():.3f}"
              f"                     {(P[m].argmax(1) == Y[m].argmax(1)).mean():.0%}"
              f"          {np.abs(VPp[m] - V[m]).mean():.2f}")
    p, y = P.ravel(), Y.ravel()
    print("calibration (predicted -> actual):")
    for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .45), (.45, .6), (.6, .8), (.8, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"  {lo:.2f}-{hi:.2f}: n={m.sum():6d} pred {p[m].mean():.2f} actual {y[m].mean():.2f}")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT, help="data + evaluator.npz folder")
    ap.add_argument("--specs", default=None, help="comma-separated agent specs for gen (default SPECS)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("gen"); a.add_argument("--games", type=int, default=15000)
    a.add_argument("--test-games", type=int, default=1000); a.add_argument("--workers", type=int, default=8)
    b = sub.add_parser("train"); b.add_argument("--net", default=None)
    b.add_argument("--steps", type=int, default=8000); b.add_argument("--batch", type=int, default=512)
    b.add_argument("--lr", type=float, default=3e-4); b.add_argument("--az-frac", type=float, default=0.3)
    sub.add_parser("test")
    A = ap.parse_args()
    OUT = A.out
    specs = A.specs.split(",") if A.specs else SPECS
    if A.cmd == "gen":
        print("test games"); gen(A.test_games, A.workers, TEST_SEED0, "test", specs)
        print("train games"); gen(A.games, A.workers, 4_000_000, "train", specs)
    elif A.cmd == "train":
        from src.az.agent import latest_net
        train(A.net or latest_net(), A.steps, A.batch, A.lr, A.az_frac)
    else:
        test(os.path.join(OUT, "evaluator.npz"))


if __name__ == "__main__":
    main()
