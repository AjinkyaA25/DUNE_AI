"""Train the PPO policy (src/ai/ppo.py) by self-play against Best AI and a League.

Each iteration:
  1. --games games on CPU workers: --ppo-seats seats play the current policy
     (sampling), the others are Best AI (heuristic) or an older AI from the
     League (--league-frac) or an earlier PPO snapshot (--self-frac, greedy).
     Every decision with 2+ moves is recorded; reward at the end = win (ties
     split) + 0.02 x VP margin (clipped to +-5). Auxiliary target (user's
     friend's net, 2026-10-19): every player's final VP (seat order from the
     decider), predicted by a 4-output head off the value trunk - far more
     signal per game than one win/loss.
  2. GAE advantages (gamma 1, lambda --lam) from the value head; PPO update
     on the GPU: clipped ratio, value loss, entropy bonus.
  3. every --eval-every iterations: the greedy policy in 1 seat vs 3 Best AI
     on a FIXED set of --eval-games seeds (paired across evaluations and with
     plain Best AI in that seat), win share + avg rounds -> log.

Dense reward (--dense w, 2026-10-20): besides the end-of-game reward, every
decision is credited with the change in w x (my VP - the opponents' average
VP) since the previous one. It is potential-based shaping with the potential
dropped to 0 at the end, so the rewards still sum to the game result (the goal
stays "win"), but a VP swing is credited to the decisions right before it
instead of only reaching them through the value head.

All moves (--k 0, 2026-10-20): the policy chooses among every legal move, not
only the heuristic's top 16 (10% of decisions, mostly Agent turns, have more).
Batches are stored ragged (one row per legal move) and padded per minibatch.

Self-play (2026-10-20, once the policy plateaued vs Best AI): --cur-frac of the
opponent seats play the CURRENT policy (sampling, and trained on too), --pool
adds frozen older PPO nets to the snapshot opponents, and --eval-opp /
--eval-base measure progress against a frozen PPO instead of Best AI.

    python -m src.selfplay.ppo_train --out models_ppo [--iters 0] [--games 400] [--workers 16]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import time
from multiprocessing import Pool

import numpy as np

from src.ai.ppo import EMB, ID_FIELDS, id_sizes
from src.selfplay import elo

BASE = os.path.join("config", "ppo", "base.json")        # heuristic the policy sits on (pinned)


# ---------------------------------------------------------------------------
# network (PyTorch mirror of src.ai.ppo.PPONet)
# ---------------------------------------------------------------------------
def make_torch_net(S, A, sizes, T, w=(256, 128, 128, EMB)):
    """w = widths (state layer 1, state layer 2, move layer, embedding size)."""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.w, self.A = w, A
            self.s1 = nn.Linear(S, w[0])
            self.s2 = nn.Linear(w[0], w[1])
            self.v = nn.Linear(w[1], 1)
            self.aux = nn.Linear(w[1], 4)            # final VP / 10 of me, next, next+1, next+2
            self.emb = nn.ModuleList([nn.Embedding(n, w[3]) for n in sizes])
            for e in self.emb:
                nn.init.normal_(e.weight, std=0.05)
            self.a1 = nn.Linear(w[1] + A + w[3] * len(sizes), w[2])
            self.a2 = nn.Linear(w[2], 1)
            nn.init.zeros_(self.a2.weight)            # correction starts at exactly 0
            nn.init.zeros_(self.a2.bias)
            self.T = T
            self.K = 16                              # shortlist size played with (0 = all moves)

        def forward(self, s, D, I, H, M):
            hs = torch.relu(self.s2(torch.relu(self.s1(s))))
            v = self.v(hs).squeeze(-1)
            e = torch.cat([emb(I[..., j]) for j, emb in enumerate(self.emb)], dim=-1)
            x = torch.cat([hs[:, None, :].expand(-1, D.shape[1], -1), D, e], dim=-1)
            c = self.a2(torch.relu(self.a1(x))).squeeze(-1)
            logits = H / self.T + c
            logits = logits.masked_fill(~M, -1e9)
            self.last_aux = self.aux(hs)
            return logits, v, c

        def export(self, path):
            p = {"s1.w": self.s1.weight, "s1.b": self.s1.bias, "s2.w": self.s2.weight, "s2.b": self.s2.bias,
                 "v.w": self.v.weight[0], "v.b": self.v.bias[0],
                 "a1.w": self.a1.weight, "a1.b": self.a1.bias, "a2.w": self.a2.weight[0], "a2.b": self.a2.bias[0]}
            for j, e in enumerate(self.emb):
                p[f"e{j}"] = e.weight
            out = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in p.items()}
            out["T"] = np.array(self.T, dtype=np.float32)
            out["K"] = np.array(self.K, dtype=np.int32)
            tmp = path + ".tmp.npz"
            np.savez(tmp, **out)
            os.replace(tmp, path)

    return Net()


def widen(net, old: dict) -> None:
    """Load a smaller network's weights into a wider `net` so it plays EXACTLY the
    same: old units keep their weights, new units get a fresh random input side
    and a zero output side (so they start silent and grow with training)."""
    import torch
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    w1, w2, w3, e = old["s1.weight"].shape[0], old["s2.weight"].shape[0], old["a1.weight"].shape[0],         old["emb.0.weight"].shape[1]
    W2, E, A, nf = net.w[1], net.w[3], net.A, len(net.emb)
    sd["s1.weight"][:w1] = old["s1.weight"]; sd["s1.bias"][:w1] = old["s1.bias"]
    sd["s2.weight"][:w2].zero_(); sd["s2.weight"][:w2, :w1] = old["s2.weight"]; sd["s2.bias"][:w2] = old["s2.bias"]
    for k in ("v", "aux"):
        sd[k + ".weight"].zero_(); sd[k + ".weight"][:, :w2] = old[k + ".weight"]; sd[k + ".bias"] = old[k + ".bias"]
    for j in range(nf):
        sd[f"emb.{j}.weight"][:, :e] = old[f"emb.{j}.weight"]
    a = sd["a1.weight"]
    a[:w3].zero_()                                   # old move units: zero weight on every new input
    o = old["a1.weight"]
    a[:w3, :w2] = o[:, :w2]
    a[:w3, W2:W2 + A] = o[:, w2:w2 + A]
    for j in range(nf):
        a[:w3, W2 + A + j * E:W2 + A + j * E + e] = o[:, w2 + A + j * e:w2 + A + (j + 1) * e]
    sd["a1.bias"][:w3] = old["a1.bias"]
    sd["a2.weight"].zero_(); sd["a2.weight"][:, :w3] = old["a2.weight"]; sd["a2.bias"] = old["a2.bias"]
    net.load_state_dict(sd)


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------
def _reward(gs, pid) -> float:
    vp = [q.victory_points for q in gs.players]
    if gs.winner is not None:
        win = float(gs.winner == pid)
    else:
        top = max(vp)
        tied = [i for i, v in enumerate(vp) if v == top]
        win = 1.0 / len(tied) if pid in tied else 0.0
    lead = vp[pid] - max(v for i, v in enumerate(vp) if i != pid)
    return win + 0.02 * max(-5, min(5, lead))


def rollout(args):
    """One training game -> list of (trajectory, reward) for the PPO seats."""
    g, net, ppo_seats, league, league_frac, snaps, self_frac, cur_frac = args
    from src.ai.agents import make_agent
    from src.ai.ppo import PPOAgent
    from src.data.card_definitions import setup_game
    from src.game.leader.leader_definitions import LEADER_NAMES
    r = random.Random(g)
    lineup = r.sample(list(LEADER_NAMES), 4)
    gs = setup_game(num_players=4, seed=g, leaders=lineup, use_bloodlines=True)
    seats = r.sample(range(4), ppo_seats)
    for s in range(4):                       # self-play: opponent seats on the current policy
        if cur_frac and s not in seats and r.random() < cur_frac:
            seats.append(s)
    agents = []
    base = json.load(open(BASE))
    for s in range(4):
        if s in seats:
            a = PPOAgent(net=net, tuning=base, seed=g * 4 + s)
            a.record = []
        else:
            x = r.random()
            if snaps and x < self_frac:
                a = PPOAgent(net=r.choice(snaps), tuning=base, seed=g * 4 + s, greedy=True)
            elif league and x < self_frac + league_frac:
                a = make_agent(f"heuristic:tuned={r.choice(league)}", seed=g * 4 + s)
            else:
                a = make_agent(f"heuristic:tuned={BASE}", seed=g * 4 + s)
        agents.append(a)
    n = 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    out = []
    vp = [q.victory_points for q in gs.players]
    for s in seats:
        rot = [vp[(s + j) % 4] / 10.0 for j in range(4)]
        out.append(([d for d in agents[s].record if d["pid"] == s], _reward(gs, s), rot))
    return out, gs.round


def eval_game(args):
    """Greedy policy (or plain heuristic) in one seat vs 3 Best AI (or 3 x opp)."""
    g, spec, opp = (args + (None,))[:3]
    opp = opp or f"heuristic:tuned={BASE}"
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.game.leader.leader_definitions import LEADER_NAMES
    r = random.Random(5_000_000 + g)
    lineup = r.sample(list(LEADER_NAMES), 4)
    gs = setup_game(num_players=4, seed=5_000_000 + g, leaders=lineup, use_bloodlines=True)
    seat = g % 4
    agents = [make_agent(spec if s == seat else opp, seed=g * 4 + s) for s in range(4)]
    n = 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    vp = [q.victory_points for q in gs.players]
    win = gs.winner if gs.winner is not None else max(range(4), key=lambda s: vp[s])
    return float(win == seat), gs.round


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def to_batch(trajs, device):
    """Ragged batch: the moves of every decision back to back (D, I, H), with
    each decision's offset and move count; rows() pads a minibatch."""
    import torch
    recs = [d for t, _, _ in trajs for d in t]
    aux = np.array([vp for t, _, vp in trajs for _ in t], np.float32)
    N = len(recs)
    S = np.stack([d["s"] for d in recs]).astype(np.float32)
    cnt = np.array([len(d["h"]) for d in recs], np.int64)
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
    D = np.concatenate([d["D"] for d in recs]).astype(np.float32)
    I = np.concatenate([d["I"] for d in recs]).astype(np.int64)
    H = np.concatenate([d["h"] for d in recs]).astype(np.float32)
    act = np.array([d["k"] for d in recs], np.int64)
    oldlp = np.array([d["logp"] for d in recs], np.float32)
    t = lambda x: torch.from_numpy(x).to(device)
    return dict(S=t(S), D=t(D), I=t(I), H=t(H), cnt=t(cnt), off=t(off), act=t(act), oldlp=t(oldlp),
                aux=t(aux)), N


def rows(B, idx):
    """Decisions idx of a ragged batch -> padded (S, D, I, H, M), padded to the
    largest move count among them."""
    import torch
    cnt, off = B["cnt"][idx], B["off"][idx]
    k = int(cnt.max())
    ar = torch.arange(k, device=cnt.device)
    M = ar[None, :] < cnt[:, None]
    pos = torch.where(M, off[:, None] + ar[None, :], torch.zeros_like(off)[:, None])
    return B["S"][idx], B["D"][pos], B["I"][pos], B["H"][pos], M


def potentials(t, dense):
    """dense x (my VP - opponents' average VP) at each decision of a trajectory."""
    if not dense:
        return np.zeros(len(t), np.float32)
    return np.array([dense * (d["vp"][0] - sum(d["vp"][1:]) / max(1, len(d["vp"]) - 1)) for d in t],
                    np.float32)


def advantages(trajs, values, lam, dense=0.0):
    """GAE with gamma 1 over each seat's decisions. Reward: the game result at
    the end, plus (dense > 0) the potential change after every decision, the
    potential being 0 at the end (so the total is still the game result)."""
    adv, ret, i = [], [], 0
    for t, R, _ in trajs:
        n = len(t)
        v = values[i:i + n]
        i += n
        phi = potentials(t, dense)
        r = np.append(phi[1:], 0.0) - phi           # shaping: phi(next) - phi(now), phi(end) = 0
        r[-1] += R
        a = np.zeros(n, np.float32)
        last = 0.0
        for j in range(n - 1, -1, -1):
            nxt = v[j + 1] if j + 1 < n else 0.0
            delta = r[j] + nxt - v[j]
            last = delta + lam * last
            a[j] = last
        adv.append(a)
        ret.append(a + v)
    return np.concatenate(adv), np.concatenate(ret)


def ppo_update(net, opt, B, adv, ret, A):
    import torch
    dev = B["S"].device
    adv_t = torch.from_numpy(adv).to(dev)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.from_numpy(ret).to(dev)
    N = adv_t.shape[0]
    stats = {"pi": 0.0, "v": 0.0, "aux": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0, "n": 0}
    for _ in range(A.epochs):
        perm = torch.randperm(N, device=dev)
        for s in range(0, N, A.batch):
            idx = perm[s:s + A.batch]
            bs = rows(B, idx)
            logits, v, _ = net(*bs)
            lp_all = torch.log_softmax(logits, dim=-1)
            lp = lp_all.gather(1, B["act"][idx][:, None]).squeeze(1)
            ratio = torch.exp(lp - B["oldlp"][idx])
            a = adv_t[idx]
            pi_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - A.clip, 1 + A.clip) * a).mean()
            v_loss = ((v - ret_t[idx]) ** 2).mean()
            aux_loss = ((net.last_aux - B["aux"][idx]) ** 2).mean()
            p_all = torch.exp(lp_all)
            ent = -(p_all * lp_all.masked_fill(~bs[4], 0.0)).sum(-1).mean()
            loss = pi_loss + A.vf * v_loss + A.aux * aux_loss - A.ent * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                stats["pi"] += float(pi_loss); stats["v"] += float(v_loss); stats["ent"] += float(ent)
                stats["aux"] += float(aux_loss)
                stats["kl"] += float((B["oldlp"][idx] - lp).mean())
                stats["clip"] += float(((ratio - 1).abs() > A.clip).float().mean())
                stats["n"] += 1
    return {k: round(v / max(1, stats["n"]), 5) for k, v in stats.items() if k != "n"}


# ---------------------------------------------------------------------------
def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models_ppo")
    ap.add_argument("--iters", type=int, default=0, help="0 = until stopped")
    ap.add_argument("--games", type=int, default=400, help="training games per iteration")
    ap.add_argument("--ppo-seats", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--T", type=float, default=1.5, help="heuristic points per logit unit")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--aux", type=float, default=0.25, help="weight of the final-VP prediction loss")
    ap.add_argument("--league", default="config/best_ai_history/*.json")
    ap.add_argument("--league-frac", type=float, default=0.2)
    ap.add_argument("--self-frac", type=float, default=0.2)
    ap.add_argument("--cur-frac", type=float, default=0.0,
                    help="share of opponent seats played (and trained) by the current policy")
    ap.add_argument("--pool", default="", help="glob of frozen PPO nets added to the snapshot opponents")
    ap.add_argument("--eval-opp", default="", help="eval opponents' spec (default Best AI)")
    ap.add_argument("--eval-base", default="", help="eval baseline's spec in the eval seat (default Best AI)")
    ap.add_argument("--widths", default="256,128,128,16",
                    help="state layer 1, state layer 2, move layer, embedding size; a checkpoint of a smaller "
                         "net is widened (same play, new units start silent); --init <checkpoint.pt> starts from one")
    ap.add_argument("--init", default="", help="checkpoint.pt to start from when --out has none")
    ap.add_argument("--k", type=int, default=16,
                    help="moves the policy chooses among: the heuristic's top k (0 = every legal move)")
    ap.add_argument("--dense", type=float, default=0.0,
                    help="dense reward: credit per VP of (my VP - opponents' average) as it happens")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=2000)
    ap.add_argument("--eval-secs", type=float, default=0,
                    help="> 0: run the eval every this many seconds instead of every --eval-every "
                         "iterations (which then only sets how often a self-play snapshot is saved)")
    ap.add_argument("--self-eval-games", type=int, default=0,
                    help="> 0: each eval also plays the policy in 1 seat vs 3 copies of the previous eval's "
                         "snapshot (25%% = no progress since then; plateau check once Best AI is beaten)")
    ap.add_argument("--snap-pool", type=int, default=5, help="newest snapshots the opponents are drawn from")
    ap.add_argument("--elo-games", type=int, default=0,
                    help="> 0: each eval's snapshot also plays this many 4-player games vs the rated field (Best AI, "
                         "--pool nets, earlier snapshots) and all rated games are refit into Elo (src.selfplay.elo)")
    ap.add_argument("--base", default=os.path.join("config", "best_ai.json"),
                    help="heuristic weights the policy starts from (copied to config/ppo/base.json once)")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    os.makedirs(os.path.dirname(BASE), exist_ok=True)
    if not os.path.exists(BASE):
        shutil.copy(A.base, BASE)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from src.ai.features import FEATURE_DIM
    from src.ai.action_features import ACTION_FEATURE_DIM
    widths = tuple(int(x) for x in A.widths.split(","))
    net = make_torch_net(FEATURE_DIM, ACTION_FEATURE_DIM, id_sizes(), A.T, widths).to(dev)
    net.K = A.k
    opt = torch.optim.Adam(net.parameters(), lr=A.lr)
    ck = os.path.join(A.out, "checkpoint.pt")
    state = {"iter": 0, "games": 0}
    src = ck if os.path.exists(ck) else A.init
    c = torch.load(src, map_location=dev) if src else None
    if c and any(v.shape != net.state_dict()[k].shape for k, v in c["net"].items() if k in net.state_dict()):
        widen(net, c["net"])
        state = c["state"]
        print(f"widened {src} to {widths} at iteration {state['iter']} ({state['games']} games)", flush=True)
    elif c:
        missing = net.load_state_dict(c["net"], strict=False).missing_keys
        try:
            opt.load_state_dict(c["opt"])
        except ValueError:                       # new head added: fresh optimiser state
            pass
        state = c["state"]
        if missing:
            print(f"new layers (not in the checkpoint): {missing}", flush=True)
        print(f"resumed at iteration {state['iter']} ({state['games']} games)", flush=True)
    cur = os.path.join(A.out, "net.npz")
    net.export(cur)
    league = sorted(glob.glob(A.league)) if A.league else []
    pool_nets = sorted(glob.glob(A.pool)) if A.pool else []
    log = open(os.path.join(A.out, "log.jsonl"), "a")
    with Pool(A.workers, maxtasksperchild=50) as pool:
        base_eval = None
        bpath = os.path.join(A.out, "eval_baseline.json")
        if os.path.exists(bpath):
            base_eval = json.load(open(bpath))
        if (A.eval_every or A.eval_secs) and base_eval is None:
            res = pool.map(eval_game, [(g, A.eval_base or f"heuristic:tuned={BASE}", A.eval_opp)
                                       for g in range(A.eval_games)], chunksize=8)
            base_eval = {"share": float(np.mean([w for w, _ in res])), "wins": [w for w, _ in res],
                         "rounds": float(np.mean([r for _, r in res]))}
            json.dump(base_eval, open(bpath, "w"))
            print(f"baseline: Best AI in the eval seat {base_eval['share']:.1%} | rounds {base_eval['rounds']:.2f}",
                  flush=True)
        elo_field = {"pool_" + os.path.basename(p)[:-4]: f"ppo:net={p}:tuned={BASE}:greedy" for p in pool_nets}
        if A.elo_games and not os.path.exists(os.path.join(A.out, "elo.json")):
            rt = elo.rate_snapshot(pool, A.out, None, None, A.elo_games, elo_field, f"heuristic:tuned={BASE}")
            print("elo field: " + ", ".join(f"{k} {v[0]:.0f}" for k, v in sorted(rt.items())), flush=True)
        t_eval, g_eval = time.time(), state["games"]
        while A.iters == 0 or state["iter"] < A.iters:
            state["iter"] += 1
            it = state["iter"]
            t0 = time.time()
            snaps = sorted(glob.glob(os.path.join(A.out, "snap_*.npz")))[-A.snap_pool:] + pool_nets
            jobs = [(100_000_000 + state["games"] + g, cur, A.ppo_seats, league, A.league_frac, snaps, A.self_frac,
                     A.cur_frac)
                    for g in range(A.games)]
            trajs, rounds = [], []
            for out, rnd in pool.imap_unordered(rollout, jobs, chunksize=2):
                trajs += [x for x in out if x[0]]
                rounds.append(rnd)
            state["games"] += A.games
            t1 = time.time()
            B, N = to_batch(trajs, dev)
            with torch.no_grad():
                vals = []
                for s in range(0, N, 16384):
                    _, v, _ = net(*rows(B, torch.arange(s, min(N, s + 16384), device=dev)))
                    vals.append(v.cpu().numpy())
            adv, ret = advantages(trajs, np.concatenate(vals), A.lam, A.dense)
            st = ppo_update(net, opt, B, adv, ret, A)
            net.export(cur)
            # how often the trained policy now overrules the heuristic's top move
            with torch.no_grad():
                idx = torch.randperm(N, device=dev)[:8192]
                bs = rows(B, idx)
                lg, _, _ = net(*bs)
                h_top = bs[3].masked_fill(~bs[4], -1e9).argmax(-1)
                overrule = float((lg.argmax(-1) != h_top).float().mean())
            rec = {"iter": it, "games": state["games"], "decisions": N,
                   "reward": round(float(np.mean([R for _, R, _ in trajs])), 4),
                   "win": round(float(np.mean([R > 0.5 for _, R, _ in trajs])), 4),
                   "rounds": round(float(np.mean(rounds)), 3), "overrule": round(overrule, 4),
                   "rollout_s": round(t1 - t0, 1), "update_s": round(time.time() - t1, 1), **st}
            snap = os.path.join(A.out, f"snap_{it:05d}.npz")
            if A.eval_every and it % A.eval_every == 0:
                shutil.copy(cur, snap)                # self-play opponent
            if A.eval_secs:
                do_eval = time.time() - t_eval >= A.eval_secs
            else:
                do_eval = bool(A.eval_every) and it % A.eval_every == 0
            if do_eval:
                if not os.path.exists(snap):
                    shutil.copy(cur, snap)
                if A.eval_secs:                       # throughput since the last eval (incl. that eval)
                    rec["games_per_hour"] = round((state["games"] - g_eval) * 3600 / (time.time() - t_eval))
                    t_eval, g_eval = time.time(), state["games"]
                res = pool.map(eval_game, [(g, f"ppo:net={snap}:tuned={BASE}:greedy", A.eval_opp)
                                           for g in range(A.eval_games)], chunksize=8)
                w = np.array([x for x, _ in res])
                d = w - np.array(base_eval["wins"])
                rec["eval"] = {"share": round(float(w.mean()), 4), "baseline": round(base_eval["share"], 4),
                               "diff": round(float(d.mean()), 4),
                               "se": round(float(d.std() / math.sqrt(len(d))), 4),
                               "rounds": round(float(np.mean([r for _, r in res])), 3)}
                prev = state.get("eval_snap")
                if A.self_eval_games and prev and os.path.exists(prev):
                    opp = f"ppo:net={prev}:tuned={BASE}:greedy"
                    res = pool.map(eval_game, [(g, f"ppo:net={snap}:tuned={BASE}:greedy", opp)
                                               for g in range(A.self_eval_games)], chunksize=8)
                    w = np.array([x for x, _ in res])
                    rec["eval_vs_prev"] = {"prev": os.path.basename(prev), "share": round(float(w.mean()), 4),
                                           "se": round(float(w.std() / math.sqrt(len(w))), 4)}
                state["eval_snap"] = snap
                if A.elo_games:
                    name = f"iter{it:05d}"
                    rt = elo.rate_snapshot(pool, A.out, name, f"ppo:net={snap}:tuned={BASE}:greedy", A.elo_games,
                                           elo_field, f"heuristic:tuned={BASE}")
                    rec["elo"] = dict(zip(("rating", "se", "games"), rt[name]))
                best =os.path.join(A.out, "best.json")
                b = json.load(open(best)) if os.path.exists(best) else {"diff": -1}
                if rec["eval"]["diff"] > b["diff"]:
                    shutil.copy(snap, os.path.join(A.out, "best.npz"))
                    json.dump({"iter": it, **rec["eval"]}, open(best, "w"))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "state": state}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)   # atomic: a kill mid-save never leaves a broken checkpoint


if __name__ == "__main__":
    main()
