"""Low-noise evolution (user, 2026-10-17: "keep trying different weights").

The champion-vs-mutant loop (evolve.py) picks the best of 3 mutants on 300
games each: ~2.5 points of luck per candidate, more than one weight change is
worth, so it mostly selected luck (round 3: 27.5% -> 23% vs the fixed AI).

Here every candidate plays 1 seat against 3 copies of a FIXED opponent (the
previous AI), many games each with the fast plain heuristic. The champion's
estimate keeps accumulating over its games, and a mutant replaces it only if
it is ahead by --z standard errors. Every --bench-every generations the
champion is checked with the lookahead agent (the one that actually plays).

    python -m src.selfplay.evolve_fixed --out config/evolve_fixed --start <json> --opp <json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from multiprocessing import Pool

import numpy as np

from src.selfplay.evolve import (mutate, mutate_leader, load_human_positions, human_agreement,
                                 leader_play, play)
import src.selfplay.evolve as E


class Est:
    """Running mean / variance of a candidate's per-game fitness."""

    def __init__(self, d=None):
        d = d or {}
        self.n, self.s, self.ss, self.w = d.get("n", 0), d.get("s", 0.0), d.get("ss", 0.0), d.get("w", 0)

    def add(self, xs, wins):
        self.n += len(xs); self.s += sum(xs); self.ss += sum(x * x for x in xs); self.w += wins

    @property
    def mean(self):
        return self.s / self.n if self.n else 0.0

    @property
    def var(self):
        return max(1e-9, self.ss / self.n - self.mean ** 2) if self.n else 1.0

    def se2(self):
        return self.var / max(1, self.n)

    def to(self):
        return {"n": self.n, "s": self.s, "ss": self.ss, "w": self.w}


def _fit(win, rnd, speed):
    return win * (1 + speed * max(0, 10 - rnd))


def eval_vs_fixed(pool, cand, opp, n, base, speed, agent):
    """Candidate in 1 seat (rotating), the fixed opponent in the other 3."""
    specs = [f"{agent}:tuned={cand}"] + [f"{agent}:tuned={opp}"] * 3
    xs, wins, rounds = [], 0, []
    for _, rnd, out in pool.imap_unordered(play, [(base + g, specs, True) for g in range(n)], chunksize=8):
        w = out[0]["win"]
        xs.append(_fit(w, rnd, speed)); wins += w; rounds.append(rnd)
    return xs, wins, rounds


def eval_leader(pool, leader, cand, opp, n, base, speed, agent):
    """Candidate plays `leader` (rotating seat), the fixed opponent the rest."""
    jobs = [(base + g, leader, f"{agent}:tuned={cand}", f"{agent}:tuned={opp}") for g in range(n)]
    xs, wins, rounds = [], 0, []
    for r in pool.imap_unordered(leader_play, jobs, chunksize=8):
        xs.append(_fit(r["win"], r["rounds"], speed)); wins += r["win"]; rounds.append(r["rounds"])
    return xs, wins, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--opp", required=True, help="fixed opponent weights (the previous AI)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--games", type=int, default=4000, help="games per candidate")
    ap.add_argument("--champ-games", type=int, default=8000, help="champion's first estimate")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--max-traits", type=int, default=3)
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--agent", default="heuristic")
    ap.add_argument("--leader-every", type=int, default=3)
    ap.add_argument("--leader-games", type=int, default=3000)
    ap.add_argument("--human", type=float, default=0.5)
    ap.add_argument("--human-games", default=os.path.join("data", "ui_games", "*.pkl"))
    ap.add_argument("--bench-agent", default="look:D2:K3:M1:ev=models_eval_leaders/evaluator.npz")
    ap.add_argument("--bench-every", type=int, default=15)
    ap.add_argument("--bench-games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--knobs", default="", help="comma list: mutate only these knobs (and their "
                    "early./mid./late. versions in --start); Leader generations are skipped")
    A = ap.parse_args()
    if A.knobs:
        E.MUTABLE = tuple(k.strip() for k in A.knobs.split(",") if k.strip())
        A.leader_every = 10 ** 9
    E.LEADERS = True
    os.makedirs(A.out, exist_ok=True)
    log_path, champ_path = os.path.join(A.out, "log.jsonl"), os.path.join(A.out, "champion.json")
    state_path, bench_path = os.path.join(A.out, "state.json"), os.path.join(A.out, "bench.jsonl")
    opp = os.path.join(A.out, "opp.json")
    if not os.path.exists(opp):
        json.dump(json.load(open(A.opp)), open(opp, "w"), indent=1)
    start_gen, st = 0, {}
    if os.path.exists(champ_path) and os.path.exists(state_path):
        st = json.load(open(state_path))
        start_gen = st["gen"]
        print(f"resuming at generation {start_gen}", flush=True)
    else:
        json.dump(json.load(open(A.start)), open(champ_path, "w"), indent=1)
    champion = json.load(open(champ_path))
    est = Est(st.get("champ"))
    lest = {k: Est(v) for k, v in (st.get("leaders") or {}).items()}   # champion per Leader seat
    human_pos = load_human_positions(A.human_games) if A.human else []
    agree_c = human_agreement(champ_path, human_pos) if human_pos else 0.0
    rng = random.Random(A.seed + start_gen)
    from src.game.leader.leader_definitions import LEADER_NAMES
    order = list(LEADER_NAMES)
    rng.shuffle(order)

    def save(gen):
        json.dump({"gen": gen, "champ": est.to(), "leaders": {k: v.to() for k, v in lest.items()}},
                  open(state_path, "w"))

    def bench(gen):
        specs = [f"{A.bench_agent}:tuned={champ_path}"] + [f"{A.bench_agent}:tuned={opp}"] * 3
        wins, fit, vps, r6s, rounds = E.run_games(pool, specs, A.bench_games, 91_000_000 + gen * 10_000, A.speed)
        rec = {"gen": gen, "games": A.bench_games, "share": round(wins[0] / A.bench_games, 4),
               "vp": round(vps[0] / A.bench_games, 3),
               "opp_vp": round(sum(vps[i] for i in (1, 2, 3)) / (3 * A.bench_games), 3),
               "rounds": round(float(np.mean(rounds)), 3),
               "ended_by_r8": round(float(np.mean([r <= 8 for r in rounds])), 4)}
        open(bench_path, "a").write(json.dumps(rec) + "\n")
        print(f"bench gen {gen} (lookahead): champion {rec['share']:.1%} vs previous AI | "
              f"rounds {rec['rounds']} | ended by R8 {rec['ended_by_r8']:.1%}", flush=True)

    with Pool(A.workers, maxtasksperchild=200) as pool:
        if est.n == 0:
            xs, w, rd = eval_vs_fixed(pool, champ_path, opp, A.champ_games, 0, A.speed, A.agent)
            est.add(xs, w)
            print(f"champion vs previous AI: {w / len(xs):.1%} (fit {est.mean:.3f}, "
                  f"{len(xs)} games, rounds {np.mean(rd):.2f})", flush=True)
            save(start_gen)
        if start_gen == 0 and A.bench_every and not os.path.exists(bench_path):
            bench(0)
        for gen in range(start_gen + 1, A.gens + 1):
            leader = None
            if A.leader_every and gen % A.leader_every == 0:
                leader = order[(gen // A.leader_every - 1) % len(order)]
                muts = [mutate_leader(champion, leader, rng, A.sigma) for _ in range(3)]
                if leader not in lest:
                    xs, w, _ = eval_leader(pool, leader, champ_path, opp, A.leader_games,
                                           gen * 1_000_000 + 900_000, A.speed, A.agent)
                    lest[leader] = Est(); lest[leader].add(xs, w)
                ref = lest[leader]
            else:
                muts = [mutate(champion, rng, A.sigma, A.max_traits) for _ in range(3)]
                ref = est
            res = []
            for k, (w_, traits) in enumerate(muts, 1):
                p = os.path.join(A.out, f"gen{gen}_m{k}.json")
                json.dump(w_, open(p, "w"), indent=1)
                base = gen * 1_000_000 + k * 100_000
                if leader:
                    xs, w, rd = eval_leader(pool, leader, p, opp, A.leader_games, base, A.speed, A.agent)
                else:
                    xs, w, rd = eval_vs_fixed(pool, p, opp, A.games, base, A.speed, A.agent)
                e = Est(); e.add(xs, w)
                agree = human_agreement(p, human_pos) if (human_pos and not leader) else agree_c
                score = e.mean + A.human * (agree - agree_c)
                zv = (score - ref.mean) / math.sqrt(e.se2() + ref.se2())
                res.append({"k": k, "traits": traits, "share": round(w / len(xs), 4), "fit": round(e.mean, 4),
                            "score": round(score, 4), "z": round(zv, 2), "rounds": round(float(np.mean(rd)), 3),
                            "agree": round(agree, 4), "est": e, "w": w_, "path": p})
            best = max(res, key=lambda r: r["z"])
            changed = best["z"] > A.z
            rec = {"gen": gen, "leader": leader, "champ_fit": round(ref.mean, 4), "champ_n": ref.n,
                   "champ_share": round(ref.w / max(1, ref.n), 4),
                   "mutants": [{k: v for k, v in r.items() if k not in ("est", "w", "path")} for r in res],
                   "new_champion": best["k"] if changed else 0}
            if changed:
                champion = best["w"]
                json.dump(champion, open(champ_path, "w"), indent=1)
                rec["changes"] = {t: champion.get(t) for t in best["traits"]}
                # re-measure on fresh seeds: the best of 3 is luckier than its true level
                fresh = Est()
                if leader:
                    xs, w, _ = eval_leader(pool, leader, champ_path, opp, A.leader_games,
                                           gen * 1_000_000 + 500_000, A.speed, A.agent)
                    fresh.add(xs, w)
                    lest[leader] = fresh
                else:
                    xs, w, _ = eval_vs_fixed(pool, champ_path, opp, A.games, gen * 1_000_000 + 500_000,
                                             A.speed, A.agent)
                    fresh.add(xs, w)
                    est = fresh
                    agree_c = best["agree"]
                    lest = {}                       # Leader seats re-measured with the new weights
                rec["fresh_share"] = round(fresh.w / fresh.n, 4)
            open(log_path, "a").write(json.dumps(rec) + "\n")
            tag = f" [{leader}]" if leader else ""
            print(f"gen {gen}{tag}: champion {rec['champ_share']:.1%} (fit {ref.mean:.3f}, n {ref.n}) | " +
                  " ".join(f"{r['share']:.1%} z{r['z']:+.1f}" for r in res) +
                  (f" -> mutant {best['k']} wins {rec['changes']}" if changed else " -> champion stays") +
                  f" | rounds {np.mean([r['rounds'] for r in res]):.2f}", flush=True)
            save(gen)
            if A.bench_every and gen % A.bench_every == 0:
                bench(gen)


if __name__ == "__main__":
    main()
