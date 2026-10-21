"""Evolve the heuristic's weights the way the user's friend's legacy_policy.py
does: each generation the champion plays three mutants (1-3 weights each
nudged by exp(N(0, sigma)), all others unchanged) in 4-player games with
rotated seats; the best mutant replaces the champion if it wins more.

    python -m src.selfplay.evolve [--start config/gap/tsmf5_dp1_wb10_tm1_op1.json]
                                  [--gens 60] [--games 400] [--workers 12]

Writes config/evolve/: champion.json (current best), gen<N>_m<k>.json, log.jsonl,
progress.html (rebuilt every generation). Every --bench-every generations the
champion also plays 3 copies of the starting weights (bench.jsonl) to measure
real progress and game pace. Resumable: rerunning continues from the last
logged generation.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
from multiprocessing import Pool

import numpy as np

OUT = os.path.join("config", "evolve")
# continuous weights only (on/off switches and model selectors are not mutated)
MUTABLE = ("res_contract", "res_spy", "res_intrigue", "res_draw", "res_solari", "res_spice",
           "res_water", "res_troops", "sm_solari", "combat", "influence", "ptw_tax", "troop_hold",
           "ci_hold", "plot_hold", "deck_k", "contracts", "skills", "card_play", "buy_mult",
           "tsmf", "alliance", "worm_bonus", "style", "deck_plan",
           "look_w",                               # look_w: lookahead agents only
           "spy_post", "spy_multi", "spy_combat", "deploy_hold", "intrigue_love", "tech_love",
           "maker_bonus", "contract_stack", "leader_play", "out_of_rewards",
           "hc_rush", "starter_trash",
           # TSMF / persuasion plan + learned Conflict forecast (2026-10-18)
           "smf_plan", "tsmf_goal", "pers_w", "buy_plan", "fight_learned")
# knobs at 0 can be switched on by a mutation (user, 2026-10-17)


def effective(weights: dict) -> dict:
    from src.ai.agents import TUNE_DEFAULTS
    return {**TUNE_DEFAULTS, **weights}


# on/off switches and selectors: never "activated" by a random mutation
SWITCHES = frozenset(("fight_model", "deck_model", "trash_model", "cmd_choice", "opp_pool",
                      "fight_calib", "trash_dagger", "look_w"))
# exploration (user, 2026-10-19: "if there are any dead nodes, make sure to try
# to activate them occasionally"): DEAD_P = chance a mutant also switches on
# one knob that is off and that evolution never touches; CROSS_P = chance a
# mutant instead copies 1-3 knob values from an older AI in LEAGUE ("if there
# is something to be learned that may have been missed, pick up on that")
DEAD_P, CROSS_P, LEAGUE = 0.0, 0.0, []


def dead_knobs(weights: dict) -> list:
    """Numeric knobs that are off (0) for these weights and outside MUTABLE."""
    from src.ai.agents import TUNE_DEFAULTS
    eff = effective(weights)
    return sorted(k for k, v in TUNE_DEFAULTS.items()
                  if k not in SWITCHES and k not in MUTABLE and isinstance(v, (int, float))
                  and not (eff.get(k) or 0))


def crossover(weights: dict, rng: random.Random, max_traits: int = 3):
    """Copy 1-3 knob values (incl. early./mid./late. and L. keys) from a League AI."""
    if not LEAGUE:
        return None
    src = rng.choice(LEAGUE)
    a, b = effective(weights), effective(json.load(open(src)))
    diff = sorted(k for k in set(a) | set(b) if k not in SWITCHES and a.get(k) != b.get(k)
                  and isinstance(b.get(k, 0), (int, float)) and isinstance(a.get(k, 0), (int, float)))
    if not diff:
        return None
    child = dict(weights)
    traits = rng.sample(diff, rng.randint(1, min(max_traits, len(diff))))
    for k in traits:
        child[k] = b.get(k, 0)
    return child, [f"{t}<-{os.path.basename(src)}" for t in traits]


def mutate(weights: dict, rng: random.Random, sigma: float = 0.12, max_traits: int = 3):
    """Change only 1-3 traits (friend's rule); keys with value 0 stay off -
    except a dead knob switched on with chance DEAD_P, or a crossover from an
    older AI with chance CROSS_P."""
    from src.ai.agents import SEED_ON, TUNE_DEFAULTS
    if CROSS_P and rng.random() < CROSS_P:
        out = crossover(weights, rng, max_traits)
        if out:
            return out
    eff = effective(weights)
    # a dead knob that was switched on stays mutable from then on
    revived = {k for k, d in TUNE_DEFAULTS.items() if not d and k not in MUTABLE and k not in SWITCHES
               and isinstance(weights.get(k), (int, float)) and weights.get(k, 0) > 0}
    keys = [k for k in weights if not k.startswith("L.")
            and (k.split(".")[-1] in MUTABLE or k in revived)
            and isinstance(weights[k], (int, float))] \
        + [k for k in MUTABLE if k not in weights]
    keys = [k for k in keys if (weights.get(k, eff.get(k, 0)) or 0) > 0 or k.split(".")[-1] in SEED_ON]
    child = dict(weights)
    traits = rng.sample(keys, rng.randint(1, min(max_traits, len(keys))))
    for k in traits:
        v = child.get(k, eff.get(k, 0)) or SEED_ON.get(k.split(".")[-1], 0.0)
        child[k] = round(max(0.01, min(200.0, v * math.exp(rng.gauss(0, sigma)))), 6)
    if DEAD_P and rng.random() < DEAD_P:
        dead = dead_knobs(child)
        if dead:
            k = rng.choice(dead)
            child[k] = round(SEED_ON.get(k, 0.5) * math.exp(rng.gauss(0, sigma)), 6)
            traits = traits + [f"{k} (dead, switched on)"]
    return child, traits


def mutate_leader(weights: dict, leader: str, rng: random.Random, sigma: float, max_traits: int = 3):
    """Change 1-3 per-Leader multipliers L.<leader>.<knob> (start 1.0 = shared weights)."""
    child = dict(weights)
    traits = rng.sample([k for k in MUTABLE if k != "look_w"], rng.randint(1, max_traits))
    for knob in traits:
        k = f"L.{leader}.{knob}"
        child[k] = round(max(0.05, min(20.0, child.get(k, 1.0) * math.exp(rng.gauss(0, 2 * sigma)))), 4)
    return child, [f"L.{leader}.{t}" for t in traits]


def play(args):
    from src.selfplay.gap_arena import game
    return game(args)


LEADERS = False          # --leaders: every game uses random real Leaders


def run_games(pool, specs, n, base, speed=0.0):
    """-> (wins, fitness, vp, vp_r6 per spec index, rounds list) over n rotated-seat
    games. Fitness = wins, each worth 1 + speed x (rounds before round 10)."""
    wins, fit, vps, r6s, rounds = (collections.Counter(), collections.Counter(), collections.Counter(),
                                   collections.Counter(), [])
    global LAST_STATS
    LAST_STATS = collections.defaultdict(collections.Counter)
    for _, rnd, out in pool.imap_unordered(play, [(base + g, specs, LEADERS) for g in range(n)]):
        rounds.append(rnd)
        for i, o in out.items():
            wins[i] += o["win"]
            fit[i] += o["win"] * (1 + speed * max(0, 10 - rnd))
            vps[i] += o["vp"]
            r6s[i] += o["vp_r6"]
            for k, x in (o.get("stats") or {}).items():
                LAST_STATS[i][k] += x
    return wins, fit, vps, r6s, rounds


LAST_STATS = None


def stats_summary(n):
    """Per spec index: average end-of-game resources, Swordmaster share, contracts..."""
    return [{k: round(v / n, 3) for k, v in sorted(LAST_STATS[i].items())} for i in range(4)] \
        if LAST_STATS else None


def leader_play(args):
    """One game where the seat holding `leader` uses spec A, the rest the champion."""
    g, leader, spec_a, spec_rest = args
    import random as _r
    from src.game.leader.leader_definitions import LEADER_NAMES
    others = [l for l in LEADER_NAMES if l != leader]
    rr = _r.Random(800000 + g)
    lineup = rr.sample(others, 3)
    seat = g % 4
    lineup.insert(seat, leader)
    from src.selfplay.gap_arena import game_lineup
    return game_lineup(g, [spec_a if s == seat else spec_rest for s in range(4)], lineup, seat)


def leader_ab(pool, leader, paths, champ_path, agent, n, base, speed):
    """Fitness of the Leader's seat for each candidate in `paths`, same seeds."""
    jobs = [(base + g, leader, f"{agent}:tuned={p}", f"{agent}:tuned={champ_path}")
            for p in paths for g in range(n)]
    res = list(pool.imap(leader_play, jobs, chunksize=1))
    out = []
    for k in range(len(paths)):
        part = res[k * n:(k + 1) * n]
        out.append({"share": sum(r["win"] for r in part) / n,
                    "fit": sum(r["win"] * (1 + speed * max(0, 10 - r["rounds"])) for r in part) / n,
                    "vp": sum(r["vp"] for r in part) / n,
                    "rounds": sum(r["rounds"] for r in part) / n})
    return out


def confirm(pool, champ_path, mut_path, n, base, speed, agent):
    """Re-test a would-be champion on fresh seeds: 2 mutant seats vs 2 champion
    seats (seats rotate). -> (mutant fitness, champion fitness) per seat-game."""
    specs = [f"{agent}:tuned={mut_path}", f"{agent}:tuned={champ_path}"] * 2
    wins, fit, vps, r6s, rounds = run_games(pool, specs, n, base, speed)
    return (fit[0] + fit[2]) / (2 * n), (fit[1] + fit[3]) / (2 * n), (wins[0] + wins[2]) / (2 * n), \
        (wins[1] + wins[3]) / (2 * n), float(np.mean(rounds))


def confirm_leader(pool, leader, champ_path, mut_path, agent, n, base, speed):
    """Re-test a Leader's new weights on fresh seeds (that seat only)."""
    r = leader_ab(pool, leader, [champ_path, mut_path], champ_path, agent, n, base, speed)
    return r[1]["fit"], r[0]["fit"], r[1]["share"], r[0]["share"], r[1]["rounds"]


def bench(pool, champ_path, start_path, n, gen, speed=0.0, agent="heuristic", opp_agent="heuristic"):
    """Champion (played by `agent`) vs 3 copies of the starting weights (played by `opp_agent`)."""
    specs = [f"{agent}:tuned={champ_path}"] + [f"{opp_agent}:tuned={start_path}"] * 3
    wins, fit, vps, r6s, rounds = run_games(pool, specs, n, 90_000_000 + gen * 10_000, speed)
    rec = {"gen": gen, "games": n, "share": round(wins[0] / n, 4), "fit": round(fit[0] / n, 4),
           "vp": round(vps[0] / n, 3), "opp_vp": round(sum(vps[i] for i in (1, 2, 3)) / (3 * n), 3),
           "vp_r6": round(r6s[0] / n, 3), "opp_vp_r6": round(sum(r6s[i] for i in (1, 2, 3)) / (3 * n), 3),
           "rounds": round(float(np.mean(rounds)), 3),
           "ended_by_r8": round(float(np.mean([r <= 8 for r in rounds])), 4)}
    print(f"bench gen {gen}: champion {rec['share']:.1%} vs start | VP r6 {rec['vp_r6']} "
          f"(start {rec['opp_vp_r6']}) | rounds {rec['rounds']}", flush=True)
    return rec


# ----------------------------------------------------------- human agreement
def load_human_positions(pattern):
    """Your decisions from the UI (data/ui_games/*.pkl): position, options, your pick."""
    import glob, pickle
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        won = (d.get("meta") or {}).get("you_won")
        for x in d.get("positions", []):
            if x.get("cands") and x.get("human"):
                out.append((x["gs"], x["pid"], x["cands"], x["human"], bool(won)))
    return out


def human_agreement(weights_path, positions) -> float:
    """Share of your decisions this weight set's heuristic would also make
    (won games count double: those are the plays to copy)."""
    from src.ai.agents import make_agent
    from ui.server import fit_label
    ag = make_agent(f"heuristic:tuned={weights_path}", seed=0)
    hit = tot = 0.0
    for gs, pid, cands, human, won in positions:
        try:
            sc = [ag.score(gs, pid, a) for a in cands]
        except Exception:
            continue
        w = 2.0 if won else 1.0
        tot += w
        hit += w * (fit_label(cands[max(range(len(cands)), key=lambda i: sc[i])]) == human)
    return hit / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=os.path.join("config", "gap", "tsmf5_dp1_wb10_tm1_op1.json"))
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--margin", type=float, default=0.02,
                    help="a mutant must beat the champion's fitness per game by this much")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-traits", type=int, default=3, help="weights changed per mutant (1..n)")
    ap.add_argument("--intrigue-persona", action="store_true",
                    help="mutant 3 of every generation is an Intrigue-loving copy of the champion")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="a win is worth 1 + speed x (rounds before round 10)")
    ap.add_argument("--agent", default="heuristic",
                    help="agent spec the weights drive, e.g. look:D3:K3:M2 (lookahead + evaluator)")
    ap.add_argument("--bench-start", default=None,
                    help="weights the benchmark opponents use (default: --start)")
    ap.add_argument("--bench-every", type=int, default=10)
    ap.add_argument("--bench-games", type=int, default=400)
    ap.add_argument("--bench-agent", default="heuristic",
                    help="agent the benchmark opponents use ('same' = --agent)")
    ap.add_argument("--leaders", action="store_true", help="games use random real Leaders")
    ap.add_argument("--personas", default="",
                    help="comma list (intrigue, tech): mutant 3 cycles through these personas")
    ap.add_argument("--human", type=float, default=0.0,
                    help="fitness += this x (agreement with your UI decisions - champion's)")
    ap.add_argument("--human-games", default=os.path.join("data", "ui_games", "*.pkl"))
    ap.add_argument("--leader-every", type=int, default=0,
                    help="every k-th generation tunes one Leader's own weights (A/B on that seat)")
    ap.add_argument("--leader-games", type=int, default=150)
    ap.add_argument("--leader-order", default="",
                    help="comma list of Leaders to tune, in order (default: all, weakest first)")
    ap.add_argument("--confirm-games", type=int, default=0,
                    help="a would-be champion replays this many fresh games and must win them too")
    ap.add_argument("--confirm-leader-games", type=int, default=0,
                    help="same for a Leader's new weights (0 = --leader-games)")
    ap.add_argument("--extra-mutant", default="",
                    help="weights file played as mutant 3 in generation 1 (a starting candidate)")
    A = ap.parse_args()
    global LEADERS
    LEADERS = A.leaders
    bench_agent = A.agent if A.bench_agent == "same" else A.bench_agent
    personas = [x for x in A.personas.split(",") if x] or (["intrigue"] if A.intrigue_persona else [])
    human_pos = load_human_positions(A.human_games) if A.human else []
    if A.human:
        print(f"human agreement: {len(human_pos)} of your UI decisions", flush=True)
    out_dir = A.out
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.jsonl")
    champ_path = os.path.join(out_dir, "champion.json")
    start_gen = 0
    if os.path.exists(champ_path) and os.path.exists(log_path):
        champion = json.load(open(champ_path))
        start_gen = sum(1 for _ in open(log_path))
        print(f"resuming at generation {start_gen}", flush=True)
    else:
        champion = json.load(open(A.start))
        json.dump(champion, open(champ_path, "w"), indent=1)
    rng = random.Random(A.seed + start_gen)
    bench_path = os.path.join(out_dir, "bench.jsonl")
    start_cfg = os.path.join(out_dir, "start.json")
    if not os.path.exists(start_cfg):
        json.dump(json.load(open(A.bench_start or A.start)), open(start_cfg, "w"), indent=1)
    from src.selfplay.evolve_progress import build

    def do_bench(gen):
        with open(bench_path, "a") as f:
            f.write(json.dumps(bench(pool, champ_path, start_cfg, A.bench_games, gen, A.speed, A.agent,
                                     bench_agent)) + "\n")

    with Pool(A.workers, maxtasksperchild=50) as pool:
        benched = {json.loads(l)["gen"] for l in open(bench_path)} if os.path.exists(bench_path) else set()
        if A.bench_every and start_gen not in benched:
            do_bench(start_gen)
        build(out_dir)
        from src.game.leader.leader_definitions import LEADER_NAMES
        order = [x for x in A.leader_order.split(",") if x] or list(LEADER_NAMES)
        for gen in range(start_gen + 1, A.gens + 1):
            if A.leader_every and gen % A.leader_every == 0:
                # tune one Leader's own weights: its seat is played by the champion
                # and by 3 mutants on the same games
                leader = order[(gen // A.leader_every - 1) % len(order)]
                muts = [mutate_leader(champion, leader, rng, A.sigma) for _ in range(3)]
                paths = [champ_path]
                for k, (w, _) in enumerate(muts, 1):
                    p = os.path.join(out_dir, f"gen{gen}_L{k}.json")
                    json.dump(w, open(p, "w"), indent=1)
                    paths.append(p)
                res = leader_ab(pool, leader, paths, champ_path, A.agent, A.leader_games, gen * 100_000, A.speed)
                best = max(range(1, 4), key=lambda i: res[i]["fit"])
                changed = res[best]["fit"] > res[0]["fit"] + A.margin
                conf = None
                if changed and A.confirm_games:
                    n_c = A.confirm_leader_games or A.leader_games
                    mf, cf, ms, cs, cr = confirm_leader(pool, leader, champ_path, paths[best], A.agent, n_c,
                                                        gen * 100_000 + 50_000, A.speed)
                    conf = {"games": n_c, "mutant_fit": round(mf, 4), "champion_fit": round(cf, 4),
                            "mutant_share": round(ms, 4), "champion_share": round(cs, 4), "rounds": round(cr, 3)}
                    changed = mf > cf
                    print(f"  confirm [{leader}]: mutant seat {ms:.1%} vs champion {cs:.1%} -> "
                          + ("confirmed" if changed else "rejected"), flush=True)
                rec = {"gen": gen, "leader": leader, "ab": [{k: round(v, 4) for k, v in r.items()} for r in res],
                       "mutated": [m[1] for m in muts], "new_champion": best if changed else 0,
                       "confirm": conf}
                if changed:
                    champion = muts[best - 1][0]
                    json.dump(champion, open(champ_path, "w"), indent=1)
                    rec["changes"] = {k: champion[k] for k in muts[best - 1][1]}
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"gen {gen} [{leader}]: champion seat {res[0]['share']:.1%} | mutants " +
                      " ".join(f"{r['share']:.1%}" for r in res[1:]) +
                      (f" -> mutant {best} wins {rec['changes']}" if changed else " -> no change"), flush=True)
                if A.bench_every and gen % A.bench_every == 0:
                    do_bench(gen)
                build(out_dir)
                continue
            mutants = [mutate(champion, rng, A.sigma, A.max_traits) for _ in range(3)]
            # count only the normal generations, so every persona gets its turn
            # (with --leader-every 2 the normal ones are all odd)
            normal_k = gen - (gen // A.leader_every if A.leader_every else 0)
            persona = personas[normal_k % len(personas)] if personas else None
            if persona == "intrigue":
                # one mutant each generation is an Intrigue lover (user, 2026-10-16)
                w = dict(champion)
                base = effective(champion).get("intrigue_love", 1.0)
                w["intrigue_love"] = round(max(1.0, base) * rng.uniform(1.4, 2.6), 4)
                w["res_intrigue"] = round(effective(champion).get("res_intrigue", 1.0) * rng.uniform(1.0, 1.5), 4)
                mutants[2] = (w, ["intrigue_love", "res_intrigue"])
            elif persona == "contract":
                # contract lover: takes contracts, stacks them to complete several in one move
                w = dict(champion)
                e = effective(champion)
                w["contracts"] = round(max(0.5, e.get("contracts", 1.0)) * rng.uniform(1.4, 2.5), 4)
                w["res_contract"] = round(e.get("res_contract", 1.0) * rng.uniform(1.3, 2.2), 4)
                w["contract_stack"] = round(rng.uniform(1.0, 3.0), 4)
                mutants[2] = (w, ["contracts", "res_contract", "contract_stack"])
            elif persona == "knobs":
                # the knobs that were stuck at 0: maker bonus spice, alliances
                w = dict(champion)
                w["maker_bonus"] = round(rng.uniform(0.3, 1.2), 4)
                w["alliance"] = round(rng.uniform(0.3, 1.2), 4)
                mutants[2] = (w, ["maker_bonus", "alliance"])
            elif persona == "tech":
                # tech lover: values tech tiles more, spams sandworms less (user, 2026-10-17)
                w = dict(champion)
                w["tech_love"] = round(max(1.0, effective(champion).get("tech_love", 1.0)) * rng.uniform(1.5, 3.0), 4)
                w["worm_bonus"] = round(effective(champion).get("worm_bonus", 0.0) * rng.uniform(0.2, 0.6), 4)
                mutants[2] = (w, ["tech_love", "worm_bonus"])
            if gen == 1 and A.extra_mutant and start_gen == 0:
                w = json.load(open(A.extra_mutant))
                mutants[2] = (w, sorted(k for k in w if w.get(k) != champion.get(k)))
                persona = "start:" + os.path.basename(A.extra_mutant)
            paths = [champ_path]
            for k, (w, _) in enumerate(mutants, 1):
                p = os.path.join(out_dir, f"gen{gen}_m{k}.json")
                json.dump(w, open(p, "w"), indent=1)
                paths.append(p)
            specs = [f"{A.agent}:tuned={p}" for p in paths]
            wins, fit, vps, r6s, rounds = run_games(pool, specs, A.games, gen * 100_000, A.speed)
            share = [wins[i] / A.games for i in range(4)]
            score = [fit[i] / A.games for i in range(4)]       # = share when --speed 0
            agree = [human_agreement(p, human_pos) for p in paths] if human_pos else None
            if agree:
                score = [score[i] + A.human * (agree[i] - agree[0]) for i in range(4)]
            best = max(range(1, 4), key=lambda i: score[i])
            changed = score[best] > score[0] + A.margin
            conf = None
            if changed and A.confirm_games:
                mf, cf, ms, cs, cr = confirm(pool, champ_path, paths[best], A.confirm_games,
                                             gen * 100_000 + 50_000, A.speed, A.agent)
                if agree:                       # same human-agreement credit as the main match
                    mf += A.human * (agree[best] - agree[0])
                conf = {"games": A.confirm_games, "mutant_fit": round(mf, 4), "champion_fit": round(cf, 4),
                        "mutant_share": round(ms, 4), "champion_share": round(cs, 4), "rounds": round(cr, 3)}
                changed = mf > cf
                print(f"  confirm mutant {best}: {ms:.1%} vs champion {cs:.1%} per seat -> "
                      + ("confirmed" if changed else "rejected"), flush=True)
            rec = {"gen": gen, "share": [round(x, 4) for x in share], "confirm": conf,
                   "fit": [round(x, 4) for x in score],
                   "vp": [round(vps[i] / A.games, 3) for i in range(4)],
                   "vp_r6": [round(r6s[i] / A.games, 3) for i in range(4)],
                   "rounds": round(float(np.mean(rounds)), 3),
                   "ended_by_r8": round(float(np.mean([r <= 8 for r in rounds])), 4),
                   "mutated": [m[1] for m in mutants], "new_champion": best if changed else 0,
                   "persona": persona, "human_agree": [round(x, 4) for x in agree] if agree else None,
                   "stats": stats_summary(A.games)}
            if changed:
                champion = mutants[best - 1][0]
                json.dump(champion, open(champ_path, "w"), indent=1)
                rec["changes"] = {k: champion[k] for k in mutants[best - 1][1]}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"gen {gen}: champion {share[0]:.1%} (fit {score[0]:.3f}) | mutants " +
                  " ".join(f"{s:.1%} ({f:.3f})" for s, f in zip(share[1:], score[1:])) +
                  (f" -> mutant {best} wins {rec['changes']}" if changed else " -> champion stays") +
                  f" | rounds {rec['rounds']}", flush=True)
            if A.bench_every and gen % A.bench_every == 0:
                do_bench(gen)
            build(out_dir)


if __name__ == "__main__":
    main()
