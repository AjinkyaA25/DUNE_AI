"""Keep improving the Best AI (user, 2026-10-18: "even if it is slightly better,
it should be the dominant AI. Run several thousand games and keep updating").

Each cycle:
  1. evolve_fixed from the current Best AI against itself, mutating only the
     --knobs (troop deploy / conservation, TSMF + persuasion plan by default);
  2. confirm the cycle's champion over --confirm fresh games (1 seat vs 3
     Best AI, seats rotated, random Leaders);
  3. promote it to config/best_ai.json when its win share is at least --z
     standard errors above the 25% an equal AI gets AND the lookahead agent
     (what the UI / website plays) with it is not below 25% over --look-games;
     the replaced Best AI is kept in config/best_ai_history/.

    python -m src.selfplay.evolve_loop [--cycles 0] [--gens 30] [--candidate <json>]

--candidate: confirm (and maybe promote) this config first, e.g. the champion
of a run that just finished. --recalib N (user, 2026-10-19: "with the increase
in games the AI should get better slowly with adjustment to judgement"): after
every promotion (and once at the start) play N deploy-report games with Best
AI, refit the Conflict odds on what really happened (fight_calib.py ->
models_fight/calibration_vK.json) and put Best AI + fight_calib K through the
same promotion check, so the combat judgement keeps tracking how Best AI plays. --cycles 0 = run until stopped. Progress goes to
config/evolve_loop/log.jsonl; the website / UI pick up a new Best AI on restart.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import src.selfplay.evolve as E

BEST = os.path.join("config", "best_ai.json")
HIST = os.path.join("config", "best_ai_history")
OUT = os.path.join("config", "evolve_loop")
KNOBS = ("deploy_hold,combat_ev,troop_hold,out_of_rewards,ci_hold,fight_learned,intrigue_count,"
         "smf_plan,tsmf,tsmf_goal,pers_w,buy_plan,buy_mult,hc_rush,sm_solari,deck_k,"
         "deck_plan,hc_judge,intrigue_love,res_intrigue,ci_plan,inf_plan,eg_plan,prep_draw,spice_judge,tech_love,res_spice,tech_judge,tech_learned,tech_deny")


def log(rec: dict) -> None:
    rec = {"time": time.strftime("%Y-%m-%d %H:%M"), **rec}
    with open(os.path.join(OUT, "log.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


LOOK = "look:D2:K3:M1:ev=models_eval_leaders/evaluator.npz"   # plain: the E6 solver x4 seats is ~15x slower


def confirm(path: str, games: int, seed: int, workers: int, agent: str = "heuristic"):
    """Win share of `path` (1 seat) vs 3 Best AI, rotated seats, random Leaders."""
    E.LEADERS = True
    specs = [f"{agent}:tuned={path}"] + [f"{agent}:tuned={BEST}"] * 3
    with Pool(workers) as pool:
        wins, _, vps, _, rounds = E.run_games(pool, specs, games, seed)
    share = wins[0] / games
    se = math.sqrt(0.25 * 0.75 / games)
    return {"share": round(share, 4), "z": round((share - 0.25) / se, 2), "vp": round(vps[0] / games, 3),
            "rounds": round(sum(rounds) / len(rounds), 3),
            "ended_by_r8": round(sum(r <= 8 for r in rounds) / len(rounds), 4)}


def maybe_promote(path: str, tag: str, A) -> bool:
    res = confirm(path, A.confirm, A.seed, A.workers)
    A.seed += 1_000_000
    win = res["z"] >= A.z
    log({"event": "confirm", "agent": "heuristic", "candidate": path, **res})
    if win and A.look_games:
        look = confirm(path, A.look_games, A.seed, A.workers, LOOK)
        A.seed += 1_000_000
        win = look["z"] >= 0.0
        log({"event": "confirm", "agent": "lookahead", "candidate": path, **look})
    log({"event": "decision", "candidate": path, "promoted": win})
    if win:
        os.makedirs(HIST, exist_ok=True)
        keep = os.path.join(HIST, f"best_ai_{time.strftime('%Y%m%d_%H%M')}_{tag}.json")
        shutil.copy(BEST, keep)
        shutil.copy(path, BEST)
        log({"event": "promoted", "new_best": path, "previous_saved_as": keep})
    return win


def recalibrate(A) -> None:
    """Refit the Conflict odds on Best AI's own deploys; try the refit as a candidate."""
    if not A.recalib:
        return
    best = json.load(open(BEST))
    # skip when this exact Best AI was refitted already (e.g. on a restart)
    import hashlib
    key = hashlib.sha1(json.dumps(best, sort_keys=True).encode()).hexdigest()
    done = os.path.join(OUT, "recalib_done.json")
    seen = json.load(open(done)) if os.path.exists(done) else []
    if key in seen:
        return
    json.dump(seen + [key], open(done, "w"))
    have = [int(f[len("calibration_v"):-5]) for f in os.listdir("models_fight")
            if f.startswith("calibration_v") and f.endswith(".json")]
    ver = max([1, int(best.get("fight_calib", 1))] + have) + 1
    spec = "heuristic:tuned=" + BEST.replace(os.sep, "/")
    with open(os.path.join(OUT, f"recalib_v{ver}.log"), "w") as fh:
        subprocess.run([sys.executable, "-u", "-m", "src.selfplay.deploy_report", spec, "--games", str(A.recalib),
                        "--workers", str(A.workers)], stdout=fh, stderr=subprocess.STDOUT, check=False)
        report = os.path.join("reports", "deploy_" + spec.replace(":", "_").replace("/", "_").replace("=", "_")[:60] + ".json")
        dest = os.path.join("models_fight", f"calibration_v{ver}.json")
        subprocess.run([sys.executable, "-u", "-m", "src.selfplay.fight_calib", "--out", dest,
                        f"{report}:{best.get('fight_learned', 0.0)}"], stdout=fh, stderr=subprocess.STDOUT, check=False)
    if not os.path.exists(dest):
        log({"event": "recalib_failed", "version": ver})
        return
    cand = os.path.join(OUT, f"recalib_v{ver}.json")
    json.dump({**best, "fight_calib": ver}, open(cand, "w"), indent=1)
    log({"event": "recalib", "version": ver, "games": A.recalib, "fit": json.load(open(dest))})
    maybe_promote(cand, f"recalib{ver}", A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=0, help="0 = until stopped")
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--games", type=int, default=6000, help="games per mutant")
    ap.add_argument("--confirm", type=int, default=10000)
    ap.add_argument("--z", type=float, default=1.0, help="promote at this many SE above 25%%")
    ap.add_argument("--look-games", type=int, default=1500, help="lookahead check before promoting (0 = skip)")
    ap.add_argument("--knobs", default=KNOBS)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--seed", type=int, default=50_000_000)
    ap.add_argument("--start-from", default=None, help="first cycle evolves from this config instead of Best AI")
    ap.add_argument("--recalib", type=int, default=0, help="deploy-report games per Conflict-odds refit (0 = off)")
    # passed to evolve_fixed (user, 2026-10-19): paired seeds, a League of older
    # AIs in the field + crossover from them, dead knobs switched on now and then
    ap.add_argument("--paired", type=int, default=0)
    ap.add_argument("--league", default="")
    ap.add_argument("--league-frac", type=float, default=0.0)
    ap.add_argument("--cross", type=float, default=0.0)
    ap.add_argument("--dead", type=float, default=0.0)
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if A.candidate:
        maybe_promote(A.candidate, "candidate", A)
    recalibrate(A)
    k = 0
    while A.cycles == 0 or k < A.cycles:
        k += 1
        out = os.path.join(OUT, f"cycle_{time.strftime('%Y%m%d_%H%M')}")
        start = os.path.join(OUT, "start.json")
        # evolve from the current Best AI (or --start-from, first cycle), against Best AI
        shutil.copy(A.start_from if (k == 1 and A.start_from) else BEST, start)
        log({"event": "cycle_start", "cycle": k, "out": out})
        cmd = [sys.executable, "-u", "-m", "src.selfplay.evolve_fixed", "--out", out, "--start", start,
               "--opp", BEST, "--gens", str(A.gens), "--games", str(A.games), "--knobs", A.knobs,
               "--workers", str(A.workers), "--seed", str(31 + 7 * k),
               "--paired", str(A.paired), "--league", A.league + "," + os.path.join(HIST, "*.json"),
               "--league-frac", str(A.league_frac), "--cross", str(A.cross), "--dead", str(A.dead)]
        with open(out + ".log", "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
        champ = os.path.join(out, "champion.json")
        if not os.path.exists(champ) or json.load(open(champ)) == json.load(open(BEST)):
            log({"event": "cycle_end", "cycle": k, "result": "no change kept"})
            continue
        if maybe_promote(champ, f"cycle{k}", A):
            recalibrate(A)


if __name__ == "__main__":
    main()
