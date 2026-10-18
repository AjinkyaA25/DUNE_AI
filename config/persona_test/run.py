"""Persona test (2026-10-17): champion vs contract lovers vs Intrigue lover, with Leaders.

    python config/persona_test/run.py [games]
"""
import json
import sys
from multiprocessing import Pool

sys.path.insert(0, ".")

if __name__ == "__main__":
    import src.selfplay.evolve as E
    E.LEADERS = True
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    agent = "look:D2:K3:M1:ev=models_eval_leaders/evaluator.npz"
    names = ["champion", "contract_moderate", "contract_strong", "intrigue"]
    paths = ["config/evolve_leaders2/champion.json"] + [f"config/persona_test/{x}.json" for x in names[1:]]
    with Pool(28, maxtasksperchild=50) as pool:
        wins, fit, vps, r6s, rounds = E.run_games(pool, [f"{agent}:tuned={p}" for p in paths], n, 55_000_000, 0.4)
    stats = E.stats_summary(n)
    out = {"games": n, "rounds": sum(rounds) / n, "ended_by_r8": sum(r <= 8 for r in rounds) / n,
           "agents": {names[i]: {"share": wins[i] / n, "fit": fit[i] / n, "vp": vps[i] / n,
                                 "vp_r6": r6s[i] / n, **stats[i]} for i in range(4)}}
    json.dump(out, open("config/persona_test/result.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
