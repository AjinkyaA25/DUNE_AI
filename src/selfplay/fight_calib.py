"""Calibrate the Conflict forecast on what really happened (user, 2026-10-18:
"correct the forecast estimation ... try to place in combats").

The deploy report (src/selfplay/deploy_report.py) keeps, for every deploy, the
AI's view of each rival - forecast, spread, P(in the fight) - its own strength
estimate and the final strengths. Measured: the AI's own final strength runs
~2.5 above its estimate (swords it adds later), and its odds are too sure at
the extremes (a 97% "rival ahead" happens ~86%: Reveal / Intrigue swings).

Fitted here, per forecast blend (fight_learned weight):

    P(rival finishes ahead of me) = P(rival in the fight) * sigmoid(a * z + b)
    z = (rival forecast - my estimate) / spread

b < 0 moves the line toward "I'm stronger than I think", a < 1.7 (the slope
of the plain normal curve) widens it. Fitted separately for rivals who have /
have not revealed (a revealed rival can only add Combat Intrigues, so those
odds can be much sharper). Used by agents.fight_ev when the fight_calib knob
is on. Ties count half.

    python -m src.selfplay.fight_calib [--out PATH] REPORT.json:WEIGHT [REPORT.json:WEIGHT ...]
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

OUT = os.path.join("models_fight", "calibration.json")


def rows(path):
    d = json.load(open(path))["decisions"]
    Z, PIN, Y, G, R = [], [], [], [], []
    for x in d:
        mine = x["my_strength_after"] + x["combat_intrigue_counted"]
        myf = x["final_strength"]
        for r in x["rivals"]:
            if "final" not in r:
                continue
            Z.append((r["forecast"] - mine) / r["spread"])
            PIN.append(r.get("p_in", 1.0 if r["can_fight"] else 0.0))
            Y.append(1.0 if r["final"] > myf else (0.5 if r["final"] == myf and myf > 0 else 0.0))
            G.append(x["game"])
            R.append(bool(r.get("revealed", False)))
    return np.array(Z), np.array(PIN), np.array(Y), np.array(G), np.array(R)


def loss(a, b, Z, PIN, Y):
    P = np.clip(PIN / (1 + np.exp(-(a * Z + b))), 1e-4, 1 - 1e-4)
    return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))


def fit(Z, PIN, Y):
    best = (1e9, 1.7, 0.0)
    for a in np.arange(0.2, 6.01, 0.1):
        for b in np.arange(-2.0, 1.01, 0.05):
            l = loss(a, b, Z, PIN, Y)
            if l < best[0]:
                best = (l, float(a), float(b))
    return best


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    dest = OUT
    if "--out" in argv:
        i = argv.index("--out")
        dest = argv[i + 1]
        del argv[i:i + 2]
    out = json.load(open(dest)) if os.path.exists(dest) else {}
    for arg in argv:
        path, w = arg.rsplit(":", 1)
        Z, PIN, Y, G, R = rows(path)
        test = G % 5 == 0                                   # held-out games
        entry = {"n": int(len(Y)), "source": path}
        groups = [("all", np.ones(len(Y), bool))]
        if R.any():                                         # rivals who have / have not revealed
            groups += [("rev", R), ("unrev", ~R)]
        for tag, g in groups:
            _, a, b = fit(Z[g & ~test], PIN[g & ~test], Y[g & ~test])
            phi = 0.5 * (1 + np.vectorize(math.erf)(Z / math.sqrt(2)))
            P0 = np.clip(PIN * phi, 1e-4, 1 - 1e-4)
            t = g & test
            before = float(-np.mean(Y[t] * np.log(P0[t]) + (1 - Y[t]) * np.log(1 - P0[t])))
            after = loss(a, b, Z[t], PIN[t], Y[t])
            P1 = PIN / (1 + np.exp(-(a * Z + b)))
            rec = {"a": round(a, 3), "b": round(b, 3), "n": int(g.sum()),
                   "test_logloss_before": round(before, 4), "test_logloss_after": round(after, 4)}
            if tag == "all":
                entry.update(rec)
            else:
                entry[tag] = rec
            print(f"weight {w} [{tag}]: n={g.sum()} a={a:.2f} b={b:+.2f} | held-out log loss {before:.4f} -> {after:.4f} | "
                  f"mean P(rival ahead) {np.mean(P0[g]):.3f} -> {np.mean(P1[g]):.3f}, actual {np.mean(Y[g]):.3f}")
            for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
                m = (P1 >= lo) & (P1 < hi) & t
                if m.sum():
                    print(f"   calibrated {lo:.1f}-{hi:.1f}: n={m.sum():5d} predicted {P1[m].mean():.3f} actual {Y[m].mean():.3f}")
        out[str(float(w))] = entry
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)


if __name__ == "__main__":
    main()
