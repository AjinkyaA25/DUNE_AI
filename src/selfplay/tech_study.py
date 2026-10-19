"""In which situations is each tech tile worth buying? (user, 2026-10-19: "for
each tech tile, there should be plays with it to understand in which
situations which tech is a good idea").

Best AI plays full games. At up to --per-game real tech-buy decisions, every
option (each affordable stack top and "decline") is tried on the same --k
determinized copies of the game (hidden cards re-randomised, common random
numbers across options) and played to the END by Best AI in every seat. An
option's result = win (ties split) + 0.03 x VP margin. Each row records the
situation, the tile, both heuristic valuations (flat and tech_judge) and the
measured gain over declining.

Every tile gets studied, not only the ones that happen to be on top of a
stack: --extra more affordable tiles are put on top of stack 0 in the copies
(counterfactual market) and tried as well (rows marked "injected").

    python -m src.selfplay.tech_study --games 500 --k 10 [--extra 2] [--workers 28]
    python -m src.selfplay.tech_study --report      (reports/tech_study.txt)
    python -m src.selfplay.tech_study --fit         (models_tech/tech_values.json)

--fit: per tile, a ridge fit of the measured gain on the situation (FEATS),
shrunk toward the tile's average; agents use it through the tech_learned knob
(the heuristic valuations didn't correlate with the measured gains: r ~ 0).
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

BEST = "config/best_ai.json"
ROWS = os.path.join("reports", "tech_study.jsonl")


def _score(g, pid) -> float:
    vp = [q.victory_points for q in g.players]
    w = g.winner
    if w is None:
        top = max(vp)
        tied = [i for i, v in enumerate(vp) if v == top]
        win = 1.0 / len(tied) if pid in tied else 0.0
    else:
        win = float(w == pid)
    lead = vp[pid] - max(v for i, v in enumerate(vp) if i != pid)
    return win + 0.03 * max(-5, min(5, lead))


def _play_out(g, agents, cap=4000):
    n = 0
    while not g.game_over and n < cap:
        q = g.player_in_reveal_buy
        if q is None:
            q = g.get_current_player_id()
        g.step(agents[q].select_action(g, q, g.get_valid_actions(q)))
        n += 1


def _situation(gs, pid) -> dict:
    from src.ai.bloodlines_heuristic import _hand_persuasion
    p = gs.players[pid]
    inf = list(p.influence.values())
    vp = [q.victory_points for q in gs.players]
    row_costs = sorted((c.cost or 0) for c in gs.imperium_row)
    P = _hand_persuasion(gs, pid)
    return {"round": gs.round, "spice": p.spice, "vp": p.victory_points,
            "lead": p.victory_points - max(v for i, v in enumerate(vp) if i != pid),
            "friendships": sum(1 for x in inf if x >= 2), "at_1": sum(1 for x in inf if x == 1),
            "at_0": sum(1 for x in inf if x == 0),
            "contracts_done": len(p.contracts_completed), "contracts_active": len(p.contracts_active),
            "icons_real": sum(1 for i in p.battle_icons if i != "wild"),
            "icons_wild": sum(1 for i in p.battle_icons if i == "wild"),
            "techs": len(p.techs), "councilor": int(p.has_councilor),
            "swordmaster": int(bool(getattr(p, "has_swordmaster", False))),
            "intrigues": len(p.intrigue_cards), "persuasion": P,
            "row_gap": min([c - P for c in row_costs if c > P] or [99])}


def _heuristic_values(gs, pid, d, discount):
    from src.ai import agents as A
    from src.ai.bloodlines_heuristic import tech_buy_value
    saved = A._TUNE
    out = {}
    for tag, k in (("flat", 0.0), ("judge", 1.0)):
        A._TUNE = {**saved, "tech_judge": k}
        try:
            out[tag] = round(tech_buy_value(gs, pid, d, discount), 3)
        finally:
            A._TUNE = saved
    return out


def _inject(g, name):
    """Put tile `name` on top of stack 0 of copy g (wherever it was)."""
    for st in g.bl.tech_stacks:
        for t in list(st):
            if t.name == name:
                st.remove(t)
                g.bl.tech_stacks[0].insert(0, t)
                return True
    return False


def study_game(args):
    g_id, k, per_game = args[:3]
    extra = args[3] if len(args) > 3 else 0
    from src.ai.agents import make_agent
    from src.ai.determinize import determinize
    from src.data.card_definitions import setup_game
    from src.game.gameState import ActionType
    from src.game.leader.leader_definitions import LEADER_NAMES
    lineup = random.Random(600000 + g_id).sample(list(LEADER_NAMES), 4)
    gs = setup_game(num_players=4, seed=600000 + g_id, leaders=lineup, use_bloodlines=True)
    agents = [make_agent(f"heuristic:tuned={BEST}", seed=g_id * 4 + s) for s in range(4)]
    rng = np.random.default_rng(g_id)
    rows, studied, n = [], 0, 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        opts = [a for a in valid if a.action_type == ActionType.RESOLVE_BL_CHOICE
                and str(getattr(a, "choice", "")).startswith("stack:")]
        decline = next((a for a in valid if a.action_type == ActionType.RESOLVE_BL_CHOICE
                        and getattr(a, "choice", "") == "decline"), None)
        if opts and decline is not None and studied < per_game:
            studied += 1
            pend = next((c for c in gs.bl.pending if c.player_id == pid), None)
            disc = getattr(pend, "discount", 0) if pend is not None else 0
            seeds = rng.integers(0, 2**31 - 1, size=k)
            sit = _situation(gs, pid)
            res = {}
            from src.game.bloodlines.techs import BLOODLINES_TECH_NAMES as _ALL
            tops = {gs.bl.tech_at(pid, int(a.choice.split(":")[1]))[0].name for a in opts
                    if gs.bl.tech_at(pid, int(a.choice.split(":")[1]))[0] is not None}
            spice = gs.players[pid].spice
            from src.game.bloodlines.techs import TECH_BY_NAME as _T
            others = [n for n in sorted(_ALL) if n not in tops and n not in {t.name for t in gs.players[pid].techs}
                      and gs.bl.tech_price(pid, _T[n], disc) <= spice
                      and any(t.name == n for st in gs.bl.tech_stacks for t in st)]
            inj = list(rng.choice(others, size=min(extra, len(others)), replace=False)) if others else []
            stack0 = next((a for a in valid if a.action_type == ActionType.RESOLVE_BL_CHOICE
                           and getattr(a, "choice", "") == "stack:0"), None)
            trials = [(a.choice, a, None) for a in opts + [decline]]
            if stack0 is not None:
                trials += [(f"inject:{n}", stack0, n) for n in inj]
            for tag, a, name in trials:
                key = repr(a)
                vals = []
                for j in range(k):
                    g = determinize(gs, pid, np.random.default_rng(int(seeds[j])))
                    if name is not None and not _inject(g, name):
                        continue
                    ga = next((x for x in g.get_valid_actions(pid) if repr(x) == key), None)
                    if ga is None:
                        continue
                    g.step(ga)
                    _play_out(g, agents)
                    vals.append(_score(g, pid))
                res[tag] = vals
            base = res.get("decline") or []
            leader = getattr(gs.players[pid].leader, "name", None)
            for tag in res:
                if tag == "decline":
                    continue
                if tag.startswith("inject:"):
                    d, more = _T[tag[7:]], 0
                else:
                    d, more = gs.bl.tech_at(pid, int(tag.split(":")[1]))
                if d is None or not res.get(tag) or not base:
                    continue
                m = min(len(res[tag]), len(base))
                gain = float(np.mean(np.array(res[tag][:m]) - np.array(base[:m])))
                rows.append({"game": g_id, "pid": pid, "leader": leader, "tech": d.name,
                             "injected": tag.startswith("inject:"),
                             "price": gs.bl.tech_price(pid, d, disc + more), "k": m,
                             "gain": round(gain, 4), "win": round(float(np.mean(res[tag])), 4),
                             "win_decline": round(float(np.mean(base)), 4),
                             "heur": _heuristic_values(gs, pid, d, disc + more), **sit})
        a = agents[pid].select_action(gs, pid, valid)
        gs.step(a)
        n += 1
    return rows


BINS = {   # the situation each tile depends on (user's examples first)
    "Panopticon": ("friendships", "at_1"),
    "CHOAM Transports": ("contracts_done", "contracts_active"),
    "Ornithopter Fleet": ("icons_real", "icons_wild"),
    "Gene-Locked Vault": ("row_gap", "intrigues"),
    "Delivery Bay": ("row_gap", "persuasion"),
}


def report(rows, out_path=os.path.join("reports", "tech_study.txt")):
    lines = [f"{len(rows)} tech-buy options studied in {len({(r['game'], r['pid'], r['round']) for r in rows})} "
             f"decisions ({len({r['game'] for r in rows})} games). gain = win share (+0.03/VP margin) when "
             f"buying vs declining, same worlds; x100 = percentage points.\n"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["tech"]].append(r)
    lines.append(f"{'tech':26s} {'n':>5s} {'gain':>7s} {'+/-':>5s} {'early R1-4':>10s} {'late R5+':>9s}"
                 f" {'corr flat':>9s} {'corr judge':>10s}")
    for t, rs in sorted(by.items(), key=lambda kv: -np.mean([r["gain"] for r in kv[1]])):
        g = np.array([r["gain"] for r in rs])
        se = g.std() / math.sqrt(len(g)) if len(g) > 1 else 0
        early = [r["gain"] for r in rs if r["round"] <= 4]
        late = [r["gain"] for r in rs if r["round"] >= 5]

        def corr(tag):
            h = np.array([r["heur"][tag] for r in rs])
            return float(np.corrcoef(h, g)[0, 1]) if len(rs) > 5 and h.std() > 0 and g.std() > 0 else float("nan")
        lines.append(f"{t:26s} {len(rs):5d} {100 * g.mean():+6.1f}  {100 * se:4.1f} "
                     f"{100 * np.mean(early) if early else float('nan'):+10.1f} "
                     f"{100 * np.mean(late) if late else float('nan'):+9.1f} {corr('flat'):9.2f} {corr('judge'):10.2f}")
    for t, keys in BINS.items():
        rs = by.get(t, [])
        if not rs:
            continue
        for key in keys:
            groups = collections.defaultdict(list)
            for r in rs:
                groups[min(r[key], 6)].append(r["gain"])
            lines.append(f"\n{t} by {key}: " + "  ".join(
                f"{v}: {100 * np.mean(gs_):+.1f} (n={len(gs_)})" for v, gs_ in sorted(groups.items())))
    # does the heuristic pick the option that the play-outs prefer?
    dec = collections.defaultdict(list)
    for r in rows:
        dec[(r["game"], r["pid"], r["round"], r["spice"])].append(r)
    for tag in ("flat", "judge"):
        hit = tot = 0
        regret = []
        for rs in dec.values():
            best = max(rs + [None], key=lambda r: r["gain"] if r else 0.0)
            pick = max(rs + [None], key=lambda r: r["heur"][tag] if r else 0.0)   # decline = 0
            tot += 1
            hit += best is pick
            regret.append((best["gain"] if best else 0.0) - (pick["gain"] if pick else 0.0))
        lines.append(f"\n{tag:5s} valuation picks the play-outs' best option in {hit / max(1, tot):.1%} of "
                     f"{tot} decisions; average regret {100 * np.mean(regret):.1f} pts")
    txt = "\n".join(lines)
    open(out_path, "w").write(txt + "\n")
    print(txt)


FEATS = ("round", "friendships", "at_1", "contracts_done", "contracts_active",
         "icons_real", "techs", "persuasion", "spice", "lead")
MODEL = os.path.join("models_tech", "tech_values.json")


def feature_vector(sit: dict) -> list:
    """Centred situation features (same order as FEATS)."""
    c = {"round": 5, "friendships": 1.5, "at_1": 1.5, "contracts_done": 1.5, "contracts_active": 1,
         "icons_real": 1, "techs": 1, "persuasion": 6, "spice": 4, "lead": 0}
    return [max(-6.0, min(6.0, float(sit.get(k, c[k])) - c[k])) for k in FEATS]


def fit(rows, lam=60.0):
    """gain ~ a + w . x per tile; the intercept shrunk toward the global mean
    with weight 20 rows, the slopes by ridge lam (few rows -> mostly the mean)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["tech"]].append(r)
    g_all = float(np.mean([r["gain"] for r in rows]))
    out = {"feats": list(FEATS), "global": g_all, "tiles": {}}
    for t, rs in by.items():
        X = np.array([feature_vector(r) for r in rs])
        y = np.array([r["gain"] for r in rs])
        n = len(y)
        a0 = (y.sum() + 20 * g_all) / (n + 20)
        Xc = X - X.mean(0)
        w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - y.mean()))
        a = a0 - float(X.mean(0) @ w)
        out["tiles"][t] = {"a": round(float(a), 5), "w": [round(float(v), 5) for v in w], "n": n}
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    json.dump(out, open(MODEL, "w"), indent=1)
    print(f"fitted {len(out['tiles'])} tiles -> {MODEL}")
    for t, m in sorted(out["tiles"].items(), key=lambda kv: -kv[1]["a"]):
        top = sorted(zip(FEATS, m["w"]), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  {t:26s} n={m['n']:4d} base {100 * m['a']:+5.1f}  " +
              ", ".join(f"{k} {100 * v:+.1f}" for k, v in top))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--extra", type=int, default=2, help="off-market tiles tried per decision")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--fit", action="store_true")
    A = ap.parse_args()
    if A.fit:
        fit([json.loads(x) for x in open(ROWS)])
        return
    if not A.report:
        os.makedirs("reports", exist_ok=True)
        with Pool(A.workers, maxtasksperchild=20) as pool, open(ROWS, "a") as fh:
            for rows in pool.imap_unordered(study_game, [(A.start + g, A.k, A.per_game, A.extra)
                                                          for g in range(A.games)]):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
    report([json.loads(x) for x in open(ROWS)])


if __name__ == "__main__":
    main()
