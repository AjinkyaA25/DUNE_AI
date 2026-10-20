#!/usr/bin/env python
"""How far is each AI version's judgement from the human plays in the video
games? (user, 2026-10-19: "learning from the video scraped games so far to
check for percent error off from 1 play to the other").

Replays every cleaned video game; at each recorded human decision (Agent
placement + card, card buy) every config scores all legal moves. Per decision:

  agree    the config's top move is the human's move
  regret   how far down the config's scale the human's move sits:
           (top score - human score) / (top score - bottom score), 0..100%
           0% = the config would play it, 100% = it is the config's worst move
  pct      share of the other legal moves the config ranks above the human's

Reported by decision type, round band and winner / others. A config whose
judgement matches good play should agree more with the winners' moves than
with the other players' moves ("winner gap" > 0).

    python video_scrape/judge_ai.py [config.json ...] [--workers 8]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from compare_ai import label, my_color  # noqa: E402

CONFIGS = None


def band(r: int) -> str:
    return "R1-3" if r <= 3 else ("R4-6" if r <= 6 else "R7+")


def judge_game(path: str):
    import replay as R
    from clean import clean_game
    from src.ai.agents import make_agent
    from src.game.gameState import ActionType
    g = json.load(open(path, encoding="utf-8"))
    if g.get("source") != "vision" or g.get("duplicate_of"):
        return []
    if len({x["color"] for x in g["players"]}) != 4 or len(g.get("rounds", [])) < 5:
        return []
    g = clean_game(g, R._card_factory())
    rp = R.Replay(g)
    # a config is a heuristic JSON, or a full agent spec (e.g. ppo:...) scored by its policy logits
    agents = {c: make_agent(c if ":" in c.split("/")[0] else f"heuristic:tuned={c}", seed=0) for c in CONFIGS}
    rows = []
    keep = (ActionType.AGENT_TURN, ActionType.ACQUIRE_CARD, ActionType.ACQUIRE_RESERVE)

    def rec(pid, chosen, valid, _rp=rp):
        gs = _rp.gs
        if chosen.action_type not in keep:
            return
        cands = [a for a in valid if a.action_type in keep]
        hi = next((i for i, a in enumerate(cands) if repr(a) == repr(chosen)), None)
        if hi is None or len(cands) < 2:
            return
        row = {"round": gs.round, "type": chosen.action_type.value, "human": label(chosen),
               "pid": pid, "n": len(cands), "by": {}}
        for c, ag in agents.items():
            try:
                sc = ag.scores(gs, pid, cands) if hasattr(ag, "scores") else [ag.score(gs, pid, a) for a in cands]
            except Exception:
                return
            top, bot = max(sc), min(sc)
            span = (top - bot) or 1.0
            row["by"][c] = {"agree": bool(sc[hi] == top), "regret": float((top - sc[hi]) / span),
                            "rank": int(1 + sum(s > sc[hi] for s in sc)),
                            "pct": float(sum(s > sc[hi] for s in sc) / (len(sc) - 1)),
                            "ai": label(cands[max(range(len(sc)), key=sc.__getitem__)])}
        rows.append(row)

    rp._record = rec
    try:
        rp.run()
    except Exception:
        return []
    win = rp.winner()
    for r in rows:
        r["game"] = g["video_id"]
        r["won"] = None if win is None else r["pid"] == win
    return rows


def init(configs):
    global CONFIGS
    CONFIGS = configs


def summary(rows, configs):
    out = {}
    for c in configs:
        groups = collections.defaultdict(list)
        for r in rows:
            x = r["by"][c]
            keys = ["all", r["type"], band(r["round"]), f"{r['type']} {band(r['round'])}"]
            if r["won"] is not None:
                keys.append("winners" if r["won"] else "others")
            for k in keys:
                groups[k].append(x)
        out[c] = {k: {"n": len(v), "agree": round(sum(x["agree"] for x in v) / len(v), 4),
                      "regret": round(sum(x["regret"] for x in v) / len(v), 4),
                      "rank": round(sum(x["rank"] for x in v) / len(v), 2),
                      "pct": round(sum(x["pct"] for x in v) / len(v), 4)}
                  for k, v in sorted(groups.items())}
        w, o = out[c].get("winners"), out[c].get("others")
        if w and o:
            out[c]["winner_gap"] = {"agree": round(w["agree"] - o["agree"], 4),
                                    "regret": round(o["regret"] - w["regret"], 4),
                                    "pct": round(o["pct"] - w["pct"], 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "video_judgement.json"))
    A = ap.parse_args()
    configs = A.configs or sorted(glob.glob(os.path.join("config", "best_ai_history", "*.json"))) + \
        [os.path.join("config", "best_ai.json")]
    games = sorted(glob.glob(os.path.join(HERE, "games", "*.json")))
    with Pool(A.workers, initializer=init, initargs=(configs,)) as pool:
        rows = [r for part in pool.imap_unordered(judge_game, games) for r in part]
    res = summary(rows, configs)
    json.dump({"configs": configs, "decisions": len(rows), "summary": res, "rows": rows},
              open(A.out, "w"), indent=0)
    print(f"{len(rows)} human decisions from {len({r['game'] for r in rows})} video games\n")
    keys = ["all", "agent_turn", "acquire_card", "acquire_reserve", "R1-3", "R4-6", "R7+", "winners", "others"]
    for c in configs:
        s = res[c]
        print(os.path.basename(c))
        for k in keys:
            if k in s:
                print(f"  {k:16s} n={s[k]['n']:5d}  agrees {s[k]['agree']:6.1%}  "
                      f"error (regret) {s[k]['regret']:6.1%}  moves ranked above the human's {s[k]['pct']:6.1%}")
        if "winner_gap" in s:
            print(f"  winners vs others: agrees {s['winner_gap']['agree']:+.1%}, "
                  f"error {-s['winner_gap']['regret']:+.1%}")
        print()
    # where the newest config and the humans part ways most
    c = configs[-1]
    swaps = collections.Counter((r["human"], r["by"][c]["ai"]) for r in rows
                                if not r["by"][c]["agree"] and r["won"])
    print(f"winners' moves {os.path.basename(c)} disagrees with most (winner chose -> AI would choose):")
    for (h, a), n in swaps.most_common(15):
        print(f"  {n:3d}  {h:34s} -> {a}")


if __name__ == "__main__":
    main()
