#!/usr/bin/env python
"""
Where do humans and the AI disagree? Replays every cleaned video game and,
at each recorded human decision, records what the heuristic AI would have
chosen in the same position. Writes reports/human_vs_ai.json and prints a
summary: agreement by decision type, most common space / buy swaps, and
game-level habits (Swordmaster / High Council timing, round of first tech,
game length) for humans vs AI self-play.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import replay as R  # noqa: E402
from clean import clean_game  # noqa: E402
from src.game.gameState import ActionType  # noqa: E402


def label(a) -> str:
    t = a.action_type
    if t == ActionType.AGENT_TURN:
        return f"agent {a.space_name}"
    if t == ActionType.ACQUIRE_CARD:
        return f"buy {a.acquire_card_name}"
    if t == ActionType.ACQUIRE_RESERVE:
        return f"buy {a.reserve_type}"
    return t.value


ME = "dinosaur"      # the channel owner's TTS name (Dinosaur11, OCR'd variants)
COLORS = ("Red", "Green", "Blue", "Yellow")


def my_color(g: dict) -> str | None:
    """Dinosaur11's colour from the chat lobby list, when it is unambiguous:
    named directly (and no other name OCR'd onto the same colour), or the one
    colour left over when the three opponents are named."""
    lc = (g.get("merge_stats") or {}).get("lobby_colours") or {}
    mine = {c for n, c in lc.items() if ME in n.lower()}
    others = {c for n, c in lc.items() if ME not in n.lower() and "dinosar" not in n.lower()}
    if len(mine) == 1:
        c = next(iter(mine))
        return None if c in others else c
    if not mine and len(others) == 3:
        rest = set(COLORS) - others
        return rest.pop() if len(rest) == 1 else None
    return None


def main() -> None:
    rows = []
    fac = R._card_factory()
    for p in sorted(glob.glob(os.path.join(HERE, "games", "*.json"))):
        g = json.load(open(p, encoding="utf-8"))
        if g.get("source") != "vision" or g.get("duplicate_of"):
            continue
        if len({x["color"] for x in g["players"]}) != 4 or len(g.get("rounds", [])) < 5:
            continue
        mc = my_color(g)
        me_seat = next((x["seat"] for x in g["players"] if x["color"] == mc), None)
        g = clean_game(g, fac)
        rp = R.Replay(g)
        start = len(rows)

        def rec(pid, chosen, valid, _rp=rp, _vid=g["video_id"]):
            gs = _rp.gs
            cands = [a for a in valid if a.action_type != ActionType.NO_OP]
            if len(cands) < 2:
                return
            best = max(cands, key=lambda a: _rp.h.score(gs, pid, a))
            rows.append({"game": _vid, "round": gs.round,
                         "type": chosen.action_type.value,
                         "human": label(chosen), "ai": label(best),
                         "agree": repr(best) == repr(chosen),
                         "same_label": label(best) == label(chosen), "pid": pid})
        rp._record = rec
        rp.run()
        win = rp.winner()
        me_pid = rp.pid_of.get(me_seat) if me_seat else None
        for r in rows[start:]:
            pid = r.pop("pid")
            r["me"] = None if me_pid is None else pid == me_pid
            r["won"] = None if win is None else pid == win
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "reports", "human_vs_ai.json"), "w"), indent=0)

    print(f"{len(rows)} human decisions compared\n")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    print("agreement by decision type (same move / same space-or-card):")
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"  {t:14s} n={len(rs):4d}  exact {sum(r['agree'] for r in rs) / len(rs):.0%}"
              f"  same target {sum(r['same_label'] for r in rs) / len(rs):.0%}")
    for kind in ("agent", "buy"):
        swaps = collections.Counter(
            (r["human"], r["ai"]) for r in rows
            if not r["same_label"] and r["human"].startswith(kind) and r["ai"].startswith(kind))
        print(f"\nmost common {kind} disagreements (human chose -> AI would choose):")
        for (h, a), n in swaps.most_common(12):
            print(f"  {n:3d}  {h[len(kind) + 1:]:28s} -> {a[len(kind) + 1:]}")
    # what humans do that the AI would rarely do, and vice versa
    for kind in ("agent", "buy"):
        hc = collections.Counter(r["human"] for r in rows if r["human"].startswith(kind))
        ac = collections.Counter(r["ai"] for r in rows if r["ai"].startswith(kind))
        diff = sorted(set(hc) | set(ac), key=lambda k: -(hc[k] - ac[k]))
        print(f"\n{kind}: humans pick MORE than the AI would        | humans pick LESS")
        more = [(k, hc[k], ac[k]) for k in diff if hc[k] - ac[k] > 0][:8]
        less = [(k, hc[k], ac[k]) for k in reversed(diff) if ac[k] - hc[k] > 0][:8]
        for i in range(max(len(more), len(less))):
            m = more[i] if i < len(more) else None
            l = less[i] if i < len(less) else None
            ms = f"{m[0][len(kind) + 1:][:24]:24s} {m[1]:3d} vs {m[2]:3d}" if m else " " * 35
            ls = f"{l[0][len(kind) + 1:][:24]:24s} {l[1]:3d} vs {l[2]:3d}" if l else ""
            print(f"  {ms}   | {ls}")


if __name__ == "__main__":
    main()
