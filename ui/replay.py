#!/usr/bin/env python
"""
Replay a game recorded by the browser table (game_logs/ui_*.json) and look
at any decision in it.

  python ui/replay.py game_logs/ui_X.json                  # list every move
  python ui/replay.py game_logs/ui_X.json --move 57        # the position before move 57:
                                                           # state, legal moves, heuristic scores
  python ui/replay.py game_logs/ui_X.json --move 57 --search 24
                                                           # ...and re-run the search there
  python ui/replay.py game_logs/ui_X.json --spec heuristic:tuned=config/gap/tsmf5.json --move 57
                                                           # score with other weights (a tweak)
  python ui/replay.py game_logs/ui_X.json --check          # replay to the end, verify it matches

The game is rebuilt from its seed and the index of each move among the legal
moves, so the replayed position is exactly the one the AI saw.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.data.card_definitions import setup_game  # noqa: E402
from src.game.gameState import ActionType  # noqa: E402


def active(gs):
    p = gs.player_in_reveal_buy
    return gs.get_current_player_id() if p is None else p


def rebuild(doc, upto=None):
    """Game state before move `upto` (None = after the last move)."""
    random.seed(doc["seed"])
    n_players = len(doc["vp"])
    gs = setup_game(num_players=n_players, seed=doc["seed"], neutral_leaders=True,
                    use_bloodlines=doc.get("bloodlines", True))
    moves = doc["moves"]
    for i, m in enumerate(moves[:upto] if upto is not None else moves):
        pid = active(gs)
        valid = gs.get_valid_actions(pid)
        if not m.get("repr"):
            raise SystemExit(f"move {i} has no recorded move (log from an older server)")
        # match the move itself: the order of legal moves can differ between
        # processes (Python's per-process string hashing)
        act = next((x for x in valid if repr(x) == m["repr"]), None)
        if pid != m["active_player"] or act is None:
            raise SystemExit(f"replay diverged at move {i}: expected {m['repr']} by seat "
                             f"{m['active_player']}, engine has seat {pid} with {len(valid)} moves")
        gs.step(act)
    return gs


def show_state(gs, seat):
    print(f"Round {gs.round}  phase {gs.phase.value}  conflict "
          f"{gs.current_conflict.name if gs.current_conflict else None}  "
          f"shield wall {'up' if gs.shield_wall_intact else 'down'}")
    for p in gs.players:
        tag = " (you)" if p.id == seat else ""
        print(f"  P{p.id}{tag}: VP {p.victory_points}  solari {p.solari} spice {p.spice} "
              f"water {p.water}  garrison {p.troops_garrison}  in conflict "
              f"{gs.troops_in_conflict.get(p.id, 0)}+{gs.sandworms_in_conflict.get(p.id, 0)}w  "
              f"agents {p.agents_available}  infl {dict(p.influence)}  "
              f"hooks {int(p.has_maker_hooks)}  SM {int(p.has_swordmaster)}  HC {int(p.has_councilor)}")
        print(f"       hand: {[c.name for c in p.hand]}  intrigues: {[c.name for c in p.intrigue_cards]}")
    print(f"  Imperium Row: {[c.name for c in gs.imperium_row]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--move", type=int)
    ap.add_argument("--search", type=int, default=0, help="re-run search with M playouts")
    ap.add_argument("--spec", default="", help="agent whose heuristic scores to show "
                                               "(default: the seat's own opponent spec)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    doc = json.load(open(a.log, encoding="utf-8"))
    seat = doc.get("human_id")
    log = doc["log"]
    if a.check:
        gs = rebuild(doc)
        vp = [p.victory_points for p in gs.players]
        print(f"replayed {len(doc['moves'])} moves: VP {vp} (recorded {doc['vp']}) "
              f"{'OK' if vp == doc['vp'] else 'MISMATCH'}")
        return
    if a.move is None:
        for e in log:
            who = "you" if e["pid"] == seat else f"P{e['pid']}"
            ex = e.get("explain") or {}
            flag = ""
            if ex.get("kind") == "search":
                ch = next((r for r in ex["rows"] if r["chosen"]), None)
                best = max(ex["rows"], key=lambda r: r["value"])
                if ch is not best:
                    flag = f"  [kept heuristic pick; best playout {best['label']} {best['value']:.2f}]"
                elif ch is not ex["rows"][0]:
                    flag = "  [search overruled heuristic]"
            print(f"{e['n']:4d} R{e['round']:<2d} {who:>4s}  {e['label']}{flag}")
        if doc.get("error"):
            print("\nENGINE/AI ERROR:\n" + doc["error"])
        return
    gs = rebuild(doc, a.move)
    pid = active(gs)
    e = log[a.move] if a.move < len(log) else None
    print(f"Before move {a.move}: seat P{pid}{' (you)' if pid == seat else ''} to act")
    show_state(gs, seat)
    if e:
        print(f"\nPlayed: {e['label']}  ({e.get('secs', 0)} s)")
        ex = e.get("explain")
        if ex and ex.get("kind") == "search":
            print("Search at the time (heuristic order; value = playout win share):")
            for r in ex["rows"]:
                print(f"   {'*' if r['chosen'] else ' '} {r['value']:+.3f}  heur {r['heur']:+7.2f}  {r['label']}")
    from src.ai.agents import make_agent
    from ui.server import label
    spec = a.spec or doc.get("opponents", {}).get(str(pid)) \
        or "search:K5:M24:tuned=config/gap/tsmf5.json"
    ag = make_agent(spec, seed=0)
    h = getattr(ag, "h", ag)
    valid = gs.get_valid_actions(pid)
    acts = [x for x in valid if x.action_type != ActionType.NO_OP]
    sc = sorted(((h.score(gs, pid, x), x) for x in acts), key=lambda t: -t[0])
    print(f"\nHeuristic scores ({spec}):")
    for s, x in sc[:15]:
        print(f"   {s:+8.2f}  {label(gs, x)}")
    if len(sc) > 15:
        print(f"   ... {len(sc) - 15} more")
    if a.search:
        from src.ai.search import RoundSearchAgent
        sa = RoundSearchAgent(k=5, m=a.search, seed=1, tuning=getattr(h, "tuning", None))
        pick = sa.select_action(gs.clone(), pid, gs.get_valid_actions(pid))
        print(f"\nSearch now (M={a.search}):")
        if sa.last:
            for i, (c, v) in enumerate(zip(sa.last["cands"], sa.last["scores"])):
                print(f"   {'*' if i == sa.last['chosen'] else ' '} {v:+.3f}  {label(gs, c)}")
        else:
            print(f"   (not a searched decision) -> {label(gs, pick)}")


if __name__ == "__main__":
    main()
