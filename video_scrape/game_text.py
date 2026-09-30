#!/usr/bin/env python
"""
Write a plain-text move list per game, for checking and filling gaps by hand.

  games_text/<id>.txt

  === Round 3 - Conflict: Siege of Arrakeen ===
  Combat: 1st Gurney Halleck, 2nd Paul Atreides, 3rd Princess Irulan
  Gurney Halleck: Arrakeen, Signet Ring, deployed 2
  Paul Atreides: Hagga Basin, Dune the Desert Planet
  Princess Irulan: reveal
  ...

Sources, best first:
  leader names     chat ("<name> picked leader X"); else the seat colour
  space / card     chat agent lines; else the video (Agent Turn row + the
                   colour that appeared on the board)
  troops deployed  the player's combat strength on screen: an agent turn
                   that raises it by 2 per troop (3 = a sandworm)
  combat placings  chat ("1st: <leader>"); else strength at the round's end
Unknown bits are written as '?', face-down cards as 'face-down card'.

Usage:
  python video_scrape/game_text.py              # every game file
  python video_scrape/game_text.py games/RHID5LtIO68_g1.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clean import clean_game  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(HERE, "games")
OUT = os.path.join(HERE, "games_text")
SETTLE = 12.0      # s after an agent turn for deployed troops to show
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def strength_at(series: list, t: float):
    val = None
    for tt, v in series:
        if tt > t:
            break
        val = v.get("strength")
    return val


def deployed(space: str) -> str:
    """Deploys aren't in the chat log and aren't readable from the finished
    videos: mark the turns where troops COULD be deployed, to fill in."""
    from src.game.board.board import COMBAT_SPACES
    return ", deployed ?" if space in COMBAT_SPACES else ""


def _deployed_from_strength(res: dict, seat: str, t: float) -> str:
    s = res.get(seat, [])
    a, b = strength_at(s, t - 4), strength_at(s, t + SETTLE)
    if a is None or b is None or b <= a:
        return ""
    d = b - a
    if d % 2 == 0 and d <= 12:
        return f", deployed {d // 2}"
    if d % 3 == 0 and d <= 9:
        return f", deployed {d // 3} sandworm{'s' if d > 3 else ''}"
    return f", strength +{d}"


def combat_line(rnd: dict, next_t: float, res: dict, label: dict) -> str:
    if rnd.get("combat"):
        parts = []
        for c in sorted(rnd["combat"], key=lambda c: c["place"]):
            who = label.get(c["player"], c["player"])
            parts.append(f"{ORD.get(c['place'], c['place'])}{' (tie)' if c.get('tie') else ''} {who}")
        return "Combat: " + ", ".join(parts)
    return "Combat: ?"
    final = {}
    for seat, series in res.items():
        v = strength_at(series, next_t - 2)
        if v:
            final[seat] = v
    if not final:
        return "Combat: ?"
    ranked = sorted(final.items(), key=lambda kv: -kv[1])
    parts, place = [], 1
    for i, (seat, v) in enumerate(ranked[:3]):
        tie = any(v == w for s2, w in ranked if s2 != seat)
        parts.append(f"{ORD[place]}{' (tie)' if tie else ''} "
                     f"{label.get(seat_player(seat, label), seat)} [{v}]")
        place += 1
    return "Combat (from strength on screen): " + ", ".join(parts)


def seat_player(seat: str, label: dict) -> str:
    return next((p for p in label if p.endswith(f"({seat})")), seat)


def render(g: dict) -> str:
    label = {}
    for p in g["players"]:
        lead = p.get("leader")
        label[p["name"]] = f"{lead} ({p['color']})" if lead else p["name"]
    seat_of = {p["name"]: p["seat"] for p in g["players"]}
    res = g.get("resources", {})
    rounds = sorted(g["rounds"], key=lambda r: r["t"])
    cs = g.get("clean_stats", {})
    lines = [f"{g.get('title') or g['video_id']}",
             f"video: {g.get('url', '')}   game file: {g['video_id']}.json",
             "players: " + ", ".join(
                 f"{label[p['name']]}" + (f" = {p['chat_name']}" if p.get("chat_name") else "")
                 for p in g["players"]),
             "sources: lines without a tag come from the chat log; [video] = read "
             "from the video (less certain); '?' = unknown, fill in; "
             "'deployed ?' = a combat space, troops may have been deployed",
             f"cleaning: {cs}",
             ""]
    acts = sorted((a for a in g["actions"] if a["kind"] in ("agent", "reveal")),
                  key=lambda a: a["t"])
    for i, r in enumerate(rounds):
        nxt = rounds[i + 1]["t"] if i + 1 < len(rounds) else float("inf")
        lines.append(f"=== Round {r['round']} - Conflict: {r.get('conflict') or '?'} ===")
        lines.append(combat_line(r, nxt if nxt != float("inf") else
                                 max([a["t"] for a in acts] or [r["t"]]) + 120,
                                 res, label))
        for a in acts:
            if not (r["t"] <= a["t"] < nxt):
                continue
            who = label.get(a["player"], a["player"])
            if a["kind"] == "reveal":
                lines.append(f"{who}: reveal")
                continue
            dep = deployed(a.get("space") or "")
            src = "" if a.get("source") == "chat" else "  [video]"
            lines.append(f"{who}: {a.get('space') or '?'}, {a.get('card') or '?'}{dep}{src}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    paths = args.games or sorted(glob.glob(os.path.join(GAMES, "*.json")))
    from replay import _card_factory
    factory = _card_factory()
    n = 0
    for p in paths:
        g = json.load(open(p, encoding="utf-8"))
        if g.get("source") != "vision":
            continue
        g = clean_game(g, factory)
        out = os.path.join(OUT, f"{g['video_id']}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(render(g))
        n += 1
    print(f"{n} move lists -> {OUT}")


if __name__ == "__main__":
    main()
