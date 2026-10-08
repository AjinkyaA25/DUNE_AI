#!/usr/bin/env python
"""
Merge a stream's TTS chat log into the vision-built game files.

The chat log (chat_extract.py) records moves exactly: agent card + space,
Imperium buys, contracts, techs, Sardaukar recruits, VP events, combat
placings. The vision scan (scan.py) records what the chat can't: reveals,
intrigues, resources, influence/VP tracks, and seat positions. This tool
combines them, per game:

  1. split the chat into games at "Phase: leader selection" / game start
  2. pair each chat game with the vision game it overlaps most in time
  3. parse the chat game (parse_log.parse)
  4. map chat player names -> vision seats by matching their agent turns
     (same space within MATCH_S seconds); names are often misread, moves
     are not
  5. merged actions = chat agent turns, buys, contracts, techs, recruits,
     VP events (players renamed to their seat) + vision reveals and
     intrigue events + vision agent turns the chat missed ([GAP?]
     stretches). Vision buys are dropped when the chat has the buys
     (the video's Row reading over-counts in streams)
  6. rounds keep the vision conflicts; chat combat placings are attached

Writes the merged file over games/<id>_gN.json (the vision-only version is
kept as games/.vision/<id>_gN.json). Idempotent: re-merging starts from the
kept vision version.

Usage:
  python video_scrape/merge_chat.py                    # every stream game
  python video_scrape/merge_chat.py KKWlGCFb8Eo
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parse_log as P  # noqa: E402

GAMES = os.path.join(HERE, "games")
VISION = os.path.join(GAMES, ".vision")
CHAT_DIRS = [os.path.join(HERE, d) for d in ("raw_streams", "raw_streams2", "raw_streams3", "raw")]
LOBBY_S = 900.0           # s of lobby (colour picks) before a game start
REAPPEAR_S = 90.0        # s: a "bought" card back in the Row = flicker
MATCH_S = 20.0            # agent turn in chat vs video: same space within
GAME_START = re.compile(r"phase\s*:?\s*leader\s*selection|phase\s*:?\s*game\s*start",
                        re.I)
CHAT_KINDS = ("agent", "buy", "contract", "fulfill_contract", "tech", "sardaukar",
              "swordmaster", "vp", "manual")


def ts(s: str) -> float:
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def _n(txt: str) -> str:
    return re.sub(r"[^a-z0-9#]", "", (txt or "").lower())


def game_starts(lines: list[str]) -> list[float]:
    """Times the games start: 'Phase: game start' / 'round start #1' (OCR
    drops spaces), grouped within 10 minutes."""
    out = []
    for l in lines:
        if "	" not in l:
            continue
        t, txt = l.split("	", 1)
        n = _n(txt)
        if "gamestart" in n or re.search(r"roundstart#?1(?!\d)", n):
            t = ts(t)
            if not out or t - out[-1] > 600:
                out.append(t)
    return out


def chat_segments(chat_path: str) -> list[tuple[float, float, list[str]]]:
    """(game start, segment end, lines) per game. A segment begins
    LOBBY_S before its game start (to include the lobby's colour picks)."""
    lines = open(chat_path, encoding="utf-8").read().splitlines()
    starts = game_starts(lines)
    segs = []
    for k, st in enumerate(starts):
        a = st - LOBBY_S
        b = starts[k + 1] - LOBBY_S if k + 1 < len(starts) else float("inf")
        seg = [l for l in lines if "	" in l and a <= ts(l.split("	", 1)[0]) < b]
        tt = [ts(l.split("	", 1)[0]) for l in seg]
        if tt:
            segs.append((st, max(tt), seg))
    return segs


COLOUR_LINE = re.compile(r"^(.*?)[\s.]*(?:is|ts|its|s|l)?[\s.]*colo\w*[\s.]*"
                         r"(red|blue|green|yellow)", re.I)
TURN_LINE = re.compile(r"^(.+?)['’`]?\s*s[\s.]*tur", re.I)


def _sim(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, _n(a).replace("u", "v"),
                                   _n(b).replace("u", "v")).ratio()


def colours_by_name(seg: list[str], start: float) -> dict[str, str]:
    """Canonical player name -> colour, from the lobby's last colour picks.
    Canonical names = the most common spellings of '<name>'s turn' lines."""
    turns = collections.Counter()
    for l in seg:
        t, txt = l.split("	", 1)
        m = TURN_LINE.match(txt.strip())
        if m and start <= ts(t) <= start + 3600:
            turns[m.group(1).strip()] += 1
    canon: list[str] = []
    for name, _ in turns.most_common():
        if len(_n(name)) >= 2 and all(_sim(name, c) < 0.75 for c in canon):
            canon.append(name)
    canon = canon[:4]
    out = {}
    for l in seg:
        t, txt = l.split("	", 1)
        if ts(t) > start + 60:
            continue
        m = COLOUR_LINE.match(txt.strip())
        if not m or not canon:
            continue
        best = max(canon, key=lambda c: _sim(m.group(1), c))
        if _sim(m.group(1), best) >= 0.6:
            out[best] = m.group(2).capitalize()     # last pick wins
    return out


def parse_segment(seg: list[str], vid: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, f"{vid}.chat.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(seg) + "\n")
        return P.parse(p)


def game_span(g: dict) -> tuple[float, float]:
    ts_ = [a["t"] for a in g["actions"]] + [r["t"] for r in g["rounds"]]
    return (min(ts_), max(ts_)) if ts_ else (0.0, 0.0)


def seat_of_player(g: dict) -> dict[str, str]:
    return {p["name"]: p["seat"] for p in g["players"]}


def map_names(chat: dict, vis: dict, colours: dict[str, str]) -> dict[str, str]:
    """chat player name -> vision player name. By colour: the lobby's colour
    picks vs the colours of the seats on screen; agent-turn matching fills
    in anyone left."""
    seat_by_colour = {p["color"]: p["name"] for p in vis["players"]}
    out = {}
    for a in chat["actions"]:
        cn = a["player"]
        if cn in out or not colours:
            continue
        best = max(colours, key=lambda c: _sim(cn, c))
        if _sim(cn, best) >= 0.6 and colours[best] in seat_by_colour:
            out[cn] = seat_by_colour[colours[best]]
    if len(set(out.values())) >= 4:
        return out
    return {**_map_by_agents(chat, vis, set(out), set(out.values())), **out}


def _map_by_agents(chat: dict, vis: dict, done_c: set, done_v: set) -> dict[str, str]:
    vis_agents = [a for a in vis["actions"] if a["kind"] == "agent" and a.get("space")]
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for a in chat["actions"]:
        if a["kind"] != "agent" or not a.get("space"):
            continue
        near = [v for v in vis_agents if v["space"] == a["space"]
                and abs(v["t"] - a["t"]) <= MATCH_S]
        for v in near:
            votes[a["player"]][v["player"]] += 1.0 / len(near)
    out, used = {}, set(done_v)
    pairs = sorted(((c, cn, vn) for cn, cnt in votes.items() for vn, c in cnt.items()),
                   reverse=True)
    for c, cn, vn in pairs:
        if cn in out or cn in done_c or vn in used or c < 1.5:
            continue
        out[cn] = vn
        used.add(vn)
    return out


def merge_one(vis_path: str, segs) -> dict | None:
    vis = json.load(open(vis_path, encoding="utf-8"))
    a0, a1 = game_span(vis)
    lines = segs          # all chat lines of the video
    lobby = [l for l in lines if a0 - LOBBY_S <= ts(l.split("\t", 1)[0]) < a0 + 60]
    game = [l for l in lines if a0 - 60 <= ts(l.split("\t", 1)[0]) <= a1 + 60]
    if len(game) < 50:
        return None
    vid = vis["video_id"]
    chat = parse_segment(lobby + [l for l in game if l not in set(lobby)], vid)
    chat["actions"] = [a for a in chat["actions"] if a0 - 30 <= a["t"] <= a1 + 60]
    colours = colours_by_name(lobby + game, a0)
    names = map_names(chat, vis, colours)
    names = {k: v for k, v in names.items()}
    if len(set(names.values())) < 3:
        # no chat merge: still drop the video's Row-flicker 'buys'
        buys = [a for a in vis["actions"] if a["kind"] == "buy"]
        keep = {id(a) for a in filter_row_buys(vid, buys)}
        vis["actions"] = [a for a in vis["actions"]
                          if a["kind"] != "buy" or id(a) in keep]
        vis.setdefault("merge_notes", []).append(
            f"chat not merged: only {len(set(names.values()))} seats matched "
            f"(lobby colours {colours})")
        return vis

    rounds = sorted(vis["rounds"], key=lambda r: r["t"])

    def round_at(t):
        return max((r["round"] for r in rounds if r["t"] <= t + 1), default=1)

    merged = []
    chat_agents = []
    n_chat_buys = 0
    for a in chat["actions"]:
        if a["kind"] not in CHAT_KINDS or a["player"] not in names:
            continue
        b = dict(a)
        b["player"] = names[a["player"]]
        b["round"] = round_at(a["t"])
        b["source"] = "chat"
        merged.append(b)
        if a["kind"] == "agent":
            chat_agents.append(b)
        if a["kind"] == "buy":
            n_chat_buys += 1
    kept_vis_agents = kept_vis_buys = 0
    n_chat = collections.Counter((c["player"], c["round"]) for c in chat_agents)
    vis_by = collections.defaultdict(list)
    for a in vis["actions"]:
        if a["kind"] == "agent":
            vis_by[a["player"], a["round"]].append(a)
    for key, lst in vis_by.items():
        extra = len(lst) - n_chat.get(key, 0)
        if extra <= 0:
            continue
        # keep the video turns farthest from any chat turn of that player
        def gap(a):
            return min((abs(c["t"] - a["t"]) for c in chat_agents
                        if c["player"] == a["player"]), default=1e9)
        for a in sorted(lst, key=gap, reverse=True)[:extra]:
            merged.append(dict(a, source="video",
                               flags=a["flags"] + ["chat missed this turn"]))
            kept_vis_agents += 1
    real_vis_buys = filter_row_buys(vid, [a for a in vis["actions"] if a["kind"] == "buy"])
    for a in vis["actions"]:
        k = a["kind"]
        if k in ("reveal", "intrigue", "gain_intrigue"):
            merged.append(dict(a, source="video"))
    if n_chat_buys < 10:
        for a in real_vis_buys:
            merged.append(dict(a, source="video"))
            kept_vis_buys += 1
    merged.sort(key=lambda a: a["t"])
    for i, a in enumerate(merged):
        a["id"] = i

    # combat placings from the chat, names -> seats (leader names too)
    leader_to = {p.get("leader"): names.get(p["name"]) for p in chat["players"]
                 if p.get("leader") and p["name"] in names}
    for cr in chat["rounds"]:
        r = round_at(cr["t"])
        tgt = next((x for x in vis["rounds"] if x["round"] == r), None)
        if tgt is None or not cr.get("combat"):
            continue
        comb = []
        for c in cr["combat"]:
            who = names.get(c["player"]) or leader_to.get(c["player"])
            if who and all(x["player"] != who for x in comb):
                comb.append(dict(c, player=who))
        if comb:
            tgt["combat"] = comb

    # player real names and leaders, by seat
    rev = {v: k for k, v in names.items()}
    for p in vis["players"]:
        cn = rev.get(p["name"])
        if cn:
            cp = next((x for x in chat["players"] if x["name"] == cn), {})
            p["chat_name"] = cn
            if cp.get("leader"):
                p["leader"] = cp["leader"]
    if chat.get("result"):
        vis["result"] = [dict(r, player=names.get(r["player"], r["player"]))
                         for r in chat["result"]]
    vis["actions"] = merged
    vis["chat_merged"] = True
    vis["merge_stats"] = {"chat_actions": sum(1 for a in merged if a.get("source") == "chat"),
                          "video_agent_turns_kept": kept_vis_agents,
                          "video_buys_kept": kept_vis_buys,
                          "seats_matched": len(set(names.values())),
                          "lobby_colours": colours}
    return vis


def filter_row_buys(vid: str, buys: list[dict]) -> list[dict]:
    """Drop video 'buys' whose card shows up in the Imperium Row again within
    REAPPEAR_S (the Row reading flickered; nothing was bought)."""
    tl_path = next((os.path.join(d, f"{re.sub(r'_g[0-9]+$', '', vid)}.timeline.json")
                    for d in CHAT_DIRS
                    if os.path.exists(os.path.join(d, f"{re.sub(r'_g[0-9]+$', '', vid)}.timeline.json"))),
                   None)
    if tl_path is None:
        return buys
    sys.path.insert(0, HERE)
    import vision as V
    lib = _lib()
    row = [(e["t"], {_n(lib.name(c["card"])) for c in e["cards"]})
           for e in json.load(open(tl_path, encoding="utf-8"))
           if e["seat"] == "G" and e["region"] == "row"]
    out = []
    for b in buys:
        name = _n(b.get("card", ""))
        back = any(b["t"] < t <= b["t"] + REAPPEAR_S and name in cards for t, cards in row)
        if not back:
            out.append(b)
    return out


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        import vision as V
        _LIB = V.Library()
    return _LIB


def find_chat(vid: str) -> str | None:
    for d in CHAT_DIRS:
        p = os.path.join(d, f"{vid}.chat.txt")
        if os.path.exists(p):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    args = ap.parse_args()
    os.makedirs(VISION, exist_ok=True)
    files = sorted(glob.glob(os.path.join(GAMES, "*_g*.json")))
    by_vid = collections.defaultdict(list)
    for f in files:
        vid = re.sub(r"_g\d+$", "", os.path.splitext(os.path.basename(f))[0])
        if not args.videos or vid in args.videos:
            by_vid[vid].append(f)
    for vid, fs in sorted(by_vid.items()):
        chat = find_chat(vid)
        if chat is None:
            print(f"{vid}: no chat log")
            continue
        segs = [l for l in open(chat, encoding="utf-8").read().splitlines()
                if "\t" in l]
        for f in fs:
            keep = os.path.join(VISION, os.path.basename(f))
            if not os.path.exists(keep):
                shutil.copy(f, keep)
            g = merge_one(keep, segs)
            if g is None:
                print(f"{os.path.basename(f)}: no chat game overlaps")
                continue
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(g, fh, indent=1, ensure_ascii=False)
            kinds = collections.Counter(a["kind"] for a in g["actions"])
            print(f"{os.path.basename(f)}: {g.get('merge_stats') or g.get('merge_notes')} "
                  f"{dict(kinds)} winner={g['result'][0]['player'] if g.get('result') else '?'}")


if __name__ == "__main__":
    main()
