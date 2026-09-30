"""
Cleaning applied to scraped game files before they are used (text move
lists, engine replay). Pure functions on the game dict; the files on disk
are left as extracted.

  names       leader names snapped to the known leaders ('Shaddam NV' ->
              'Shaddam Corrino IV'), card names to engine cards
              ('Dialomgcu' -> 'Diplomacy')
  setup       round-1 readings before the first real agent turn are the
              opening-hand set-aside / lobby leftovers: dropped
  reveals     a reveal logged before that player's last agent turn of the
              round is a stale reading: dropped; one reveal per player/round
  duplicates  the same player, card and space logged twice within 25 s
              (chat lines re-read after a scroll): one kept
  blanks      video agent turns with neither a space nor a readable card:
              dropped (nothing to learn from, nothing to fill in)
"""
from __future__ import annotations

import difflib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_DOWN = "face-down card"
DUP_S = 25.0

_LEADER_IDS = (
    "glossuRabban", "ilbanRichese", "helenaRichese", "letoAtreides", "paulAtreides",
    "arianaThorvald", "memnonThorvald", "armandEcaz", "ilesaEcaz", "rhomburVernius",
    "tessiaVernius", "yunaMoritani", "hundroMoritani", "stabanTuek", "amberMetulli",
    "gurneyHalleck", "margotFenring", "irulanCorrino", "jessica", "feydRauthaHarkonnen",
    "shaddamCorrino", "muadDib", "vladimirHarkonnen", "bl_Chani", "bl_Duncan",
    "bl_Esmar", "bl_Hasimir", "bl_Kota", "bl_Liet", "bl_Mohiam", "bl_Piter", "bl_Yrkoon")
_LEADERS = None


def _n(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def leader_names() -> list[str]:
    global _LEADERS
    if _LEADERS is None:
        idx = json.load(open(os.path.join(HERE, "cards", "index.json"), encoding="utf-8"))
        out = []
        for i in _LEADER_IDS:
            names = idx.get(i, {}).get("names") or []
            if names:
                out.append(names[0])
        _LEADERS = out
    return _LEADERS


def snap_leader(name: str | None) -> str | None:
    if not name:
        return name
    known = leader_names()
    base = re.sub(r"\s*\(community\)", "", name, flags=re.I).strip()
    table = {_n(k): k for k in known}
    m = difflib.get_close_matches(_n(base), list(table), n=1, cutoff=0.7)
    return table[m[0]] if m else name


def clean_game(g: dict, factory: dict | None = None) -> dict:
    """Return a cleaned copy of a game dict."""
    g = json.loads(json.dumps(g))
    stats = {"setup_noise": 0, "stale_reveals": 0, "duplicates": 0, "blanks": 0,
             "names_fixed": 0}
    for p in g["players"]:
        if p.get("leader"):
            fixed = snap_leader(p["leader"])
            if fixed != p["leader"]:
                stats["names_fixed"] += 1
                p["leader"] = fixed
    acts = sorted(g["actions"], key=lambda a: a["t"])
    if factory:
        from replay import resolve_name, norm
        for a in acts:
            c = a.get("card")
            if not c or c == FACE_DOWN:
                continue
            k = resolve_name(c, factory)
            if k not in factory:              # badly garbled: looser match
                m = difflib.get_close_matches(norm(c), list(factory), n=1, cutoff=0.62)
                k = m[0] if m else k
            if k in factory:
                if factory[k].name != c:
                    a["card"] = factory[k].name
                    stats["names_fixed"] += 1
            elif a.get("source") != "chat":
                # a video reading of a card outside the pool is a misread
                a["card"] = "?"
                stats["unknown_video_cards"] = stats.get("unknown_video_cards", 0) + 1

    # setup noise: before the first agent turn with a known space in round 1
    first = min((a["t"] for a in acts if a["kind"] == "agent" and a.get("space")
                 and a["round"] == 1), default=None)
    keep = []
    for a in acts:
        if first is not None and a["round"] == 1 and a["t"] < first - 1 \
                and a["kind"] in ("agent", "reveal", "buy", "intrigue", "gain_intrigue"):
            stats["setup_noise"] += 1
            continue
        if a["kind"] == "agent" and not a.get("space") and \
                (not a.get("card") or a["card"] == FACE_DOWN):
            stats["blanks"] += 1
            continue
        keep.append(a)
    acts = keep

    # duplicates
    keep = []
    for a in acts:
        if a["kind"] == "agent" and any(
                b["kind"] == "agent" and b["player"] == a["player"]
                and b.get("card") == a.get("card") and b.get("space") == a.get("space")
                and abs(b["t"] - a["t"]) <= DUP_S for b in keep[-12:]):
            stats["duplicates"] += 1
            continue
        keep.append(a)
    acts = keep

    # reveals: one per player per round, after that player's agent turns
    last_agent = {}
    for a in acts:
        if a["kind"] == "agent":
            last_agent[a["player"], a["round"]] = a["t"]
    seen = set()
    keep = []
    for a in reversed(acts):
        if a["kind"] == "reveal":
            key = (a["player"], a["round"])
            if key in seen or a["t"] < last_agent.get(key, -1):
                stats["stale_reveals"] += 1
                continue
            seen.add(key)
        keep.append(a)
    acts = list(reversed(keep))
    for i, a in enumerate(acts):
        a["id"] = i
    g["actions"] = acts
    g["clean_stats"] = stats
    return g
