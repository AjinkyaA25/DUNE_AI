#!/usr/bin/env python
"""
Turn an OCR'd TTS chat log (chat_extract.py output) into a structured,
editable game file.

The game file is JSON: players, result, and a flat list of `actions`. Each
action is one thing a player did (agent placement, buy, contract, tech,
Sardaukar recruit, ...) with the resource changes the mod logged right after
it collected as `effects`. Every action keeps its video timestamp and the raw
OCR lines it came from, so edits in the editor can be checked against the
video. Actions the parser is unsure of carry `flags`.

Usage:
  python video_scrape/parse_log.py video_scrape/raw/vjcSLuiye7A.chat.txt
  python video_scrape/parse_log.py --all          # every *.chat.txt in raw/
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(HERE, "raw")
GAMES = os.path.join(HERE, "games")

ME = "Dinosaur11"

SPACES = [
    "Arrakeen", "Spice Refinery", "Research Station", "Sietch Tabr",
    "Sardaukar", "Dutiful Service", "Heighliner", "Deliver Supplies",
    "Espionage", "Secrets", "Desert Tactics", "Fremkit", "High Council",
    "Swordmaster", "Imperial Privilege", "Assembly Hall", "Gather Support",
    "Accept Contract", "Shipping", "Imperial Basin", "Hagga Basin",
    "Deep Desert", "Tuek's Sietch", "Tech Negotiation",
]
SARDAUKAR_SKILLS = ["Hardy", "Driven", "Charismatic", "Canny", "Fierce",
                    "Loyal", "Desperate"]
COLORS = ["Red", "Blue", "Green", "Yellow"]
VP_SOURCES = ["Emperor Friendship", "Spacing Guild Friendship",
              "Bene Gesserit Friendship", "Fremen Friendship",
              "Emperor Alliance", "Spacing Guild Alliance",
              "Bene Gesserit Alliance", "Fremen Alliance",
              "The Spice Must Flow", "Crysknife Objective",
              "Ornithopter Objective", "Desert Mouse Objective",
              "Muad'Dib Objective",
              "High Council", "Conflict"]
FACTIONS = {"emperor": "Emperor", "spacing guild": "Spacing Guild",
            "bene gesserit": "Bene Gesserit", "fremen": "Fremen"}


def _load_card_names() -> list[str]:
    with open(os.path.join(ROOT, "src", "data", "uprising_cards.json"),
              encoding="utf-8") as f:
        return sorted({c["name"] for c in json.load(f)})


CARDS = _load_card_names()


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _ocr_fix(s: str) -> str:
    """Common rapidocr confusions on the TTS chat font."""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return s


def best_name(raw: str, vocab: list[str], cutoff: float = 0.8
              ) -> tuple[str, bool]:
    """Snap an OCR'd name to the closest vocabulary entry.

    Returns (name, verified). Unverified names (e.g. Bloodlines cards the
    engine doesn't know yet) are returned cleaned up but as-is.
    """
    raw = raw.strip(" .,'\"")
    k = _key(raw)
    if not k:
        return raw, False
    # 'u' for 'v' is the single most common misread ("Seruice", "Driuen")
    candidates = {k, k.replace("u", "v")}
    best, score = None, 0.0
    for name in vocab:
        nk = _key(name)
        for c in candidates:
            r = difflib.SequenceMatcher(None, c, nk).ratio()
            if r > score:
                best, score = name, r
    if best and score >= cutoff:
        return best, True
    return raw, False


# ---------------------------------------------------------------- patterns
NUM = r"(\d+)"
P = {
    "turn_of": re.compile(r"^(?P<name>.+?)\s*'\s*s\s*turn\b", re.I),
    "turn_leader": re.compile(r"^Tur[nm][:;e\s]\s*(?P<leader>.+)$", re.I),
    "agent": re.compile(r"^Sending an agent to\s*:?\s*(?P<space>[^(]+)"
                        r"(?:\((?P<card>[^)]*)\)?)?", re.I),
    "buy": re.compile(r"^Acquire Imperi\w* ?/?card\s*:?\s*(?P<card>.+)$", re.I),
    "contract": re.compile(r"^Acquired contract\s*:?\s*(?P<name>.+)$", re.I),
    "fulfill": re.compile(r"^Fulfilled contract\s*:?\s*(?P<name>.+)$", re.I),
    "tech_cost": re.compile(r"^Acquire tech for (?P<n>\d+) spice", re.I),
    "tech": re.compile(r"^Acquire tech\s*:\s*(?P<name>.+)$", re.I),
    "sard_recruit": re.compile(r"^Recruit Sardaukar Commander\s*:?\s*"
                               r"(?P<space>.+)$", re.I),
    "sard_skill": re.compile(r"^Acquired Sardaukar Commander sk\w+\s*:?\s*"
                             r"(?P<skill>.+)$", re.I),
    "swordmaster": re.compile(r"^Recruit Swordmaster", re.I),
    "vp": re.compile(r"^Gaining VP\s*\((?P<src>.+?)\)?\.?$", re.I),
    "round": re.compile(r"^Phase\s*:\s*round\s*start\s*#?\s*(?P<n>\d+)", re.I),
    "phase": re.compile(r"^Phase\s*:\s*(?P<p>[a-z ]+)$", re.I),
    "conflict": re.compile(r"^Round combat is\s*:?\s*(?P<c>.+)$", re.I),
    "placing": re.compile(r"^\W*(?P<pl>\d)\s*(?:st|nd|rd|th)\s*(?P<tie>ex "
                          r"aequo)?\s*:\s*(?P<leader>[^-–→]+)$", re.I),
    "elo": re.compile(r"^(?P<pl>\d)\s*(?:st|nd|rd|th)\s*:\s*(?P<name>.+?)\s*-\s*"
                      r"(?P<a>\d{3,4})\s*\D+(?P<b>\d{3,4})", re.I),
    "color": re.compile(r"^(?P<name>.+?)\s*is\s+colou?r\s*(?P<color>[A-Za-z]+)",
                        re.I),
    "picked_leader": re.compile(r"^(?P<name>.+?)\s+picked leader\s+"
                                r"(?P<leader>.+?)\.?$", re.I),
    "picked_pos": re.compile(r"^(?P<name>.+?)\s+picked position\s*(?P<n>\d)",
                             re.I),
    "manual": re.compile(r"^(?P<leader>.+?)\s+(?P<verb>spent|recei\w*)\s+"
                         r"(?P<n>\d+)\s*(?P<what>.+?)\s*manually", re.I),
    "fixed": re.compile(r"^Fixed\s+(?P<rest>.+)$", re.I),
    "delta": re.compile(r"^(?P<sign>[+\-])\s*(?P<n>\d+)\s*(?P<what>.+?)\.?$"),
    "influence": re.compile(r"influence with the (?P<f>[a-z ]+)", re.I),
    "draw": re.compile(r"^Draw\s*(?P<n>\d+)\s*(?P<intrigue>intrigue)?\s*card",
                       re.I),
    "transfer": re.compile(r"^Transf\w+ \w+ (?P<n>\d+)\W*(?P<what>.+?)\s*:\s*"
                           r"(?P<route>.+)$", re.I),
    "chat": re.compile(r"^(?P<name>[A-Za-z0-9_ .]{3,24})\s*[:;]\s*(?P<msg>.+)$"),
}


def _norm_player(raw: str, known: list[str]) -> str:
    raw = raw.strip()
    if not known:
        return raw
    best = difflib.get_close_matches(_key(raw), [_key(k) for k in known],
                                     n=1, cutoff=0.7)
    if best:
        return next(k for k in known if _key(k) == best[0])
    return raw


def _resource(what: str) -> str | None:
    w = what.lower()
    for word, res in (("solari", "solari"), ("spice", "spice"),
                      ("water", "water"), ("persuasion", "persuasion"),
                      ("sword", "swords"), ("troop", "troops"),
                      ("intrigue", "intrigue"), ("card", "cards")):
        if word in w:
            return res
    m = P["influence"].search(what)
    if m:
        f = m.group("f").lower().rstrip("s ").strip()
        for k, v in FACTIONS.items():
            if f.startswith(k[:5]):
                return f"inf_{v}"
    return None


def parse(chat_path: str) -> dict:
    vid = os.path.basename(chat_path).split(".")[0]
    info_path = os.path.join(RAW, f"{vid}.info.json")
    title = ""
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            title = json.load(f).get("title", "")

    lines: list[tuple[float, str]] = []
    with open(chat_path, encoding="utf-8") as f:
        for row in f:
            ts, _, text = row.rstrip("\n").partition("\t")
            mm, ss = ts.split(":")
            lines.append((int(mm) * 60 + float(ss), _ocr_fix(text.strip())))

    # --- pass 1: players (name, color, leader, seat) from setup lines
    players: dict[str, dict] = {}
    def pkey(name: str) -> str:
        """Player key tolerant of OCR misspellings (u/v, dropped letters)."""
        k = _key(name).replace("u", "v")
        for existing in players:
            if difflib.SequenceMatcher(None, k, existing).ratio() >= 0.85:
                return existing
        return k

    spellings: dict[str, dict[str, int]] = {}
    for t, text in lines:
        m = P["turn_of"].match(text)
        if m:
            nm = m.group("name").strip()
            k = pkey(nm)
            if k in players:
                spellings.setdefault(k, {}).setdefault(nm, 0)
                spellings[k][nm] += 1
        for key in ("color", "picked_leader", "picked_pos"):
            m = P[key].match(text)
            if m:
                name = m.group("name").strip()
                p = players.setdefault(pkey(name), {"name": name})
                if key == "color":
                    p["color"] = m.group("color").capitalize()
                    p["_color_t"] = t
                elif key == "picked_leader":
                    ld = re.sub(r"\s*\(.*$", "", m.group("leader")).strip()
                    p.setdefault("leader", ld)
                else:
                    p["seat"] = int(m.group("n"))
    # display the spelling OCR produced most often ("Dinosaur11", not
    # "Dinosaur 1 1"), then fill leaders the setup lines missed from the
    # "<name>'s turn." -> "Turn: <leader>" pairs seen during play
    for k, p in players.items():
        if spellings.get(k):
            p["name"] = max(spellings[k], key=spellings[k].get)
    known = [p["name"] for p in players.values()]
    votes: dict[str, dict[str, int]] = {}
    for (_, a), (_, b) in zip(lines, lines[1:]):
        m1, m2 = P["turn_of"].match(a), P["turn_leader"].match(b)
        if m1 and m2 and "?" not in b:
            k = pkey(m1.group("name"))
            ld = m2.group("leader").strip(" .")
            votes.setdefault(k, {}).setdefault(ld, 0)
            votes[k][ld] += 1
    for k, p in players.items():
        if "leader" not in p and votes.get(k):
            p["leader"] = max(votes[k], key=votes[k].get)
    # player-name overlays ("... Benten") get glued onto log lines
    name_tail = re.compile(r"\s+(?:%s)\s*$" % "|".join(
        re.escape(n) for n in known)) if known else None
    leader_to_player = {_key(p["leader"]): p["name"]
                        for p in players.values() if "leader" in p}
    leaders = [p["leader"] for p in players.values() if "leader" in p]

    # --- pass 2: actions
    actions: list[dict] = []
    result: list[dict] = []
    rounds: list[dict] = []
    cur_player = None
    turn_no = 0
    rnd = 0
    last: dict | None = None
    prev_text = None
    pending_tech_cost = None

    def new(kind: str, t: float, raw: str, **kw) -> dict:
        nonlocal last
        a = {"id": len(actions), "t": round(t, 1), "round": rnd,
             "turn": turn_no, "player": cur_player, "kind": kind, **kw,
             "effects": [], "raw": [raw], "flags": []}
        actions.append(a)
        last = a
        return a

    i = 0
    while i < len(lines):
        t, text = lines[i]
        i += 1
        if name_tail:
            text = name_tail.sub("", text)
        if not text or text == prev_text:
            continue            # stitched duplicate
        prev_text = text

        if text == "[GAP?]":
            if last is not None and rnd > 0:
                last["flags"].append("log gap after this action: check video")
            continue

        m = P["turn_of"].match(text)
        if m:
            cur_player = _norm_player(m.group("name"), known)
            turn_no += 1
            last = None
            continue
        m = P["turn_leader"].match(text)
        if m and "?" not in text:
            who = leader_to_player.get(_key(best_name(m.group("leader"),
                                                      leaders, 0.7)[0]))
            if who and who != cur_player:
                # "<name>'s turn." line was lost to OCR; the leader line
                # still tells us whose turn it is
                cur_player = who
                turn_no += 1
                last = None
            continue

        m = P["round"].match(text)
        if m:
            n = int(m.group("n"))
            if n != rnd:
                rnd = n
                rounds.append({"round": n, "t": round(t, 1), "conflict": None})
            last = None
            continue
        m = P["conflict"].match(text)
        if m and rounds:
            rounds[-1]["conflict"] = m.group("c").strip(" \"'.")
            continue

        m = P["agent"].match(text)
        if m:
            raw = text
            # long lines wrap: "Sending an agent to: Dutiful Service (Command"
            # + next line "Center)."
            if "(" in text and ")" not in text and i < len(lines):
                nt = lines[i][1]
                if not any(p.match(nt) for k, p in P.items()
                           if k not in ("chat", "delta")) and \
                        not nt.startswith(("+", "-")):
                    raw = text + " " + nt
                    i += 1
                    m = P["agent"].match(raw)
            space, sv = best_name(m.group("space"), SPACES)
            card_raw = (m.group("card") or "").strip()
            card, cv = best_name(card_raw, CARDS) if card_raw else ("", False)
            a = new("agent", t, raw, space=space, card=card)
            if not sv:
                a["flags"].append(f"unknown space '{space}'")
            if card_raw and not cv:
                a["flags"].append(f"card '{card}' not in engine card list")
            if not card_raw or ")" not in raw:
                a["flags"].append("card name may be cut off")
            continue

        m = P["buy"].match(text)
        if m:
            card, ok = best_name(m.group("card"), CARDS)
            a = new("buy", t, text, card=card)
            if not ok:
                a["flags"].append(f"card '{card}' not in engine card list")
            continue
        m = P["contract"].match(text)
        if m:
            new("contract", t, text, name=m.group("name").strip(" \"'.“”"))
            continue
        m = P["fulfill"].match(text)
        if m:
            new("fulfill_contract", t, text,
                name=m.group("name").strip(" \"'.“”"))
            continue
        m = P["tech_cost"].match(text)
        if m:
            pending_tech_cost = int(m.group("n"))
            continue
        m = P["tech"].match(text)
        if m:
            new("tech", t, text, name=m.group("name").strip(" \"'.“”"),
                cost=pending_tech_cost)
            pending_tech_cost = None
            continue
        m = P["sard_recruit"].match(text)
        if m:
            space, _ = best_name(m.group("space"), SPACES)
            new("sardaukar", t, text, space=space, skill=None)
            continue
        m = P["sard_skill"].match(text)
        if m:
            skill, ok = best_name(m.group("skill"), SARDAUKAR_SKILLS, 0.6)
            if last is not None and last["kind"] == "sardaukar" and \
                    last["player"] == cur_player:
                last["skill"] = skill
                last["raw"].append(text)
            else:
                new("sardaukar", t, text, space=None, skill=skill)
            continue
        if P["swordmaster"].match(text):
            new("swordmaster", t, text)
            continue
        m = P["vp"].match(text)
        if m:
            new("vp", t, text,
                source=best_name(m.group("src"), VP_SOURCES, 0.75)[0])
            continue

        m = P["placing"].match(text)
        if m and rounds:
            lead, _ = best_name(m.group("leader"), leaders, 0.7)
            rounds[-1].setdefault("combat", []).append(
                {"place": int(m.group("pl")), "tie": bool(m.group("tie")),
                 "player": leader_to_player.get(_key(lead), lead)})
            continue
        m = P["elo"].match(text)
        if m:
            name = _norm_player(m.group("name"), known)
            if not any(r["player"] == name for r in result):
                result.append({"place": int(m.group("pl")), "player": name,
                               "elo_before": int(m.group("a")),
                               "elo_after": int(m.group("b"))})
            continue

        m = P["manual"].match(text)
        if m:
            lead, _ = best_name(m.group("leader"), leaders, 0.7)
            who = leader_to_player.get(_key(lead), lead)
            sign = -1 if m.group("verb").lower() == "spent" else 1
            a = new("manual", t, text, resource=_resource(m.group("what")),
                    amount=sign * int(m.group("n")))
            a["player"] = who
            a["flags"].append("manual adjustment (possible fix/takeback)")
            continue
        m = P["fixed"].match(text)
        if m:
            a = new("manual", t, text, note=m.group("rest"))
            a["flags"].append("manual fix (possible takeback)")
            continue

        m = P["delta"].match(text)
        if m:
            res = _resource(m.group("what"))
            if res and last is not None:
                amt = int(m.group("n")) * (1 if m.group("sign") == "+" else -1)
                last["effects"].append({"res": res, "n": amt})
                last["raw"].append(text)
            continue
        m = P["draw"].match(text)
        if m:
            if last is not None:
                last["effects"].append(
                    {"res": "intrigue" if m.group("intrigue") else "cards",
                     "n": int(m.group("n"))})
                last["raw"].append(text)
            continue
        m = P["transfer"].match(text)
        if m:
            if last is not None:
                what = "sardaukar" if "sard" in m.group("what").lower() \
                    else "troops"
                route = m.group("route").lower()
                dest = "garrison" if "garr" in route else \
                    "tech negotiation" if "tech" in route else route
                last["effects"].append({"res": f"{what}->{dest}",
                                        "n": int(m.group("n"))})
                last["raw"].append(text)
            continue
        # anything else (player chat, mod notices) is ignored

    # sanity flag: two agents inside one turn is usually a takeback
    for a, b in zip(actions, actions[1:]):
        if a["kind"] == b["kind"] == "agent" and a["turn"] == b["turn"]:
            b["flags"].append("second agent in same turn: possible takeback "
                              "of the previous action")
    for a in actions:
        del a["turn"]
    for p in players.values():
        if "color" in p:
            p["color"] = best_name(p["color"][:6], COLORS, 0.5)[0]
    # colors can be re-picked during setup and a re-pick can fall in a log
    # gap: if two players share a color, the stale one gets the leftover
    free = [c for c in COLORS
            if c not in {p.get("color") for p in players.values()}]
    if len(free) == 1:
        by_color: dict[str, list[dict]] = {}
        for p in players.values():
            by_color.setdefault(p.get("color"), []).append(p)
        for c, ps in by_color.items():
            if c and len(ps) == 2:
                min(ps, key=lambda p: p["_color_t"])["color"] = free[0]
    for p in players.values():
        p.pop("_color_t", None)

    me = next((p for p in players.values() if _key(p["name"]) == _key(ME)),
              None)
    return {
        "video_id": vid,
        "url": f"https://youtu.be/{vid}",
        "title": title,
        "me": me["name"] if me else ME,
        "players": sorted(players.values(), key=lambda p: p.get("seat", 9)),
        "rounds": rounds,
        "result": sorted(result, key=lambda r: r["place"]),
        "actions": actions,
        "edited": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite games you have already edited")
    args = ap.parse_args()
    paths = args.chat or []
    if args.all:
        paths = sorted(glob.glob(os.path.join(RAW, "*.chat.txt")))
    os.makedirs(GAMES, exist_ok=True)
    for p in paths:
        g = parse(p)
        out = os.path.join(GAMES, f"{g['video_id']}.json")
        if os.path.exists(out) and not args.force:
            with open(out, encoding="utf-8") as f:
                if json.load(f).get("edited"):
                    print(f"skip {out}: already edited (use --force)")
                    continue
        with open(out, "w", encoding="utf-8") as f:
            json.dump(g, f, indent=1, ensure_ascii=False)
        mine = [a for a in g["actions"] if a["player"] == g["me"]]
        nflag = sum(bool(a["flags"]) for a in g["actions"])
        print(f"{out}: {len(g['players'])} players, {len(g['rounds'])} rounds,"
              f" {len(g['actions'])} actions ({len(mine)} yours),"
              f" {nflag} flagged")


if __name__ == "__main__":
    main()
