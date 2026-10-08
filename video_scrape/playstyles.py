#!/usr/bin/env python
"""
Playstyles in the video games: what each player spent on, and when.

Bloodlines players mostly spend on four objectives:
  swordmaster  the 3rd agent (Swordmaster space, solari)
  high_council the council seat (High Council space, solari; techs -1 spice)
  commanders   Sardaukar commanders (2 solari each; a board recruit = a skill)
  techs        tech tiles (spice, bought on green spaces)

For every player in every replayed video game this builds a profile:
round of Swordmaster / High Council, commander recruits (round, skill),
techs (round, name, cost), cards bought, final VP and whether they won.
Exact tech / commander events come from the merged chat log (62 games);
Swordmaster and High Council come from agent spaces (every game).

Each profile gets playstyle tags from its early game (rounds 1-4), the
"what did players do when they went for X" groups are summarised, and the
by-round adoption curve shows how the styles converge late.

Usage:
  python video_scrape/playstyles.py            # -> reports/playstyles.json
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

EARLY = 4            # "early" = by the end of round 4
STYLES = ("swordmaster", "high_council", "commanders", "techs")
SKILLS = ("Hardy", "Driven", "Charismatic", "Canny", "Fierce", "Loyal", "Desperate")


def _skill(raw: str | None) -> str | None:
    """Snap an OCR'd skill name to the 7 skills."""
    if not raw:
        return None
    import difflib
    m = difflib.get_close_matches(raw.strip(' "\'').title(), SKILLS, n=1, cutoff=0.5)
    return m[0] if m else None


def _skill_from_raw(lines: list[str]) -> str | None:
    for ln in lines:
        m = re.search(r"s[hk]?[il1]{1,2}[l1]?s?\s*[:;]\s*\"?([A-Za-z]+)", ln)
        if m:
            s = _skill(m.group(1))
            if s:
                return s
    return None


def profiles_for_game(g: dict, winner_colour: str | None, me_colour: str | None) -> list[dict]:
    colour = {p["name"]: p["color"] for p in g["players"]}
    seat = {p["name"]: p["seat"] for p in g["players"]}
    has_chat = any(a["kind"] in ("tech", "sardaukar", "swordmaster") for a in g["actions"])
    out = {}
    for name in colour:
        out[name] = {"game": g["video_id"], "player": name, "colour": colour[name],
                     "seat": seat[name], "me": colour[name] == me_colour,
                     "won": None if winner_colour is None else colour[name] == winner_colour,
                     "final_vp": (g.get("end_vp") or {}).get("vp", {}).get(colour[name]),
                     "rounds": len(g.get("rounds", [])), "exact_bloodlines": has_chat,
                     "sm_round": None, "hc_round": None, "commanders": [], "techs": [],
                     "buys": [], "spaces": collections.Counter()}
    seen_cmd = set()
    for a in g["actions"]:
        pr = out.get(a.get("player"))
        if pr is None:
            continue
        r = a.get("round") or 0
        k = a["kind"]
        if k == "agent" and a.get("space"):
            pr["spaces"][a["space"]] += 1
            if a["space"] == "Swordmaster" and pr["sm_round"] is None:
                pr["sm_round"] = r
            if a["space"] == "High Council" and pr["hc_round"] is None:
                pr["hc_round"] = r
        elif k == "swordmaster" and pr["sm_round"] is None:
            pr["sm_round"] = r
        elif k == "sardaukar":
            # the chat prints a recruit on 2-3 lines; one event per
            # player / space / round
            key = (a.get("player"), a.get("space"), r)
            if key in seen_cmd:
                continue
            seen_cmd.add(key)
            sk = _skill(a.get("skill")) or _skill_from_raw(a.get("raw", []))
            pr["commanders"].append({"round": r, "space": a.get("space"), "skill": sk})
        elif k == "tech":
            if any(t["name"] == a.get("name") for t in pr["techs"]):
                continue
            pr["techs"].append({"round": r, "name": a.get("name"), "cost": a.get("cost")})
        elif k == "buy" and a.get("card"):
            pr["buys"].append({"round": r, "card": a["card"]})
    res = []
    for pr in out.values():
        pr["spaces"] = dict(pr["spaces"])
        pr["tags"] = tags(pr)
        res.append(pr)
    return res


def tags(pr: dict) -> list[str]:
    """Playstyles a player committed to early (rounds 1-EARLY)."""
    t = []
    if pr["sm_round"] is not None and pr["sm_round"] <= EARLY:
        t.append("swordmaster")
    if pr["hc_round"] is not None and pr["hc_round"] <= EARLY:
        t.append("high_council")
    if pr["exact_bloodlines"]:
        if sum(1 for c in pr["commanders"] if c["round"] <= EARLY) >= 2:
            t.append("commanders")
        if sum(1 for x in pr["techs"] if x["round"] <= EARLY) >= 2:
            t.append("techs")
    return t


def summarise(group: list[dict], everyone: list[dict]) -> dict:
    """What a group of players did: outcome, other objectives, buys."""
    n = len(group)
    if not n:
        return {"n": 0}
    known = [p for p in group if p["won"] is not None]
    exact = [p for p in group if p["exact_bloodlines"]]
    vps = [p["final_vp"] for p in group if p["final_vp"] is not None]

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    techs = collections.Counter(x["name"] for p in exact for x in p["techs"])
    skills = collections.Counter(c["skill"] for p in exact for c in p["commanders"] if c["skill"])
    buys = collections.Counter(b["card"] for p in group for b in p["buys"])
    base_buys = collections.Counter(b["card"] for p in everyone for b in p["buys"])
    tot, base_tot = max(1, sum(buys.values())), max(1, sum(base_buys.values()))
    # cards this group buys more than everyone else (lift), seen 3+ times
    lift = sorted(((c, round((buys[c] / tot) / (base_buys[c] / base_tot), 2), buys[c])
                   for c in buys if buys[c] >= 3), key=lambda x: -x[1])[:10]
    return {
        "n": n,
        "win_rate": round(sum(p["won"] for p in known) / len(known), 3) if known else None,
        "avg_final_vp": avg(vps),
        "has_swordmaster": round(sum(p["sm_round"] is not None for p in group) / n, 2),
        "avg_sm_round": avg(p["sm_round"] for p in group),
        "has_high_council": round(sum(p["hc_round"] is not None for p in group) / n, 2),
        "avg_hc_round": avg(p["hc_round"] for p in group),
        "avg_commanders": avg(len(p["commanders"]) for p in exact),
        "avg_commanders_after_r4": avg(sum(c["round"] > EARLY for c in p["commanders"]) for p in exact),
        "avg_techs": avg(len(p["techs"]) for p in exact),
        "avg_techs_after_r4": avg(sum(x["round"] > EARLY for x in p["techs"]) for p in exact),
        "top_techs": techs.most_common(8),
        "top_skills": skills.most_common(7),
        "cards_favoured": lift,
    }


def adoption(profiles: list[dict], max_round: int = 9) -> dict:
    """Share of players holding each objective by the end of each round."""
    exact = [p for p in profiles if p["exact_bloodlines"]]
    out = {}
    for r in range(1, max_round + 1):
        out[r] = {
            "swordmaster": round(sum(p["sm_round"] is not None and p["sm_round"] <= r
                                     for p in profiles) / len(profiles), 2),
            "high_council": round(sum(p["hc_round"] is not None and p["hc_round"] <= r
                                      for p in profiles) / len(profiles), 2),
            "1+ commander": round(sum(any(c["round"] <= r for c in p["commanders"])
                                      for p in exact) / max(1, len(exact)), 2),
            "1+ tech": round(sum(any(x["round"] <= r for x in p["techs"])
                                 for p in exact) / max(1, len(exact)), 2),
        }
    return out


def collect() -> list[dict]:
    import replay as R
    from clean import clean_game
    from compare_ai import my_color
    fac = R._card_factory()
    rep = {r["game"]: r for r in json.load(open(os.path.join(
        ROOT, "data", "video_games", "replay_report.json")))}
    profiles = []
    for gid, r in sorted(rep.items()):
        if r.get("skipped"):
            continue
        g = json.load(open(os.path.join(HERE, "games", gid + ".json"), encoding="utf-8"))
        rp = R.Replay(clean_game(g, fac))
        rp.run()
        w = rp.winner()
        wc = next((rp.colour_of[s] for s, pid in rp.pid_of.items() if pid == w), None)
        profiles += profiles_for_game(g, wc, my_color(g))
        print(f"  {gid}", flush=True)
    return profiles


def main() -> None:
    profiles = collect()
    groups = {
        "early_swordmaster": [p for p in profiles if "swordmaster" in p["tags"]],
        "early_high_council": [p for p in profiles if "high_council" in p["tags"]],
        "commander_heavy": [p for p in profiles if "commanders" in p["tags"]],
        "tech_heavy": [p for p in profiles if "techs" in p["tags"]],
        "high_council_and_techs": [p for p in profiles
                                   if {"high_council", "techs"} <= set(p["tags"])],
        "swordmaster_then_commanders": [p for p in profiles if "swordmaster" in p["tags"]
                                        and p["exact_bloodlines"]
                                        and sum(c["round"] > (p["sm_round"] or 0)
                                                for c in p["commanders"]) >= 2],
        "no_early_commitment": [p for p in profiles if not p["tags"]],
        "everyone": profiles,
    }
    report = {"early_means_by_round": EARLY,
              "players": len(profiles),
              "games": len({p["game"] for p in profiles}),
              "exact_bloodlines_players": sum(p["exact_bloodlines"] for p in profiles),
              "groups": {k: summarise(v, profiles) for k, v in groups.items()},
              "adoption_by_round": adoption(profiles),
              "profiles": profiles}
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out = os.path.join(ROOT, "reports", "playstyles.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1, default=str)
    for k, s in report["groups"].items():
        print(k, {x: s.get(x) for x in ("n", "win_rate", "avg_final_vp", "avg_sm_round",
                                         "avg_hc_round", "avg_commanders", "avg_techs")})
    print("wrote", out)


if __name__ == "__main__":
    main()
