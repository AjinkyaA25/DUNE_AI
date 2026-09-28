#!/usr/bin/env python
"""
Turn a card timeline (scan.py) into an editable game file (editor.py format).

What the timeline shows, per seat (screen quadrant), and what it means:
  Agent Turn row gains a card  -> "agent" action (that card; the board space
                                  is found by where the seat's colour
                                  appeared on the board at that moment)
  Reveal Turn row fills        -> "reveal" action (the revealed cards)
  discard top changes to a card that wasn't just played
                               -> "buy" action
  an Intrigue leaves the hand (not into a row) -> "intrigue" (played)
  an Intrigue joins the hand   -> "gain_intrigue"
  all rows clear at once       -> end of round

Single readings that flicker (A -> B -> A within a few seconds: a card being
dragged or hovered, a deck shuffle animation) are smoothed away first.
Anything uncertain carries a flag so it shows up in the editor's "Only
flagged" view.

Usage:
  python video_scrape/timeline_events.py video_scrape/raw_tournament/NFT8aY_T3ZE.webm
  python video_scrape/timeline_events.py VIDEO --names "TL=rl_agent,TR=Chromex"
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vision as V  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(HERE, "games")

MIN_AGENT_PX = 60   # colour pixels an agent adds to its slot (~20x20 piece)
REAPPEAR = 60.0     # s: an Intrigue that comes back this soon was only hidden
GLITCH = 4.0        # s: a state shorter than this between two equal states is noise
ROUND_GAP = 20.0    # s: rows clearing within this window = one round end
SEATS = ("TL", "TR", "BL", "BR")

# Agent slot of every board space (reference frame coords) and each seat's
# colour swatch on its player board.
SPACES = {
    "Sardaukar": (815, 272, 852, 305), "Dutiful Service": (815, 320, 852, 352),
    "Heighliner": (815, 390, 852, 420), "Deliver Supplies": (815, 437, 852, 469),
    "Espionage": (815, 507, 852, 537), "Secrets": (815, 555, 852, 585),
    "Desert Tactics": (815, 624, 852, 654), "Fremkit": (815, 671, 852, 702),
    "High Council": (897, 256, 935, 286), "Imperial Privilege": (897, 302, 935, 333),
    "Swordmaster": (992, 297, 1032, 334), "Assembly Hall": (1062, 256, 1100, 286),
    "Gather Support": (1062, 302, 1100, 333), "Shipping": (1154, 261, 1194, 292),
    "Accept Contract": (1154, 304, 1194, 334), "Research Station": (939, 402, 979, 431),
    "Arrakeen": (1037, 391, 1077, 421), "Spice Refinery": (1116, 377, 1155, 407),
    "Sietch Tabr": (892, 461, 931, 491), "Imperial Basin": (1102, 455, 1142, 486),
    "Hagga Basin": (989, 479, 1028, 510), "Deep Desert": (901, 514, 941, 544),
}
BOARD = (752, 238, 1232, 718)
SWATCH = {"TL": (507, 72, 540, 105), "TR": (1437, 72, 1472, 107),
          "BL": (507, 582, 540, 615), "BR": (1437, 582, 1472, 617)}
HUES = {"Red": ((0, 8), (171, 180)), "Yellow": ((18, 35),),
        "Green": ((45, 85),), "Blue": ((95, 128),)}


# ---------------------------------------------------------------------------
# colours on the board
# ---------------------------------------------------------------------------
def colour_mask(img: np.ndarray, colour: str) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(h.shape, bool)
    for lo, hi in HUES[colour]:
        m |= (h >= lo) & (h <= hi)
    return m & (s > 110) & (v > 90)


def seat_colours(frame: np.ndarray) -> dict[str, str]:
    out = {}
    for seat, (x0, y0, x1, y1) in SWATCH.items():
        crop = frame[y0:y1, x0:x1]
        out[seat] = max(HUES, key=lambda c: colour_mask(crop, c).mean())
    return out


def placed_space(before: np.ndarray, after: np.ndarray, colour: str
                 ) -> tuple[str | None, int]:
    """Which space's agent slot gained the most `colour` pixels between two
    aligned frames? Only slots are looked at, so troops deployed to the
    Conflict or spies placed at the same time don't mislead it.
    Returns (space, pixels gained) or (None, 0)."""
    best, gain = None, 0
    for space, (x0, y0, x1, y1) in SPACES.items():
        x0, y0, x1, y1 = x0 - 4, y0 - 4, x1 + 4, y1 + 4
        g = int((colour_mask(after[y0:y1, x0:x1], colour)
                 & ~colour_mask(before[y0:y1, x0:x1], colour)).sum())
        if g > gain:
            best, gain = space, g
    return (best, gain) if gain >= MIN_AGENT_PX else (None, 0)


# ---------------------------------------------------------------------------
# timeline smoothing
# ---------------------------------------------------------------------------
def sequences(tl: list[dict]) -> dict[tuple, list[tuple[float, list]]]:
    seqs: dict[tuple, list] = collections.defaultdict(list)
    for e in tl:
        seqs[(e["seat"], e["region"])].append((e["t"], e["cards"]))
    return seqs


def names_of(cards: list) -> list[str]:
    return [c["card"] for c in cards]


def smooth(seq: list[tuple[float, list]]) -> list[tuple[float, list]]:
    """Drop A -> B -> A flickers where B lasted < GLITCH seconds, and hand
    re-orderings (same multiset)."""
    changed = True
    while changed and len(seq) > 2:
        changed = False
        out = [seq[0]]
        i = 1
        while i < len(seq):
            t, cards = seq[i]
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            prev = out[-1]
            if (nxt is not None and nxt[0] - t < GLITCH
                    and sorted(names_of(nxt[1])) == sorted(names_of(prev[1]))):
                i += 2          # skip the glitch and the return to A
                changed = True
                continue
            if sorted(names_of(cards)) == sorted(names_of(prev[1])):
                i += 1          # same cards, just re-ordered
                continue
            out.append((t, cards))
            i += 1
        seq = out
    return seq


def multiset_diff(a: list[str], b: list[str]) -> tuple[list[str], list[str]]:
    """(added, removed) going from a to b."""
    ca, cb = collections.Counter(a), collections.Counter(b)
    return list((cb - ca).elements()), list((ca - cb).elements())


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
class Frames:
    """Aligned frames at arbitrary times (cached per second)."""

    def __init__(self, video: str):
        self.cap = cv2.VideoCapture(video)
        self.aligner = V.Aligner()
        self.M = None
        self.cache: dict[int, np.ndarray] = {}

    def at(self, t: float) -> np.ndarray | None:
        k = int(round(t * 2))
        if k in self.cache:
            return self.cache[k]
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = self.cap.read()
        if not ok:
            return None
        r = self.aligner.transform(frame)
        if r is not None:
            self.M = r[0]
        if self.M is None:
            return None
        warped, _ = self.aligner.align(frame, self.M)
        self.cache[k] = warped
        return warped


def agent_icons() -> tuple[dict[str, set], dict[str, str]]:
    """Engine card agent icons (by printed name) and board space icons, to
    sanity-check detected (card, space) pairs. Bloodlines cards aren't in the
    engine yet, so they go unchecked."""
    try:
        from src.data.card_definitions import (create_imperium_cards,
                                               create_starter_cards)
        from src.game.board.board import SPACE_ICONS
    except Exception:
        return {}, {}
    icons: dict[str, set] = {}
    for c in create_imperium_cards() + create_starter_cards():
        icons.setdefault(c.name.replace(",", ""), set()).update(
            s.value for s in c.access_symbols)
    return icons, dict(SPACE_ICONS)


def build(video: str, tl: list[dict], names: dict[str, str]) -> dict:
    lib = V.Library()
    kind_of = {}
    for kind, temps in lib.templates.items():
        for t in temps:
            kind_of.setdefault(t.card, kind)

    def disp(card: str) -> str:
        return "face-down card" if card == "_back" else lib.name(card)

    card_icons, space_icons = agent_icons()
    frames = Frames(video)
    seqs = {k: smooth(v) for k, v in sequences(tl).items()}
    t_first = min(e["t"] for e in tl) if tl else 0.0
    for k, seq in seqs.items():          # rows start the game empty
        if k[1] in ("agent", "reveal", "discard") and seq and seq[0][1]:
            seq.insert(0, (t_first - 1, []))

    # seat colours from a frame a few minutes in
    t_probe = min(e["t"] for e in tl) + 120 if tl else 300
    probe = frames.at(t_probe)
    colours = seat_colours(probe) if probe is not None else \
        dict(zip(SEATS, ("Red", "Green", "Blue", "Yellow")))
    player = {s: names.get(s) or f"{colours[s]} ({s})" for s in SEATS}

    # --- round ends: moments when the agent/reveal rows of all seats empty
    clears = []
    for (seat, rg), seq in seqs.items():
        if rg not in ("agent", "reveal"):
            continue
        for (t0, c0), (t1, c1) in zip(seq, seq[1:]):
            if c0 and not c1:
                clears.append(t1)
    clears.sort()
    round_ends: list[float] = []
    for t in clears:
        if round_ends and t - round_ends[-1] < ROUND_GAP:
            continue
        # a real round end clears (nearly) every seat's rows together
        n = sum(1 for u in clears if t <= u < t + ROUND_GAP)
        if n >= 4:
            round_ends.append(t)

    def round_at(t: float) -> int:
        return 1 + sum(1 for r in round_ends if r <= t)

    actions: list[dict] = []

    def act(t, seat, kind, flags=(), raw=(), **fields):
        a = {"id": len(actions), "t": round(t, 1), "round": round_at(t),
             "player": player[seat], "kind": kind, **fields,
             "effects": [], "raw": list(raw), "flags": list(flags)}
        actions.append(a)
        return a

    for seat in SEATS:
        colour = colours[seat]
        # agent turns: within a round the row only grows, so a card is new
        # only when its count passes the most seen so far this round (a row
        # that briefly reads empty - a card hovered - adds nothing)
        seen: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        agent_seq = seqs.get((seat, "agent"), [])
        for (t0, c0), (t1, c1) in zip(agent_seq, agent_seq[1:]):
            r = round_at(t1)
            # compare printed names: starter cards have several library
            # variants (e.g. Emperor / Muad'Dib Signet Ring art)
            now = collections.Counter(disp(n) for n in names_of(c1))
            added = list((now - seen[r]).elements())
            seen[r] |= now
            for name in added:
                card = next(n for n in names_of(c1) if disp(n) == name)
                before, after = frames.at(t1 - 6), frames.at(t1 + 2)
                space, _ = (placed_space(before, after, colour)
                            if before is not None and after is not None
                            else (None, 0))
                flags = []
                if space is None:
                    flags.append("board space not found")
                    space = ""
                else:
                    ic = card_icons.get(disp(card).replace(",", ""))
                    if ic and "spy" not in ic and space_icons.get(space) not in ic:
                        flags.append(f"{disp(card)} can't be sent to {space}: "
                                     "card or space misread")
                rival = next((c["rivals"] for c in c1
                              if c["card"] == card and c["rivals"]), None)
                if rival:
                    flags.append("card may be " + " / ".join(rival))
                act(t1, seat, "agent", flags, [f"Agent Turn row +{disp(card)}"],
                    space=space, card=disp(card))
        # reveals: one per round, with the fullest reading of the row
        best: dict[int, tuple[float, list]] = {}
        for t, cards in seqs.get((seat, "reveal"), []):
            r = round_at(t)
            if cards and (r not in best or len(cards) > len(best[r][1])):
                best[r] = (best[r][0] if r in best else t, cards)
        for r, (t, cards) in best.items():
            act(t, seat, "reveal", [], [f"Reveal Turn row: {len(cards)} cards"],
                cards=", ".join(disp(n) for n in names_of(cards)))
        # buys: discard top becomes a card that isn't one this seat played
        # this round (those land on the discard at cleanup)
        played: dict[int, set] = collections.defaultdict(set)
        for (t, cards) in seqs.get((seat, "agent"), []) + seqs.get((seat, "reveal"), []):
            played[round_at(t)].update(disp(n) for n in names_of(cards))
        prev_top = None
        for t, cards in seqs.get((seat, "discard"), []):
            top = cards[0]["card"] if cards else None
            near_end = any(abs(t - r) < ROUND_GAP for r in round_ends)
            if top and top != prev_top and top != "_back" and not near_end \
                    and disp(top) not in played[round_at(t)]:
                flags = ["card may be " + " / ".join(cards[0]["rivals"])] \
                    if cards[0]["rivals"] else []
                act(t, seat, "buy", flags, [f"discard pile top: {disp(top)}"],
                    card=disp(top))
            prev_top = top
        # intrigues entering / leaving the hand. Ignored: one-for-one swaps
        # in a single reading (the same card re-identified as a lookalike)
        # and a card that leaves and comes back within REAPPEAR seconds
        # (hovered / covered, not played).
        hand = seqs.get((seat, "hand"), [])
        events = []                      # (t, "+"/"-", card)
        for (t0, c0), (t1, c1) in zip(hand, hand[1:]):
            if not c1 or not c0:
                continue        # hand fully hidden/emptied: not informative
            i0 = [disp(n) for n in names_of(c0) if kind_of.get(n) == "intrigue"]
            i1 = [disp(n) for n in names_of(c1) if kind_of.get(n) == "intrigue"]
            added, removed = multiset_diff(i0, i1)
            if added and len(added) == len(removed):
                continue
            events += [(t1, "-", c) for c in removed] + [(t1, "+", c) for c in added]
        dropped = set()
        for i, (t, sign, card) in enumerate(events):
            if sign != "-" or i in dropped:
                continue
            back = next((j for j in range(i + 1, len(events))
                         if events[j][1] == "+" and events[j][2] == card
                         and events[j][0] - t <= REAPPEAR and j not in dropped), None)
            if back is not None:
                dropped |= {i, back}
        for i, (t, sign, card) in enumerate(events):
            if i in dropped:
                continue
            if sign == "-":
                act(t, seat, "intrigue", [], [f"Intrigue left hand: {card}"],
                    card=card)
            else:
                act(t, seat, "gain_intrigue", [], [f"Intrigue joined hand: {card}"],
                    card=card)

    actions.sort(key=lambda a: a["t"])
    for i, a in enumerate(actions):
        a["id"] = i

    vid = os.path.splitext(os.path.basename(video))[0]
    info_path = os.path.splitext(video)[0] + ".info.json"
    title = ""
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            title = json.load(f).get("title", "")
    return {
        "video_id": vid,
        "url": f"https://youtu.be/{vid}",
        "title": title,
        "me": names.get("me"),
        "source": "vision",
        "players": [{"name": player[s], "color": colours[s], "seat": s}
                    for s in SEATS],
        "rounds": [{"round": i + 1,
                    "t": (round_ends[i - 1] if i else (min(e["t"] for e in tl) if tl else 0)),
                    "conflict": ""} for i in range(len(round_ends) + 1)],
        "result": [],
        "actions": actions,
        "edited": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--timeline", help="default: <video>.timeline.json")
    ap.add_argument("--names", default="",
                    help='seat names, e.g. "TL=rl_agent,TR=Chromex,me=TL"')
    ap.add_argument("--out")
    args = ap.parse_args()

    tl_path = args.timeline or os.path.splitext(args.video)[0] + ".timeline.json"
    with open(tl_path, encoding="utf-8") as f:
        tl = json.load(f)
    names = dict(kv.split("=", 1) for kv in args.names.split(",") if "=" in kv)
    if "me" in names:
        names["me"] = names.get(names["me"], names["me"])
    game = build(args.video, tl, names)

    vid = game["video_id"]
    out = args.out or os.path.join(GAMES, f"{vid}.json")
    if os.path.exists(out) and not args.out:
        with open(out, encoding="utf-8") as f:
            if json.load(f).get("edited"):
                raise SystemExit(f"{out} has hand edits; pass --out to write elsewhere")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(game, f, indent=1, ensure_ascii=False)
    kinds = collections.Counter(a["kind"] for a in game["actions"])
    flagged = sum(1 for a in game["actions"] if a["flags"])
    print(f"{out}: {len(game['rounds'])} rounds, {len(game['actions'])} actions "
          f"{dict(kinds)}, {flagged} flagged")


if __name__ == "__main__":
    main()
