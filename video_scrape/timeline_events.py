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
AGENT_SETTLE = 10.0  # s after an agent card lands until its payouts are in
BUY_MATCH = 20.0    # s: discard-pile buy vs card leaving the Row
COUNT_HOLD = 8.0    # s an intrigue count must hold to count as a change
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
        if "cards" in e:                 # card regions (numbers are separate)
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

    # the Conflict card is the better clock when it was scanned: a new card
    # is flipped at the start of every round (reading kept only if it held
    # for a while - a card being hovered/moved flickers)
    conflicts: list[tuple[float, str]] = []      # (start time, card)
    cseq = seqs.get(("G", "conflict"), [])
    for i, (t, cards) in enumerate(cseq):
        if not cards:
            continue
        nxt = cseq[i + 1][0] if i + 1 < len(cseq) else float("inf")
        if nxt - t < ROUND_GAP:
            continue
        name = cards[0]["card"]
        if not conflicts or conflicts[-1][1] != name:
            conflicts.append((t, name))
    if len(conflicts) >= 2:
        round_ends = [t for t, _ in conflicts[1:]]

    def round_at(t: float) -> int:
        return 1 + sum(1 for r in round_ends if r <= t)

    def conflict_of(r: int) -> str:
        if len(conflicts) >= 2 and r - 1 < len(conflicts):
            return disp(conflicts[r - 1][1])
        return ""

    # board numbers per seat (spice, solari, water, persuasion, strength)
    nums: dict[str, list] = collections.defaultdict(list)
    for e in tl:
        if e["region"] == "numbers":
            nums[e["seat"]].append((e["t"], e["values"]))

    def numbers_at(seat: str, t: float) -> dict | None:
        best = None
        for tt, v in nums.get(seat, []):
            if tt > t:
                break
            best = v
        return best

    def resource_effects(seat: str, t0: float, t1: float) -> list[dict]:
        a, b = numbers_at(seat, t0), numbers_at(seat, t1)
        if not a or not b:
            return []
        out = []
        for res in ("spice", "solari", "water"):
            if a.get(res) is not None and b.get(res) is not None                     and b[res] != a[res]:
                out.append({"res": res, "n": b[res] - a[res]})
        return out

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
                a = act(t1, seat, "agent", flags, [f"Agent Turn row +{disp(card)}"],
                        space=space, card=disp(card))
                # resources before the card went down vs once the turn settled
                a["effects"] = resource_effects(seat, t1 - 6, t1 + AGENT_SETTLE)
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
        # intrigues: driven by the NUMBER of intrigues in the hand, which
        # survives lookalike misreads (the names of a cut-off hand flicker
        # between similar cards; the count doesn't). A count change counts
        # once it has held for COUNT_HOLD seconds; the card's name is taken
        # from what disappeared / appeared across that change.
        hand = [(t, [disp(n) for n in names_of(c) if kind_of.get(n) == "intrigue"])
                for t, c in seqs.get((seat, "hand"), []) if c]
        stable_i: list[tuple[float, list]] = []
        for k, (t, ints) in enumerate(hand):
            nxt = hand[k + 1][0] if k + 1 < len(hand) else float("inf")
            if nxt - t >= COUNT_HOLD or (stable_i and len(ints) == len(stable_i[-1][1])):
                if not stable_i or len(ints) != len(stable_i[-1][1]):
                    stable_i.append((t, ints))
                else:
                    stable_i[-1] = (stable_i[-1][0], ints)   # same count: newest names
        # drop glitches: a count that comes back within REAPPEAR seconds
        # (hand hidden / cut off for a moment), and big jumps (3+) that don't
        # last (a whole hand misread as intrigues)
        changed = True
        while changed and len(stable_i) >= 2:
            changed = False
            for k in range(1, len(stable_i)):
                t, ints = stable_i[k]
                prev = stable_i[k - 1][1]
                nxt = stable_i[k + 1] if k + 1 < len(stable_i) else None
                lasts = (nxt[0] if nxt else float("inf")) - t
                back = nxt is not None and len(nxt[1]) == len(prev) and lasts <= REAPPEAR
                jump = abs(len(ints) - len(prev)) >= 3 and lasts <= REAPPEAR
                if back or jump:
                    del stable_i[k]
                    # merge the neighbours if they now have the same count
                    if k < len(stable_i) and len(stable_i[k][1]) == len(stable_i[k - 1][1]):
                        del stable_i[k]
                    changed = True
                    break
        for (t0, i0), (t1, i1) in zip(stable_i, stable_i[1:]):
            added, removed = multiset_diff(i0, i1)
            n = len(i1) - len(i0)
            if n < 0:
                which = removed[:-n] + ["?"] * max(0, -n - len(removed))
                for card in which:
                    act(t1, seat, "intrigue", [] if card != "?" else ["which Intrigue?"],
                        [f"Intrigues in hand {len(i0)} -> {len(i1)}"], card=card)
            else:
                which = added[:n] + ["?"] * max(0, n - len(added))
                for card in which:
                    act(t1, seat, "gain_intrigue", [] if card != "?" else ["which Intrigue?"],
                        [f"Intrigues in hand {len(i0)} -> {len(i1)}"], card=card)

    # --- the Imperium Row (when scanned): a card leaving it is a purchase.
    # Confirms the discard-pile buys and recovers ones the discard missed
    # (covered pile, bought straight to hand, ...).
    row = seqs.get(("G", "row"), [])
    if row:
        rev_times = {s_: [t for t, c in seqs.get((s_, "reveal"), []) if c]
                     for s_ in SEATS}
        buys = [a for a in actions if a["kind"] == "buy"]
        for (t0, c0), (t1, c1) in zip(row, row[1:]):
            if not c0 or not c1:
                continue                   # row hidden / being refilled
            _, left = multiset_diff([disp(n) for n in names_of(c0)],
                                    [disp(n) for n in names_of(c1)])
            for name in left:
                hit = next((a for a in buys if a["card"] == name
                            and abs(a["t"] - t1) <= BUY_MATCH), None)
                if hit is not None:
                    hit["raw"].append(f"left the Imperium Row at {t1:.0f}s")
                    buys.remove(hit)
                    continue
                # buyer = whoever revealed most recently before this
                seat = max(SEATS, key=lambda s_: max(
                    (t for t in rev_times[s_] if t <= t1 + 2), default=-1e9))
                act(t1, seat, "buy", ["buyer guessed from the last reveal"],
                    [f"left the Imperium Row: {name}"], card=name)
        for a in buys:
            a["flags"].append("not seen leaving the Row (reserve card, or misread)")

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
                    "conflict": conflict_of(i + 1)}
                   for i in range(len(round_ends) + 1)],
        "result": [],
        "resources": {s_: [[t, v] for t, v in nums.get(s_, [])] for s_ in SEATS},
        # shared board over time: influence per colour + faction, VP per colour
        "board": [[e["t"], e["values"]] for e in tl if e["region"] == "board"],
        "actions": actions,
        "edited": False,
    }


LEVEL1 = ("skirmishA", "skirmishB", "skirmishC", "bl_Skirmish")


def split_games(tl: list[dict]) -> list[list[dict]]:
    """Split a multi-game video (a stream) into games: a new game starts when
    the Conflict card goes back to a level-I Skirmish after later conflicts
    had been seen. Videos without Conflict readings stay one game."""
    starts = [tl[0]["t"] if tl else 0.0]
    seen_later = False
    for e in tl:
        if e["region"] != "conflict" or not e.get("cards"):
            continue
        c = e["cards"][0]["card"]
        if c in LEVEL1:
            if seen_later:
                starts.append(e["t"] - 60)      # setup happens just before
                seen_later = False
        else:
            seen_later = True
    starts.append(float("inf"))
    return [[e for e in tl if a <= e["t"] < b] for a, b in zip(starts, starts[1:])]


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
    parts = [p for p in split_games(tl) if sum(1 for e in p if e.get("cards")) >= 50]
    for gi, part in enumerate(parts or [tl]):
        game = build(args.video, part, names)
        vid = game["video_id"]
        if len(parts) > 1:                       # stream: one file per game
            game["video_id"] = f"{vid}_g{gi + 1}"
            game["game_in_video"] = gi + 1
        out = args.out or os.path.join(GAMES, f"{game['video_id']}.json")
        if os.path.exists(out) and not args.out:
            with open(out, encoding="utf-8") as f:
                if json.load(f).get("edited"):
                    print(f"{out} has hand edits; skipped (pass --out)")
                    continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(game, f, indent=1, ensure_ascii=False)
        kinds = collections.Counter(a["kind"] for a in game["actions"])
        flagged = sum(1 for a in game["actions"] if a["flags"])
        print(f"{out}: {len(game['rounds'])} rounds, {len(game['actions'])} actions "
              f"{dict(kinds)}, {flagged} flagged")


if __name__ == "__main__":
    main()
