#!/usr/bin/env python
"""
Recognise cards on screen in a spectator-view TTS video (no chat log).

Camera zoom/pan differs between videos (and changes inside a video), so each
frame is first aligned to a reference frame (layout/reference.jpg) by matching
keypoints on the four player boards. In reference coordinates every player's
hand strip, Agent Turn row and Reveal Turn row sit in fixed regions. Cards are
found by sliding every library card (card_library.py) over the region at the
on-screen card size and keeping the best non-overlapping matches.

Imperium cards and intrigue cards are printed at different sizes on the table
(and have different proportions), so each library image is matched at the
height of its own kind.

Debug usage (draws what was recognised onto one frame):
  python video_scrape/vision.py VIDEO --at 1021 --out frame.png
"""
from __future__ import annotations

import argparse
import sys
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CARDS = os.path.join(HERE, "cards")
LAYOUT = os.path.join(HERE, "layout")
REFERENCE = os.path.join(LAYOUT, "reference.jpg")  # NFT8aY_T3ZE @28:22

# Player-board rectangles in the reference frame: static art, used for
# alignment keypoints (the card rows below them change every turn).
BOARDS = [(205, 35, 745, 240), (1240, 35, 1770, 240),
          (205, 545, 745, 750), (1240, 545, 1770, 750)]
MIN_INLIERS = 40

# On-screen card height (px, reference frame) per kind.
CARD_H = {"imperium": 74, "intrigue": 69}
MIN_SCORE = 0.6
SURE_SCORE = 0.8    # above this, accept without the runner-up check
MIN_MARGIN = 0.07   # else must beat the best different card by this much
RIVAL_SCORE = 0.45  # runner-up candidates are collected down to this score
ROW_TOL = 6         # px: cards in one strip/row share the same top edge
# dark cards that match empty table well and can never be in a hand/row
NEVER_HELD = {"reclaimedForces", "muadDibFirstPlayer"}
EMPTY = "_empty"
BACK_ID = "_back"
BLANK_STD = 15
CLIP_TOP = 44  # in a region cut off by the video edge, match card tops only

# Screen regions (x0, y0, x1, y1) on a 1920x1080 frame, per seat.
# Seats are screen quadrants; which player sits where is read separately.
REGIONS = {
    "TL": {"hand": (200, 405, 750, 530), "agent": (385, 250, 730, 335),
           "reveal": (320, 335, 730, 420), "discard": (662, 145, 730, 235)},
    "TR": {"hand": (1240, 405, 1775, 530), "agent": (1270, 250, 1600, 335),
           "reveal": (1270, 335, 1740, 420), "discard": (1422, 145, 1490, 235)},
    "BL": {"hand": (200, 915, 750, 1075), "agent": (385, 765, 730, 850),
           "reveal": (320, 850, 730, 935), "discard": (662, 655, 730, 745)},
    "BR": {"hand": (1240, 915, 1775, 1075), "agent": (1270, 765, 1600, 850),
           "reveal": (1270, 850, 1740, 935), "discard": (1422, 655, 1490, 745)},
    # shared: the 5-card Imperium Row under the central board, and the
    # current Conflict card (shown sideways under the Conflict deck)
    "G": {"row": (920, 795, 1205, 895), "conflict": (880, 605, 945, 700)},
}
REGION_KINDS = {"hand": ("imperium", "intrigue"), "agent": ("imperium",),
                "reveal": ("imperium",), "discard": ("imperium",),
                "row": ("imperium",), "conflict": ("intrigue",)}
# regions whose cards are printed at a different size than CARD_H
REGION_HEIGHT = {"row": 68, "conflict": 66}
# regions that can only hold these library cards
CONFLICT_CARDS = frozenset((
    "battleForArrakeen", "battleForImperialBasin", "battleForSpiceRefinery",
    "bl_Skirmish", "bl_StormsInTheSouth", "choamSecurity", "propaganda",
    "protectTheSietches", "secureImperialBasin", "seizeSpiceRefinery",
    "shadowContest", "siegeOfArrakeen", "skirmishA", "skirmishB", "skirmishC",
    "spiceFreighters", "testOfLoyalty", "tradeDispute"))
REGION_ONLY = {"conflict": CONFLICT_CARDS}


# engine card -> mod id where the mod spells the card differently
MOD_SPELLING = {
    "councilorAmbition": "Councilor's Ambition",
    "bl_DeliverLogistics": "Delivery Logistics",
    "nothernWatermaster": "Northern Watermaster",
    "publicSpectable": "Public Spectacle",
    "shaddamFavor": "Shaddam's Favor",
    "smugglerHarvester": "Smuggler's Harvester",
    "smugglerHaven": "Smuggler's Haven",
    "spacingGuildFavor": "Spacing Guild's Favor",
    "theBeastSpoils": "The Beast's Spoils",
    "trecherousManeuver": "Treacherous Maneuver",
}


def region_only(rg: str, lib: "Library") -> frozenset:
    return REGION_ONLY.get(rg) or game_cards(lib)


# never in a hand / row / played: conflict cards, Steersman's Navigation
# cards (named like techs), per-player-count components, community variants
NOT_HAND_CARDS = CONFLICT_CARDS | frozenset((
    "bl_AdvancedDataAnalysis", "bl_ChoamTransports", "bl_DeliveryBay",
    "bl_ForbiddenWeapons", "bl_GeneLockedVault", "bl_Glowglobes",
    "bl_NavigationChamber", "bl_OrnithopterFleet", "bl_Panopticon",
    "bl_PlanetaryArray", "bl_YrkoonPersuasion", "muadDib4to6p", "reshuffle"))


def game_cards(lib: "Library") -> frozenset:
    """Library ids that can show up in a hand, the Row or a played row.
    NOT limited to the engine's card list: the tournament games use cards the
    engine doesn't have yet (e.g. Ixian Probe), and hiding those would turn
    them into lookalike misreads."""
    if lib._game_cards is None:
        lib._game_cards = frozenset(
            c for c in lib.index
            if c not in NOT_HAND_CARDS and not c.endswith("_com")
            and not c.startswith("placeSpy")) | {BACK_ID, EMPTY}
    return lib._game_cards


def _kind(w: int, h: int) -> str | None:
    """Card kind from library-image proportions."""
    a = w / h
    if 0.69 <= a <= 0.74:
        return "imperium"
    if 0.63 <= a < 0.69:
        return "intrigue"  # also conflict cards (same card size)
    return None  # leaders, techs, board spaces, skill tiles


@dataclass
class Template:
    card: str
    kind: str
    img: np.ndarray  # scaled to on-screen size


class Library:
    def __init__(self, cards_dir: str = CARDS):
        self._game_cards = None
        self.full: dict = {}          # (card, id(template)) -> library image
        self._scaled: dict = {}
        with open(os.path.join(cards_dir, "index.json"), encoding="utf-8") as f:
            self.index = json.load(f)
        self.templates: dict[str, list[Template]] = {k: [] for k in CARD_H}
        for card, e in self.index.items():
            for fn in e["files"]:
                im = cv2.imdecode(np.fromfile(os.path.join(cards_dir, fn),
                                              np.uint8), cv2.IMREAD_COLOR)
                kind = _kind(im.shape[1], im.shape[0])
                if kind is None or card in NEVER_HELD:
                    continue
                h = CARD_H[kind]
                w = round(im.shape[1] * h / im.shape[0])
                t = Template(card, kind, cv2.resize(im, (w, h),
                                                    interpolation=cv2.INTER_AREA))
                self.templates[kind].append(t)
                self.full[card, id(t)] = im

        # empty-slot artwork (e.g. the discard pile placeholder) so it wins
        # its spot instead of the nearest card; removed from results
        for fn in os.listdir(LAYOUT):
            if fn.startswith("empty_"):
                im = cv2.imread(os.path.join(LAYOUT, fn))
                self.templates["imperium"].append(Template(EMPTY, "imperium", im))

    def at(self, kind: str, h: int) -> list[Template]:
        """This kind's templates rescaled to height h (cached)."""
        key = (kind, h)
        if key not in self._scaled:
            self._scaled[key] = [
                Template(t.card, t.kind, cv2.resize(
                    self.full[t.card, id(t)], (round(self.full[t.card, id(t)].shape[1]
                                                     * h / self.full[t.card, id(t)].shape[0]), h),
                    interpolation=cv2.INTER_AREA))
                for t in self.templates[kind] if (t.card, id(t)) in self.full]
        return self._scaled[key]

    def name(self, card: str) -> str:
        if card in MOD_SPELLING:
            return MOD_SPELLING[card]
        names = self.index.get(card, {}).get("names") or [card]
        return names[0]


class Aligner:
    """Maps video frames onto the reference layout (scale + translation)."""

    def __init__(self, reference: str = REFERENCE):
        ref = cv2.imread(reference, cv2.IMREAD_GRAYSCALE)
        self.size = (ref.shape[1], ref.shape[0])
        mask = np.zeros_like(ref)
        for x0, y0, x1, y1 in BOARDS:
            mask[y0:y1, x0:x1] = 255
        self.orb = cv2.ORB_create(4000)
        self.kp, self.desc = self.orb.detectAndCompute(ref, mask)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def transform(self, frame: np.ndarray) -> tuple[np.ndarray, int] | None:
        """Reference->frame similarity transform and its inlier count, or
        None when the board isn't visible (menus, zoomed-in card, scene cut)."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(g, None)
        if desc is None:
            return None
        good = [p[0] for p in self.bf.knnMatch(self.desc, desc, k=2)
                if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < MIN_INLIERS:
            return None
        src = np.float32([self.kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3)
        if M is None or inl.sum() < MIN_INLIERS:
            return None
        return M, int(inl.sum())

    def align(self, frame: np.ndarray, M: np.ndarray | None = None
              ) -> tuple[np.ndarray, int] | None:
        """Frame resampled into reference coordinates, plus the reference y
        where the video frame ends (zoomed-in videos cut off the bottom hand
        strips)."""
        if M is None:
            r = self.transform(frame)
            if r is None:
                return None
            M = r[0]
        inv = cv2.invertAffineTransform(M)
        bottom = int(inv[1, 1] * frame.shape[0] + inv[1, 2])
        return (cv2.warpAffine(frame, inv, self.size, flags=cv2.INTER_AREA),
                bottom)


@dataclass
class Hit:
    card: str
    kind: str
    score: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in frame coords
    rivals: list[str] | None = None  # set when the match is a close call


def find_cards(frame: np.ndarray, region: tuple[int, int, int, int],
               lib: Library, kinds: tuple[str, ...],
               bottom: int | None = None, height: int | None = None,
               only: frozenset | None = None) -> list[Hit]:
    """Best non-overlapping card matches inside one region, left to right.

    If the video ends above the region's bottom edge, cards are only partly
    visible: match just the top of each card (title + art)."""
    x0, y0, x1, y1 = region
    clipped = bottom is not None and bottom < y1
    if clipped:
        y1 = bottom
    if y1 - y0 < CLIP_TOP:
        return []
    roi = frame[y0:y1, x0:x1]
    cands: list[Hit] = []
    for kind in kinds:
        for t in (lib.at(kind, height) if height else lib.templates[kind]):
            if only is not None and t.card not in only:
                continue
            img = t.img[:CLIP_TOP] if clipped else t.img
            th, tw = img.shape[:2]
            if tw > roi.shape[1]:
                continue
            r = cv2.matchTemplate(roi, img, cv2.TM_CCOEFF_NORMED)
            # a card can appear more than once (two Dune the Desert Planets):
            # keep every column-wise local peak above threshold
            col = r.max(axis=0)
            rows = r.argmax(axis=0)
            for x in np.flatnonzero(col >= RIVAL_SCORE):
                lo, hi = max(0, x - tw // 2), x + tw // 2 + 1
                if col[x] < col[lo:hi].max():
                    continue
                y = int(rows[x])
                cands.append(Hit(t.card, kind, float(col[x]),
                                 (x0 + int(x), y0 + y, x0 + int(x) + tw,
                                  y0 + y + th)))
    cands.sort(key=lambda h: -h.score)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    chosen: list[Hit] = []
    for h in cands:
        if h.score < MIN_SCORE:
            break
        bx0, by0, bx1, by1 = h.box
        # empty table scores ~0.6 against dark cards; real cards have
        # texture (grey-level std 30+ vs <5 for empty slots)
        if gray[by0:by1, bx0:bx1].std() < BLANK_STD:
            continue
        if any(_overlap_x(h.box, c.box) >= 0.4 for c in chosen):
            continue
        # background texture matches many cards about equally well; a real
        # card usually clearly beats the best *different* card at that spot
        if h.score < SURE_SCORE:
            rivals = sorted(((c.score, lib.name(c.card)) for c in cands
                             if _same_spot(h.box, c.box)
                             and lib.name(c.card) != lib.name(h.card)),
                            reverse=True)
            if h.score - (rivals[0][0] if rivals else RIVAL_SCORE) < MIN_MARGIN:
                h.rivals = list(dict.fromkeys(n for _, n in rivals))[:3]
        chosen.append(h)
    # weaker matches (incl. close calls: intrigue art is hard to tell apart
    # at this size) are kept only in line with the confident cards; off-row
    # hits are table texture or board edges
    chosen = [h for h in chosen if h.card != EMPTY]
    sure_y = [h.box[1] for h in chosen if h.score >= SURE_SCORE]
    if sure_y:
        row_y = float(np.median(sure_y))
        chosen = [h for h in chosen if h.score >= SURE_SCORE
                  or abs(h.box[1] - row_y) <= ROW_TOL]
    else:
        chosen = [h for h in chosen if h.rivals is None]
    return sorted(chosen, key=lambda h: h.box[0])


def _same_spot(a, b) -> bool:
    return abs(a[0] - b[0]) <= 4 and abs(a[1] - b[1]) <= 6


def _overlap_x(a, b) -> float:
    inter = min(a[2], b[2]) - max(a[0], b[0])
    return max(0, inter) / min(a[2] - a[0], b[2] - b[0])


def read_frame(frame: np.ndarray, lib: Library, bottom: int | None = None
               ) -> dict[str, dict[str, list[Hit]]]:
    return {seat: {rg: find_cards(frame, box, lib, REGION_KINDS[rg], bottom,
                                  REGION_HEIGHT.get(rg), region_only(rg, lib))
                   for rg, box in regs.items()}
            for seat, regs in REGIONS.items()}


def grab(video: str, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"could not read {video} at {t}s")
    return frame


def draw(frame: np.ndarray, state, lib: Library) -> np.ndarray:
    out = frame.copy()
    for seat, regs in REGIONS.items():
        for rg, (x0, y0, x1, y1) in regs.items():
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 1)
            for h in state[seat][rg]:
                col = (0, 255, 0) if h.rivals is None else (0, 165, 255)
                cv2.rectangle(out, h.box[:2], h.box[2:], col, 2)
                cv2.putText(out, f"{h.card[:12]} {h.score:.2f}",
                            (h.box[0], h.box[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.33, col, 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--at", type=float, required=True, help="seconds")
    ap.add_argument("--out", help="write annotated frame here")
    args = ap.parse_args()

    lib = Library()
    aligned = Aligner().align(grab(args.video, args.at))
    if aligned is None:
        raise SystemExit("board not visible in this frame")
    frame, bottom = aligned
    state = read_frame(frame, lib, bottom)
    for seat, regs in state.items():
        for rg, hits in regs.items():
            print(f"{seat} {rg:6s} " + ", ".join(
                f"{lib.name(h.card)} ({h.score:.2f}"
                + (f" ?{'/'.join(h.rivals)}" if h.rivals else "") + ")"
                for h in hits))
    if args.out:
        cv2.imwrite(args.out, draw(frame, state, lib))


if __name__ == "__main__":
    main()
