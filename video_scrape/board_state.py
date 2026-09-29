#!/usr/bin/env python
"""
Read the shared board by player colour (reference-frame coordinates):
  - influence: each faction panel on the left is a vertical track; a
    player's disc sits LEVEL_STEP px higher per influence point
  - VP: the track on the right edge of the board (0 at the bottom)
  - troops in the Conflict: cubes in each colour's quadrant of the
    Conflict area

Usage (debug):  python video_scrape/board_state.py VIDEO --at 1702 --out f.png
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision as V  # noqa: E402
from timeline_events import HUES  # noqa: E402

# stricter than the agent-piece masks: the board prints gold icons (hue
# 13-24) right next to the yellow discs (hue 32-34)
DISC_HUES = {"Red": ((0, 8), (171, 180)), "Yellow": ((27, 37),),
             "Green": ((45, 85),), "Blue": ((95, 128),)}


def colour_mask(img: np.ndarray, colour: str) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(h.shape, bool)
    for lo, hi in DISC_HUES[colour]:
        m |= (h >= lo) & (h <= hi)
    return m & (s > 110) & (v > 130)

FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")
# influence panels: x range of the track, y of level 0 per faction
INF_X = (767, 809)
INF_LEVEL0 = {"emperor": 353, "spacing_guild": 470, "bene_gesserit": 587,
              "fremen": 704}
LEVEL_STEP = 17.3
INF_MAX = 6
# VP track (right edge of the board)
VP_X = (1196, 1228)
VP_ZERO_Y = 693
VP_STEP = 25.0
# Conflict quadrants per colour
TROOP_ZONE = {"Red": (944, 563, 990, 610), "Blue": (944, 618, 990, 663),
              "Green": (1137, 563, 1182, 610), "Yellow": (1137, 618, 1182, 663)}
CUBE_AREA = 55.0          # colour pixels of one troop cube
MIN_BLOB = 18


def _blobs(mask: np.ndarray):
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((2, 2), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    return [(st[k, cv2.CC_STAT_AREA], cen[k]) for k in range(1, n)
            if st[k, cv2.CC_STAT_AREA] >= MIN_BLOB]


def read_influence(frame: np.ndarray) -> dict:
    """{colour: {faction: level}} (None when that disc wasn't found)."""
    out = {c: {} for c in HUES}
    x0, x1 = INF_X
    for f, y0 in INF_LEVEL0.items():
        top = int(y0 - LEVEL_STEP * (INF_MAX + 0.5))
        crop = frame[top:int(y0 + LEVEL_STEP / 2), x0:x1]
        for c in HUES:
            bl = _blobs(colour_mask(crop, c))
            if not bl:
                out[c][f] = None
                continue
            area, (cx, cy) = max(bl, key=lambda b: b[0])
            y = top + cy
            out[c][f] = int(np.clip(round((y0 - y) / LEVEL_STEP), 0, INF_MAX))
    return out


def read_vp(frame: np.ndarray) -> dict:
    """{colour: vp} from the disc heights on the VP track (0-12 on board)."""
    out = {}
    x0, x1 = VP_X
    top = int(VP_ZERO_Y - VP_STEP * 12.5)
    crop = frame[top:int(VP_ZERO_Y + VP_STEP / 2), x0:x1]
    for c in HUES:
        bl = _blobs(colour_mask(crop, c))
        if not bl:
            out[c] = None
            continue
        area, (cx, cy) = max(bl, key=lambda b: b[0])
        out[c] = int(round((VP_ZERO_Y - (top + cy)) / VP_STEP))
    return out


def read_troops(frame: np.ndarray) -> dict:
    """{colour: troops in the Conflict} from the colour's cube area."""
    out = {}
    for c, (x0, y0, x1, y1) in TROOP_ZONE.items():
        m = colour_mask(frame[y0:y1, x0:x1], c)
        area = sum(a for a, _ in _blobs(m))
        out[c] = int(round(area / CUBE_AREA))
    return out


def read_board(frame: np.ndarray) -> dict:
    # troops: cube colour is too close to the tinted Conflict zones in
    # compressed video; combat strength comes from the board numbers instead
    return {"influence": read_influence(frame), "vp": read_vp(frame)}


def draw(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for f, y0 in INF_LEVEL0.items():
        for lv in range(INF_MAX + 1):
            y = int(y0 - LEVEL_STEP * lv)
            cv2.line(out, (INF_X[0], y), (INF_X[1], y), (0, 255, 255), 1)
    for v in range(13):
        y = int(VP_ZERO_Y - VP_STEP * v)
        cv2.line(out, (VP_X[0], y), (VP_X[1], y), (255, 255, 0), 1)
    for c, (x0, y0, x1, y1) in TROOP_ZONE.items():
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--at", type=float, required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.video.endswith(".jpg"):
        frame = cv2.imread(args.video)
    else:
        a = V.Aligner().align(V.grab(args.video, args.at))
        if a is None:
            raise SystemExit("board not visible")
        frame = a[0]
    for k, v in read_board(frame).items():
        print(k, v)
    if args.out:
        cv2.imwrite(args.out, draw(frame))


if __name__ == "__main__":
    main()
