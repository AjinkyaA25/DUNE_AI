#!/usr/bin/env python
"""
Read the numbers printed on each player board (spice, solari, water,
persuasion, combat strength) in spectator-view videos.

General OCR can't read these (one or two big digits on coloured tokens), so
this is a tiny nearest-neighbour matcher: each number crop is binarised
into a glyph, and compared against labelled glyph prototypes
(layout/digits.npz, built with `harvest` + `label` below).

Usage:
  python video_scrape/board_numbers.py harvest            # crops -> clusters sheet
  python video_scrape/board_numbers.py label labels.json  # cluster id -> value
  python video_scrape/board_numbers.py read VIDEO --at 1702
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision as V  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOS = os.path.join(HERE, "layout", "digits.npz")
WORK = os.path.join(HERE, "raw_tournament", "digit_harvest")

# token centres in the reference frame (x, y), per seat
FIELDS = ("spice", "solari", "water", "persuasion", "strength")
_TOP = {"spice": (566, 132), "solari": (609, 123), "water": (651, 138),
        "persuasion": (460, 214), "strength": (462, 90)}
_RIGHT = {"spice": (1324, 132), "solari": (1368, 123), "water": (1410, 138),
          "persuasion": (1516, 214), "strength": (1516, 90)}
TOKENS = {
    "TL": _TOP, "TR": _RIGHT,
    "BL": {k: (x, y + 516) for k, (x, y) in _TOP.items()},
    "BR": {k: (x, y + 516) for k, (x, y) in _RIGHT.items()},
}
R = 16                       # half-size of a token crop
GW, GH = 20, 20              # glyph grid (fits 1-2 digits)
# text is light on these tokens, dark on the others
LIGHT_TEXT = {"spice", "persuasion", "strength"}
MAX_DIST = 0.22              # mean pixel mismatch to accept a prototype


def glyph(crop: np.ndarray, field: str) -> np.ndarray | None:
    """Binary text mask of a token crop, cropped to the text and resized."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    inner = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= (0.62 * w) ** 2 / 4 * 1.6
    vals = g[inner]
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (g > thr) if field in LIGHT_TEXT else (g <= thr)
    m &= inner
    m = m.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for k in range(1, n):          # text blobs: not touching the crop edge
        x, y, bw, bh, area = stats[k]
        if area >= 6 and bh >= h * 0.25 and x > 0 and y > 0 \
                and x + bw < w and y + bh < h:
            keep[lab == k] = 1
    ys, xs = np.nonzero(keep)
    if len(xs) == 0:
        return None
    t = keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    # keep aspect: pad to GW:GH box
    th, tw = t.shape
    scale = min(GW / tw, GH / th)
    t = cv2.resize(t, (max(1, round(tw * scale)), max(1, round(th * scale))),
                   interpolation=cv2.INTER_AREA)
    out = np.zeros((GH, GW), np.float32)
    oy, ox = (GH - t.shape[0]) // 2, (GW - t.shape[1]) // 2
    out[oy:oy + t.shape[0], ox:ox + t.shape[1]] = t
    return out


def crops(frame: np.ndarray):
    for seat, fields in TOKENS.items():
        for f, (x, y) in fields.items():
            yield seat, f, frame[y - R:y + R, x - R:x + R]


class Reader:
    def __init__(self, path: str = PROTOS):
        z = np.load(path)
        self.P = z["protos"].reshape(len(z["protos"]), -1)
        self.values = z["values"]

    def read(self, crop: np.ndarray, field: str) -> tuple[int | None, float]:
        g = glyph(crop, field)
        if g is None:
            return None, 1.0
        d = np.abs(self.P - g.reshape(1, -1)).mean(axis=1)
        k = int(np.argmin(d))
        v = int(self.values[k])
        return (v if v >= 0 and d[k] <= MAX_DIST else None), float(d[k])

    def read_frame(self, frame: np.ndarray) -> dict:
        out: dict = {}
        for seat, f, c in crops(frame):
            out.setdefault(seat, {})[f] = self.read(c, f)[0]
        return out


# ---------------------------------------------------------------------------
# building the prototypes
# ---------------------------------------------------------------------------
def harvest(step: float = 20.0) -> None:
    """Sample every video, cluster the glyphs, write a labelling sheet."""
    os.makedirs(WORK, exist_ok=True)
    aligner = V.Aligner()
    gl, src = [], []
    for video in sorted(glob.glob(os.path.join(HERE, "raw_tournament", "*.webm"))):
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fi in range(int(120 * fps), n, int(step * fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            a = aligner.align(frame)
            if a is None:
                continue
            for seat, f, c in crops(a[0]):
                g = glyph(c, f)
                if g is not None:
                    gl.append(g)
                    src.append(c)
        print(os.path.basename(video), len(gl), flush=True)
    G = np.stack(gl).reshape(len(gl), -1)
    # greedy leader clustering
    centers, members = [], []
    for i, g in enumerate(G):
        if centers:
            d = np.abs(np.stack(centers) - g).mean(axis=1)
            k = int(np.argmin(d))
            if d[k] < 0.12:
                members[k].append(i)
                continue
        centers.append(g)
        members.append([i])
    order = sorted(range(len(centers)), key=lambda k: -len(members[k]))
    protos = np.stack([G[members[k]].mean(axis=0) for k in order])
    sizes = [len(members[k]) for k in order]
    np.savez(os.path.join(WORK, "clusters.npz"), protos=protos, sizes=sizes)
    # labelling sheet: cluster id, 4 example crops each
    tiles = []
    for rank, k in enumerate(order):
        ex = [cv2.resize(src[i], (48, 48)) for i in members[k][:4]]
        ex += [np.zeros((48, 48, 3), np.uint8)] * (4 - len(ex))
        row = np.hstack(ex)
        lab = np.zeros((48, 60, 3), np.uint8)
        cv2.putText(lab, str(rank), (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        tiles.append(np.hstack([lab, row]))
    per = 40
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        cols = [np.vstack(chunk[i:i + 10] + [np.zeros_like(chunk[0])] * (10 - len(chunk[i:i + 10])))
                for i in range(0, len(chunk), 10)]
        cv2.imwrite(os.path.join(WORK, f"sheet_{s // per:02d}.png"), np.hstack(cols))
    print(f"{len(G)} glyphs -> {len(centers)} clusters; sheets in {WORK}")


def label(labels_path: str) -> None:
    """labels.json: {"cluster rank": value, ...}; -1 = not a number."""
    z = np.load(os.path.join(WORK, "clusters.npz"))
    with open(labels_path, encoding="utf-8") as f:
        lab = {int(k): int(v) for k, v in json.load(f).items()}
    keep = sorted(lab)
    np.savez(PROTOS, protos=z["protos"][keep].reshape(len(keep), GH, GW),
             values=np.array([lab[k] for k in keep]))
    print(f"{len(keep)} prototypes -> {PROTOS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("harvest")
    lp = sub.add_parser("label")
    lp.add_argument("labels")
    rp = sub.add_parser("read")
    rp.add_argument("video")
    rp.add_argument("--at", type=float, required=True)
    args = ap.parse_args()
    if args.cmd == "harvest":
        harvest()
    elif args.cmd == "label":
        label(args.labels)
    else:
        a = V.Aligner().align(V.grab(args.video, args.at))
        if a is None:
            raise SystemExit("board not visible")
        for seat, vals in Reader().read_frame(a[0]).items():
            print(seat, vals)


if __name__ == "__main__":
    main()
