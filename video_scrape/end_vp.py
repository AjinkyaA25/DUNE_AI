#!/usr/bin/env python
"""
Final VP from the end-screen frames saved by end_frames.py.

Each frame is aligned to the board reference and the VP track is read
(board_state.read_vp). Readings from a game's frames are combined per
colour (most common value). A disc hidden under another reads as None; a
hidden disc sits at the same height as a visible one, so the winner is only
called when no hidden colour could be tied with the top score.

Usage:
  python video_scrape/end_vp.py               # every end_frames/<game>.jpg
  python video_scrape/end_vp.py --eval        # vs the chat's final standings
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_state as BS  # noqa: E402
import vision as V  # noqa: E402


def read_game(gid: str, aligner) -> dict:
    reads = collections.defaultdict(list)
    n = 0
    for p in sorted(glob.glob(os.path.join(HERE, "end_frames", gid + "*.jpg"))):
        if os.path.basename(p)[len(gid):] not in (".jpg", "_1.jpg", "_2.jpg"):
            continue
        a = aligner.align(cv2.imread(p))
        if a is None:
            continue
        n += 1
        for c, v in BS.read_vp(a[0]).items():
            if v is not None:
                reads[c].append(v)
    vp = {c: collections.Counter(v).most_common(1)[0][0] for c, v in reads.items()}
    hidden = [c for c in BS.HUES if c not in vp]
    winner = None
    if vp:
        top = max(vp.values())
        best = [c for c, v in vp.items() if v == top]
        # a hidden disc could be under the top one: only call it when
        # every colour was seen
        if len(best) == 1 and not hidden:
            winner = best[0]
    return {"frames": n, "vp": vp, "hidden": hidden, "winner": winner}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    al = V.Aligner()
    gids = sorted({os.path.basename(p)[:-4] for p in
                   glob.glob(os.path.join(HERE, "end_frames", "*.jpg"))
                   if not p.endswith(("_1.jpg", "_2.jpg"))})
    out = {}
    right = wrong = undecided = 0
    for gid in gids:
        r = read_game(gid, al)
        out[gid] = r
        line = f"{gid:22s} frames={r['frames']} vp={r['vp']} hidden={r['hidden']} -> {r['winner']}"
        if a.eval:
            g = json.load(open(os.path.join(HERE, "games", gid + ".json"), encoding="utf-8"))
            first = [x["player"] for x in g.get("result") or [] if x.get("place") == 1]
            col = {p["name"]: p["color"] for p in g["players"]}
            truth = col.get(first[0]) if len(first) == 1 else None
            if truth is None:
                continue
            if r["winner"] is None:
                undecided += 1
            elif r["winner"] == truth:
                right += 1
            else:
                wrong += 1
            line += f"   truth={truth} {'OK' if r['winner'] == truth else ('?' if r['winner'] is None else 'WRONG')}"
        print(line, flush=True)
    if a.eval:
        print(f"right {right}  wrong {wrong}  undecided {undecided}")
    json.dump(out, open(os.path.join(HERE, "end_frames", "end_vp.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
