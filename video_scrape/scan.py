#!/usr/bin/env python
"""
Scan a spectator-view TTS video into a card timeline.

Samples the video (default 1 frame/s), aligns each sample to the reference
layout, and re-reads a card region (hand strip, Agent Turn row, Reveal Turn
row, top of discard pile) only when its pixels changed and then held still
for one more sample, so hover animations and drags aren't read half-way.

Output: raw_tournament/<id>.timeline.json, a list of region readings
  {"t": 612.0, "seat": "TL", "region": "agent",
   "cards": [{"card": "signetRing", "score": 0.9, "rivals": null}, ...]}
written only when a region's card list changes. timeline_events.py turns
that into game actions.

Usage:
  python video_scrape/scan.py video_scrape/raw_tournament/NFT8aY_T3ZE.webm
  python video_scrape/scan.py VIDEO --start 300 --end 900 --step 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision as V  # noqa: E402

CHANGE = 6.0   # mean abs grey diff (on a 1/4-scale region) that counts as change
STILL = 3.0    # ...and below this between two samples counts as settled
REALIGN_EVERY = 10  # seconds between full keypoint re-alignments


def signature(frame: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, ((x1 - x0) // 4, (y1 - y0) // 4),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def diff(a: np.ndarray | None, b: np.ndarray) -> float:
    return 1e9 if a is None else float(np.abs(a - b).mean())


def samples(video: str, start: float, end: float | None, step: float):
    """Yield (t, frame) every `step` seconds, decoding sequentially (much
    faster than seeking)."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    n = int(round(start * fps))
    every = max(1, int(round(step * fps)))
    while True:
        t = n / fps
        if end is not None and t > end:
            return
        ok, frame = cap.read()
        if not ok:
            return
        yield t, frame
        for _ in range(every - 1):
            if not cap.grab():
                return
        n += every


def scan(video: str, start: float = 0, end: float | None = None,
         step: float = 1.0, log=print) -> list[dict]:
    lib = V.Library()
    aligner = V.Aligner()
    M, last_align = None, -1e9
    stable: dict[tuple, np.ndarray] = {}   # signature at last read
    pending: dict[tuple, np.ndarray] = {}  # changed, waiting to settle
    current: dict[tuple, list] = {}        # last card list per region
    out: list[dict] = []
    t0, reads = time.time(), 0

    for t, frame in samples(video, start, end, step):
        if M is None or t - last_align >= REALIGN_EVERY:
            r = aligner.transform(frame)
            last_align = t
            if r is None:  # menu/overlay: keep last transform if we had one
                if M is None:
                    continue
            else:
                M = r[0]
        warped, bottom = aligner.align(frame, M)
        for seat, regs in V.REGIONS.items():
            for rg, box in regs.items():
                key = (seat, rg)
                sig = signature(warped, box)
                if key in pending:
                    if diff(pending[key], sig) > STILL:  # still moving
                        pending[key] = sig
                        continue
                    del pending[key]
                elif diff(stable.get(key), sig) > CHANGE:
                    pending[key] = sig
                    continue
                else:
                    continue
                # settled after a change: read it
                stable[key] = sig
                hits = V.find_cards(warped, box, lib, V.REGION_KINDS[rg],
                                    bottom)
                reads += 1
                cards = [{"card": h.card, "score": round(h.score, 3),
                          "rivals": h.rivals} for h in hits]
                names = [c["card"] for c in cards]
                if names != [c["card"] for c in current.get(key, [])]:
                    current[key] = cards
                    out.append({"t": round(t, 1), "seat": seat, "region": rg,
                                "cards": cards})
        if int(t) % 60 == 0 and abs(t - round(t)) < step / 2:
            log(f"  {t / 60:5.1f} min  reads={reads}  changes={len(out)}  "
                f"({time.time() - t0:.0f}s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--out")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.video)[0] + ".timeline.json"
    tl = scan(args.video, args.start, args.end, args.step)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tl, f, indent=0)
    print(f"wrote {len(tl)} region changes -> {out}")


if __name__ == "__main__":
    main()
