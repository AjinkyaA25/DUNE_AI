#!/usr/bin/env python
"""
Pull the TTS game log out of a recorded Dune Imperium Uprising video.

The TTS mod prints every action to the chat box in the bottom-left corner
("Sending an agent to: Deep Desert (Signet Ring).", "+7 spice units", ...).
That box only shows the last ~7 lines, so we sample frames, OCR the box only
when its pixels change, and stitch overlapping snapshots into one continuous
log. Output: one line per chat line, prefixed with the video timestamp.

Usage:
  python video_scrape/chat_extract.py video_scrape/raw/vjcSLuiye7A.webm
  python video_scrape/chat_extract.py VIDEO --start 60 --end 600 --fps 3
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import time

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Chat box on a 1920x1080 recording (scaled for other resolutions).
CHAT_BOX = (8, 908, 400, 1028)
BASE_W, BASE_H = 1920, 1080


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _same(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return na == nb
    # OCR typos change a character or two; different log lines like
    # "Phase: combat" vs "Phase: combat end" differ in length
    if abs(len(na) - len(nb)) > max(2, 0.12 * max(len(na), len(nb))):
        return False
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.85


def ocr_lines(ocr: RapidOCR, crop: np.ndarray) -> list[str]:
    """OCR the chat crop and return its text lines top-to-bottom."""
    big = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    res, _ = ocr(big, use_cls=False)
    if not res:
        return []
    # group fragments into rows by vertical centre
    items = []
    for box, txt, conf in res:
        if float(conf) < 0.5:
            continue
        ys = [p[1] for p in box]
        items.append(((min(ys) + max(ys)) / 2, box[0][0], txt))
    items.sort()
    rows: list[list[tuple[float, float, str]]] = []
    for it in items:
        if rows and abs(rows[-1][0][0] - it[0]) < 18:
            rows[-1].append(it)
        else:
            rows.append([it])
    lines = [" ".join(t for _, _, t in sorted(r, key=lambda z: z[1])).strip()
             for r in rows]
    # floating TTS player-name tags sometimes overlap the box; drop
    # fragments that are too short to be a log line
    return [l for l in lines if len(_norm(l)) >= 4]


def stitch(log: list[str], snap: list[str]) -> list[str]:
    """Return the lines of `snap` that are new relative to the tail of `log`.

    The newest chat lines are at the bottom, so we look for the longest
    prefix of `snap` that matches a suffix of `log`; everything after it
    is new. If nothing overlaps (lines scrolled past between samples) the
    whole snapshot is new and we mark the gap.
    """
    if not log:
        return snap
    tail = [l for l in log[-len(snap) - 4:] if l != "[GAP?]"]
    # 1) prefix of snap aligned to suffix of tail, tolerating one misread
    #    line once the overlap is long enough to be unambiguous
    for k in range(min(len(snap), len(tail)), 0, -1):
        miss = sum(not _same(tail[len(tail) - k + i], snap[i])
                   for i in range(k))
        if miss == 0 or (k >= 4 and miss <= 1):
            return snap[k:]
    # 2) the snapshot starts with already-logged lines (possibly one misread
    #    among them): new lines begin after that leading block. Only the
    #    leading block counts -- later lines like "+2 solaris" or "Turn: X"
    #    repeat all game and would otherwise swallow genuinely new lines.
    seen = [any(_same(s, t) for t in tail) for s in snap]
    cut, misses = -1, 0
    for i, v in enumerate(seen):
        if v:
            cut, misses = i, 0
        else:
            misses += 1
            if misses >= 2:
                break
    if cut >= 1 or (cut == 0 and len(snap) == 1):
        return snap[cut + 1:]
    return ["[GAP?]"] + snap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=2.0,
                    help="frames per second to sample")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=None, help="seconds")
    ap.add_argument("--diff", type=float, default=4.0,
                    help="mean abs pixel change in the chat box that "
                         "triggers a fresh OCR")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.video)[0] + ".chat.txt"
    cap = cv2.VideoCapture(args.video)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sx, sy = w / BASE_W, h / BASE_H
    x0, y0, x1, y1 = (int(CHAT_BOX[0] * sx), int(CHAT_BOX[1] * sy),
                      int(CHAT_BOX[2] * sx), int(CHAT_BOX[3] * sy))
    end = args.end if args.end is not None else nframes / vfps
    step = max(1, int(round(vfps / args.fps)))

    ocr = RapidOCR()
    log: list[str] = []
    stamps: list[float] = []
    prev = None
    n_ocr = 0
    t_start = time.time()
    fi = int(args.start * vfps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    while fi / vfps <= end:
        ok = cap.grab()
        if not ok:
            break
        if fi % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            crop = frame[y0:y1, x0:x1]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if prev is None or float(np.mean(cv2.absdiff(gray, prev))) > args.diff:
                prev = gray
                snap = ocr_lines(ocr, crop)
                n_ocr += 1
                new = stitch(log, snap)
                t = fi / vfps
                log.extend(new)
                stamps.extend([t] * len(new))
                if n_ocr % 50 == 0:
                    el = time.time() - t_start
                    print(f"  {t/60:5.1f} min  ocr={n_ocr}  lines={len(log)}"
                          f"  ({el:.0f}s elapsed)", flush=True)
        fi += 1

    with open(out, "w", encoding="utf-8") as f:
        for t, line in zip(stamps, log):
            f.write(f"{int(t//60):02d}:{t%60:05.2f}\t{line}\n")
    print(f"wrote {len(log)} lines -> {out}  ({n_ocr} OCR calls, "
          f"{time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
