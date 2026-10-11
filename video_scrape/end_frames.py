#!/usr/bin/env python
"""
Grab each game's end screen so its winner can be verified by eye.

The board-VP reader picks the wrong winner in ~30% of the games checked
against the chat's final standings (13 of 43), and the full videos are
deleted after extraction. This downloads only the last round of each game
(720p), finds the first frame showing the mod's "Play another round?"
prompt (game over, final VP on every board), and saves it as
end_frames/<game>.jpg (plus <game>_1/_2.jpg, 25 s and 60 s later). Without the prompt, the last frame is saved instead.

Usage:
  python video_scrape/end_frames.py TODO.json [--workers 4]
TODO.json: [[game_id, last_round_t, last_action_t, rounds], ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "end_frames")
CLIPS = os.path.join(HERE, "end_clips")


def download(vid: str, a: float, b: float, dst: str) -> str | None:
    import imageio_ffmpeg
    r = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--js-runtimes", "node",
         "--remote-components", "ejs:github",
         "--ffmpeg-location", imageio_ffmpeg.get_ffmpeg_exe(),
         "-f", "232/bv*[height<=720][protocol^=m3u8]/bv*[height<=720]",
         "--download-sections", f"*{int(a)}-{int(b)}",
         "-o", dst, "--no-playlist", "--quiet", "--no-warnings",
         f"https://youtu.be/{vid}"], capture_output=True, text=True)
    return None if r.returncode == 0 and os.path.exists(dst) else r.stderr[-300:]


def find_end(path: str, ocr) -> tuple[float, object]:
    cap = cv2.VideoCapture(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = n / fps
    last = None
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        last = (t, f)
        h, w = f.shape[:2]
        res, _ = ocr(f[int(h * .5):int(h * .9), int(w * .3):int(w * .7)])
        if any("another" in x[1].lower() for x in (res or [])):
            # the prompt pops up as the final VP lands; a few seconds on
            # the boards have settled
            # the camera often moves after the game ends: also keep two later
            # frames in case a board is off-screen in the first
            frames = []
            for dt in (3, 25, 60):
                cap.set(cv2.CAP_PROP_POS_MSEC, (t + dt) * 1000)
                ok, f2 = cap.read()
                if ok:
                    frames.append(f2)
            return t, frames or [f]
        t += 8
    return -1.0, [last[1]] if last else None


def one(item, ocr):
    gid, lr, la, _ = item
    vid = re.sub(r"_g\d+$", "", gid)
    out = os.path.join(OUT, gid + ".jpg")
    if os.path.exists(out):
        return f"{gid}: done already"
    clip = os.path.join(CLIPS, gid + ".mp4")
    if not os.path.exists(clip):
        err = download(vid, lr, max(la, lr) + 240, clip)
        if err:
            return f"{gid}: download FAILED {err}"
    t, f = find_end(clip, ocr)
    if f is None:
        return f"{gid}: unreadable clip"
    for k, fr in enumerate(f):
        cv2.imwrite(out if k == 0 else out[:-4] + f"_{k}.jpg", cv2.resize(fr, (1280, 720)))
    os.remove(clip)
    return f"{gid}: prompt at +{t:.0f}s" if t >= 0 else f"{gid}: NO prompt (last frame saved)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("todo")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CLIPS, exist_ok=True)
    from rapidocr_onnxruntime import RapidOCR
    todo = json.load(open(a.todo, encoding="utf-8"))
    local = threading.local()

    def run(item):
        if not hasattr(local, "ocr"):
            local.ocr = RapidOCR()
        return one(item, local.ocr)

    with ThreadPoolExecutor(a.workers) as ex:
        for i, msg in enumerate(ex.map(run, todo)):
            print(f"[{i + 1}/{len(todo)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
