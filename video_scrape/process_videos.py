#!/usr/bin/env python
"""
Extract everything from downloaded videos, then DELETE each video.

For every <id>.webm/.mp4/.mkv in a folder (default raw_streams/):
  1. scan.py            -> <id>.timeline.json   (cards, Row, Conflict, numbers)
  2. chat_extract.py    -> <id>.chat.txt        (TTS chat log, when on screen)
  3. timeline_events.py -> games/<id>[_gN].json  (one file per game)
  4. delete the video file (kept: info.json, timeline, chat log, games)
A video is only deleted once steps 1-3 all succeeded; failures keep the video
and are logged, so a re-run picks them up. Videos run in parallel.

Usage:
  python video_scrape/process_videos.py                       # raw_streams/
  python video_scrape/process_videos.py --dir video_scrape/raw_tournament
  python video_scrape/process_videos.py --workers 12 --keep   # don't delete
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
VIDEO_EXT = (".webm", ".mp4", ".mkv")


def _run(args: list[str], log: str) -> bool:
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(args) + "\n")
        f.flush()
        r = subprocess.run([PY, "-u"] + args, stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return r.returncode == 0


def process(video: str, keep: bool) -> str:
    base = os.path.splitext(video)[0]
    vid = os.path.basename(base)
    log = base + ".process.log"
    if os.path.exists(base + ".done"):
        return f"{vid}: already done"
    ok = os.path.exists(base + ".timeline.json") or \
        _run([os.path.join(HERE, "scan.py"), video], log)
    ok = ok and (os.path.exists(base + ".chat.txt") or
                 _run([os.path.join(HERE, "chat_extract.py"), video], log))
    ok = ok and _run([os.path.join(HERE, "timeline_events.py"), video], log)
    if not ok:
        return f"{vid}: FAILED (video kept, see {log})"
    open(base + ".done", "w").close()
    if not keep:
        os.remove(video)
    return f"{vid}: done{'' if keep else ', video deleted'}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "raw_streams"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--keep", action="store_true", help="don't delete videos")
    args = ap.parse_args()
    videos = sorted(p for p in glob.glob(os.path.join(args.dir, "*"))
                    if p.endswith(VIDEO_EXT) and ".f" not in os.path.basename(p))
    # videos with checkpointed progress first, so a restart finishes them soonest
    videos.sort(key=lambda p: not glob.glob(os.path.splitext(p)[0] + ".*.partial")
                and not os.path.exists(os.path.splitext(p)[0] + ".timeline.json"))
    print(f"{len(videos)} videos, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(process, v, args.keep) for v in videos]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
