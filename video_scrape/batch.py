#!/usr/bin/env python
"""
Download every video in a playlist and extract its chat log, in parallel.

Skips videos whose .chat.txt already exists, so it is safe to re-run when
new games are added to the playlist.

Usage:
  python video_scrape/batch.py https://www.youtube.com/playlist?list=...
  python video_scrape/batch.py URL --workers 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")


def playlist_ids(url: str) -> list[str]:
    out = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--flat-playlist", "--print",
         "%(id)s\t%(availability)s", url],
        capture_output=True, text=True, check=True).stdout
    ids = []
    for line in out.splitlines():
        vid, _, avail = line.partition("\t")
        if vid and avail in ("public", "unlisted", "NA", ""):
            ids.append(vid)
    return ids


def process(vid: str) -> str:
    chat = os.path.join(RAW, f"{vid}.chat.txt")
    if os.path.exists(chat):
        return f"{vid}: already extracted"
    video = next((os.path.join(RAW, f) for f in os.listdir(RAW)
                  if f.startswith(vid + ".") and f.endswith((".webm", ".mp4",
                                                             ".mkv"))), None)
    if video is None:
        r = subprocess.run(
            [sys.executable, "-m", "yt_dlp",
             "--ffmpeg-location", imageio_ffmpeg.get_ffmpeg_exe(),
             "-f", "bv*[height<=1080]+ba/b[height<=1080]",
             "-o", os.path.join(RAW, "%(id)s.%(ext)s"),
             "--write-info-json", "--no-playlist", "--quiet",
             f"https://youtu.be/{vid}"], capture_output=True, text=True)
        if r.returncode != 0:
            return f"{vid}: download failed: {r.stderr.strip()[-300:]}"
        video = next(os.path.join(RAW, f) for f in os.listdir(RAW)
                     if f.startswith(vid + ".") and
                     f.endswith((".webm", ".mp4", ".mkv")))
    log = os.path.join(RAW, f"{vid}.extract.log")
    with open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "chat_extract.py"), video,
             "--fps", "2.5"], stdout=fh, stderr=subprocess.STDOUT)
    return f"{vid}: {'ok' if r.returncode == 0 else 'FAILED, see ' + log}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("playlist")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    ids = playlist_ids(args.playlist)
    print(f"{len(ids)} videos", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process, v) for v in ids]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
