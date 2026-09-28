#!/usr/bin/env python
"""
Build the card reference library from the Dune Uprising TTS mod.

The mod save lists every card with a stable id in GMNotes
("duneTheDesertPlanet") and a CardID = sheet*100 + index into a sprite sheet
(CustomDeck FaceURL, NumWidth x NumHeight cards). TTS caches those sheets in
Mods/Images under the URL stripped of punctuation, so the whole library can be
cut out locally without downloading anything.

Output: video_scrape/cards/<id>__<sheet>.png (one image per variant; the mod
has English and French printings of some cards) and cards/index.json mapping
id -> {name, files}.

Usage:
  python video_scrape/card_library.py
  python video_scrape/card_library.py --mod-dir "D:/.../Tabletop Simulator/Mods"
"""
from __future__ import annotations

import argparse
import json
import os
import re

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cards")
MOD_ID = "3522149839"
CARD_H = 280
BACK = "_back"  # library id for any face-down card
DEFAULT_MODS = os.path.expanduser(
    "~/OneDrive/Documents/My Games/Tabletop Simulator/Mods")


def walk(o, cards: dict, decks: dict, backs: dict) -> None:
    if isinstance(o, dict):
        if o.get("Name") in ("Card", "CardCustom") and "CardID" in o:
            gm = (o.get("GMNotes") or "").strip()
            if gm and gm != "back":
                cards.setdefault(o["CardID"], (gm, o.get("Nickname", "")))
        for k, v in (o.get("CustomDeck") or {}).items():
            decks[int(k)] = (v["FaceURL"], v["NumWidth"], v["NumHeight"])
            backs.setdefault(v["BackURL"], int(k))
        for v in o.values():
            walk(v, cards, decks, backs)
    elif isinstance(o, list):
        for v in o:
            walk(v, cards, decks, backs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod-dir", default=DEFAULT_MODS)
    args = ap.parse_args()

    save = os.path.join(args.mod_dir, "Workshop", f"{MOD_ID}.json")
    with open(save, encoding="utf-8") as f:
        d = json.load(f)
    cards: dict[int, tuple[str, str]] = {}
    decks: dict[int, tuple[str, int, int]] = {}
    backs: dict[str, int] = {}  # BackURL -> first sheet using it
    walk(d, cards, decks, backs)

    img_dir = os.path.join(args.mod_dir, "Images")
    cached = os.listdir(img_dir)

    def sheet_file(url: str) -> str | None:
        key = re.sub(r"[^A-Za-z0-9]", "", url)
        hit = [f for f in cached if os.path.splitext(f)[0] == key]
        return os.path.join(img_dir, hit[0]) if hit else None

    os.makedirs(OUT, exist_ok=True)
    sheets: dict[int, np.ndarray | None] = {}
    index: dict[str, dict] = {}
    missing = 0
    blank: list[str] = []

    def save(crop: np.ndarray, gm: str, nick: str, fn: str) -> None:
        # cards are ~70px tall in a 1080p video; 280px is plenty
        if crop.shape[0] > CARD_H:
            crop = cv2.resize(crop, (round(crop.shape[1] * CARD_H / crop.shape[0]),
                                     CARD_H), interpolation=cv2.INTER_AREA)
        cv2.imencode(".png", crop)[1].tofile(os.path.join(OUT, fn))
        e = index.setdefault(gm, {"names": [], "files": []})
        if nick and nick not in e["names"]:
            e["names"].append(nick)
        e["files"].append(fn)

    for cid, (gm, nick) in sorted(cards.items()):
        sid, pos = divmod(cid, 100)
        if sid not in decks:
            missing += 1
            continue
        url, nw, nh = decks[sid]
        if sid not in sheets:
            path = sheet_file(url)
            # cv2.imread can't open non-ASCII Windows paths; decode bytes
            sheets[sid] = (cv2.imdecode(np.fromfile(path, np.uint8),
                                        cv2.IMREAD_COLOR) if path else None)
        sheet = sheets[sid]
        if sheet is None or pos >= nw * nh:
            missing += 1
            continue
        h, w = sheet.shape[:2]
        cw, ch = w / nw, h / nh
        r, c = divmod(pos, nw)
        crop = sheet[int(r * ch):int((r + 1) * ch), int(c * cw):int((c + 1) * cw)]
        if crop.std() < 5:  # blank cell (sheet smaller than declared)
            blank.append(gm)
            continue
        save(crop, gm, nick, f"{gm}__{sid}.png")

    # card backs (all non-unique: one image per deck) so face-down cards in
    # hidden hands are recognised as such instead of as the nearest face
    for url, sid in backs.items():
        path = sheet_file(url)
        if path:
            im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            save(im, BACK, "face-down card", f"{BACK}__{sid}.png")

    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1, ensure_ascii=False, sort_keys=True)
    print(f"{len(index)} cards, {sum(len(e['files']) for e in index.values())} "
          f"images -> {OUT}  ({missing} card ids without a sheet)")
    if blank:
        print("blank crops skipped:", ", ".join(sorted(set(blank))))


if __name__ == "__main__":
    main()
