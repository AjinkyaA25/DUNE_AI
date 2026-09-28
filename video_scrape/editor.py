#!/usr/bin/env python
"""
Local editor for the extracted game files (video_scrape/games/*.json).

Opens a page with the YouTube video next to the move list: click a move to
jump the video there, then fix, delete or insert moves (takebacks, OCR
mistakes, things the chat log never showed). Saving writes the game file back
with "edited": true -- parse_log.py will not overwrite an edited game unless
you pass --force -- and keeps the previous version under games/.history/.

Usage:
  python video_scrape/editor.py            # then open http://localhost:8765
  python video_scrape/editor.py --port 9000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from parse_log import CARDS, GAMES, SARDAUKAR_SKILLS, SPACES, VP_SOURCES

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(GAMES, ".history")


def _game_path(vid: str) -> str | None:
    path = os.path.join(GAMES, f"{os.path.basename(vid)}.json")
    return path if os.path.exists(path) else None


def _vocab() -> dict:
    """Names for the editor's autocomplete: engine names plus every name
    already used in any game file (so Bloodlines cards you have typed once
    are offered everywhere)."""
    cards, contracts, techs = set(CARDS), set(), set()
    for p in glob.glob(os.path.join(GAMES, "*.json")):
        with open(p, encoding="utf-8") as f:
            for a in json.load(f)["actions"]:
                if a.get("card"):
                    cards.add(a["card"])
                if a["kind"] in ("contract", "fulfill_contract") and a.get("name"):
                    contracts.add(a["name"])
                if a["kind"] == "tech" and a.get("name"):
                    techs.add(a["name"])
    return {"spaces": SPACES, "cards": sorted(cards), "skills": SARDAUKAR_SKILLS,
            "vp": VP_SOURCES, "contracts": sorted(contracts),
            "techs": sorted(techs)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype="application/json") -> None:
        data = body if isinstance(body, bytes) else (
            json.dumps(body, ensure_ascii=False).encode("utf-8")
            if ctype == "application/json" else body.encode("utf-8"))
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a) -> None:  # keep the console quiet
        pass

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "editor.html"), encoding="utf-8") as f:
                return self._send(200, f.read(), "text/html")
        if self.path == "/api/games":
            out = []
            for p in sorted(glob.glob(os.path.join(GAMES, "*.json"))):
                with open(p, encoding="utf-8") as f:
                    g = json.load(f)
                out.append({"video_id": g["video_id"], "title": g["title"],
                            "edited": g.get("edited", False),
                            "flags": sum(bool(a["flags"]) for a in g["actions"])})
            return self._send(200, out)
        if self.path == "/api/vocab":
            return self._send(200, _vocab())
        if self.path.startswith("/api/game/"):
            path = _game_path(self.path.rsplit("/", 1)[1])
            if not path:
                return self._send(404, {"error": "no such game"})
            with open(path, encoding="utf-8") as f:
                return self._send(200, json.load(f))
        self._send(404, {"error": "not found"})

    def do_PUT(self) -> None:
        if not self.path.startswith("/api/game/"):
            return self._send(404, {"error": "not found"})
        vid = self.path.rsplit("/", 1)[1]
        path = _game_path(vid)
        if not path:
            return self._send(404, {"error": "no such game"})
        try:
            g = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert g["video_id"] == vid and isinstance(g["actions"], list)
        except Exception as e:  # noqa: BLE001 - report any bad payload
            return self._send(400, {"error": f"bad game data: {e}"})
        os.makedirs(HISTORY, exist_ok=True)
        shutil.copy(path, os.path.join(
            HISTORY, f"{vid}.{time.strftime('%Y%m%d-%H%M%S')}.json"))
        g["edited"] = True
        for i, a in enumerate(g["actions"]):
            a["id"] = i
        with open(path, "w", encoding="utf-8") as f:
            json.dump(g, f, indent=1, ensure_ascii=False)
        self._send(200, {"ok": True, "actions": len(g["actions"])})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"game editor running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
