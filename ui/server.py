#!/usr/bin/env python
"""
Play Dune Imperium: Uprising (+ Bloodlines) against the AI in the browser.

  python ui/server.py                 # then open http://localhost:8765
  python ui/server.py --port 9000

The engine runs here; the page (ui/static/) polls /api/state and posts your
moves. AI seats move in a background thread and every AI decision carries an
explanation: for search agents the candidate moves it played out and their
estimated win chance, for heuristic agents its top-scored moves.

Like play_game.py, every finished game is saved to game_logs/ and your own
decisions to data/human_games/ (training shard format).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import re
import socket
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import play_game as PG  # noqa: E402  (labels, training-shard + log helpers)
from src.ai.action_features import encode_action  # noqa: E402
from src.ai.agents import make_agent  # noqa: E402
from src.ai.features import encode_state  # noqa: E402
from src.data.card_definitions import setup_game  # noqa: E402
from src.game.board.board import UPRISING_BOARD  # noqa: E402
from src.game.gameState import ActionType, GameAction, GameState  # noqa: E402

STATIC = os.path.join(HERE, "static")
CARD_DIR = os.path.join(ROOT, "video_scrape", "cards")

OPPONENTS = [
    {"spec": "search:K5:M24", "name": "Search (24 playouts)",
     "note": "strongest; ~5 s per decision"},
    {"spec": "search:K5:M8", "name": "Search (8 playouts)",
     "note": "strong; ~2 s per decision"},
    {"spec": "heuristic:tuned=config/heuristic_tuned.json", "name": "Heuristic (human-tuned)",
     "note": "instant; plays most like the humans in your videos"},
    {"spec": "heuristic", "name": "Heuristic (default)", "note": "instant"},
    {"spec": "random", "name": "Random", "note": "for testing"},
]
# what the spaces with engine-coded effects do (shown on the board)
SPECIAL = {   # rulebook wording, shortened (rulebook_spaces_text.txt)
    "High Council": "1st time: Council seat (+2 persuasion every reveal); after: 2 spice, intrigue, 3 troops",
    "Swordmaster": "8 solari (6 once anyone has one) → 3rd agent; once per game",
    "Gather Support": "pay 0 or 2 solari → 2 troops (+1 water if paid)",
    "Spice Refinery": "pay 0 or 1 spice → 2 solari (4 if paid); controller +1 solari",
    "Sietch Tabr": "Maker Hooks + troop + water, OR water + may remove Shield Wall",
    "Hagga Basin": "bonus spice + 2 spice, OR summon a sandworm (with Hooks)",
    "Deep Desert": "bonus spice + 4 spice, OR summon 2 sandworms (with Hooks)",
    "Imperial Basin": "+ bonus spice; controller +1 spice",
    "Arrakeen": "controller +1 solari",
}
COLORS = ["#c0392b", "#27ae60", "#2e6fd8", "#d4a017"]   # red green blue yellow
COLOR_NAMES = ["Red", "Green", "Blue", "Yellow"]


# ---------------------------------------------------------------------------
# Card art (extracted from the TTS mod by video_scrape/card_library.py)
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _card_images() -> Dict[str, str]:
    idx_path = os.path.join(CARD_DIR, "index.json")
    if not os.path.exists(idx_path):
        return {}
    idx = json.load(open(idx_path, encoding="utf-8"))
    out = {}
    for cid, v in idx.items():
        if cid.startswith("_") or not v.get("files"):
            continue
        for n in v.get("names", [])[:1]:          # English name first
            out.setdefault(_norm(n), v["files"][0])
        out.setdefault(_norm(cid), v["files"][0])
    return out


CARD_IMG = _card_images()


def card_img(name: str) -> Optional[str]:
    f = CARD_IMG.get(_norm(name))
    return f"/cards/{f}" if f else None


# ---------------------------------------------------------------------------
# Labels / serialisation
# ---------------------------------------------------------------------------

def label(gs: GameState, a: GameAction) -> str:
    t = a.action_type
    if t == ActionType.AGENT_TURN:
        mods = [m for m, on in (("Gather Intelligence", a.use_gather_intelligence),
                                ("Infiltrate", a.use_infiltrate)) if on]
        if a.space_option:
            mods.append(a.space_option)
        return f"{a.card_name} → {a.space_name}" + (f" [{', '.join(mods)}]" if mods else "")
    if t == ActionType.RESOLVE_CONTRACT:
        try:
            c = gs.contracts_on_board[a.contract_index]
            return f"Take contract: {c.name}"
        except Exception:
            return f"Take contract #{a.contract_index}"
    if t == ActionType.RESOLVE_OPTIONAL:
        return "Accept optional cost → reward" if a.accept_optional else "Decline optional cost"
    if t == ActionType.RESOLVE_BL_CHOICE:
        return f"Choose: {a.choice}"
    if t == ActionType.ACTIVATE_TECH:
        return f"Activate tech: {a.choice or a.card_name}"
    if t == ActionType.REVEAL_TURN:
        return "Reveal turn"
    if t == ActionType.END_REVEAL:
        return "Done buying — end turn"
    return PG.action_label(a)


def card_info(c) -> Dict:
    g = lambda k, d=None: getattr(c, k, d)   # noqa: E731
    syms = g("access_symbols") or []
    return {
        "name": g("name"), "cost": g("cost"),
        "persuasion": g("persuasion") or 0, "swords": g("swords") or 0,
        "access": sorted(getattr(s, "value", str(s)) for s in syms),
        "agent": g("agent_effects") or [], "reveal": g("reveal_effects") or [],
        "img": card_img(g("name")),
    }


def active_player(gs: GameState) -> int:
    p = gs.player_in_reveal_buy
    return gs.get_current_player_id() if p is None else p


# ---------------------------------------------------------------------------
# Game session
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, seat: int, specs: List[str], seed: Optional[int],
                 bloodlines: bool, players: int = 4):
        self.seed = seed if seed is not None else random.randint(0, 999_999)
        random.seed(self.seed)
        self.gs = setup_game(num_players=players, seed=self.seed,
                             neutral_leaders=True, use_bloodlines=bloodlines)
        self.seat = seat
        self.specs = {}
        self.agents = {}
        opp = iter(specs)
        for i in range(players):
            if i == seat:
                continue
            spec = next(opp, OPPONENTS[0]["spec"])
            self.specs[i] = spec
            self.agents[i] = make_agent(spec, seed=self.seed * 8 + i)
        self.id = str(uuid.uuid4())
        self.log: List[Dict] = []
        self.thinking: Optional[int] = None
        self.think_started = 0.0
        self.error: Optional[str] = None
        self.lock = threading.RLock()
        self.moves = []                       # play_game-format move log
        self.train = {"feats": [], "rounds": [], "actA": [], "ci": []}
        self.saved: Optional[Dict] = None
        self._thread: Optional[threading.Thread] = None
        self.kick()

    # -- AI loop -----------------------------------------------------------
    def kick(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_ai, daemon=True)
        self._thread.start()

    def _run_ai(self) -> None:
        try:
            while True:
                with self.lock:
                    gs = self.gs
                    if gs.game_over:
                        self._finish()
                        return
                    pid = active_player(gs)
                    if pid == self.seat:
                        return
                    agent = self.agents[pid]
                    valid = gs.get_valid_actions(pid)
                    self.thinking, self.think_started = pid, time.time()
                # think outside the lock so the page stays responsive
                view = gs.clone()
                t0 = time.time()
                act = agent.select_action(view, pid, view.get_valid_actions(pid))
                secs = time.time() - t0
                expl = self._explain(agent, view, pid, act)
                with self.lock:
                    real = next((a for a in valid if repr(a) == repr(act)), None)
                    if real is None:              # should not happen; stay safe
                        real = valid[0]
                    self._apply(pid, real, explain=expl, secs=secs)
                    self.thinking = None
        except Exception:
            self.error = traceback.format_exc()
            self.thinking = None
            print(self.error, file=sys.stderr)

    def _explain(self, agent, gs: GameState, pid: int, act: GameAction) -> Optional[Dict]:
        last = getattr(agent, "last", None)
        if last and last.get("cands"):
            h = getattr(agent, "h", None)
            rows = []
            for i, (c, v) in enumerate(zip(last["cands"], last["scores"])):
                rows.append({"label": label(gs, c), "value": round(float(v), 3),
                             "heur": round(float(h.score(gs, pid, c)), 2) if h else None,
                             "chosen": i == last["chosen"]})
            note = (f"Played each move out to the end of the game {getattr(agent, 'm', '?')} "
                    "times; value ≈ win chance (+0.02 per VP of margin).")
            best = max(range(len(rows)), key=lambda i: rows[i]["value"])
            if not rows[best]["chosen"]:
                note += (" Search keeps the heuristic's pick (first row) unless another move "
                         "beats it by more than the playout noise; this gap wasn't clear enough.")
            return {"kind": "search", "rows": rows, "note": note}
        score = getattr(agent, "score", None)
        if callable(score):
            acts = [a for a in gs.get_valid_actions(pid) if a.action_type != ActionType.NO_OP]
            if len(acts) < 2:
                return None
            sc = sorted(((float(score(gs, pid, a)), a) for a in acts),
                        key=lambda t: -t[0])[:5]
            return {"kind": "heuristic",
                    "rows": [{"label": label(gs, a), "value": round(s, 2),
                              "chosen": repr(a) == repr(act)} for s, a in sc],
                    "note": "Heuristic scores of the top moves (higher = preferred)."}
        return None

    # -- moves -------------------------------------------------------------
    def _apply(self, pid: int, act: GameAction, explain=None, secs=0.0) -> None:
        gs = self.gs
        before = gs.get_state_dict()
        entry = {"n": len(self.log), "round": gs.round, "phase": gs.phase.value,
                 "pid": pid, "label": label(gs, act), "type": act.action_type.value,
                 "explain": explain, "secs": round(secs, 1)}
        _, reward, _, info = gs.step(act)
        if info.get("error"):
            entry["error"] = info["error"]
        self.log.append(entry)
        self.moves.append({"move_num": len(self.moves), "round": before["round"],
                           "phase": before["phase"], "active_player": pid,
                           "is_human": pid == self.seat, "action": PG.action_to_dict(act),
                           "reward": reward, "error": info.get("error")})

    def human_move(self, index: int) -> Optional[str]:
        with self.lock:
            gs = self.gs
            if gs.game_over or active_player(gs) != self.seat:
                return "not your turn"
            valid = gs.get_valid_actions(self.seat)
            if not 0 <= index < len(valid):
                return "bad action index"
            act = valid[index]
            self._record_training(valid, act)
            self._apply(self.seat, act)
        self.kick()
        return None

    def _record_training(self, valid, act) -> None:
        """Your decisions in the shard format train_from_human.py reads
        (same as play_game.py)."""
        gs, pid = self.gs, self.seat
        non = [a for a in valid if a.action_type != ActionType.NO_OP]
        if len(non) < 1:
            return
        sc = PG._pol_scorer()
        hs = [sc.score(gs, pid, a) for a in non]
        keep = sorted(range(len(non)), key=lambda k: hs[k], reverse=True)[:PG.POLICY_KMAX]
        ai = next((k for k, a in enumerate(non) if a is act), 0)
        if ai not in keep:
            keep = keep[:PG.POLICY_KMAX - 1] + [ai]
        chs = [hs[k] for k in keep]
        lo, span = min(chs), (max(chs) - min(chs)) or 1.0
        self.train["feats"].append(encode_state(gs, pid))
        self.train["rounds"].append(gs.round)
        self.train["actA"].append(np.stack([encode_action(gs, pid, non[k], (h - lo) / span)
                                            for k, h in zip(keep, chs)]))
        self.train["ci"].append(keep.index(ai))

    def _finish(self) -> None:
        if self.saved is not None:
            return
        gs = self.gs
        vp = [p.victory_points for p in gs.players]
        winner = gs.winner if gs.winner is not None else int(np.argmax(vp))
        log = {"game_id": self.id, "timestamp": datetime.now().isoformat(),
               "seed": self.seed, "num_players": len(gs.players), "human_id": self.seat,
               "agents": {str(k): v for k, v in self.specs.items()},
               "winner": winner, "final_vp": vp, "moves": self.moves}
        self.saved = {"log": PG.save_log(log, "game_logs")}
        if self.train["feats"]:
            self.saved["training"] = PG.save_training_shard(
                self.train["feats"], self.train["rounds"], self.train["actA"],
                self.train["ci"], human_won=(winner == self.seat), rounds_played=gs.round,
                out_dir="data/human_games", game_id=self.id)

    # -- view --------------------------------------------------------------
    def view(self, reveal: bool) -> Dict:
        with self.lock:
            gs = self.gs
            d = gs.get_state_dict()
            act = None if gs.game_over else active_player(gs)
            players = []
            for p, pd in zip(gs.players, d["players"]):
                mine = p.id == self.seat
                show = mine or reveal
                players.append({
                    "id": p.id, "color": COLORS[p.id % 4], "colorName": COLOR_NAMES[p.id % 4],
                    "you": mine, "agent": None if mine else self.specs.get(p.id),
                    "agentName": None if mine else next(
                        (o["name"] for o in OPPONENTS if o["spec"] == self.specs.get(p.id)),
                        self.specs.get(p.id)),
                    "vp": p.victory_points, "solari": p.solari, "spice": p.spice,
                    "water": p.water, "influence": pd["influence"],
                    "alliances": pd["alliances"], "agentsAvail": p.agents_available,
                    "agentsTotal": p.agents_total, "garrison": p.troops_garrison,
                    "supply": p.troops_supply, "spies": p.spies_available,
                    "swordmaster": pd["has_swordmaster"], "councilor": pd["has_councilor"],
                    "inConflict": gs.troops_in_conflict.get(p.id, 0),
                    "worms": gs.sandworms_in_conflict.get(p.id, 0),
                    "strength": gs.combat_strength.get(p.id, 0),
                    "deck": pd["deck_size"], "handSize": pd["hand_size"],
                    "discard": [card_info(c) for c in p.discard][-12:],
                    "inPlay": [card_info(c) for c in p.in_play],
                    "hand": [card_info(c) for c in p.hand] if show else None,
                    "intrigues": [{"name": c.name} for c in p.intrigue_cards] if show else None,
                    "intrigueCount": len(p.intrigue_cards),
                    "techs": pd.get("techs", []), "battleIcons": pd.get("battle_icons", []),
                    "contractsDone": pd.get("contracts_completed", 0),
                    "revealed": p.id in gs.players_revealed,
                    "persuasion": gs.persuasion_pool.get(p.id, 0),
                    "research": gs.research_track.get(p.id, 0) if gs.research_track else 0,
                })
            cc = gs.current_conflict
            conflict = None
            if cc:
                conflict = {"name": cc.name, "level": cc.conflict_level,
                            "first": cc.first_place_reward,
                            "second": cc.second_place_reward,
                            "third": cc.third_place_reward or None,
                            "location": cc.location,
                            "img": card_img(cc.name.split(" (")[0])}
            spaces = []
            for name, s in UPRISING_BOARD.items():
                spaces.append({"name": name, "type": s.space_type.value,
                               "special": SPECIAL.get(name),
                               "cost": s.mandatory_cost, "gate": s.influence_gate,
                               "effects": s.effects, "combat": s.is_combat_space,
                               "occupant": gs.agent_on_space.get(name),
                               "controlledBy": gs.controlled_by.get(name),
                               "makerSpice": gs.maker_bonus_spice.get(name, 0)})
            actions = []
            if act == self.seat and not self.thinking:
                for i, a in enumerate(gs.get_valid_actions(self.seat)):
                    actions.append({"i": i, "type": a.action_type.value,
                                    "label": label(gs, a), "card": a.card_name,
                                    "space": a.space_name,
                                    "buy": a.acquire_card_name or a.reserve_type,
                                    "intrigue": a.intrigue_card_name})
            return {
                "id": self.id, "seed": self.seed, "seat": self.seat,
                "round": gs.round, "phase": gs.phase.value, "active": act,
                "gameOver": gs.game_over, "winner": gs.winner,
                "thinking": self.thinking,
                "thinkingFor": round(time.time() - self.think_started, 1) if self.thinking is not None else 0,
                "error": self.error, "players": players, "conflict": conflict,
                "conflictsLeft": d["conflict_deck_remaining"],
                "shieldWall": gs.shield_wall_intact, "alliance": d["alliance_holder"],
                "spaces": spaces,
                "row": [card_info(c) for c in gs.imperium_row],
                "reserve": {"prepare_the_way": len(gs.reserve_prepare_the_way),
                            "spice_must_flow": len(gs.reserve_spice_must_flow)},
                "contracts": [{"name": c["name"], "rewards": c.get("rewards"),
                               "trigger": c.get("trigger_condition")}
                              for c in d.get("contracts_on_board", [])],
                "actions": actions, "log": self.log[-400:], "saved": self.saved,
            }


SESSION: Optional[Session] = None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):     # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def _file(self, path: str) -> None:
        if not os.path.isfile(path):
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_GET(self):
        path = self.path.split("?")[0]
        q = dict(p.split("=", 1) for p in self.path.split("?", 1)[1].split("&")
                 if "=" in p) if "?" in self.path else {}
        if path == "/":
            return self._file(os.path.join(STATIC, "index.html"))
        if path.startswith("/static/"):
            return self._file(os.path.join(STATIC, os.path.basename(path)))
        if path.startswith("/cards/"):
            return self._file(os.path.join(CARD_DIR, os.path.basename(path)))
        if path == "/api/options":
            return self._json({"opponents": OPPONENTS})
        if path == "/api/state":
            if SESSION is None:
                return self._json({"none": True})
            return self._json(SESSION.view(reveal=q.get("reveal") == "1"))
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        global SESSION
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/new":
            seed = body.get("seed")
            SESSION = Session(seat=int(body.get("seat", 0)),
                              specs=body.get("opponents") or [],
                              seed=int(seed) if seed not in (None, "") else None,
                              bloodlines=bool(body.get("bloodlines", True)))
            return self._json({"ok": True})
        if self.path == "/api/act":
            if SESSION is None:
                return self._json({"error": "no game"}, 400)
            err = SESSION.human_move(int(body["i"]))
            return self._json({"ok": err is None, "error": err}, 200 if err is None else 400)
        self._send(404, b"not found", "text/plain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    # serve both loopbacks: browsers try ::1 first for "localhost", and an
    # IPv4-only server costs ~2 s per request while they fall back
    class V6(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        v6 = V6(("::1", args.port), Handler)
        threading.Thread(target=v6.serve_forever, daemon=True).start()
    except OSError:
        pass
    print(f"Dune AI table: http://localhost:{args.port}  ({len(CARD_IMG)} card images)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
