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
import collections
import copy
import glob
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
from urllib.parse import quote, unquote
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
from src.game.board.board import ALL_OBSERVATION_POSTS, LEADER_SPACES, UPRISING_BOARD  # noqa: E402
from src.game.effects import EffectResolver  # noqa: E402
from src.game.gameState import ActionType, GameAction, GameState, Phase  # noqa: E402

STATIC = os.path.join(HERE, "static")

# Public mode (a website): every visitor gets their own game against the Best
# AI; no AI hands, evaluator, replays, watch mode or files written to disk.
#   DUNE_PUBLIC=1 python ui/server.py      (or --public)
PUBLIC = os.environ.get("DUNE_PUBLIC") == "1"
MAX_GAMES = int(os.environ.get("DUNE_MAX_GAMES", "20"))      # games in memory at once
IDLE_SECS = int(os.environ.get("DUNE_IDLE_SECS", "3600"))    # a game nobody looked at for this long is dropped
LOG_DIR = os.path.join(ROOT, "game_logs")            # ui_<time>_<id>.json, autosaved
POS_DIR = os.path.join(ROOT, "data", "ui_games")     # your decisions as positions
# the decision types the weight fitter learns from (video_scrape/tune_heuristic.py)
FIT_TYPES = (ActionType.AGENT_TURN, ActionType.ACQUIRE_CARD,
             ActionType.ACQUIRE_RESERVE, ActionType.REVEAL_TURN)


def fit_label(a: GameAction) -> str:
    """Same move label as tune_heuristic / compare_ai use."""
    t = a.action_type
    if t == ActionType.AGENT_TURN:
        return f"agent {a.space_name}"
    if t == ActionType.ACQUIRE_CARD:
        return f"buy {a.acquire_card_name}"
    if t == ActionType.ACQUIRE_RESERVE:
        return f"buy {a.reserve_type}"
    return t.value


def _dump(path: str, write) -> None:
    tmp = path + ".tmp"
    write(tmp)
    for i in range(5):              # Windows: a virus scanner / indexer may hold
        try:                        # the file for a moment; never let a save
            os.replace(tmp, path)   # stop the game
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    print(f"autosave skipped (file busy): {path}", file=sys.stderr)

CARD_DIR = os.path.join(ROOT, "video_scrape", "cards")
HUMAN_AZ_DIR = os.path.join(ROOT, "data", "human_az")    # your moves for AlphaZero training

# Position evaluator (src/az/evaluator.py): win chance + expected VP per seat
# from your point of view. Reloaded when the file changes (retraining).
EVAL_PATH = os.path.join(ROOT, "models_eval", "evaluator.npz")
_EVAL: Dict = {"mtime": None, "ev": None}


def evaluator():
    if PUBLIC:
        return None
    try:
        m = os.path.getmtime(EVAL_PATH)
    except OSError:
        return None
    if _EVAL["mtime"] != m:
        from src.az.evaluator import Evaluator
        try:
            _EVAL["ev"], _EVAL["mtime"] = Evaluator(EVAL_PATH), m
        except Exception:            # half-written file: keep the old one
            return _EVAL["ev"]
    return _EVAL["ev"]


def _az_gen() -> str:
    from src.az.agent import latest_net
    n = latest_net()
    return n.rsplit("_", 1)[1].split(".")[0] if n else "?"


# The only bot the page offers: the PPO policy (2026-10-20) - a network trained by
# self-play (PPO) that corrects the evolved heuristic's move scores. Frozen copies
# (models_ui/ppo.npz = models_ppo/best.npz at iteration 1340 / 932k games, and the
# heuristic it sits on, config/ppo_ui.json) so a game never changes AI mid-way.
# Refresh after more training: cp models_ppo/best.npz models_ui/ppo.npz
# 53.2% vs 3x the E6 endgame solver, 67.3% vs 3x the evolved heuristic; instant moves.
BEST_AI = {"spec": "ppo:net=models_ui/ppo.npz:tuned=config/ppo_ui.json:greedy",
           "name": "Best AI", "note": "PPO network on top of the evolved heuristic"}
# :E6:EK3:EM4 = endgame solver from round 6 (play-outs to the end; 38.4% vs 31.9% plain, 2026-10-19)
E6_AI = {"spec": "look:D2:K3:M1:ev=models_eval_leaders/evaluator.npz:tuned=config/best_ai.json:E6:EK3:EM4",
         "name": "Endgame solver (E6)", "note": "evolved heuristic + lookahead, solves the game from round 6 (slow)"}

OPPONENTS = [
    BEST_AI,
    E6_AI,
    {"spec": "look:D3:K4:M4:tuned=config/evolve_look/champion.json",
     "name": "Lookahead heuristic (3 turns ahead + evaluator)",
     "note": "tries its top 4 moves in 4 reshuffled worlds, plays 3 more turns and scores the position "
             "with the win-chance evaluator; weights = the current lookahead-evolution champion "
             "(45.5% vs 18% for each original heuristic at gen 0); ~0.1-0.5 s per decision"},
    {"spec": "look:D3:K4:M4:tuned=config/gap/intrigue_lover.json",
     "name": "Intrigue lover (lookahead)",
     "note": "the lookahead champion with Intrigue cards valued ~2.9x: grabs Intrigues "
             "(Secrets, Assembly Hall, BG), plays them, and likes stealing at Secrets"},
    {"spec": "look:D3:K3:M1:tuned=config/evolve_look/champion.json",
     "name": "Lookahead heuristic (fast, as evolved)",
     "note": "the exact settings the evolution plays with: top 3 moves, 1 world, 3 turns ahead; under 0.1 s"},
    {"spec": "search:K5:M24:tuned=config/gap/evolved_champion.json",
     "name": "Evolved heuristic + search (24 playouts) - strongest",
     "note": "heuristic weights evolved like your friend's legacy policy (60 generations x 400 games); "
             "won 44% vs 37% for the original heuristic in a 4-way arena; ~5 s per decision"},
    {"spec": "heuristic:tuned=config/gap/evolved_champion.json", "name": "Evolved heuristic",
     "note": "instant; the evolved weights without search"},
    {"spec": "search:K5:M24:tuned=config/gap/tsmf5_dp1_wb10_tm1_op1.json",
     "name": "Search (24 playouts, latest knobs)",
     "note": "TSMF +5 (confirmed under search) + worm bonus 10 + deck plan 1 "
             "(each +2-4 pts in heuristic arenas); + deck-aware trashing + reads discard piles; ~5 s per decision"},
    {"spec": "az:S128:coach=1.5", "name": "AlphaZero (coached, 128 sims)",
     "note": f"self-trained net (latest generation at server start: {_az_gen()}) + 128-sim search, "
             "tilted by your advice: Swordmaster / High Council, TSMF from round 5, influence, "
             "Panopticon when short on influence, collect what a Conflict's extra VP costs, "
             "deploy just enough; ~2 s"},
    {"spec": "az:S128:coach=1.5:style=smf", "name": "AlphaZero coached · Spice Must Flow seeker",
     "note": "coached + draws cards, High Council and Assembly Hall for persuasion, buys TSMF over other cards"},
    {"spec": "az:S128:coach=1.5:style=sword", "name": "AlphaZero coached · Swordmaster seeker",
     "note": "coached + collects and saves 8 solari for an early Swordmaster"},
    {"spec": "az:S128:coach=1.5:style=commanders", "name": "AlphaZero coached · Sardaukar commander seeker",
     "note": "coached + commander spaces, recruits (board and supply), Sardaukar High Command"},
    {"spec": "az:S128:coach=1.5:style=faction", "name": "AlphaZero coached · Faction-influence seeker",
     "note": "coached + influence cards (draws / Assembly Hall / High Council for the expensive ones), faction spaces"},
    {"spec": "az:S128", "name": "AlphaZero (raw, 128 sims)",
     "note": "the network + search with no advice: what it has learned on its own"},
    {"spec": "search:K5:M24:tuned=config/gap/tsmf5.json",
     "name": "Search (24 playouts, balanced + TSMF)",
     "note": "balanced weights + TSMF +5: 28.5% vs 20.1% in the 2026-10-12 search arena"},
    {"spec": "search:K5:M24:tuned=config/heuristic_phased_final.json",
     "name": "Search (24 playouts, tuned) - strongest",
     "note": "beats plain search 62.5% / 37.5%; ~5 s per decision"},
    {"spec": "search:K5:M24", "name": "Search (24 playouts)",
     "note": "strong; ~5 s per decision"},
    {"spec": "search:K5:M48", "name": "Search (48 playouts)",
     "note": "strongest; ~10 s per decision"},
    {"spec": "heuristic:tuned=config/heuristic_winners.json", "name": "Winners' style",
     "note": "instant; weights fitted to the winners in your videos"},
    {"spec": "heuristic:tuned=config/heuristic_me.json", "name": "Your style",
     "note": "instant; weights fitted to Dinosaur11's own moves"},
    {"spec": "search:K5:M8", "name": "Search (8 playouts)",
     "note": "strong; ~2 s per decision"},
    {"spec": "heuristic:tuned=config/heuristic_phased_final.json", "name": "Heuristic (tuned to win)",
     "note": "instant; your-style weights + win-rate tuning"},
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
    "Tuek's Sietch": "Esmar Tuek only: + bonus spice, then 1 spice OR draw 1 card; Esmar +1 solari "
                     "(an opponent: Esmar draws an Intrigue); no Spy post",
}
# seats in turn order (first player moves on each round): red, green, yellow, blue
COLORS = ["#c0392b", "#27ae60", "#d4a017", "#2e6fd8"]
COLOR_NAMES = ["Red", "Green", "Yellow", "Blue"]
ICON_NAMES = {"crysknife": "Crysknife", "desert_mouse": "Desert Mouse",
              "ornithopter": "Ornithopter", "wild": "Wild"}


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
    # Some names are both an Imperium card and a Bloodlines leader (Piter de
    # Vries, Chani, Duncan Idaho, Esmar Tuek, Liet Kynes): the leader art is
    # "bl_<Name>", so take the plain card id first.
    order = sorted(idx, key=lambda k: (k.startswith("bl_"), k.endswith("_com"), k))
    for cid in order:
        v = idx[cid]
        if cid.startswith("_") or not v.get("files"):
            continue
        for n in v.get("names", [])[:1]:          # English name first
            out.setdefault(_norm(n), v["files"][0])
        out.setdefault(_norm(cid), v["files"][0])
    return out


CARD_IMG = _card_images()


# Images the TTS library lacks, from dunecardshub.com, saved under the engine's
# names: hub_images/cards (Imperium / Intrigue) and hub_images/contracts.
HUB_DIR = os.path.join(ROOT, "hub_images")
mimetypes.add_type("image/webp", ".webp")


def _hub_index(sub: str) -> Dict[str, str]:
    d = os.path.join(HUB_DIR, sub)
    return {_norm(os.path.splitext(f)[0]): f for f in os.listdir(d)} if os.path.isdir(d) else {}


HUB_CARDS, HUB_CONTRACTS = _hub_index("cards"), _hub_index("contracts")


def card_img(name: str) -> Optional[str]:
    f = CARD_IMG.get(_norm(name))
    if f:
        return f"/cards/{f}"
    f = HUB_CARDS.get(_norm(name))
    return f"/hub/cards/{quote(f)}" if f else None


LEADER_DIR = os.path.join(HUB_DIR, "leaders")


def leader_img(name: str) -> Optional[str]:
    f = os.path.join(LEADER_DIR, f"{name}.webp")
    return f"/hub/leaders/{quote(name)}.webp" if os.path.isfile(f) else None


def _leader_text() -> Dict[str, Dict[str, str]]:
    """Passive + Signet text per Leader from src/game/leader/LEADERS.md."""
    out = {}
    path = os.path.join(ROOT, "src", "game", "leader", "LEADERS.md")
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not cells[1].isdigit():
            continue
        name = re.sub(r"\s*\(.*\)$", "", cells[0])
        out[name] = {"complexity": int(cells[1]),
                     "passive": cells[2].replace("**", ""), "signet": cells[3].replace("**", "")}
    return out


LEADER_TEXT = _leader_text()


def leader_list(bloodlines: bool) -> List[Dict]:
    from src.game.leader.leader_definitions import create_leaders
    return [{"name": l.name, "img": leader_img(l.name), **LEADER_TEXT.get(l.name, {})}
            for l in create_leaders(include_bloodlines=bloodlines)]


def leader_view(gs, p, show: bool) -> Optional[Dict]:
    ld = p.leader
    if ld is None or ld.name == "Neutral":
        return None
    st = ld.state
    name = "Reverend Mother Jessica" if (ld.name == "Lady Jessica" and st.get("flipped")) else ld.name
    status = []
    if "feyd" in st:
        status.append(f"Training track: {st['feyd']}")
    if "tactics" in st or ld.name == "Chani":
        from src.game.leader.leader_definitions import CHANI_START, CHANI_END
        status.append(f"Tactics track: {st.get('tactics', CHANI_START) - CHANI_START} of "
                      f"{CHANI_END - CHANI_START} (from the 1-4p marker)")
    if ld.name == "Lady Jessica":
        status.append(f"Memories: {st.get('memories', 0)}" + (" · flipped" if st.get("flipped") else ""))
    if "twisted_deck" in st:
        status.append(f"Twisted deck {len(st['twisted_deck'])} · discard {len(st['twisted_discard'])}")
    if "nav" in st:
        k = st.get("nav_next", 0)
        played = ", ".join(f"#{i}" for i in st["nav"][:k]) or "none"
        status.append(f"Navigation played: {played} · {len(st['nav']) - k} face down"
                      + (f" (next #{st['nav'][k]})" if show and k < len(st["nav"]) else ""))
        if st.get("nav_reveal_persuasion"):
            status.append("+1 persuasion every Reveal turn (Navigation card 3 in slot 4)")
    if ld.name == "Esmar Tuek":
        bonus = gs.maker_bonus_spice.get("Tuek's Sietch", 0)
        status.append(f"Tuek's Sietch bonus spice: {bonus}")
    if ld.name == "Kota Odax of Ix" and getattr(gs, "bl", None) is not None:
        t = gs.bl.kota_tile(p.id)
        status.append(f"Secret Project: {t.name if (t is not None and show) else ('face-down tech' if t else 'already taken')}")
    if ld.name == "Shaddam Corrino IV":
        n = len(getattr(gs, "shaddam_contracts", {}).get(p.id, []))
        status.append(f"Sardaukar contracts set aside: {n}")
    nav = None
    if st.get("nav"):                  # face down until played; the owner sees his
        k = st.get("nav_next", 0)
        nav = [{"slot": j + 1, "played": j < k, "img": nav_img(c) if (show or j < k) else None}
               for j, c in enumerate(st["nav"])]
    return {"name": name, "img": leader_img(name), **LEADER_TEXT.get(name, {}), "status": status,
            "nav": nav}


# Round 1 Skirmishes share one name; each has its own card art
SKIRMISH_IMG = {"Skirmish (Crysknife)": "skirmishA", "Skirmish (Ornithopter)": "skirmishB",
                "Skirmish (Desert Mouse)": "skirmishC", "Skirmish (Wild)": "bl_Skirmish"}


def conflict_img(name: str) -> Optional[str]:
    if name in SKIRMISH_IMG:
        return card_img(SKIRMISH_IMG[name])
    return card_img(name.split(" (")[0])


def nav_img(i: int) -> str:
    """Steersman's Navigation card i (hub_images/navigation, from dunecardshub)."""
    return f"/hub/navigation/{quote(f'Navigation card {i}.webp')}"


def choice_img(gs: GameState, a: GameAction) -> Optional[str]:
    """Picture for a choice button: Navigation cards (Steersman's setup) and
    the bottom Tech tiles Kota may keep."""
    if a.action_type != ActionType.RESOLVE_CHOICE or not a.choice:
        return None
    m = re.match(r"claim:(\d+)$", a.choice)
    if m:
        items = gs._reward_items(a.player_id)
        it = items[int(m.group(1))] if int(m.group(1)) < len(items) else {}
        return nav_img(it["effect"]["card"]) if it.get("kind") == "nav" else None
    m = re.match(r"llf_(draw|trash):(\d+)$", a.choice)      # Long Live the Fighters
    if m:
        cards = getattr(gs.players[a.player_id], "llf_cards", None) or []
        i = int(m.group(2))
        return card_img(cards[i].name) if i < len(cards) else None
    m = re.match(r"(drop|nav):(\d+)$", a.choice)
    if m:
        return nav_img(int(m.group(2)))
    m = re.match(r"secret:(\d+)$", a.choice)
    if m and getattr(gs, "bl", None) is not None:
        st = gs.bl.tech_stacks[int(m.group(1))]
        return tech_img(st[-1].name) if st else None
    return None


def contract_img(name: str) -> Optional[str]:
    f = HUB_CONTRACTS.get(_norm(name))
    return f"/hub/contracts/{quote(f)}" if f else None


# ---------------------------------------------------------------------------
# Bloodlines: tech tiles, Sardaukar commanders, skills
# ---------------------------------------------------------------------------

TECH_DIR = os.path.join(ROOT, "Tech tiles")
SKILL_DIR = os.path.join(ROOT, "sard_skills")


def _img_index(folder: str) -> Dict[str, str]:
    if not os.path.isdir(folder):
        return {}
    return {_norm(os.path.splitext(f)[0]): f for f in os.listdir(folder)}


TECH_IMG, SKILL_IMG = _img_index(TECH_DIR), _img_index(SKILL_DIR)


def _match(idx: Dict[str, str], name: str) -> Optional[str]:
    import difflib
    k = _norm(name)
    if k in idx:
        return idx[k]
    near = difflib.get_close_matches(k, list(idx), n=1, cutoff=0.8)   # "recievers"
    return idx[near[0]] if near else None


TECH_ALIAS = {"Rapid Dropships": "dropships", "Navigation Chamber": "nav chamber",
              "Ornithopter Fleet": "orni fleet", "Spy Satellites": "spy sats"}


def tech_img(name: str) -> Optional[str]:
    f = _match(TECH_IMG, TECH_ALIAS.get(name, name))
    return f"/tech/{quote(f)}" if f else None


def skill_img(name: str) -> Optional[str]:
    f = _match(SKILL_IMG, name)
    return f"/skill/{quote(f)}" if f else None


SKILL_TEXT = {   # src/game/bloodlines/rules.py
    "Hardy": "Reveal, a commander in the Conflict: +1 troop",
    "Driven": "Reveal, a commander in the Conflict: +1 spice",
    "Charismatic": "Reveal, a commander in the Conflict: +1 persuasion",
    "Desperate": "Reveal, a commander in the Conflict: may trash this skill for +3 swords",
    "Canny": "Agent on a green space: +2 swords (needs a commander in the Conflict)",
    "Fierce": "+1 sword, +1 more if an opponent has a sandworm in the Conflict "
              "(needs a commander in the Conflict)",
    "Loyal": "3+ Emperor influence: +2 swords (needs a commander in the Conflict)",
}


def _fx(effects) -> str:
    out = []
    for e in effects or []:
        if isinstance(e, dict):
            out.append(", ".join(f"{v} {k}" if not isinstance(v, (dict, list)) else k
                                 for k, v in e.items()).replace("_", " "))
    return "; ".join(out)


def tech_info(d, gs=None, pid=None) -> Dict:
    parts = []
    for title, key in (("Acquire", "acquire"), ("Reveal", "reveal"),
                       ("Command (6+ persuasion)", "command"), ("Round start", "round_start"),
                       ("Win a conflict", "on_win_conflict"),
                       ("Complete a contract", "on_complete_contract")):
        t = _fx(getattr(d, key, None))
        if t:
            parts.append(f"{title}: {t}")
    act = getattr(d, "activation", None)
    if act:
        cost = ", ".join(f"{v} {k}" for k, v in (act.get("cost") or {}).items())
        parts.append("Once per round: " + (f"pay {cost} -> " if cost else "")
                     + _fx(act.get("effects")))
    if getattr(d, "note", ""):
        parts.append(d.note)
    price = d.cost
    if gs is not None and pid is not None and gs.bl:
        try:
            price = gs.bl.tech_price(pid, d, 0)
        except Exception:
            pass
    return {"name": d.name, "cost": d.cost, "price": price,
            "text": " · ".join(parts), "img": tech_img(d.name)}


def bl_choice_label(gs: GameState, a: GameAction) -> str:
    ch = a.choice or ""
    pend = next((c for c in (gs.bl.pending if gs.bl else []) if c.player_id == a.player_id), None)
    kind = pend.kind if pend else ""
    if ch == "decline":
        return {"tech_buy": "Don't buy a tech", "tech_offer": "Don't take a tech",
                "commander": "Don't recruit a commander",
                "plasteel": "Keep Plasteel Blades (no extra skill)"}.get(kind, "Decline")
    if ch == "stack:-1":                                  # Kota Odax's Secret Project tile
        d = gs.bl.kota_tile(a.player_id)
        if d is not None:
            price = gs.bl.tech_price(a.player_id, d, (pend.discount if pend else 0) + 1)
            return f"Buy tech: {d.name} (Secret Project, {price} spice)"
    if ch.startswith("stack:"):
        st = gs.bl.tech_stacks[int(ch.split(":")[1])]
        if st:
            d = st[0]
            price = gs.bl.tech_price(a.player_id, d, pend.discount if pend else 0)
            return f"Buy tech: {d.name} ({price} spice)"
    if ch.startswith("board:"):
        cost = gs.bl.commander_cost(a.player_id)
        skill = ch.split(":", 1)[1]
        return f"Recruit Sardaukar commander ({cost} solari)" + (f" + skill {skill}" if skill else "")
    if ch == "supply":
        return f"Re-recruit a commander from your supply ({gs.bl.commander_cost(a.player_id)} solari)"
    if kind == "plasteel":
        return ("Keep Plasteel Blades" if ch == "decline"
                else f"Trash Plasteel Blades → gain skill {ch.split(':', 1)[1]}")
    if kind == "desperate":
        return "Trash Desperate for +3 swords" if ch == "use" else "Keep Desperate"
    if kind == "complete_contract":
        return f"Complete contract: {ch.split(':', 1)[1]} (CHOAM Demands)"
    if kind == "ada_spy":
        return f"Trash your Spy at {ch.split(':', 1)[1]} (Advanced Data Analysis)"
    if kind == "vault":
        return "Gene-Locked Vault: take an Intrigue" if ch == "intrigue" else "Gene-Locked Vault: draw a card"
    if kind == "forbidden":
        p = gs.players[a.player_id]
        lose = any(v > 0 for v in p.influence.values())
        return (f"Forbidden Weapons: +3 swords{', lose 1 influence' if lose else ' (no influence to lose)'}" if ch == "swords"
                else "Forbidden Weapons: lose all your spice, trash the tile")
    if kind == "trash_hand":
        return f"Trash {ch} from hand"
    if kind == "disposal":
        return f"Trash {ch} (Disposal Facility)"
    if kind == "effect" and pend:
        return f"Choose: {_fx([pend.payloads.get(ch, {})]) or ch}"
    return f"Choose: {ch}"


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
    if t == ActionType.RESOLVE_CHOICE:
        if a.choice == "recruit:supply":
            return f"Re-recruit a commander from your supply ({gs.bl.commander_cost(a.player_id)} solari)"
        if a.choice and a.choice.startswith("recruit:board:"):
            sk = a.choice[len("recruit:board:"):]
            return f"Recruit Sardaukar commander ({gs.bl.commander_cost(a.player_id)} solari)" + (f" + skill {sk}" if sk else "")
        if a.choice and a.choice.startswith("techbuy:"):
            st = gs.bl.tech_stacks[int(a.choice[8:])]
            return f"Buy tech: {st[0].name} ({gs.bl.tech_price(a.player_id, st[0])} spice)" if st else "Buy tech"
        if a.choice and a.choice.startswith("claim:"):
            items = gs._reward_items(a.player_id)
            i = int(a.choice[6:])
            it = items[i] if 0 <= i < len(items) else {"kind": "?", "n": 0}
            n = it["n"]
            if it["kind"] == "nav":
                return f"Play Navigation card {it['effect']['card']} (slot {it['effect']['slot']}) now"
            if it["kind"] == "effect" and "bl_complete_contract" in (it.get("effect") or {}):
                what = "complete one of your contracts"
            elif it["kind"] == "effect":
                what = ", ".join(f"{v} {k.replace('_', ' ')}" if isinstance(v, (int, float)) else k.replace("_", " ")
                                 for k, v in (it.get("effect") or {}).items())
            else:
                what = f"draw {n} card{'s' if n != 1 else ''}" if it["kind"] == "draw"                     else f"take {n} Intrigue card{'s' if n != 1 else ''}"
            return f"Take reward now: {what}" + (f" (from {it['src']})" if it.get("src") else "")
        pc = gs.pending_choice_for(a.player_id)
        opt = next((o for o in pc.options if o[0] == a.choice), None) if pc else None
        return opt[1] if opt else f"Choose: {a.choice}"
    if t == ActionType.RESOLVE_CONTRACT:
        try:
            if a.contract_index >= 100:             # Shaddam's set-aside Sardaukar contracts
                c = gs.shaddam_contracts[a.player_id][a.contract_index - 100]
                return f"Take Sardaukar contract: {c.name}"
            c = gs.contracts_on_board[a.contract_index]
            return f"Take contract: {c.name}"
        except Exception:
            return f"Take contract #{a.contract_index}"
    if t == ActionType.RESOLVE_OPTIONAL:
        op = next((o for o in gs.pending_optional_payments if o.player_id == a.player_id), None)
        if op is None:
            return "Accept optional cost → reward" if a.accept_optional else "Decline optional cost"
        cost = ", ".join(f"recall {v} spies" if k == "recall_spy" else f"{v} {k}"
                         for k, v in op.cost.items())
        if op.discard:
            cost = (cost + ", " if cost else "") + f"discard {op.discard}"
        gain = ", ".join(f"{v} {k}".replace("_", " ") if isinstance(v, (int, float)) else k
                         for k, v in op.reward.items())
        return f"Pay {cost} → gain {gain}" if a.accept_optional else f"Don't pay ({cost} → {gain})"
    if t == ActionType.RESOLVE_BL_CHOICE:
        return bl_choice_label(gs, a)
    if t == ActionType.ACTIVATE_TECH:
        return f"Activate tech: {a.choice or a.card_name}"
    if t == ActionType.REVEAL_TURN:
        return "Reveal turn"
    if t == ActionType.END_REVEAL:
        return "Done buying — end turn"
    if t == ActionType.END_TURN:
        return "End turn"
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

def buy_power(gs, q: int) -> Optional[Dict]:
    """Chance q reaches 5 / 7 / 9 persuasion on this round's Reveal, from
    public info only (src/ai/opp_model.py); None once they've revealed."""
    if q in gs.players_revealed or gs.phase.name != "PLAYER_TURNS":
        return None
    from src.ai.opp_model import reveal_persuasion_samples
    smp = reveal_persuasion_samples(gs, q)
    return {"mean": round(float(smp.mean()), 1),
            "p": {str(c): round(float((smp >= c).mean()), 2) for c in (5, 7, 9)}}


def name_counts(cards) -> List[List]:
    c = collections.Counter(x.name for x in cards)
    return [[n, k, card_img(n)] for n, k in sorted(c.items(), key=lambda t: (-t[1], t[0]))]


def vp_sources(p) -> List[Dict]:
    """Where a player's VP came from (Player.vp_log), grouped by source in the
    order first earned; anything the log misses (a game started before it
    existed) shows as 'Earlier (not recorded)' so the total always matches."""
    out: Dict[str, Dict] = {}
    for src, d, rnd in getattr(p, "vp_log", []):
        e = out.setdefault(src, {"src": src, "vp": 0, "rounds": []})
        e["vp"] += d
        e["rounds"].append([rnd, d])
    rest = p.victory_points - sum(e["vp"] for e in out.values())
    if rest:
        out["Earlier (not recorded)"] = {"src": "Earlier (not recorded)", "vp": rest, "rounds": []}
    return [e for e in out.values() if e["vp"] or len(e["rounds"]) > 1]


def last_combat(gs) -> Optional[Dict]:
    """The Conflict just resolved: each fighter's strength and what they got."""
    r = getattr(gs, "_last_combat_result", None)
    if r is None:
        return None
    return {"name": r.conflict_name, "winner": r.winner,
            "rows": [{"pid": pid, "strength": st, "reward": r.rewards_awarded.get(pid)}
                     for pid, st in r.rankings]}


class ReplayAgent:
    """Plays back a recorded self-play game: the index of each move among the
    legal moves, in order (src/az/selfplay.py writes <run>/games/*.jsonl.gz).
    One instance is shared by every seat."""

    def __init__(self, moves):
        self.moves, self.k = list(moves), 0

    def select_action(self, gs, pid, valid):
        vi = self.moves[self.k] if self.k < len(self.moves) else 0
        self.k += 1
        return valid[vi] if 0 <= vi < len(valid) else valid[0]


def list_replays(limit: int = 300) -> List[Dict]:
    """Recorded self-play games, newest first, from every run folder."""
    import gzip
    files = sorted(glob.glob(os.path.join(ROOT, "models_*", "games", "*.jsonl.gz")),
                   key=os.path.getmtime, reverse=True)
    out = []
    for f in files:
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        run = os.path.basename(os.path.dirname(os.path.dirname(f)))
        gen = os.path.basename(f).split("_")[0].lstrip("g")
        for i, line in enumerate(lines):
            r = json.loads(line)
            out.append({"file": os.path.relpath(f, ROOT), "line": i, "run": run, "gen": gen,
                        "rounds": r.get("rounds"), "vp": r.get("vp"), "winner": r.get("winner"),
                        "styles": r.get("styles"), "moves": len(r.get("moves", []))})
            if len(out) >= limit:
                return out
    return out


def load_replay(file: str, line: int) -> Dict:
    import gzip
    path = os.path.normpath(os.path.join(ROOT, file))
    if not path.startswith(ROOT) or not path.endswith(".jsonl.gz"):
        raise ValueError("not a recorded game file")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, l in enumerate(fh):
            if i == int(line):
                return json.loads(l)
    raise ValueError("game not found")


class Session:
    def __init__(self, seat: int, specs: List[str], seed: Optional[int],
                 bloodlines: bool, players: int = 4, pace: float = 1.0,
                 replay: Optional[Dict] = None, use_leaders: bool = True,
                 my_leader: Optional[str] = None):
        if replay is not None:                     # watch a recorded self-play game
            seat, seed, bloodlines = -1, int(replay["seed"]), True
            specs = [f"replay ({replay.get('styles', ['?'] * 4)[i] if replay.get('styles') else 'self-play'})"
                     for i in range(players)]
        self.replay = replay
        self.seed = seed if seed is not None else random.randint(0, 999_999)
        random.seed(self.seed)
        lineup = None
        if use_leaders and replay is None:
            # your pick (or random); the AIs get random other Leaders
            from src.game.leader.leader_definitions import create_leaders
            pool = [l.name for l in create_leaders(include_bloodlines=bloodlines)]
            rng = random.Random(self.seed)
            rng.shuffle(pool)
            mine_at = seat if seat >= 0 else None
            if mine_at is not None and my_leader in pool:
                pool.remove(my_leader)
                pool.insert(mine_at, my_leader)
            lineup = pool[:players]
        if lineup:
            self.gs = setup_game(num_players=players, seed=self.seed, leaders=lineup,
                                 use_bloodlines=bloodlines)
        else:
            self.gs = setup_game(num_players=players, seed=self.seed,
                                 neutral_leaders=True, use_bloodlines=bloodlines)
        self.gs.ui_notes = []
        self.revealed: Dict[int, Dict] = {}       # this round's Reveal: persuasion per player
        # seat -1 = watch: every seat is an AI; the page shows it from Red's chair
        self.spectate = seat < 0
        self.seat = 0 if self.spectate else seat
        self.pace = max(0.0, pace)            # watch mode: seconds per move at least
        self.paused = False
        self.gs.manual_end_turn = set() if self.spectate else {seat}   # you end your own Agent turn
        self.specs = {}
        self.agents = {}
        opp = iter(specs)
        for i in range(players):
            if i == seat and not self.spectate:
                continue
            spec = next(opp, OPPONENTS[0]["spec"])
            self.specs[i] = spec
            if replay is None:
                self.agents[i] = make_agent(spec, seed=self.seed * 8 + i)
        if replay is not None:
            shared = ReplayAgent(replay["moves"])
            self.agents = {i: shared for i in range(players)}
        self.id = str(uuid.uuid4())
        self.log: List[Dict] = []
        self.thinking: Optional[int] = None
        self.think_started = 0.0
        self.error: Optional[str] = None
        self.lock = threading.RLock()
        self.moves = []                       # play_game-format move log
        self.started = datetime.now()
        self.positions: List[Dict] = []       # your decisions (fitter format)
        self.az_rec: List = []                # your decisions (AlphaZero examples)
        self.eval_hist: List[Dict] = []       # your-view win chances after each move
        self._move_eval = (None, {})
        self._eval_snapshot()
        from src.ai.agents import HeuristicAgent
        self.coach = HeuristicAgent(seed=0)   # "the AI would have played..."
        stamp = self.started.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(LOG_DIR, f"ui_{stamp}_{self.id[:8]}.json")
        self.pos_path = os.path.join(POS_DIR, f"ui_{stamp}_{self.id[:8]}.pkl")
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

    def _eval_all(self, ev, gs):
        """(win chances, expected VP): your view, or in watch mode each
        player's estimate from their own seat (renormalised)."""
        if not self.spectate:
            return ev(gs, self.seat)
        w, v = np.zeros(gs.num_players), np.zeros(gs.num_players)
        for q in range(gs.num_players):
            wq, vq = ev(gs, q)
            w[q], v[q] = wq[q], vq[q]
        return w / max(w.sum(), 1e-9), v

    def _eval_snapshot(self) -> None:
        ev = evaluator()
        if ev is None:
            return
        try:
            w, _ = self._eval_all(ev, self.gs)
        except Exception:
            return
        self.eval_hist.append({"n": len(self.moves), "round": self.gs.round,
                               "win": [round(float(x), 3) for x in w]})

    def _move_evals(self, ev, valid) -> Dict[int, float]:
        """Your win chance right after each legal move (cached per position)."""
        key = (len(self.moves), len(valid), self.gs.round, self.gs.phase.value)
        if self._move_eval[0] == key:
            return self._move_eval[1]
        out, obs, idx = {}, [], []
        from src.az.encode import encode_obs
        for i, a in enumerate(valid):
            if a.action_type == ActionType.NO_OP or len(idx) >= 80:
                continue
            try:
                g = self.gs.clone()
                g.step(next(v for v in g.get_valid_actions(self.seat) if repr(v) == repr(a)))
                obs.append(ev.encode(g, self.seat)); idx.append(i)
            except Exception:
                continue
        if obs:
            for i, (w, _) in zip(idx, ev.batch(obs, self.seat, self.gs.num_players)):
                out[i] = round(float(w[self.seat]), 4)
        self._move_eval = (key, out)
        return out

    def _run_ai(self) -> None:
        try:
            while True:
                with self.lock:
                    gs = self.gs
                    if gs.game_over:
                        self._finish()
                        return
                    pid = active_player(gs)
                    if pid == self.seat and not self.spectate:
                        return
                    if self.paused:
                        self.thinking = None
                        return
                    agent = self.agents[pid]
                    valid = gs.get_valid_actions(pid)
                    if not valid:
                        raise RuntimeError(f"engine stuck: no valid actions for seat {pid} "
                                           f"(round {gs.round}, phase {gs.phase.value})")
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
                    self.autosave()
                if self.spectate and secs < self.pace:
                    time.sleep(self.pace - secs)      # watchable speed
        except Exception:
            self.error = traceback.format_exc()
            self.thinking = None
            print(self.error, file=sys.stderr)
            try:
                with self.lock:
                    self.autosave()
            except Exception:
                pass

    def _explain(self, agent, gs: GameState, pid: int, act: GameAction) -> Optional[Dict]:
        if PUBLIC:                             # the AI's candidates would leak its hand
            return None
        az = getattr(agent, "last_az", None)
        if az:
            tot = float(az["visits"].sum()) or 1.0
            order = sorted(range(len(az["acts"])), key=lambda i: (-az["visits"][i], -az["prior"][i]))[:6]
            rows = [{"label": label(gs, az["acts"][i]), "value": round(float(az["visits"][i]) / tot, 3),
                     "prior": round(float(az["prior"][i]), 3),
                     "chosen": repr(az["acts"][i]) == repr(act)} for i in order]
            note = (f"Share of the {getattr(agent, 'sims', '?')} search simulations spent on each move "
                    f"(it plays the most-visited). Network's own first instinct: "
                    + ", ".join(f"{r['label'][:28]} {r['prior']:.0%}" for r in rows[:3])
                    + f". Its value for this position: {az['value']:.2f}.")
            return {"kind": "search", "rows": rows, "note": note}
        last = getattr(agent, "last", None)
        if last and last.get("ppo"):                               # PPOAgent
            rows = [{"label": label(gs, c), "value": round(p, 3), "heur": round(h, 2),
                     "chosen": repr(c) == repr(act)} for c, p, h in zip(last["cands"], last["prob"], last["heur"])]
            ht = last["heur_top"]
            note = ("The network's probability for each move (it plays the most likely). heur = the evolved "
                    "heuristic's score, which the network starts from and corrects"
                    + ("" if repr(ht) == repr(act) else f"; the heuristic alone would have played {label(gs, ht)}")
                    + f". Its win-chance estimate here: {min(1.0, max(0.0, last['value'])):.0%}.")
            return {"kind": "search", "rows": rows, "note": note}
        if last and last.get("cands") and "win" in last:          # LookaheadAgent
            rows = [{"label": label(gs, c), "value": round(float(w), 3),
                     "heur": round(float(t), 2), "chosen": i == last["chosen"]}
                    for i, (c, w, t) in enumerate(zip(last["cands"], last["win"], last["total"]))]
            note = (f"Tried the heuristic's top {len(rows)} moves in {getattr(agent, 'm', '?')} reshuffled "
                    f"world(s), played {getattr(agent, 'depth', '?')} more turns and asked the evaluator: "
                    "value = win chance there. It plays the best heuristic score + "
                    f"{getattr(agent, 'look_w', 0):.2f} x win-chance points gained over the heuristic's pick "
                    "(the heur column shows that total).")
            return {"kind": "search", "rows": rows, "note": note}
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
        valid = gs.get_valid_actions(pid)
        vi = next((i for i, a in enumerate(valid) if repr(a) == repr(act)), None)
        entry = {"n": len(self.log), "round": gs.round, "phase": gs.phase.value,
                 "pid": pid, "label": label(gs, act), "type": act.action_type.value,
                 "explain": explain, "secs": round(secs, 1), "vi": vi,
                 "n_valid": len(valid), "vp": [q.victory_points for q in gs.players]}
        bought = self._bought(gs, act)
        if bought:
            entry["bought"] = bought
        inf_before = [dict(q.influence) for q in gs.players]
        owned_before = self._owned(gs, pid)
        icons = self._move_icons(gs, act, bought)
        pool_before = gs.persuasion_pool.get(pid, 0)
        round_before = gs.round
        _, reward, _, info = gs.step(act)
        # how much each player revealed this round (persuasion before spending)
        if gs.round != round_before:
            self.revealed = {}
        if act.action_type == ActionType.REVEAL_TURN:
            self.revealed[pid] = {"persuasion": gs.persuasion_pool.get(pid, 0),
                                  "strength": gs.combat_strength.get(pid, 0)}
        elif pid in self.revealed and gs.player_in_reveal_buy == pid \
                and "acquire" not in act.action_type.value:
            gain = gs.persuasion_pool.get(pid, 0) - pool_before
            if gain > 0:                       # drawn / later persuasion this Reveal turn
                self.revealed[pid]["persuasion"] += gain
        if pid in self.revealed:
            self.revealed[pid]["strength"] = gs.combat_strength.get(pid, 0)
        notes = getattr(gs, "ui_notes", None)
        if notes:
            entry["label"] += " · " + " · ".join(notes)
            gs.ui_notes = []
        if info.get("error"):
            entry["error"] = info["error"]
        # techs / skills / contracts this move gained, for the move strip
        for kind, names in self._owned(gs, pid).items():
            for nm in names - owned_before[kind]:
                img = {"tech": tech_img, "skill": skill_img, "contract": contract_img}[kind](nm)
                icons.append({"kind": kind, "name": nm, "img": img})
        if act.action_type == ActionType.REVEAL_TURN:
            # swords revealed (cards' sword icons + reveal effects + techs), not
            # the total strength with troops
            q = gs.players[pid]
            swords = gs.swords_this_reveal.get(pid, 0) + sum(
                t.passive_reveal_bonus.get("swords", 0) for t in getattr(q, "techs", [])
                if hasattr(t, "has_passive_reveal_bonus") and t.has_passive_reveal_bonus())
            icons.append({"kind": "reveal", "text": f"{gs.persuasion_pool.get(pid, 0)}◆ {swords}⚔"})
        entry["icons"] = icons
        # the 4th space of each influence track pays a reward when reached
        from src.game.gameState import INFLUENCE_4_BONUS
        names = {"fremen": "Fremen: +1 water", "bene_gesserit": "Bene Gesserit: +1 Intrigue",
                 "spacing_guild": "Spacing Guild: +3 solari", "emperor": "Emperor: place a Spy"}
        for q, ib in zip(gs.players, inf_before):
            for f in INFLUENCE_4_BONUS:
                if ib.get(f, 0) < 4 <= q.influence.get(f, 0):
                    entry["label"] += f" · P{q.id + 1} reached 4 {names[f]}"
        self.log.append(entry)
        self.moves.append({"move_num": len(self.moves), "round": before["round"],
                           "phase": before["phase"], "active_player": pid,
                           "is_human": pid == self.seat, "action": PG.action_to_dict(act),
                           "reward": reward, "error": info.get("error"),
                           "vi": vi, "repr": repr(act)})
        self._eval_snapshot()
        h = self.eval_hist
        if len(h) >= 2 and h[-1]["n"] == len(self.moves) and h[-2]["n"] == len(self.moves) - 1:
            entry["ev"] = round(h[-1]["win"][self.seat] - h[-2]["win"][self.seat], 3)

    def _reveal_now(self, gs: GameState, pid: int) -> Optional[Dict]:
        """Persuasion (and strength) this player would have if they revealed
        right now: the engine's own Reveal turn played on a copy - cards at
        their current value (Bene Gesserit Operative counts 3 only with 2 Spies
        out, Command Center 1 until 2 troops retreated), techs, Charismatic
        with a commander in the Conflict, High Council, Assembly Hall (not if
        the Agent left it - Duncan's Into the Fray), Steersman's Navigation."""
        if pid in gs.players_revealed or gs.phase != Phase.PLAYER_TURNS:
            return None
        key = (len(self.log), pid)
        cache = self.__dict__.setdefault("_reveal_cache", {})
        if key not in cache:
            if len(cache) > 16:
                cache.clear()
            try:
                g2 = copy.deepcopy(gs)
                g2.ui_notes = []
                g2._step_reveal_turn(GameAction(ActionType.REVEAL_TURN, pid))
                for conds, q in list(g2._pending_reveal_conditionals):   # resolved at the end of the turn
                    if q == pid:
                        EffectResolver.resolve_single_effect(conds, g2.players[q], g2)
                g2.update_combat_strength(pid)
                cache[key] = {"persuasion": g2.persuasion_pool.get(pid, 0),
                              "strength": g2.combat_strength.get(pid, 0)}
            except Exception:
                cache[key] = None
        return cache[key]

    @staticmethod
    def _owned(gs: GameState, pid: int) -> Dict[str, set]:
        p = gs.players[pid]
        return {"tech": {getattr(getattr(t, "d", t), "name", str(t)) for t in getattr(p, "techs", [])},
                "skill": set(getattr(p, "skills", set()) or set()),
                "contract": {c.name for c in list(p.contracts_active) + list(p.contracts_completed)}}

    @staticmethod
    def _move_icons(gs: GameState, act: GameAction, bought: Optional[Dict]) -> List[Dict]:
        """Pictures for the move strip: the card played and where, the Intrigue
        played, the card bought (deploys show on the board's Conflict)."""
        t = act.action_type
        out = []
        if t == ActionType.AGENT_TURN and act.card_name:
            out.append({"kind": "agent", "name": act.card_name, "img": card_img(act.card_name),
                        "space": act.space_name})
        elif t == ActionType.PLAY_INTRIGUE and act.intrigue_card_name:
            out.append({"kind": "intrigue", "name": act.intrigue_card_name, "img": card_img(act.intrigue_card_name)})
        if bought:
            out.append({"kind": "buy", "name": bought["card"], "img": bought.get("img"), "free": bought.get("free")})
        return out

    @staticmethod
    def _bought(gs: GameState, act: GameAction) -> Optional[Dict]:
        """The card a move acquires (Row buy, reserve, or a free acquire)."""
        t = act.action_type
        if t == ActionType.ACQUIRE_CARD:
            card = next((c for c in gs.imperium_row if c.name == act.acquire_card_name), None)
            if card is None:
                rc = gs.players[act.player_id].reserved_card
                card = rc if rc is not None and rc.name == act.acquire_card_name else None
            return {"card": act.acquire_card_name, "cost": getattr(card, "cost", None),
                    "img": card_img(act.acquire_card_name)}
        if t == ActionType.ACQUIRE_RESERVE:
            name = "Prepare the Way" if act.reserve_type == "prepare_the_way" else "The Spice Must Flow"
            return {"card": name, "cost": None, "img": card_img(name), "reserve": True}
        if t == ActionType.RESOLVE_CHOICE:
            pc = gs.pending_choice_for(act.player_id)
            opt = next((o for o in pc.options if o[0] == act.choice), None) if pc else None
            spec = (opt[2] or {}).get("_acquire_named") if opt else None
            if spec:
                card = next((c for c in gs.imperium_row if c.name == spec["card"]), None)
                return {"card": spec["card"], "cost": getattr(card, "cost", None),
                        "img": card_img(spec["card"]), "free": True}
        return None

    def human_move(self, index: int) -> Optional[str]:
        with self.lock:
            gs = self.gs
            if self.spectate:
                return "watching: every seat is an AI"
            if gs.game_over or active_player(gs) != self.seat:
                return "not your turn"
            valid = gs.get_valid_actions(self.seat)
            if not 0 <= index < len(valid):
                return "bad action index"
            act = valid[index]
            pick = None
            if not PUBLIC:
                self._record_training(valid, act)
                self._record_az(act)
                pick = self._record_position(valid, act)
            self._apply(self.seat, act)
            if pick:
                self.log[-1].update(pick)
            self.autosave()
        self.kick()
        return None

    def _record_position(self, valid, act) -> Optional[Dict]:
        """Save the decision (position, options, your choice, the AI's pick)
        so the weight fitter and comparison tools can use your games."""
        gs, pid = self.gs, self.seat
        cands = [a for a in valid if a.action_type != ActionType.NO_OP]
        if len(cands) < 2:
            return None
        scores = [float(self.coach.score(gs, pid, a)) for a in cands]
        best = max(range(len(cands)), key=lambda i: scores[i])
        top = sorted(range(len(cands)), key=lambda i: -scores[i])[:5]
        pick = {"ai_pick": label(gs, cands[best]),
                "agree": repr(cands[best]) == repr(act),
                "ai_top": [{"label": label(gs, cands[i]), "score": round(scores[i], 2)}
                           for i in top]}
        if act.action_type in FIT_TYPES:
            self.positions.append({
                "game": f"ui_{self.id[:8]}", "source": "ui", "gs": gs.clone(),
                "pid": pid, "cands": cands, "human": fit_label(act),
                "type": "buy" if "acquire" in act.action_type.value else act.action_type.value,
                "me": True, "won": None, "round": gs.round,
                "ai_pick": fit_label(cands[best])})
        return pick

    def autosave(self) -> None:
        """Write the game so far (every move, the AI's reasoning, your
        decisions); called after each of your moves and at the end."""
        if PUBLIC:
            return
        import pickle
        gs = self.gs
        vp = [p.victory_points for p in gs.players]
        winner = None
        if gs.game_over:
            winner = gs.winner if gs.winner is not None else int(np.argmax(vp))
        mine = [e for e in self.log if e.get("pid") == self.seat and "agree" in e]
        doc = {"game_id": self.id, "source": "ui", "started": self.started.isoformat(),
               "saved": datetime.now().isoformat(), "seed": self.seed,
               "bloodlines": bool(gs.bl), "human_id": self.seat,
               # what ui/replay.py needs to rebuild the exact game
               "leaders": [getattr(p.leader, "name", None) for p in gs.players],
               "manual_end_turn": sorted(getattr(gs, "manual_end_turn", ()) or ()),
               "opponents": {str(k): v for k, v in self.specs.items()},
               "finished": gs.game_over, "round": gs.round, "winner": winner,
               "you_won": None if winner is None else winner == self.seat, "vp": vp,
               "agreement_with_ai": round(sum(e["agree"] for e in mine) / len(mine), 3) if mine else None,
               "error": self.error,
               "replay": f"python ui/replay.py {os.path.relpath(self.log_path, ROOT)} --move N",
               "log": self.log, "moves": self.moves}
        os.makedirs(LOG_DIR, exist_ok=True)
        _dump(self.log_path, lambda t: json.dump(doc, open(t, "w", encoding="utf-8"),
                                                  indent=0, default=str))
        try:
            self._save_az()
        except Exception as e:                      # never let this stop a game
            print(f"human AlphaZero examples not saved: {e}", file=sys.stderr)
        if self.positions:
            for x in self.positions:
                x["won"] = None if winner is None else winner == self.seat
            os.makedirs(POS_DIR, exist_ok=True)
            meta = {k: doc[k] for k in ("game_id", "seed", "human_id", "opponents",
                                        "finished", "winner", "you_won", "vp")}
            _dump(self.pos_path, lambda t: pickle.dump({"meta": meta, "positions": self.positions},
                                                       open(t, "wb")))

    def _record_az(self, act) -> None:
        """Your decision as an AlphaZero example (src/az): the position from
        your seat, the legal moves as the network sees them, the one you
        chose. Saved to data/human_az/ so training can learn from you."""
        if not self.gs.bl:
            return
        try:
            from src.az.encode import encode_actions, encode_obs
            from src.az.mcts import legal
            acts = legal(self.gs, self.seat)
            if len(acts) < 2:
                return
            k = next((i for i, a in enumerate(acts) if repr(a) == repr(act)), None)
            if k is None:
                return
            dense, idx = encode_actions(self.gs, self.seat, acts)
            self.az_rec.append((encode_obs(self.gs, self.seat).astype(np.float16),
                                dense.astype(np.float16), idx.astype(np.int16), k))
        except Exception:
            pass

    def _save_az(self) -> None:
        """data/human_az/<game>.npz in the self-play shard format; the value
        target is the final result (NaN until the game is over - training
        then only uses the move)."""
        if not getattr(self, "az_rec", None):
            return
        from src.az.mcts import terminal_values
        gs = self.gs
        n = gs.num_players
        if gs.game_over:
            zt = terminal_values(gs)
            z = np.array([zt[(self.seat + k) % n] for k in range(n)], np.float32)
        else:
            z = np.full(n, np.nan, np.float32)
        S, D, I, P, off = [], [], [], [], [0]
        for obs, dense, idx, k in self.az_rec:
            pi = np.zeros(len(dense), np.float32)
            pi[k] = 1.0
            S.append(obs); D.append(dense); I.append(idx); P.append(pi)
            off.append(off[-1] + len(dense))
        os.makedirs(HUMAN_AZ_DIR, exist_ok=True)
        path = os.path.join(HUMAN_AZ_DIR, f"{self.started.strftime('%Y%m%d_%H%M%S')}_{self.id[:8]}.npz")
        arrays = dict(s=np.stack(S), z=np.tile(z, (len(S), 1)), off=np.asarray(off, np.int64),
                      dense=np.concatenate(D), idx=np.concatenate(I), pi=np.concatenate(P),
                      vs=np.zeros((0, 1), np.float16), vz=np.zeros((0, 4), np.float32))

        def write(t):
            with open(t, "wb") as f:                # closed before the rename (Windows)
                np.savez(f, **arrays)
        _dump(path, write)

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
        if PUBLIC:
            self.saved = {}
            return
        gs = self.gs
        vp = [p.victory_points for p in gs.players]
        winner = gs.winner if gs.winner is not None else int(np.argmax(vp))
        self.autosave()
        self.saved = {"log": os.path.relpath(self.log_path, ROOT)}
        if self.positions:
            self.saved["positions"] = os.path.relpath(self.pos_path, ROOT)
        if self.train["feats"]:
            self.saved["training"] = PG.save_training_shard(
                self.train["feats"], self.train["rounds"], self.train["actA"],
                self.train["ci"], human_won=(winner == self.seat), rounds_played=gs.round,
                out_dir="data/human_games", game_id=self.id)

    # -- view --------------------------------------------------------------
    def view(self, reveal: bool) -> Dict:
        reveal = reveal and not PUBLIC
        with self.lock:
            gs = self.gs
            d = gs.get_state_dict()
            act = None if gs.game_over else active_player(gs)
            players = []
            for p, pd in zip(gs.players, d["players"]):
                mine = p.id == self.seat and not self.spectate
                show = mine or reveal or self.spectate   # watching: nothing is hidden
                players.append({
                    "id": p.id, "color": COLORS[p.id % 4], "colorName": COLOR_NAMES[p.id % 4],
                    "you": mine, "agent": None if mine else self.specs.get(p.id),
                    "agentName": None if mine else next(
                        (o["name"] for o in OPPONENTS if o["spec"] == self.specs.get(p.id)),
                        self.specs.get(p.id)),
                    "vp": p.victory_points, "vpSources": vp_sources(p), "solari": p.solari, "spice": p.spice,
                    "water": p.water, "influence": pd["influence"],
                    "alliances": pd["alliances"], "agentsAvail": p.agents_available,
                    "agentsTotal": p.agents_total, "garrison": p.troops_garrison,
                    "supply": p.troops_supply, "spies": p.spies_available,
                    "swordmaster": pd["has_swordmaster"], "councilor": pd["has_councilor"],
                    "inConflict": gs.troops_in_conflict.get(p.id, 0),
                    "worms": gs.sandworms_in_conflict.get(p.id, 0),
                    "strength": gs.combat_strength.get(p.id, 0),
                    "deck": pd["deck_size"], "handSize": pd["hand_size"],
                    "discard": [card_info(c) for c in p.discard],
                    # public: everything owned minus discard / played / trashed
                    # = what hand + deck hold (for you: just your deck)
                    "unseen": name_counts(p.deck if mine else p.hand + p.deck),
                    "buyPower": buy_power(gs, p.id) if not mine else None,
                    "inPlay": [card_info(c) for c in p.in_play],
                    "hand": [card_info(c) for c in p.hand] if show else None,
                    "intrigues": [{"name": c.name, "img": card_img(c.name)} for c in p.intrigue_cards] if show else None,
                    # Glowglobes: may look at the top card of your deck at any time
                    "glowglobes": bool(show and gs.bl and gs.bl.has(p.id, "Glowglobes")),
                    "topCard": card_info(p.deck[-1]) if (show and p.deck and gs.bl
                                                        and gs.bl.has(p.id, "Glowglobes")) else None,
                    "hooks": bool(getattr(p, "has_maker_hooks", False)),
                    "intrigueCount": len(p.intrigue_cards),
                    # once-per-round (flip) techs show greyed out after use until the next round
                    "techs": [{**tech_info(getattr(t, "d", t)),
                               "flip": getattr(getattr(t, "d", t), "activation", None) is not None,
                               "used": bool(getattr(t, "used_this_round", False))}
                              for t in getattr(p, "techs", [])],
                    "skills": sorted(getattr(p, "skills", set())),
                    "skillImgs": [[k, skill_img(k)] for k in sorted(getattr(p, "skills", set()))],
                    "cmdGarrison": getattr(p, "commanders_garrison", 0),
                    "cmdSupply": getattr(p, "commanders_supply", 0),
                    "cmdInConflict": gs.bl.commanders_in_conflict.get(p.id, 0) if gs.bl else 0,
                    "contractsActive": pd.get("contracts_active", 0),
                    "contractList": [{"name": c.name, "img": contract_img(c.name), "done": done,
                                      "rewards": getattr(c, "rewards", None)}
                                     for done, cs in ((False, p.contracts_active), (True, p.contracts_completed))
                                     for c in cs],
                    "battleIcons": [ICON_NAMES.get(i, i) for i in pd.get("battle_icons", [])],
                    "purchases": [{"round": e["round"], **e["bought"]} for e in self.log
                                  if e.get("pid") == p.id and e.get("bought")],
                    "contractsDone": pd.get("contracts_completed", 0),
                    "revealed": p.id in gs.players_revealed,
                    "revealedThisRound": self.revealed.get(p.id),
                    "revealNow": self._reveal_now(gs, p.id) if show else None,
                    "duncanAgent": getattr(gs, "duncan_in_conflict", {}).get(p.id, 0),
                    "leader": leader_view(gs, p, show),
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
                            "img": conflict_img(cc.name),
                            "icon": ICON_NAMES.get(getattr(cc.battle_icon, "value", None),
                                                   getattr(cc.battle_icon, "value", None))}
            spaces = []
            for name, s in UPRISING_BOARD.items():
                if name in LEADER_SPACES and not gs._leader_space_active(name):
                    continue                     # Tuek's Sietch: only with Esmar Tuek
                spaces.append({"name": name, "type": s.space_type.value,
                               "special": SPECIAL.get(name),
                               "cost": s.mandatory_cost, "gate": s.influence_gate,
                               "effects": s.effects, "combat": s.is_combat_space,
                               "occupant": gs.agent_on_space.get(name),
                               "controlledBy": gs.controlled_by.get(name),
                               "makerSpice": gs.maker_bonus_spice.get(name, 0)})
            actions = []
            if act == self.seat and not self.thinking and not self.spectate:
                for i, a in enumerate(gs.get_valid_actions(self.seat)):
                    actions.append({"i": i, "type": a.action_type.value,
                                    "label": label(gs, a), "card": a.card_name,
                                    "space": a.space_name,
                                    "buy": a.acquire_card_name or a.reserve_type,
                                    "intrigue": a.intrigue_card_name, "post": a.spy_post_name,
                                    "img": choice_img(gs, a)})
            evaluation = None
            ev = evaluator()
            if ev is not None:
                try:
                    w, v = self._eval_all(ev, gs)
                    evaluation = {
                        "win": [round(float(x), 3) for x in w],
                        "vp": [round(float(x), 1) for x in v],
                        "hist": self.eval_hist[-400:],
                        "moves": self._move_evals(ev, gs.get_valid_actions(self.seat)) if actions else {},
                        "trained": ev.meta.get("trained"), "net": ev.meta.get("net"),
                        "test": ev.meta.get("test")}
                except Exception:
                    evaluation = None
            return {
                "evaluation": evaluation, "spectate": self.spectate, "paused": self.paused,
                "replay": None if self.replay is None else {k: self.replay.get(k) for k in ("rounds", "vp", "styles", "net")},
                "id": self.id, "seed": self.seed, "seat": self.seat,
                "round": gs.round, "phase": gs.phase.value, "active": act,
                "gameOver": gs.game_over, "winner": gs.winner,
                "thinking": self.thinking,
                "thinkingFor": round(time.time() - self.think_started, 1) if self.thinking is not None else 0,
                "error": self.error, "players": players, "conflict": conflict,
                "lastCombat": last_combat(gs),
                "conflictsLeft": d["conflict_deck_remaining"],
                "shieldWall": gs.shield_wall_intact, "alliance": d["alliance_holder"],
                "spaces": spaces,
                "spyPosts": {post: [p.id for p in gs.players
                                    for _ in range(p.spies_on_board.get(post, 0))]
                             for post in ALL_OBSERVATION_POSTS},
                "row": [card_info(c) for c in gs.imperium_row],
                "reserve": {"prepare_the_way": len(gs.reserve_prepare_the_way),
                            "spice_must_flow": len(gs.reserve_spice_must_flow)},
                "bl": None if not gs.bl else {
                    "techs": [{"stack": i, "left": len(st),
                               "top": tech_info(st[0], gs, self.seat) if st else None}
                              for i, st in enumerate(gs.bl.tech_stacks)],
                    "commanders": [{"space": sp, "present": bool(on)}
                                   for sp, on in gs.bl.commander_on_space.items()],
                    "skills": [{"name": k, "text": t, "img": skill_img(k),
                                "heldBy": [q.id for q in gs.players
                                           if k in getattr(q, "skills", set())]}
                               for k, t in SKILL_TEXT.items()],
                    "skillRow": [{"name": k, "text": SKILL_TEXT.get(k, ""), "img": skill_img(k),
                                  "youHave": k in getattr(gs.players[self.seat], "skills", set())}
                                 for k in getattr(gs.bl, "skill_row", [])],
                    "skillDeckLeft": len(getattr(gs.bl, "skill_deck", [])),
                    "commanderCost": gs.bl.commander_cost(self.seat),
                },
                "contracts": [{"name": c["name"], "rewards": c.get("rewards"),
                               "trigger": c.get("trigger_condition"),
                               "img": contract_img(c["name"])}
                              for c in d.get("contracts_on_board", [])],
                "actions": actions, "log": self.log[-400:], "saved": self.saved,
                "choicePrompt": (gs.pending_choice_for(self.seat).prompt
                                 if act == self.seat and gs.pending_choice_for(self.seat) else None),
            }


# one game per browser (cookie); in local use that's simply your game
SESSIONS: Dict[str, Session] = {}
SEEN: Dict[str, float] = {}
SESSIONS_LOCK = threading.Lock()
COOKIE = "dune_sid"


def _prune_sessions() -> None:
    now = time.time()
    with SESSIONS_LOCK:
        for sid in [k for k, t in SEEN.items() if now - t > IDLE_SECS]:
            SESSIONS.pop(sid, None)
            SEEN.pop(sid, None)


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
        if getattr(self, "_new_sid", None):
            self.send_header("Set-Cookie", f"{COOKIE}={self._new_sid}; Path=/; Max-Age=86400; "
                                           f"HttpOnly; SameSite=Lax" + ("; Secure" if PUBLIC else ""))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def _sid(self) -> Optional[str]:
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE and re.fullmatch(r"[0-9a-f]{32}", v):
                return v
        return None

    def _session(self) -> Optional["Session"]:
        sid = self._sid()
        if sid is None:
            return None
        with SESSIONS_LOCK:
            ses = SESSIONS.get(sid)
            if ses is not None:
                SEEN[sid] = time.time()
            return ses

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
        if path.startswith("/tech/"):
            return self._file(os.path.join(TECH_DIR, unquote(os.path.basename(path))))
        if path.startswith("/skill/"):
            return self._file(os.path.join(SKILL_DIR, unquote(os.path.basename(path))))
        if path.startswith("/hub/"):
            sub, _, f = path[5:].partition("/")
            if sub in ("cards", "contracts", "leaders", "navigation"):
                return self._file(os.path.join(HUB_DIR, sub, unquote(os.path.basename(f))))
        if path.startswith("/cards/"):
            return self._file(os.path.join(CARD_DIR, os.path.basename(path)))
        if path == "/api/options":
            return self._json({"public": PUBLIC, "opponents": [BEST_AI], "leaders": leader_list(True),
                               "upLeaders": [l["name"] for l in leader_list(False)]})
        if path == "/api/replays":
            return self._json({"games": [] if PUBLIC else list_replays()})
        if path == "/api/state":
            ses = self._session()
            if ses is None:
                return self._json({"none": True})
            return self._json(ses.view(reveal=q.get("reveal") == "1"))
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 10_000:
            return self._send(413, b"too large", "text/plain")
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json({"error": "bad request"}, 400)
        if self.path == "/api/new":
            seed = body.get("seed")
            rp = None if PUBLIC else body.get("replay")
            seat = int(body.get("seat", 0))
            if PUBLIC and not 0 <= seat <= 3:
                return self._json({"error": "pick a seat"}, 400)
            try:
                replay = load_replay(rp["file"], rp["line"]) if rp else None
            except Exception as e:
                return self._json({"error": f"can't load that game: {e}"}, 400)
            _prune_sessions()
            sid = self._sid()
            with SESSIONS_LOCK:
                if sid not in SESSIONS and len(SESSIONS) >= MAX_GAMES:
                    return self._json({"error": "The table is full right now - try again in a few minutes."}, 503)
            if sid is None:
                sid = self._new_sid = uuid.uuid4().hex
            ses = Session(seat=seat, pace=float(body.get("pace", 1.0)),
                              # only the Best AI, even if a request names another bot
                              specs=[BEST_AI["spec"]] * 4,
                              seed=int(seed) if seed not in (None, "") else None,
                              bloodlines=bool(body.get("bloodlines", True)), replay=replay,
                              use_leaders=bool(body.get("leaders", True)),
                              my_leader=body.get("leader") or None)
            with SESSIONS_LOCK:
                SESSIONS[sid], SEEN[sid] = ses, time.time()
            return self._json({"ok": True})
        ses = self._session()
        if self.path == "/api/pause":
            if ses is None:
                return self._json({"error": "no game"}, 400)
            ses.paused = bool(body.get("paused"))
            if not ses.paused:
                ses.kick()
            return self._json({"ok": True, "paused": ses.paused})
        if self.path == "/api/act":
            if ses is None:
                return self._json({"error": "no game"}, 400)
            err = ses.human_move(int(body["i"]))
            return self._json({"ok": err is None, "error": err}, 200 if err is None else 400)
        self._send(404, b"not found", "text/plain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    ap.add_argument("--public", action="store_true", help="website mode (same as DUNE_PUBLIC=1)")
    args = ap.parse_args()
    global PUBLIC
    PUBLIC = PUBLIC or args.public
    if PUBLIC:                                  # behind a host's proxy: listen on every interface
        srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
        srv.daemon_threads = True
        print(f"Dune AI table (public mode) on port {args.port}", flush=True)
        srv.serve_forever()
        return
    # serve both loopbacks: browsers try ::1 first for "localhost", and an
    # IPv4-only server costs ~2 s per request while they fall back
    class Server(ThreadingHTTPServer):
        # Windows lets a second process bind a port with SO_REUSEADDR, so two
        # servers would silently share it; claim it exclusively instead
        allow_reuse_address = False

        def server_bind(self):
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

    class V6(Server):
        address_family = socket.AF_INET6

    try:
        srv = Server(("127.0.0.1", args.port), Handler)
    except OSError:
        print(f"A server is already running on port {args.port}: open "
              f"http://localhost:{args.port}, or stop that one (Ctrl+C in its window) first.")
        return
    try:
        v6 = V6(("::1", args.port), Handler)
        threading.Thread(target=v6.serve_forever, daemon=True).start()
    except OSError:
        pass
    print(f"Dune AI table: http://localhost:{args.port}  ({len(CARD_IMG)} card images)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
