#!/usr/bin/env python
"""
Replay scraped games (games/*.json) inside the engine and write training data.

A scraped game is a partial record: which card each player sent to which
space, reveals, buys, the conflict each round, and board snapshots
(resources, influence, VP). The engine needs every decision, so this is a
GUIDED replay:

  seats      video seats -> engine players by the order of the first agent
             turns in round 1 (engine player 0 starts round 1; the first
             player rotates each round like on the table)
  turns      on a seat's turn, play its next recorded agent turn for this
             round (same card, same space); once those run out it reveals.
             A card the engine copy doesn't have in hand is moved there
             (drawn from its deck/discard, or created) - the video is truth
  buys       after a reveal, the seat's recorded buys for the round are made
             (the card is put in the Row if the engine's Row differs;
             missing persuasion is topped up)
  the rest   choices the video doesn't show (troops deployed, influence
             picks, trashes, spies, combat intrigues) are made by the
             heuristic player
  re-sync    at every round start: the recorded conflict card, and each
             seat's spice/solari/water, influence and VP from the video

Every recorded move becomes a training sample in the self-play shard format
(state features, the move + its alternatives for the policy head, and the
discounted win target of the player who made it), so train_from_human.py
can learn from them directly. Games whose winner can't be read are skipped.

Usage:
  python video_scrape/replay.py                       # every vision game
  python video_scrape/replay.py games/NFT8aY_T3ZE.json --out data/video_games
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.ai.action_features import encode_action  # noqa: E402
from src.ai.agents import HeuristicAgent  # noqa: E402
from src.ai.features import encode_state  # noqa: E402
from src.data.card_definitions import (  # noqa: E402
    conflict_level_1_pool, conflict_level_2_pool, conflict_level_3_pool,
    create_imperium_cards, create_reserve_prepare_the_way,
    create_reserve_spice_must_flow, create_starter_cards, setup_game)
from src.game.bloodlines.cards import (  # noqa: E402
    create_bloodlines_imperium_cards, create_community_imperium_cards,
    create_foldspace)
from src.game.gameState import ActionType  # noqa: E402

FACE_DOWN = "facedowncard"   # vision's name for a card seen from the back
GAMMA = 0.90            # same discounted win target as self-play
POLICY_KMAX = 20
MOVE_CAP = 4000
FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _card_factory() -> dict:
    cards = {}
    for c in (create_starter_cards() + create_imperium_cards()
              + create_bloodlines_imperium_cards()
              + create_community_imperium_cards() + [create_foldspace()]
              + create_reserve_prepare_the_way(1) + create_reserve_spice_must_flow(1)):
        cards.setdefault(norm(c.name), c)
    return cards


_CONFLICTS = None


def conflict_by_name(name: str):
    global _CONFLICTS
    if _CONFLICTS is None:
        _CONFLICTS = {}
        for c in conflict_level_1_pool() + conflict_level_2_pool() + conflict_level_3_pool():
            _CONFLICTS.setdefault(norm(c.name.split(" (")[0]), c)
    return _CONFLICTS.get(norm(name))


def resolve_name(name: str, factory: dict) -> str:
    """Engine spelling key for a scraped card name (OCR typos like
    'Chaom Demands' / 'Possibte Futures' map to the closest known card)."""
    import difflib
    n = norm(name)
    if n in factory or not n:
        return n
    best = difflib.get_close_matches(n, list(factory), n=1, cutoff=0.82)
    return best[0] if best else n


class Replay:
    def __init__(self, game: dict, seed: int = 0):
        self.game = game
        self.gs = setup_game(num_players=4, seed=seed, neutral_leaders=True,
                             use_bloodlines=True)
        self.h = HeuristicAgent(seed=seed)
        self.factory = _card_factory()
        self.stats = collections.Counter()
        self.samples = []          # (features, pid, round, PA or None, ci)
        seats = self._seat_order()
        self.pid_of = {s: i for i, s in enumerate(seats)}
        self.seat_of = {i: s for s, i in self.pid_of.items()}
        self.colour_of = {p["seat"]: p["color"] for p in game["players"]}
        # recorded moves per (seat, round)
        self.agents = collections.defaultdict(list)
        self.buys = collections.defaultdict(list)
        for a in sorted(game["actions"], key=lambda a: a["t"]):
            seat = next((p["seat"] for p in game["players"]
                         if p["name"] == a["player"]), None)
            if seat is None:
                continue
            if a.get("card"):
                key = resolve_name(a["card"], self.factory)
                if key in self.factory:
                    a["card"] = self.factory[key].name
            if a["kind"] == "agent" and a.get("space") and a.get("card"):
                self.agents[seat, a["round"]].append(a)
            elif a["kind"] == "buy" and a.get("card") and norm(a["card"]) != FACE_DOWN:
                self.buys[seat, a["round"]].append(a)
        self.round_t = {r["round"]: r["t"] for r in game["rounds"]}
        self.round_conflict = {r["round"]: r.get("conflict") for r in game["rounds"]}
        self.synced_round = 0

    # -- setup ----------------------------------------------------------------
    def _seat_order(self) -> list[str]:
        """Seats in turn order, from the first agent turns of round 1."""
        seats = [p["seat"] for p in self.game["players"]]
        first = {}
        for a in sorted(self.game["actions"], key=lambda a: a["t"]):
            if a["kind"] == "agent" and a["round"] == 1:
                seat = next((p["seat"] for p in self.game["players"]
                             if p["name"] == a["player"]), None)
                if seat and seat not in first:
                    first[seat] = a["t"]
        order = sorted(first, key=first.get)
        return order + [s for s in seats if s not in order]

    # -- re-sync ----------------------------------------------------------------
    def _snapshot(self, series, t):
        best = None
        for tt, v in series:
            if tt > t + 5:
                break
            best = v
        return best

    def _resync(self, rnd: int) -> None:
        gs = self.gs
        t = self.round_t.get(rnd)
        if t is None:
            return
        name = self.round_conflict.get(rnd)
        c = conflict_by_name(name) if name else None
        if c is not None and (gs.current_conflict is None
                              or norm(gs.current_conflict.name.split(" (")[0]) != norm(name)):
            gs.current_conflict = c
            self.stats["conflict_set"] += 1
        res = self.game.get("resources", {})
        board = self._snapshot(self.game.get("board", []), t)
        for seat, pid in self.pid_of.items():
            p = gs.players[pid]
            v = self._snapshot(res.get(seat, []), t)
            if v:
                for k in ("spice", "solari", "water"):
                    if v.get(k) is not None:
                        setattr(p, k, v[k])
                self.stats["resources_synced"] += 1
            colour = self.colour_of.get(seat)
            if board and colour:
                for f in FACTIONS:
                    lv = board["influence"].get(colour, {}).get(f)
                    if lv is not None:
                        p.influence[f] = lv
                vp = board["vp"].get(colour)
                if vp is not None:
                    p.victory_points = vp
        if board:
            for f in FACTIONS:
                gs._check_and_update_alliance(f)

    # -- forcing the record ------------------------------------------------------
    def _force_in_hand(self, p, card_name: str) -> bool:
        n = norm(card_name)
        if any(norm(c.name) == n for c in p.hand):
            return True
        for pile in (p.deck, p.discard):
            c = next((c for c in pile if norm(c.name) == n), None)
            if c is not None:
                pile.remove(c)
                break
        else:
            proto = self.factory.get(n)
            if proto is None:
                self.stats["unknown_card"] += 1
                return False
            import copy
            c = copy.copy(proto)
            self.stats["card_created"] += 1
        if p.hand:                                   # keep the hand size
            p.deck.append(p.hand.pop())
        p.hand.append(c)
        self.stats["card_forced"] += 1
        return True

    def _match_agent(self, valid, rec):
        n, space = norm(rec["card"]), rec["space"]
        if n == FACE_DOWN:
            # the card was face-down on the table (a card still turning over,
            # or a leader ability like Ilesa Ecaz's): keep the space, play the
            # player's best card for it
            m = [a for a in valid if a.action_type == ActionType.AGENT_TURN
                 and a.space_name == space]
            if m:
                self.stats["face_down_space_only"] += 1
            return max(m, key=lambda a: self.h.score(self.gs, a.player_id, a)) if m else None
        m = [a for a in valid if a.action_type == ActionType.AGENT_TURN
             and norm(a.card_name) == n and a.space_name == space]
        if not m:
            return None
        return max(m, key=lambda a: self.h.score(self.gs, a.player_id, a))

    def _repair_agent(self, pid: int, rec):
        """The card is in hand but the recorded space isn't legal for it.
        Short on a cost (resources drifted mid-round): top it up. Wrong
        icons: the space was misread (or reached via a Spy the engine copy
        lacks) - keep the card, the player's real choice, on its most likely
        legal space."""
        gs, p = self.gs, self.gs.players[pid]
        card = next((c for c in p.hand if norm(c.name) == norm(rec["card"])), None)
        if card is None:
            return None
        ok, why = gs.can_send_agent(pid, rec["space"], card)
        if not ok and why.startswith("Cannot afford"):
            from src.game.board.board import SPACE_MANDATORY_COSTS
            cost = dict(SPACE_MANDATORY_COSTS.get(rec["space"], {}))
            if rec["space"] == "Swordmaster":
                cost["solari"] = gs._swordmaster_cost()
            for k, v in cost.items():
                setattr(p, k, max(getattr(p, k), v))
            self.stats["cost_topped_up"] += 1
            act = self._match_agent(gs.get_valid_actions(pid), rec)
            if act is not None:
                return act
        same_card = [a for a in gs.get_valid_actions(pid)
                     if a.action_type == ActionType.AGENT_TURN
                     and norm(a.card_name) == norm(rec["card"])]
        if same_card:
            self.stats["space_corrected"] += 1
            return max(same_card, key=lambda a: self.h.score(gs, pid, a))
        return None

    def _buy(self, pid: int, rec) -> bool:
        gs, p = self.gs, self.gs.players[pid]
        n = norm(rec["card"])
        for rtype, stack in (("spice_must_flow", gs.reserve_spice_must_flow),
                             ("prepare_the_way", gs.reserve_prepare_the_way)):
            if stack and norm(stack[-1].name) == n:
                gs.persuasion_pool[pid] = max(gs.persuasion_pool[pid], stack[-1].cost)
                return self._step(pid, next(a for a in gs.get_valid_actions(pid)
                                            if a.action_type == ActionType.ACQUIRE_RESERVE
                                            and a.reserve_type == rtype))
        card = next((c for c in gs.imperium_row if norm(c.name) == n), None)
        if card is None:
            src = next((c for c in gs.imperium_deck if norm(c.name) == n), None)
            if src is not None:
                gs.imperium_deck.remove(src)
            else:
                proto = self.factory.get(n)
                if proto is None:
                    self.stats["unknown_card"] += 1
                    return False
                import copy
                src = copy.copy(proto)
            if gs.imperium_row:
                gs.imperium_deck.append(gs.imperium_row.pop())
            gs.imperium_row.append(src)
            card = src
            self.stats["row_forced"] += 1
        if gs.persuasion_pool[pid] < card.cost:
            self.stats["persuasion_topped_up"] += 1
            gs.persuasion_pool[pid] = card.cost
        act = next((a for a in gs.get_valid_actions(pid)
                    if a.action_type == ActionType.ACQUIRE_CARD
                    and norm(a.acquire_card_name) == n), None)
        return act is not None and self._step(pid, act, record=True)

    # -- stepping ------------------------------------------------------------
    def _record(self, pid: int, chosen, valid) -> None:
        gs = self.gs
        feats = encode_state(gs, pid)
        cands = [a for a in valid if a.action_type != ActionType.NO_OP]
        hs = [self.h.score(gs, pid, a) for a in cands]
        order = sorted(range(len(cands)), key=lambda i: -hs[i])[:POLICY_KMAX]
        ci = next((i for i, a in enumerate(cands) if repr(a) == repr(chosen)), None)
        if ci is None:
            # forced move the engine didn't list as legal: keep it as the
            # only candidate (no policy signal, keeps shard arrays aligned)
            PA = encode_action(gs, pid, chosen, 1.0)[None, :]
            self.samples.append((feats, pid, gs.round, PA, 0))
            return
        if ci not in order:
            order = order[:POLICY_KMAX - 1] + [ci]
        lo, hi = min(hs[i] for i in order), max(hs[i] for i in order)
        span = (hi - lo) or 1.0
        PA = np.stack([encode_action(gs, pid, cands[i], (hs[i] - lo) / span)
                       for i in order])
        self.samples.append((feats, pid, gs.round, PA, order.index(ci)))

    def _step(self, pid: int, action, record: bool = False) -> bool:
        if record:
            self._record(pid, action, self.gs.get_valid_actions(pid))
        _, _, _, info = self.gs.step(action)
        if info.get("error"):
            self.stats["step_error"] += 1
            return False
        return True

    def run(self) -> None:
        gs = self.gs
        n_rounds = len(self.game["rounds"])
        moves = 0
        bought = set()
        while not gs.game_over and moves < MOVE_CAP and gs.round <= n_rounds:
            moves += 1
            if gs.round != self.synced_round and gs.phase.value == "player_turns":
                self.synced_round = gs.round
                self._resync(gs.round)
            pid = gs.player_in_reveal_buy
            if pid is None:
                pid = gs.get_current_player_id()
            valid = gs.get_valid_actions(pid)
            seat = self.seat_of[pid]
            pending = gs._has_mandatory_pending_for(pid)

            # reveal-buy phase: this seat's recorded buys, then end
            if gs.player_in_reveal_buy == pid and not pending:
                for i, rec in enumerate(self.buys.get((seat, gs.round), [])):
                    if (seat, gs.round, i) in bought:
                        continue
                    bought.add((seat, gs.round, i))
                    self.stats["buy_ok" if self._buy(pid, rec) else "buy_failed"] += 1
                    break
                else:
                    end = next((a for a in gs.get_valid_actions(pid)
                                if a.action_type == ActionType.END_REVEAL), None)
                    if end is not None:
                        self._step(pid, end)
                        continue
                    # a buy queued a follow-up choice: the heuristic resolves it
                    self._step(pid, self.h.select_action(gs, pid, gs.get_valid_actions(pid)))
                continue

            if gs.phase.value == "player_turns" and not pending \
                    and gs.get_current_player_id() == pid:
                queue = self.agents.get((seat, gs.round), [])
                if queue and gs.players[pid].agents_available > 0:
                    rec = queue[0]
                    act = self._match_agent(valid, rec)
                    if act is None and norm(rec["card"]) != FACE_DOWN and                             self._force_in_hand(gs.players[pid], rec["card"]):
                        valid = gs.get_valid_actions(pid)
                        act = self._match_agent(valid, rec)
                    if act is None:
                        act = self._repair_agent(pid, rec)
                        valid = gs.get_valid_actions(pid)
                    queue.pop(0)
                    if act is not None:
                        self.stats["agent_ok"] += 1
                        self._step(pid, act, record=True)
                        continue
                    self.stats["agent_unmatched"] += 1
                    self.stats["agent_by_heuristic"] += 1
                elif not queue:
                    rev = next((a for a in valid
                                if a.action_type == ActionType.REVEAL_TURN), None)
                    if rev is not None:
                        self.stats["reveal"] += 1
                        self._step(pid, rev, record=True)
                        continue
            # anything the video doesn't show: the heuristic decides
            self.stats["heuristic_moves"] += 1
            self._step(pid, self.h.select_action(gs, pid, valid))

    # -- output ----------------------------------------------------------------
    def winner(self):
        """Engine pid of the winner: the chat's final standings when the
        game has them, else the last VP readings of the board."""
        res = self.game.get("result") or []
        first = [r["player"] for r in res if r.get("place") == 1]
        if len(first) == 1:
            seat = next((p["seat"] for p in self.game["players"]
                         if p["name"] == first[0]), None)
            if seat in self.pid_of:
                self.stats["winner_from_chat"] += 1
                return self.pid_of[seat]
        last = {}
        for t, b in self.game.get("board", []):
            for colour, vp in b["vp"].items():
                if vp is not None:
                    last[colour] = vp
        if len(last) < 3:
            return None
        top = max(last.values())
        best = [c for c, v in last.items() if v == top]
        pids = [self.pid_of[s] for s, c in self.colour_of.items() if c in best]
        if len(pids) == 1:
            return pids[0]
        # tied on the video's VP: break it with the replayed engine's final
        # standing (VP, then the game's tiebreakers)
        gs = self.gs
        if not gs.game_over:
            gs.check_victory_conditions()
        key = lambda q: (gs.players[q].victory_points, gs.players[q].spice,
                         gs.players[q].solari, gs.players[q].water)
        self.stats["winner_tiebreak_by_engine"] += 1
        return max(pids, key=key)


def replay_file(path: str, out_dir: str) -> dict:
    with open(path, encoding="utf-8") as f:
        game = json.load(f)
    colours = [p_["color"] for p_ in game["players"]]
    if len(set(colours)) != len(colours):
        # seat colours misread: agent spaces and VP can't be attributed
        return {"game": game["video_id"], "skipped": "seat colours not distinct"}
    if game.get("duplicate_of"):
        return {"game": game["video_id"], "skipped": f"duplicate of {game['duplicate_of']}"}
    rp = Replay(game)
    rp.run()
    win = rp.winner()
    rep = {"game": game["video_id"], "rounds": len(game["rounds"]),
           "samples": len(rp.samples), "winner_pid": win, **rp.stats}
    if win is None or not rp.samples:
        rep["skipped"] = "winner unknown" if win is None else "no samples"
        return rep
    rounds_played = len(game["rounds"])
    X = np.stack([s[0] for s in rp.samples]).astype(np.float32)
    y = np.array([GAMMA ** max(0, rounds_played - s[2]) if s[1] == win else 0.0
                  for s in rp.samples], dtype=np.float32)
    # trust a game less when much of it had to be forced
    forced = rp.stats["card_created"] + rp.stats["agent_unmatched"] \
        + rp.stats["buy_failed"]
    quality = max(0.3, 1.0 - forced / max(1, len(rp.samples)))
    w = np.full(len(y), quality, dtype=np.float32)
    extra = {}
    pol = [s for s in rp.samples if s[3] is not None]
    if len(pol) == len(rp.samples):
        from src.ai.action_features import ACTION_FEATURE_DIM
        n = len(pol)
        PA = np.zeros((n, POLICY_KMAX, ACTION_FEATURE_DIM), np.float32)
        PM = np.zeros((n, POLICY_KMAX), np.float32)
        for i, s in enumerate(pol):
            k = min(len(s[3]), POLICY_KMAX)
            PA[i, :k], PM[i, :k] = s[3][:k], 1.0
        extra = dict(PA=PA, PM=PM, PCI=np.array([s[4] for s in pol], np.int32))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"video_{game['video_id']}.npz"),
                        X=X, y=y, w=w, **extra)
    rep["quality"] = round(quality, 2)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "video_games"))
    args = ap.parse_args()
    paths = args.games or sorted(
        p for p in glob.glob(os.path.join(HERE, "games", "*.json"))
        if json.load(open(p, encoding="utf-8")).get("source") == "vision")
    reports = []
    for p in paths:
        r = replay_file(p, args.out)
        reports.append(r)
        print(json.dumps(r), flush=True)
    with open(os.path.join(args.out, "replay_report.json"), "w") as f:
        json.dump(reports, f, indent=1)


if __name__ == "__main__":
    main()
