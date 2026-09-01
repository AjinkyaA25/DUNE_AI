"""
Opening book — soft priors that bias the agent toward strong early-game lines.

The book NEVER forces a move: `bonus(gs, pid, action)` just returns an additive
score bonus (0 if nothing matches) that HeuristicAgent / GreedyValueAgent fold
into their action scoring, and that generate.py uses to bias self-play
exploration.  Edit `config/openings.json` freely — reload picks it up.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from src.game.gameState import GameAction, ActionType

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config", "openings.json",
)


class OpeningBook:
    def __init__(self, rules: Optional[List[dict]] = None):
        self.rules = rules or []

    # -- loading -------------------------------------------------------

    @classmethod
    def load(cls, path: str = _DEFAULT_PATH) -> "OpeningBook":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(data.get("rules", []))
        except (FileNotFoundError, json.JSONDecodeError):
            return cls([])

    _cached: Optional["OpeningBook"] = None

    @classmethod
    def default(cls) -> "OpeningBook":
        if cls._cached is None:
            cls._cached = cls.load()
        return cls._cached

    @classmethod
    def clear_cache(cls) -> None:
        cls._cached = None

    # -- matching -----------------------------------------------------

    def _rule_active(self, rule: dict, gs, pid: int) -> bool:
        m = rule.get("match", {})
        if "max_round" in m and gs.round > m["max_round"]:
            return False
        if "min_round" in m and gs.round < m["min_round"]:
            return False
        if "num_players" in m and gs.num_players not in m["num_players"]:
            return False
        if m.get("my_leader") not in (None, getattr(gs.players[pid].leader, "name", None)):
            return False
        for fac, lim in m.get("my_influence_below", {}).items():
            if gs.players[pid].influence.get(fac, 0) >= lim:
                return False
        return True

    @staticmethod
    def _pref_matches(pref: dict, a: GameAction) -> bool:
        if "space" in pref:
            if a.action_type != ActionType.AGENT_TURN or a.space_name != pref["space"]:
                return False
            if "card" in pref and a.card_name != pref["card"]:
                return False
            return True
        if "buy_card" in pref:
            return (a.action_type == ActionType.ACQUIRE_CARD
                    and a.acquire_card_name == pref["buy_card"])
        if "buy_reserve" in pref:
            return (a.action_type == ActionType.ACQUIRE_RESERVE
                    and a.reserve_type == pref["buy_reserve"])
        if "action" in pref:
            return a.action_type.value == pref["action"]
        return False

    def bonus(self, gs, pid: int, a: GameAction) -> float:
        total = 0.0
        for rule in self.rules:
            if not self._rule_active(rule, gs, pid):
                continue
            for pref in rule.get("prefer", []):
                if self._pref_matches(pref, a):
                    total += float(pref.get("weight", 1.0))
        return total
