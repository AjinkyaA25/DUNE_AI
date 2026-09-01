"""
Board space definitions for Dune Imperium: Uprising — the SINGLE SOURCE OF TRUTH.

`GameState` imports the derived lookup tables at the bottom of this module
(`SPACE_ICONS`, `SPACE_EFFECTS`, `SPACE_MANDATORY_COSTS`, `COMBAT_SPACES`, ...)
rather than defining its own.  Keep every board fact here.

Standard (2–4 player) board — 22 spaces, from the rulebook "BOARD SPACE GUIDE"
(see `rulebook_spaces_text.txt`).  Six-player-only spaces (Carthag, Hardy
Warriors, personal boards, ...) are intentionally excluded.

Effect-dict vocabulary is the one understood by `EffectResolver`
(`src/game/effects.py`).  Spaces whose behaviour needs a player choice
(Swordmaster, High Council, Gather Support, Sietch Tabr, Spice Refinery, the
Maker spaces) carry `special=True` and an empty `effects` list; `GameState`
generates the concrete `GameAction` variants for those.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.player.player import Player
    from src.game.gameState import GameState


class BoardSpaceType(Enum):
    CITY          = "city"
    EMPEROR       = "emperor"
    SPACING_GUILD = "spacing_guild"
    BENE_GESSERIT = "bene_gesserit"
    FREMEN        = "fremen"
    LANDSRAAD     = "landsraad"
    DESERT        = "desert"      # Maker / Spice-Trade spaces


class BoardSpace:
    """One space on the Arrakis board.  `GameState.agent_on_space` owns occupancy."""

    def __init__(
        self,
        name: str,
        space_type: BoardSpaceType,
        agent_icon: str,
        effects: List[Dict],
        *,
        is_combat_space: bool         = False,
        is_maker_space: bool          = False,
        is_controlled_location: bool  = False,
        mandatory_cost: Optional[Dict] = None,
        influence_gate: Optional[Tuple[str, int]] = None,
        shield_wall_protected: bool   = False,
        one_time_use: bool            = False,
        special: bool                 = False,
        grants_faction_influence: Optional[str] = None,
    ):
        self.name                   = name
        self.space_type             = space_type
        self.agent_icon             = agent_icon
        self.effects                = effects
        self.is_combat_space        = is_combat_space
        self.is_maker_space         = is_maker_space
        self.is_controlled_location = is_controlled_location
        self.mandatory_cost         = mandatory_cost or {}
        self.influence_gate         = influence_gate
        self.shield_wall_protected  = shield_wall_protected
        self.one_time_use           = one_time_use
        self.special                = special
        # Faction whose influence track advances +1 when an Agent is placed here.
        # Defaults to the faction implied by the icon; override / clear explicitly.
        if grants_faction_influence is not None:
            self.grants_faction_influence = grants_faction_influence or None
        else:
            self.grants_faction_influence = {
                BoardSpaceType.EMPEROR:       "emperor",
                BoardSpaceType.SPACING_GUILD: "spacing_guild",
                BoardSpaceType.BENE_GESSERIT: "bene_gesserit",
                BoardSpaceType.FREMEN:        "fremen",
            }.get(space_type)

    # ------------------------------------------------------------------

    @property
    def faction(self) -> Optional[str]:
        return {
            BoardSpaceType.EMPEROR:       "emperor",
            BoardSpaceType.SPACING_GUILD: "spacing_guild",
            BoardSpaceType.BENE_GESSERIT: "bene_gesserit",
            BoardSpaceType.FREMEN:        "fremen",
        }.get(self.space_type)

    def check_influence_gate(self, player: "Player") -> Tuple[bool, str]:
        if not self.influence_gate:
            return True, ""
        faction, required = self.influence_gate
        if player.influence.get(faction, 0) < required:
            return False, (
                f"{self.name} requires {required} {faction} influence "
                f"(player has {player.influence.get(faction, 0)})"
            )
        return True, ""

    def to_dict(self) -> Dict:
        return {
            "name":                   self.name,
            "space_type":             self.space_type.value,
            "agent_icon":             self.agent_icon,
            "effects":                self.effects,
            "is_combat_space":        self.is_combat_space,
            "is_maker_space":         self.is_maker_space,
            "is_controlled_location": self.is_controlled_location,
            "mandatory_cost":         self.mandatory_cost,
            "influence_gate":         self.influence_gate,
            "shield_wall_protected":  self.shield_wall_protected,
            "one_time_use":           self.one_time_use,
            "special":                self.special,
        }

    def __repr__(self) -> str:
        return f"BoardSpace({self.name}, {self.space_type.value})"

    def __eq__(self, other) -> bool:
        return isinstance(other, BoardSpace) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


# ---------------------------------------------------------------------------
# Factory — the 22 standard Uprising board spaces
# ---------------------------------------------------------------------------

def build_uprising_board() -> Dict[str, BoardSpace]:
    spaces: Dict[str, BoardSpace] = {}

    def add(s: BoardSpace) -> None:
        spaces[s.name] = s

    C, E, S = BoardSpaceType.CITY, BoardSpaceType.EMPEROR, BoardSpaceType.SPACING_GUILD
    B, F, L, D = (BoardSpaceType.BENE_GESSERIT, BoardSpaceType.FREMEN,
                  BoardSpaceType.LANDSRAAD, BoardSpaceType.DESERT)

    # ── City ────────────────────────────────────────────────────────────────
    add(BoardSpace("Arrakeen", C, "city",
                   [{"troops": 1, "draw": 1}],
                   is_combat_space=True, is_controlled_location=True,
                   shield_wall_protected=True))
    add(BoardSpace("Spice Refinery", C, "city",
                   [], special=True,          # 0 or 1 spice → 2 or 4 solari
                   is_combat_space=True, is_controlled_location=True,
                   shield_wall_protected=True))
    add(BoardSpace("Research Station", C, "city",
                   [{"troops": 2, "draw": 2}],
                   is_combat_space=True, mandatory_cost={"water": 2}))
    add(BoardSpace("Sietch Tabr", C, "city",
                   [], special=True,          # hooks+troop+water  OR  water+destroy shield wall
                   is_combat_space=True, influence_gate=("fremen", 2)))

    # ── Emperor ─────────────────────────────────────────────────────────────
    add(BoardSpace("Sardaukar", E, "emperor",
                   [{"intrigue": 1, "troops": 4}],
                   mandatory_cost={"spice": 4}))         # NOT a Combat space
    add(BoardSpace("Dutiful Service", E, "emperor",
                   [{"contract": 1}]))

    # ── Spacing Guild ───────────────────────────────────────────────────────
    add(BoardSpace("Heighliner", S, "spacing_guild",
                   [{"troops": 5}],
                   is_combat_space=True, mandatory_cost={"spice": 5}))
    add(BoardSpace("Deliver Supplies", S, "spacing_guild",
                   [{"water": 1}]))

    # ── Bene Gesserit ───────────────────────────────────────────────────────
    add(BoardSpace("Espionage", B, "bene_gesserit",
                   [{"draw": 1, "spy": 1}], mandatory_cost={"spice": 1}))
    add(BoardSpace("Secrets", B, "bene_gesserit",
                   [{"intrigue": 1, "secrets_steal": 1}]))

    # ── Fremen ──────────────────────────────────────────────────────────────
    add(BoardSpace("Desert Tactics", F, "fremen",
                   [{"troops": 1, "trash": 1}],
                   is_combat_space=True, mandatory_cost={"water": 1}))
    add(BoardSpace("Fremkit", F, "fremen",
                   [{"draw": 1}], is_combat_space=True))

    # ── Landsraad ───────────────────────────────────────────────────────────
    add(BoardSpace("High Council", L, "landsraad",
                   [], special=True, mandatory_cost={"solari": 5}))
    add(BoardSpace("Swordmaster", L, "landsraad",
                   [], special=True, one_time_use=True))
    add(BoardSpace("Imperial Privilege", L, "landsraad",
                   [{"uplift": 1, "draw": 1}],
                   influence_gate=("emperor", 2), mandatory_cost={"solari": 3}))
    add(BoardSpace("Assembly Hall", L, "landsraad",
                   [{"intrigue": 1}]))
    add(BoardSpace("Gather Support", L, "landsraad",
                   [], special=True))

    # ── Spice Trade / Desert ────────────────────────────────────────────────
    add(BoardSpace("Accept Contract", D, "desert",
                   [{"draw": 1, "contract": 1}]))
    add(BoardSpace("Shipping", D, "desert",
                   [{"solari": 5, "influence_any": 1}],
                   influence_gate=("spacing_guild", 2), mandatory_cost={"spice": 3}))
    add(BoardSpace("Imperial Basin", D, "desert",
                   [{"spice": 1}],
                   is_combat_space=True, is_maker_space=True,
                   is_controlled_location=True, shield_wall_protected=True))
    add(BoardSpace("Hagga Basin", D, "desert",
                   [], special=True,          # bonus spice + (2 spice OR 1 sandworm)
                   is_combat_space=True, is_maker_space=True,
                   mandatory_cost={"water": 1}))
    add(BoardSpace("Deep Desert", D, "desert",
                   [], special=True,          # bonus spice + (4 spice OR 2 sandworms)
                   is_combat_space=True, is_maker_space=True,
                   mandatory_cost={"water": 3}))

    return spaces


UPRISING_BOARD: Dict[str, BoardSpace] = build_uprising_board()


# ---------------------------------------------------------------------------
# Observation posts (spy placement locations)
#
# THE single source of truth for spy topology: each post -> the set of board
# spaces a Spy there can Infiltrate / Gather Intelligence on.  `GameState`
# derives everything (`ALL_OBSERVATION_POSTS`, `SPACE_TO_OBSERVATION_POSTS`)
# from this dict.
#
# Verified against the physical board (user, 2026-09-01):
#   City area — 3 posts:
#     - Arrakeen Post              -> Arrakeen, Spice Refinery
#     - Research Station Right Post -> Spice Refinery, Research Station
#     - Research Station Left Post  -> Research Station, Sietch Tabr
#   Landsraad area — 2 posts:
#     - Landsraad Post             -> High Council, Swordmaster, Imperial Privilege
#     - Green Post                 -> Assembly Hall, Gather Support
#
# Spice Refinery and Research Station each border 2 posts.  A player with a Spy
# on BOTH posts bordering a space may, on the same Agent turn, spend one Spy to
# Infiltrate AND one Spy to Gather Intelligence (draw a card) — but may NOT
# spend both Spies to draw two cards.  (GameState offers the combined
# (gather+infiltrate) spy-mod only when the space is occupied and 2 Spies border it.)
#
# The remaining posts below are still approximate, pending board review.
# ---------------------------------------------------------------------------

OBSERVATION_POST_CONNECTIONS: Dict[str, set] = {
    # ── City (verified 2026-09-01) ──────────────────────────────────────────
    "Arrakeen Post":               {"Arrakeen", "Spice Refinery"},
    "Research Station Right Post":  {"Spice Refinery", "Research Station"},
    "Research Station Left Post":   {"Research Station", "Sietch Tabr"},
    # ── Landsraad (verified 2026-09-01) ────────────────────────────────────
    "Landsraad Post":      {"High Council", "Swordmaster", "Imperial Privilege"},
    "Green Post":          {"Assembly Hall", "Gather Support"},
    # ── Still approximate, pending board review ────────────────────────────
    "Carthag Post":        {"Espionage", "Secrets"},
    "Imperial Basin Post": {"Imperial Basin", "Accept Contract", "Shipping"},
    "Hagga Basin Post":    {"Hagga Basin", "Deep Desert"},
    "Emperor Post":        {"Sardaukar", "Dutiful Service"},
    "Spacing Guild Post":  {"Heighliner", "Deliver Supplies"},
    "Fremen Post":         {"Desert Tactics", "Fremkit"},
}

ALL_OBSERVATION_POSTS: set = set(OBSERVATION_POST_CONNECTIONS)

SPACE_TO_OBSERVATION_POSTS: Dict[str, set] = {}
for _post, _spaces in OBSERVATION_POST_CONNECTIONS.items():
    for _s in _spaces:
        SPACE_TO_OBSERVATION_POSTS.setdefault(_s, set()).add(_post)


# ---------------------------------------------------------------------------
# Derived lookup tables consumed by GameState
# ---------------------------------------------------------------------------

ALL_BOARD_SPACES:  set            = set(UPRISING_BOARD)
SPACE_ICONS:       Dict[str, str]      = {n: s.agent_icon for n, s in UPRISING_BOARD.items()}
SPACE_EFFECTS:     Dict[str, List[Dict]] = {n: s.effects for n, s in UPRISING_BOARD.items()}
SPACE_MANDATORY_COSTS: Dict[str, Dict] = {
    n: s.mandatory_cost for n, s in UPRISING_BOARD.items() if s.mandatory_cost
}
COMBAT_SPACES:     set = {n for n, s in UPRISING_BOARD.items() if s.is_combat_space}
MAKER_SPACES:      set = {n for n, s in UPRISING_BOARD.items() if s.is_maker_space}
CONTROLLED_LOCATIONS: set = {n for n, s in UPRISING_BOARD.items() if s.is_controlled_location}
SHIELD_WALL_PROTECTED: set = {n for n, s in UPRISING_BOARD.items() if s.shield_wall_protected}
SPECIAL_SPACES:    set = {n for n, s in UPRISING_BOARD.items() if s.special}
INFLUENCE_GATED_SPACES: Dict[str, Tuple[str, int]] = {
    n: s.influence_gate for n, s in UPRISING_BOARD.items() if s.influence_gate
}
SPACE_FACTION_INFLUENCE: Dict[str, str] = {
    n: s.grants_faction_influence
    for n, s in UPRISING_BOARD.items() if s.grants_faction_influence
}

CONTROL_BONUS: Dict[str, Dict[str, int]] = {
    "Arrakeen":       {"solari": 1},
    "Spice Refinery": {"solari": 1},
    "Imperial Basin": {"spice":  1},
}

SWORDMASTER_COST_FIRST: int = 8
SWORDMASTER_COST_AFTER: int = 6
HIGH_COUNCIL_COST:      int = 5

# Maker-space choices: (base_spice, sandworm_count)  [bonus spice always taken]
MAKER_SPACE_OPTIONS: Dict[str, Tuple[int, int]] = {
    "Hagga Basin": (2, 1),
    "Deep Desert": (4, 2),
}
