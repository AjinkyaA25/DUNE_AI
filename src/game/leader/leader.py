"""
Leader framework for Dune Imperium: Uprising (+ Bloodlines).

Each Leader has two abilities (rulebook p.5):
  * a left-side ability used passively during play (modelled here as trigger hooks)
  * a right-side Signet Ring ability, activated when the player plays their
    Signet Ring starter card on an Agent turn.

`GameState` calls the `on_*` hooks at the corresponding moments.  Every hook is
optional; a Leader supplies only the ones it needs.  Hook callables take
`(player, game_state, ...)` and mutate state directly (usually via
EffectResolver or a pending choice the player resolves).

Card text: src/game/leader/LEADERS.md (verified against the printed cards).
Per-game leader state (Feyd's / Chani's track, memories, Navigation cards, ...)
lives in `state`, which is copied with the game state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.player.player import Player
    from src.game.gameState import GameState

Hook = Callable[["Player", "GameState"], None]


@dataclass
class Leader:
    name: str
    complexity: int = 1                       # 1-3 icons (strategic complexity)
    # left-side ability, expressed as trigger hooks
    on_deck_setup:     Optional[Hook] = None  # before the first hand is dealt (deck changes)
    on_setup:          Optional[Hook] = None  # once, after setup (resources, decks of cards)
    on_round_start:    Optional[Hook] = None
    on_agent_turn_start: Optional[Hook] = None
    on_agent_placed:   Optional[Callable[["Player", "GameState", str], None]] = None
    # another player sent an Agent: (owner, gs, placer_id, space)
    on_other_agent_placed: Optional[Callable[["Player", "GameState", int, str], None]] = None
    on_card_played:    Optional[Callable[["Player", "GameState", object], None]] = None
    on_reveal:         Optional[Hook] = None   # start of the Reveal turn (pre-reveal hand)
    on_reveal_done:    Optional[Hook] = None   # after the reveal: strength known
    on_combat_win:     Optional[Callable[["Player", "GameState", object], None]] = None
    on_intrigue_gained: Optional[Hook] = None
    on_spy_placed:     Optional[Hook] = None
    on_influence_2:    Optional[Callable[["Player", "GameState", str], None]] = None
    on_trash:          Optional[Callable[["Player", "GameState", object], None]] = None
    # troops retreated from / lost in the Conflict: (player, gs, n)
    on_units_left_conflict: Optional[Callable[["Player", "GameState", int], None]] = None
    on_after_step:     Optional[Hook] = None   # after every engine step (turn-level counters)
    # Liet Kynes: (player, gs, n) for every sandworm he would have summoned
    on_sandworm_replaced: Optional[Callable[["Player", "GameState", int], None]] = None
    # right-side Signet Ring ability (runs after the space's effects, so it
    # can see where the Agent went: gs._current_agent_space)
    signet:            Optional[Hook] = None
    # static modifiers the engine reads
    swordmaster_discount: int = 0             # Duncan Idaho
    ignore_gates: Set[str] = field(default_factory=set)      # Liet Kynes: Sietch Tabr
    no_sandworms: bool = False                # Liet Kynes
    card_spy_icon: bool = False               # Gaius Helen Mohiam
    must_gather: bool = False                 # Gaius Helen Mohiam
    # free-form notes / verification flags
    notes: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    _per_turn_flags: Dict[str, bool] = field(default_factory=dict, repr=False)

    def __setstate__(self, d):
        """Unpickle Leaders saved before newer fields existed (old UI games)."""
        import dataclasses
        for f in dataclasses.fields(self):
            if f.name not in d:
                if f.default is not dataclasses.MISSING:
                    d[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    d[f.name] = f.default_factory()
        self.__dict__.update(d)

    # -- helpers the engine calls -------------------------------------------

    def apply_setup(self, player: "Player", gs: "GameState") -> None:
        if self.on_setup:
            self.on_setup(player, gs)

    def trigger(self, name: str, player: "Player", gs: "GameState", *args) -> None:
        # errors are NOT swallowed: a broken ability must show up in the
        # simulation audit instead of silently doing nothing
        fn = getattr(self, name, None)
        if fn is None:
            return
        log = getattr(gs, "leader_log", None)          # audit trail (leader_audit.py)
        if log is not None:
            log.append((player.id, self.name, name))
        fn(player, gs, *args)

    def reset_per_turn(self) -> None:
        self._per_turn_flags.clear()

    def once_per_turn(self, key: str) -> bool:
        """Return True the first time this is called with `key` in a turn."""
        if self._per_turn_flags.get(key):
            return False
        self._per_turn_flags[key] = True
        return True

    def to_dict(self) -> Dict:
        return {"name": self.name, "complexity": self.complexity}
