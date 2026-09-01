"""
Leader framework for Dune Imperium: Uprising.

Each Leader has two abilities (rulebook p.5):
  * a left-side ability used passively during play (modelled here as trigger hooks)
  * a right-side Signet Ring ability, activated when the player plays their
    Signet Ring starter card on an Agent turn.

`GameState` calls the `on_*` hooks at the corresponding moments.  Every hook is
optional; a Leader supplies only the ones it needs.  Hook callables take
`(player, game_state)` and mutate state directly (usually via EffectResolver).

NOTE: ability wordings here are best-effort reconstructions of the printed cards.
Anything marked `APPROX` should be checked against the physical Leader card.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.player.player import Player
    from src.game.gameState import GameState

Hook = Callable[["Player", "GameState"], None]


@dataclass
class Leader:
    name: str
    complexity: int = 1                       # 1–3 icons (strategic complexity)
    # left-side ability, expressed as trigger hooks
    on_setup:          Optional[Hook] = None  # once, during game setup
    on_agent_turn_start: Optional[Hook] = None
    on_agent_placed:   Optional[Callable[["Player", "GameState", str], None]] = None
    on_card_played:    Optional[Callable[["Player", "GameState", object], None]] = None
    on_reveal:         Optional[Hook] = None
    on_combat_win:     Optional[Callable[["Player", "GameState", object], None]] = None
    on_intrigue_gained: Optional[Hook] = None
    on_spy_placed:     Optional[Hook] = None
    # right-side Signet Ring ability
    signet:            Optional[Hook] = None
    # free-form notes / verification flags
    notes: str = ""
    _per_turn_flags: Dict[str, bool] = field(default_factory=dict, repr=False)

    # -- helpers the engine calls -------------------------------------------

    def apply_setup(self, player: "Player", gs: "GameState") -> None:
        if self.on_setup:
            self.on_setup(player, gs)

    def trigger(self, name: str, player: "Player", gs: "GameState", *args) -> None:
        fn = getattr(self, name, None)
        if fn is None:
            return
        try:
            fn(player, gs, *args)
        except Exception:
            # A misbehaving leader hook must never crash the game loop.
            pass

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
