"""
Intrigue card class for Dune Imperium Uprising.

Timing rules (FAQ):
  - Plot Intrigue cards: may be played at any time on your Agent OR Reveal turn,
    provided they respect the atomicity rules (no interrupting atomic blocks).
  - Combat Intrigue cards: may only be played during the Combat Phase, and only
    if you have at least one unit (troop or sandworm) in the Conflict.
  - "When you win" cards: may be played immediately after winning a Conflict.
  - Endgame cards: played during Endgame scoring (before final VP count).
  - Intrigue cards with two effects must be resolved "in one piece" — no
    player-initiated actions may interrupt between the two effects.
  - Faction bonuses (e.g. drawing an Intrigue from the BG track) may interrupt
    Intrigue resolution because they are not player-initiated.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.player.player import Player
    from src.game.gameState import GameState


class IntrigueTiming(Enum):
    """When a particular Intrigue card may be played."""
    PLOT    = "plot"      # Any time on your Agent or Reveal turn
    COMBAT  = "combat"    # During Combat Phase (requires unit in Conflict)
    WIN     = "win"       # Immediately after winning a Conflict
    ENDGAME = "endgame"   # During Endgame scoring before final VP count


class IntrigueCard:
    """
    A one-time-use Intrigue card.

    After being played, the card is moved to the global Intrigue discard pile
    via GameState.discard_intrigue_card() — NOT the player's personal discard.

    Effects are stored as a list of effect dicts, same format as Card effects.
    Multi-sentence Intrigue cards store effects in order; the 'atomic' flag
    indicates whether ALL effects must be resolved before other actions
    (most Intrigues are atomic — see FAQ ruling).

    Cost→Reward effects are indicated by a dict with both "cost" and "reward"
    keys rather than as separate dicts, to enforce the arrow rule.
    """

    def __init__(
        self,
        name: str,
        timing: IntrigueTiming,
        effects: List[Dict],
        atomic: bool          = True,
        condition: Optional[Dict] = None,
    ):
        """
        Args:
            name:      Card name.
            timing:    When this card may be played (IntrigueTiming).
            effects:   Ordered list of effect dicts resolved by EffectResolver.
                       Each dict follows the standard effect format
                       (e.g. {"solari": 2}, {"troops": 1}).
            atomic:    If True, all effects in this list must be resolved
                       without interruption (the default for most Intrigues).
                       Set False only for cards explicitly ruled separable (e.g.
                       Changing Allegiances cost vs. reward halves).
            condition: Optional precondition dict that must be satisfied before
                       the card can be played (e.g. {"min_units_in_conflict": 1}
                       for Combat cards).  Checked by can_play().
        """
        self.name      = name
        self.timing    = timing
        self._timing_str = (
            "/".join(t.value for t in timing)
            if isinstance(timing, (set, frozenset, tuple, list))
            else timing.value)
        self.effects   = effects
        self.atomic    = atomic
        self.condition = condition or {}

    # ------------------------------------------------------------------
    # Legality
    # ------------------------------------------------------------------

    def can_play(
        self,
        player: "Player",
        game_state: "GameState",
        *,
        is_agent_turn: bool  = False,
        is_reveal_turn: bool = False,
        in_combat: bool      = False,
        just_won_combat: bool = False,
    ) -> tuple[bool, str]:
        """
        Return (True, "") if this card can be played right now, else
        (False, reason).

        Callers should also verify that no atomic block is currently
        resolving (game_state.is_in_atomic_block()) before allowing play.
        """
        # timing may be a single IntrigueTiming or a collection (dual-timing cards)
        ts = self.timing if isinstance(self.timing, (set, frozenset, tuple, list)) \
            else (self.timing,)

        from src.game.gameState import Phase
        ok_any = False
        if IntrigueTiming.PLOT in ts and (is_agent_turn or is_reveal_turn):
            ok_any = True
        if IntrigueTiming.COMBAT in ts and in_combat:
            units = (game_state.troops_in_conflict.get(player.id, 0)
                     + game_state.sandworms_in_conflict.get(player.id, 0))
            if units >= 1:
                ok_any = True
        if IntrigueTiming.WIN in ts and just_won_combat:
            ok_any = True
        if IntrigueTiming.ENDGAME in ts and game_state.phase == Phase.GAME_OVER:
            ok_any = True
        if not ok_any:
            return False, f"Intrigue '{self.name}' cannot be played right now"

        # Check custom condition (e.g. cost preconditions)
        ok, reason = self._check_condition(player, game_state)
        if not ok:
            return False, reason

        return True, ""

    def _check_condition(
        self, player: "Player", game_state: "GameState"
    ) -> tuple[bool, str]:
        """Evaluate self.condition against current state."""
        c = self.condition

        if "min_solari" in c and player.solari < c["min_solari"]:
            return False, f"Requires {c['min_solari']} Solari"
        if "min_spice" in c and player.spice < c["min_spice"]:
            return False, f"Requires {c['min_spice']} Spice"
        if "min_water" in c and player.water < c["min_water"]:
            return False, f"Requires {c['min_water']} Water"
        if "has_council_seat" in c and not getattr(player, "has_council_seat", False):
            return False, "Requires a High Council seat"
        if "has_alliance" in c:
            if not any(player.alliances.values()):
                return False, "Requires at least one Alliance token"
        if "min_intrigue_in_hand" in c and len(player.intrigue_cards) < c["min_intrigue_in_hand"]:
            return False, f"Requires {c['min_intrigue_in_hand']} Intrigue card(s) in hand"
        if "opponent_troops_in_conflict" in c:
            opponents_with_troops = sum(
                1 for pid in range(game_state.num_players)
                if pid != player.id and game_state.troops_in_conflict.get(pid, 0) > 0
            )
            if opponents_with_troops < 1:
                return False, "Requires at least one opponent troop in the Conflict"

        return True, ""

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, player: "Player", game_state: "GameState") -> None:
        """
        Execute all effects on this card.
        If atomic=True, wraps resolution in an atomic block so other effects
        cannot be interleaved (enforced by game_state.is_in_atomic_block()).
        """
        from src.game.effects import EffectResolver

        if self.atomic:
            game_state.begin_atomic()

        try:
            for effect in self.effects:
                EffectResolver.resolve_single_effect(effect, player, game_state)
        finally:
            if self.atomic:
                game_state.end_atomic()

        game_state.discard_intrigue_card(self)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "name":    self.name,
            "timing":  self._timing_str,
            "effects": self.effects,
            "atomic":  self.atomic,
        }

    def __repr__(self) -> str:
        return f"IntrigueCard({self.name}, {self._timing_str})"

    def __eq__(self, other) -> bool:
        return isinstance(other, IntrigueCard) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
