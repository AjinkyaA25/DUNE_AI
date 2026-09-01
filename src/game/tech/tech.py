"""
Tech tile class for Dune Imperium: Rise of Ix / Uprising.

Key rules encoded here:
  - "Once per round" abilities (flip icon) can be used on EITHER an Agent
    OR a Reveal turn — NOT just Agent turns (Rise of Ix Rulebook errata,
    Official FAQ).
  - Tech tiles without a "once per round" indicator may be used during any
    of your turns (Agent or Reveal).
  - Endgame scoring effects (e.g. Spy Satellites second ability) are activated
    during Endgame scoring, not during normal turns.
  - Acquire effects trigger immediately when the tile is acquired (before
    other effects that turn — same rule as card Acquire effects).
  - You cannot mix Solari and Spice to pay for a Tech tile cost.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.player.player import Player
    from src.game.gameState import GameState


class TechTrigger(Enum):
    """When a Tech tile's ability can be activated."""
    AGENT_OR_REVEAL = "agent_or_reveal"   # Any of your turns (default)
    ONCE_PER_ROUND  = "once_per_round"    # Flip icon — Agent OR Reveal (errata)
    ENDGAME         = "endgame"           # During Endgame scoring only
    PASSIVE         = "passive"           # Always active; no activation needed
    ACQUIRE         = "acquire"           # Fires once when tile is acquired


class Tech:
    """
    A Tech tile acquired by a player.

    Abilities are stored as trigger → effect-list mappings.  Multiple
    triggers can appear on the same tile (e.g. Spy Satellites has a
    once-per-turn ability AND an Endgame ability).

    'once_per_round' tiles must be flipped to track usage; reset during
    the Recall phase.  The 'used_this_round' flag tracks this.
    """

    def __init__(
        self,
        name: str,
        cost: int,
        abilities: List[Dict],
        trigger: TechTrigger          = TechTrigger.AGENT_OR_REVEAL,
        passive_reveal_bonus: Optional[Dict] = None,
        acquire_effect: Optional[Dict]       = None,
        flavor: str                          = "",
    ):
        """
        Args:
            name:                 Tile name.
            cost:                 Solari cost (default) or Spice cost if noted.
                                  Cannot mix currencies (Rise of Ix rulebook).
            abilities:            List of effect dicts activated on trigger.
            trigger:              TechTrigger indicating when ability fires.
            passive_reveal_bonus: Dict of always-on reveal bonuses (e.g. swords).
            acquire_effect:       Effect that fires once when the tile is acquired
                                  (row refills before this, per FAQ).
            flavor:               Flavor text for display purposes only.
        """
        self.name                 = name
        self.cost                 = cost
        self.abilities            = abilities
        self.trigger              = trigger
        self.passive_reveal_bonus = passive_reveal_bonus or {}
        self.acquire_effect       = acquire_effect
        self.flavor               = flavor

        self.used_this_round: bool = False  # Tracks once-per-round flip state

    # ------------------------------------------------------------------
    # Usage checks
    # ------------------------------------------------------------------

    def can_activate(
        self,
        *,
        is_agent_turn: bool  = False,
        is_reveal_turn: bool = False,
        is_endgame: bool     = False,
    ) -> tuple[bool, str]:
        """
        Return (True, "") if this tile's ability can be activated now.
        Note: errata means once-per-round tiles work on Agent OR Reveal turns.
        """
        t = self.trigger

        if t == TechTrigger.PASSIVE:
            return False, "Passive abilities are always active; no activation needed"

        if t == TechTrigger.ACQUIRE:
            return False, "Acquire effects fire only when the tile is taken"

        if t == TechTrigger.ENDGAME:
            if not is_endgame:
                return False, "Endgame Tech abilities can only activate during Endgame scoring"
            return True, ""

        # AGENT_OR_REVEAL and ONCE_PER_ROUND both work on either turn type (errata)
        if not (is_agent_turn or is_reveal_turn):
            return False, "Tech abilities require an Agent or Reveal turn"

        if t == TechTrigger.ONCE_PER_ROUND:
            if self.used_this_round:
                return False, f"{self.name} has already been used this round (flip icon)"

        return True, ""

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate(self, player: "Player", game_state: "GameState") -> bool:
        """
        Resolve all abilities on this tile.
        Returns True if activation succeeded.
        """
        from src.game.effects import EffectResolver

        for effect in self.abilities:
            EffectResolver.resolve_single_effect(effect, player, game_state)

        if self.trigger == TechTrigger.ONCE_PER_ROUND:
            self.used_this_round = True

        return True

    def trigger_acquire(self, player: "Player", game_state: "GameState") -> None:
        """Fire the acquire effect immediately when the tile is taken."""
        if not self.acquire_effect:
            return
        from src.game.effects import EffectResolver
        EffectResolver.resolve_single_effect(self.acquire_effect, player, game_state)

    def reset_for_round(self) -> None:
        """Called during the Recall phase to flip once-per-round tiles back."""
        self.used_this_round = False

    # ------------------------------------------------------------------
    # Passive helpers (used by Player.get_passive_reveal_bonuses)
    # ------------------------------------------------------------------

    def has_passive_reveal_bonus(self) -> bool:
        return bool(self.passive_reveal_bonus)

    def has_trigger(self, condition: str) -> bool:
        """Check if any ability matches a condition string (for Player.trigger_tech_effects)."""
        for ability in self.abilities:
            if condition in ability:
                return True
        return False

    @property
    def trigger_effect(self) -> Dict:
        """Return the first ability dict (legacy interface from Player)."""
        return self.abilities[0] if self.abilities else {}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "name":                 self.name,
            "cost":                 self.cost,
            "trigger":              self.trigger.value,
            "abilities":            self.abilities,
            "passive_reveal_bonus": self.passive_reveal_bonus,
            "used_this_round":      self.used_this_round,
        }

    def __repr__(self) -> str:
        return f"Tech({self.name}, trigger={self.trigger.value})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Tech) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
