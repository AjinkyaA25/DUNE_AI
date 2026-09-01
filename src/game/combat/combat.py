"""
Combat utility for Dune Imperium Uprising.

Rules encoded here:
  - Each troop    = 2 strength
  - Each sandworm = 3 strength
  - Each sword revealed this turn = 1 strength
  - Swords only count if the player has >= 1 unit (troop or sandworm).
    Exception: swords already in the pool are kept even if a combat intrigue
    causes all units to retreat — see Duncan Loyal Blade FAQ ruling.
  - Sandworms CANNOT retreat (Desert Ambush FAQ): "Sandworms can never enter
    a garrison, thus they can never be retreated."  Any retreat logic MUST
    skip sandworm counts.
  - Combat rewards resolve in player turn order when order matters (FAQ).
    GameState.resolve_combat() handles ordering; this module is for maths only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from src.game.player.player import Player
    from src.game.gameState import GameState


class Combat:

    @staticmethod
    def calculate_player_strength(
        player: 'Player',
        troops_committed: int,
        sandworms_committed: int,
        swords: int,
        game_state: 'GameState',
    ) -> int:
        """
        Calculate total combat strength.

        Swords require at least 1 unit to count toward strength.
        Strength from swords is NOT lost if units subsequently retreat
        (they were "locked in" during the Reveal turn — see Full Scale
        Assault FAQ and Duncan Loyal Blade ruling).

        Modifiers from Tech tiles and Intrigue cards are applied by
        callers via update_combat_strength() after effects resolve.

        Args:
            player:              The player.
            troops_committed:    Troops currently in the Conflict.
            sandworms_committed: Sandworms currently in the Conflict.
            swords:              Swords accumulated this Reveal turn.
            game_state:          Current game state (for future modifiers).

        Returns:
            Total strength value (always >= 0).
        """
        total_units = troops_committed + sandworms_committed
        if total_units == 0:
            return 0

        strength = (troops_committed * 2) + (sandworms_committed * 3) + swords

        # Apply any active combat strength modifiers (Tech tiles, leader abilities)
        strength += Combat._get_active_modifiers(player, game_state)

        return max(0, strength)

    @staticmethod
    def _get_active_modifiers(
        player: 'Player', game_state: 'GameState'
    ) -> int:
        """
        Sum any active combat strength modifiers for the player.
        Currently handles passive Tech tile reveal bonuses that provide
        'swords' (treated as bonus combat strength during Reveal turns).
        Intrigue-card modifiers (Staged Incident, etc.) are applied directly
        to combat_strength via GameState.update_combat_strength().
        """
        bonus = 0
        for tech in player.techs:
            if tech.has_passive_reveal_bonus():
                bonus += tech.passive_reveal_bonus.get("swords", 0)
        return bonus

    @staticmethod
    def can_retreat(unit_type: str) -> bool:
        """
        Return True if a unit of this type can be retreated.
        Sandworms CANNOT retreat (Desert Ambush FAQ).
        Valid unit_type values: "troop", "sandworm".
        """
        return unit_type == "troop"

    @staticmethod
    def apply_retreat(
        game_state: 'GameState',
        target_player_id: int,
        unit_count: int,
        chooser_player_id: int,
    ) -> int:
        """
        Force `unit_count` troops to retreat from the Conflict to supply for
        target_player_id.  The chooser_player_id selects WHICH troops (e.g.
        Desert Ambush, Disruption Tactics — the PLAYING player chooses).

        Sandworms are never retreated; this method only touches troops.
        Returns the actual number of troops retreated.
        """
        troops_available = game_state.troops_in_conflict.get(target_player_id, 0)
        actual_retreat   = min(unit_count, troops_available)

        if actual_retreat > 0:
            game_state.troops_in_conflict[target_player_id] -= actual_retreat
            game_state.players[target_player_id].troops_supply += actual_retreat
            game_state.update_combat_strength(target_player_id)

        return actual_retreat
