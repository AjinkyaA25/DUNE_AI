"""
Conflict cards and combat result types for Dune Imperium Uprising
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple


class BattleIcon(Enum):
    CRYSKNIFE = "crysknife"
    DESERT_MOUSE = "desert_mouse"
    ORNITHOPTER = "ornithopter"
    WILD = "wild"  # Only used during Endgame; matches any icon


class Conflict:
    """
    A Conflict card revealed at the start of each round.
    Players compete for its rewards during Phase 3: Combat.
    The winner takes the card into their supply for battle icon matching.
    """

    def __init__(
        self,
        name: str,
        conflict_level: int,
        first_place_reward: Dict,
        second_place_reward: Dict,
        third_place_reward: Optional[Dict] = None,
        battle_icon: Optional[BattleIcon] = None,
        location: Optional[str] = None,
    ):
        """
        Args:
            name: Card name
            conflict_level: 1, 2, or 3 (used to build the deck)
            first_place_reward: Reward dict for highest-strength player
            second_place_reward: Reward dict for second-highest
            third_place_reward: Reward dict for third-highest (4-player only)
            battle_icon: The icon on this card for matching in supply
            location: "Arrakeen", "Spice Refinery", or "Imperial Basin" if this
                      conflict is for a controlled location (winner gains control)

        Reward dict keys (same format as card/space effects):
            "solari", "spice", "water", "vp", "troops", "draw", "intrigue",
            "influence_emperor", "influence_spacing_guild",
            "influence_bene_gesserit", "influence_fremen",
            "control" (True = winner places control marker at self.location),
            "may_pay_spice_for_vp": {"cost": 3, "vp": 1}  (optional conversion)
        """
        self.name = name
        self.conflict_level = conflict_level
        self.first_place_reward = first_place_reward
        self.second_place_reward = second_place_reward
        self.third_place_reward = third_place_reward
        self.battle_icon = battle_icon
        self.location = location

    def __repr__(self):
        return f"Conflict({self.name}, Level {self.conflict_level})"

    def __eq__(self, other):
        if not isinstance(other, Conflict):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class CombatResult:
    """
    The outcome of a combat resolution.
    Produced by GameState.resolve_combat().
    """

    def __init__(
        self,
        conflict_name: str,
        rankings: List[Tuple[int, int]],
        rewards_awarded: Dict[int, Dict],
        has_sandworm: Dict[int, bool],
        winner: Optional[int],
        conflict_card_goes_to: Optional[int],
    ):
        """
        Args:
            conflict_name: Name of the resolved conflict
            rankings: [(player_id, strength), ...] sorted descending by strength
            rewards_awarded: {player_id: reward_dict} — rewards AFTER sandworm doubling
            has_sandworm: {player_id: bool} — whether player had sandworm in conflict
            winner: player_id of first-place winner, or None if tied for first
            conflict_card_goes_to: player_id who takes the card into supply (None on tie)
        """
        self.conflict_name = conflict_name
        self.rankings = rankings
        self.rewards_awarded = rewards_awarded
        self.has_sandworm = has_sandworm
        self.winner = winner
        self.conflict_card_goes_to = conflict_card_goes_to

    def __repr__(self):
        return f"CombatResult({self.conflict_name}, winner={self.winner})"
