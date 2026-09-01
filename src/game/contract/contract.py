"""
Contract cards for Dune Imperium Uprising
Players take contracts and complete them for rewards
"""
from typing import Dict, Optional, List
from enum import Enum


class ContractType(Enum):
    """Types of contracts"""
    BOARD_SPACE = "board_space"  # Completes when visiting specific board space
    HARVEST = "harvest"  # Completes when gaining 3+ or 4+ spice at maker spaces
    ACQUIRE_CARD = "acquire_card"  # Completes when acquiring specific card
    ALLIANCE = "alliance"  # Completes when gaining any alliance
    IMMEDIATE = "immediate"  # Completes immediately when taken


class Contract:
    """
    Contract cards provide rewards when completed
    Different types complete under different conditions

    Contracts have NO costs - they just trigger and give rewards
    """

    def __init__(
            self,
            name: str,
            contract_type: ContractType,
            rewards: Dict,
            trigger_condition: Optional[Dict] = None
    ):
        """
        Initialize a contract

        Args:
            name: Contract name
            contract_type: Type of contract
            rewards: What you get for completing
            trigger_condition: Condition for completion (depends on type)

        Trigger conditions by type:
            BOARD_SPACE: {"board_space": "Arrakeen"}
            HARVEST: {"min_spice": 4}  # 3 or 4
            ACQUIRE_CARD: {"card_name": "The Spice Must Flow"}
            ALLIANCE: {}  # No specific condition
            IMMEDIATE: {} or {"requires_intrigue": True}

        Rewards can include:
            - "solari": X - Solari
            - "spice": X - Spice
            - "water": X - Water
            - "draw": X - Draw X cards
            - "troops": X - Gain X troops to garrison
            - "intrigue": X - Draw X intrigue cards
            - "spy": X - Place X spies (normal placement - not on occupied spaces)
            - "spy_special": X - Place X spies (special - can place on occupied spaces)
            - "influence_emperor": X - Gain X emperor influence
            - "influence_spacing_guild": X - Gain X spacing guild influence
            - "influence_bene_gesserit": X - Gain X bene gesserit influence
            - "influence_fremen": X - Gain X fremen influence
            - "uplift": X - Uplift X agents
            - "trash_intrigue_for": {"intrigue": 1, "draw": 1} - Trash intrigue for benefit
        """
        self.name = name
        self.contract_type = contract_type
        self.rewards = rewards
        self.trigger_condition = trigger_condition or {}

        # Track who completed it (if anyone)
        self.completed_by: Optional[int] = None

    def check_completion_on_board_visit(
            self,
            board_space_name: str,
            player: 'Player',
            game_state: 'GameState'
    ) -> bool:
        """
        Check if contract completes when visiting a board space

        Args:
            board_space_name: Board space being visited
            player: Player visiting the space
            game_state: Current game state

        Returns:
            True if contract should complete
        """
        if self.contract_type == ContractType.BOARD_SPACE:
            required_space = self.trigger_condition.get("board_space")
            return board_space_name == required_space

        return False

    def check_completion_on_harvest(
            self,
            board_space_name: str,
            spice_gained: int,
            player: 'Player',
            game_state: 'GameState'
    ) -> bool:
        """
        Check if harvest contract completes

        Must be at a maker space (Hagga Basin, Imperial Basin, Deep Desert)
        Must gain 3+ or 4+ spice (from board + effects) in single action

        Args:
            board_space_name: Board space being visited
            spice_gained: Total spice gained this action
            player: Player performing harvest
            game_state: Current game state

        Returns:
            True if harvest contract should complete
        """
        if self.contract_type != ContractType.HARVEST:
            return False

        # Check if at a maker space
        maker_spaces = ["Hagga Basin", "Imperial Basin", "Deep Desert"]
        if board_space_name not in maker_spaces:
            return False

        # Check if gained enough spice
        min_spice = self.trigger_condition.get("min_spice", 3)
        return spice_gained >= min_spice

    def check_completion_on_acquire(
            self,
            card_name: str,
            player: 'Player',
            game_state: 'GameState'
    ) -> bool:
        """
        Check if contract completes when acquiring a card

        Args:
            card_name: Name of card being acquired
            player: Player acquiring the card
            game_state: Current game state

        Returns:
            True if contract should complete
        """
        if self.contract_type != ContractType.ACQUIRE_CARD:
            return False

        required_card = self.trigger_condition.get("card_name")
        return card_name == required_card

    def check_completion_on_alliance(
            self,
            faction: str,
            player: 'Player',
            game_state: 'GameState'
    ) -> bool:
        """
        Check if contract completes when gaining an alliance

        Args:
            faction: Faction alliance was gained with
            player: Player gaining alliance
            game_state: Current game state

        Returns:
            True if contract should complete
        """
        return self.contract_type == ContractType.ALLIANCE

    def is_immediate(self) -> bool:
        """
        Check if this is an immediate contract (completes when taken)

        Returns:
            True if immediate contract
        """
        return self.contract_type == ContractType.IMMEDIATE

    def requires_intrigue(self) -> bool:
        """
        Check if immediate contract requires player to have an intrigue

        Returns:
            True if requires intrigue card
        """
        return self.trigger_condition.get("requires_intrigue", False)

    def to_dict(self) -> Dict:
        """Serialize contract to dictionary"""
        return {
            "name": self.name,
            "type": self.contract_type.value,
            "trigger_condition": self.trigger_condition,
            "rewards": self.rewards,
            "completed_by": self.completed_by
        }

    def __repr__(self):
        return f"Contract({self.name}, Type: {self.contract_type.value})"

    def __eq__(self, other):
        if not isinstance(other, Contract):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)