"""
Base Card class for Dune Imperium Uprising
These are cards that go in player decks (shuffled and redrawn)
"""
from typing import List, Dict, Set
from enum import Enum


class CardType(Enum):
    """Types of cards in the game"""
    STARTER = "starter"      # Starting 10 cards (cost = 0)
    IMPERIUM = "imperium"    # Cards from Imperium row
    RESERVE = "reserve"      # Always-available reserve cards


class AccessSymbol(Enum):
    """
    Access symbols that allow cards to be played at specific board spaces
    Each card can have multiple access symbols
    """
    CITY = "city"
    DESERT = "desert"
    LANDSRAAD = "landsraad"
    EMPEROR = "emperor"
    SPACING = "spacing_guild"  # Spacing Guild
    BENE = "bene_gesserit"     # Bene Gesserit
    FREMEN = "fremen"
    SPY = "spy"


class CardTag(Enum):
    """
    Card tags for synergies and conditional effects
    Tags are SEPARATE from access symbols

    Example: A card can have Fremen ACCESS (can play at Fremen spaces)
             but NO Fremen TAG (doesn't count for "if you have Fremen card" effects)
    """
    SPACING_GUILD = "spacing_guild"
    BENE_GESSERIT = "bene_gesserit"
    EMPEROR = "emperor"
    FREMEN = "fremen"


class Card:
    """
    Base card class for player deck cards
    These cards cycle through: hand → play → discard → deck (reshuffled)
    """

    def __init__(
        self,
        name: str,
        card_type: CardType = CardType.IMPERIUM,
        cost: int = 0
    ):
        """
        Initialize a card

        Args:
            name: Card name
            card_type: STARTER (cost=0), IMPERIUM, or RESERVE
            cost: Persuasion cost to acquire from Imperium row (0 for starters)
        """
        self.name = name
        self.card_type = card_type
        self.cost = cost  # Persuasion cost to acquire (0 for starter cards)

        # ===== ACCESS SYMBOLS =====
        # Determines which board spaces this card can access
        # A card can have multiple access symbols
        self.access_symbols: Set[AccessSymbol] = set()

        # ===== CARD TAGS =====
        # Tags for card synergies (SEPARATE from access symbols)
        # Used for conditional effects like "if you have a Fremen card in play"
        # A card can have multiple tags
        self.tags: Set[CardTag] = set()

        # ===== AGENT (ON PLAY) EFFECTS =====
        # Effects that trigger when card is played as an agent action
        # Example: [{"solari": 2}, {"water": 1}]
        # Can include conditional effects: [{"if_tag_bene": {"spice": 2}}]
        self.agent_effects: List[Dict] = []

        # ===== REVEAL EFFECTS =====
        # Every card can be revealed

        # Resources gained on reveal
        self.reveal_resources: Dict[str, int] = {}  # e.g., {"solari": 1, "water": 1}

        # Persuasion gained on reveal (used to buy cards from Imperium row)
        self.persuasion = 0

        # Swords gained on reveal
        self.swords = 0

        # Other reveal effects (draw cards, gain influence, etc.)
        self.reveal_effects: List[Dict] = []

        # ===== SPECIAL PROPERTIES =====

        # Sandworm placement: Can this card put a sandworm in combat?
        self.can_place_sandworm_in_combat = False

        # Trash effects (when card is trashed/destroyed)
        self.trash_effects: List[Dict] = []

        # Acquire (on-buy) effects, and a free-form note on any rules approximation.
        self.acquire_effects: List[Dict] = []
        self.notes: str = ""

    def add_access_symbol(self, symbol: AccessSymbol):
        """
        Add an access symbol to this card

        Args:
            symbol: AccessSymbol to add
        """
        self.access_symbols.add(symbol)

    def add_tag(self, tag: CardTag):
        """
        Add a tag to this card for synergy effects

        Args:
            tag: CardTag to add
        """
        self.tags.add(tag)

    def has_access_to(self, required_symbol: AccessSymbol) -> bool:
        """
        Check if card has access to a board space with given symbol requirement

        Args:
            required_symbol: Required AccessSymbol

        Returns:
            True if card has this access symbol
        """
        return required_symbol in self.access_symbols

    def has_tag(self, tag: CardTag) -> bool:
        """
        Check if card has a specific tag

        Args:
            tag: CardTag to check

        Returns:
            True if card has this tag
        """
        return tag in self.tags

    def can_play_as_agent(self) -> bool:
        """
        Check if card can be played as agent action
        A card can be played as agent if it has at least one access symbol
        """
        return len(self.access_symbols) > 0

    def to_dict(self) -> Dict:
        """Serialize card to dictionary for AI observation"""
        return {
            "name": self.name,
            "type": self.card_type.value,
            "cost": self.cost,
            "access_symbols": [symbol.value for symbol in self.access_symbols],
            "tags": [tag.value for tag in self.tags],
            "persuasion": self.persuasion,
            "swords": self.swords,
            "can_place_sandworm": self.can_place_sandworm_in_combat
        }

    def __repr__(self):
        symbols = [s.value for s in self.access_symbols]
        tags = [t.value for t in self.tags]
        return (f"Card({self.name}, Cost: {self.cost}, "
                f"Symbols: {symbols}, Tags: {tags}, Persuasion: {self.persuasion})")

    def __eq__(self, other):
        """Cards are equal if they have the same name"""
        if not isinstance(other, Card):
            return False
        return self.name == other.name

    def __hash__(self):
        """Hash by name for use in sets/dicts"""
        return hash(self.name)