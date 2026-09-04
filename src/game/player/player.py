"""
Player class for Dune Imperium Uprising
Represents a player's complete state including resources, cards, and progress
"""
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.cards.card import Card, CardTag
    from src.game.cards.intrigue import IntrigueCard
    from src.game.contract import Contract
    from src.game.tech import Tech


class Player:
    """
    Represents a player in Dune Imperium Uprising
    Tracks all player resources, cards, troops, spies, and state
    """

    def __init__(self, player_id: int):
        """
        Initialize a player

        Args:
            player_id: Unique identifier (0-3 for 4-player game)
        """
        self.id = player_id

        # ===== LEADER =====
        self.leader = None  # set during setup (src/game/leader)

        # ===== RESOURCES =====
        self.solari = 0
        self.spice = 0
        self.water = 0
        self.victory_points = 1  # All players start with 1 VP

        # ===== DECK MANAGEMENT (PLAYER CARDS) =====
        # These cards cycle: hand → in_play → discard → deck (reshuffled)
        self.deck: List['Card'] = []  # Draw pile
        self.hand: List['Card'] = []  # Current hand
        self.discard: List['Card'] = []  # Discard pile (visible to all)
        self.in_play: List['Card'] = []  # Cards played this turn
        self.trash: List['Card'] = []  # Removed from game permanently (visible to all)

        # ===== INTRIGUE CARDS (SEPARATE FROM DECK) =====
        # Intrigue cards are one-time use
        # After played, they go to GLOBAL intrigue discard (not player's discard)
        self.intrigue_cards: List['IntrigueCard'] = []

        # ===== AGENTS =====
        self.agents_total = 2  # Total agents (starts at 2, can increase to 3 with Swordmaster)
        self.agents_available = 2  # Agents not yet placed this round
        self.has_swordmaster = False  # Whether player has reached Swordmaster (unlocks 3rd agent)

        # ===== TROOPS =====
        # Total troop pool: 12 troops
        # Start: 3 in garrison, 9 in supply
        self.troops_supply = 9  # Troops not yet deployed (starts at 9)
        self.troops_garrison = 3  # Troops in garrison (starts at 3)
        # Troops recruited during the current Agent turn — the pool that may be
        # deployed to a Conflict when visiting a Combat space. Reset each Agent turn.
        self.troops_recruited_this_turn = 0
        # Per-Agent-turn flags used by Uprising card conditions (reset each turn).
        self.recalled_spy_this_turn = False
        self.agent_to_maker_this_turn = False
        self.agent_to_faction_this_turn = False
        # Set by gain_spice(); Leverage may only be played after gaining spice
        # this turn.  Reset at the start of each Agent/Reveal turn.
        self.gained_spice_this_turn = False
        # Cards acquired during this player's current Reveal/buy turn (Call to Arms).
        self.cards_acquired_this_turn = 0
        # Manipulate: a Row card set aside for THIS player at a discount.
        self.reserved_card = None
        self.reserved_discount = 0
        # Per-turn deploy budget: 2 (once, from any deploy icon) + every troop
        # recruited this turn.  Multiple deploy icons do NOT stack the +2.
        self.deploy_budget_this_turn = 0
        self.deployed_this_turn = 0
        # Undercover Asset / Insider Information: ignore a board space's Influence
        # requirement.  Emperor's Invitation: the card you play may reach an
        # Emperor space regardless of its icons.  Cleared at end of Agent turn
        # and at Recall.
        self.ignore_influence_gates_this_turn = False
        self.grant_emperor_access_this_turn = False

        # ===== SPIES (BLOODLINES MECHANIC) =====
        # Players start with 3 spies
        # Spies persist across rounds until recalled
        # Spies can be placed on board space spy placements
        self.spies_available = 3  # Spies not yet placed (starts at 3)
        self.spies_on_board: Dict[str, int] = {}  # {spy_space_name: spy_count}
        # Example: {"Spacing Guild Spy": 1, "Carthag Spy": 1}
        # Note: spy_space_name refers to the spy placement, not the board space itself

        # ===== INFLUENCE TRACKS (0-6 for each faction) =====
        self.influence: Dict[str, int] = {
            "emperor": 0,
            "spacing_guild": 0,
            "bene_gesserit": 0,
            "fremen": 0
        }

        # ===== FACTION FRIENDSHIPS =====
        # Gained at influence 2, lost if influence drops below 2
        # Worth 1 VP each
        self.faction_friendships: Dict[str, bool] = {
            "emperor": False,
            "spacing_guild": False,
            "bene_gesserit": False,
            "fremen": False
        }

        # ===== ALLIANCES =====
        # Gained at influence 4, worth 1 VP each
        # Can be taken by another player with higher influence
        # Lost if influence drops below 4 or another player overtakes
        self.alliances: Dict[str, bool] = {
            "emperor": False,
            "spacing_guild": False,
            "bene_gesserit": False,
            "fremen": False
        }

        # ===== HIGH COUNCIL (UPRISING) =====
        self.has_councilor = False    # Set when first visiting High Council; grants +2 Persuasion per Reveal turn

        # ===== SANDWORM (UPRISING) =====
        self.has_maker_hooks = False  # Required to spawn sandworms (from Sietch Tabr)

        # ===== BATTLE ICONS (UPRISING) =====
        # Face-up, unmatched battle-icon tokens: "crysknife" / "desert_mouse" /
        # "ornithopter" / "wild".  Each player starts with ONE (2 desert_mouse +
        # 2 crysknife dealt among the 4 players).  Winning a Conflict adds its
        # icon; two matching non-wild icons immediately convert to 1 VP; "wild"
        # only matches at Endgame.  (GameState._award_battle_icon owns the logic.)
        self.battle_icons: List[str] = []

        # ===== CONTRACTS (UPRISING) =====
        self.contracts_active: List['Contract'] = []  # Contracts player has taken but not completed
        self.contracts_completed: List['Contract'] = []  # Contracts player has completed

        # ===== TECHS (UPRISING) =====
        self.techs: List['Tech'] = []  # Tech tiles player has acquired

    # ===== RESOURCE METHODS =====

    def gain_solari(self, amount: int):
        """Gain solari"""
        self.solari += amount

    def gain_spice(self, amount: int):
        """Gain spice"""
        self.spice += amount
        if amount > 0:
            self.gained_spice_this_turn = True   # Leverage precondition

    def gain_water(self, amount: int):
        """Gain water"""
        self.water += amount

    def gain_vp(self, amount: int):
        """Gain victory points"""
        self.victory_points += amount

    def lose_vp(self, amount: int):
        """Lose victory points"""
        self.victory_points = max(0, self.victory_points - amount)

    def can_afford(self, cost_dict: Dict[str, int]) -> bool:
        """
        Check if player has the resources to pay a cost

        Args:
            cost_dict: Dictionary like {"solari": 3, "water": 1}

        Returns:
            True if player can afford the cost
        """
        for resource, amount in cost_dict.items():
            if resource == "solari" and self.solari < amount:
                return False
            elif resource == "spice" and self.spice < amount:
                return False
            elif resource == "water" and self.water < amount:
                return False

        return True

    def pay_cost(self, cost_dict: Dict[str, int]):
        """
        Deduct resources from player

        Args:
            cost_dict: Dictionary like {"solari": 3, "water": 1}
        """
        for resource, amount in cost_dict.items():
            if resource == "solari":
                self.solari -= amount
            elif resource == "spice":
                self.spice -= amount
            elif resource == "water":
                self.water -= amount

    # ===== DECK MANAGEMENT METHODS =====

    def draw_cards(self, num_cards: int):
        """
        Draw cards from deck into hand.

        DEPRECATED: Call GameState.draw_cards_for_player() instead.
        That method uses the game's seeded RNG for reproducible shuffles.
        This fallback uses Python's default (non-seeded) random, which breaks
        determinism in RL training.  It exists only so Player objects remain
        usable outside a full GameState (e.g. unit tests).
        """
        import random
        for _ in range(num_cards):
            if not self.deck:
                if not self.discard:
                    break
                self.deck = list(self.discard)
                random.shuffle(self.deck)   # non-seeded; prefer GameState path
                self.discard = []
            if self.deck:
                self.hand.append(self.deck.pop())

    def play_card_as_agent(self, card: 'Card'):
        """
        Move card from hand to in-play area
        Does NOT resolve effects - that's handled by GameState/EffectResolver

        Args:
            card: Card to play
        """
        if card not in self.hand:
            raise ValueError(f"Card {card.name} not in player's hand")

        if not card.can_play_as_agent():
            raise ValueError(f"Card {card.name} has no access symbols")

        self.hand.remove(card)
        self.in_play.append(card)

    def reveal_cards(self, cards: List['Card']):
        """
        Move cards from hand to in-play during reveal turn

        Args:
            cards: List of cards to reveal
        """
        for card in cards:
            if card not in self.hand:
                raise ValueError(f"Card {card.name} not in hand")
            self.hand.remove(card)
            self.in_play.append(card)

    def discard_card(self, card: 'Card', game_state: 'GameState' = None) -> bool:
        """
        Move card from hand to discard pile
        Checks for "when discarded" effects

        Args:
            card: Card to discard
            game_state: Optional game state for resolving discard effects

        Returns:
            True if card was discarded successfully
        """
        if card not in self.hand:
            return False

        self.hand.remove(card)
        self.discard.append(card)

        # Check if card has "when discarded" effect
        if game_state and hasattr(card, 'discard_effects') and card.discard_effects:
            from src.game.effects import EffectResolver
            for effect in card.discard_effects:
                EffectResolver.resolve_single_effect(effect, self, game_state)

        return True

    def trash_card(self, card: 'Card', from_location: str = "any"):
        """
        Remove card from game permanently (trash)
        Card goes to player's trash pile (visible to all)

        Args:
            card: Card to trash
            from_location: Where card can be trashed from
                          "any" - hand, in_play, or discard
                          "in_play" - only from in_play
                          "hand" - only from hand
                          "discard" - only from discard

        Raises:
            ValueError: If card is not in allowed location

        Note: Cards in deck cannot be trashed
        """
        if from_location == "in_play":
            if card not in self.in_play:
                raise ValueError(f"Card {card.name} not in play")
            self.in_play.remove(card)

        elif from_location == "hand":
            if card not in self.hand:
                raise ValueError(f"Card {card.name} not in hand")
            self.hand.remove(card)

        elif from_location == "discard":
            if card not in self.discard:
                raise ValueError(f"Card {card.name} not in discard")
            self.discard.remove(card)

        elif from_location == "any":
            # Remove from wherever it is (hand, in_play, or discard)
            # Cannot trash from deck
            if card in self.hand:
                self.hand.remove(card)
            elif card in self.in_play:
                self.in_play.remove(card)
            elif card in self.discard:
                self.discard.remove(card)
            else:
                raise ValueError(f"Card {card.name} not in hand, in play, or discard")

        self.trash.append(card)

    def recall_cards(self):
        """
        End of round: move all in-play cards to discard
        Called during recall phase
        """
        self.discard.extend(self.in_play)
        self.in_play = []

    # ===== INTRIGUE METHODS =====

    def add_intrigue(self, intrigue: 'IntrigueCard'):
        """
        Add an intrigue card to player's hand

        Args:
            intrigue: IntrigueCard to add
        """
        self.intrigue_cards.append(intrigue)

    def play_intrigue(self, intrigue: 'IntrigueCard'):
        """
        Play an intrigue card (one-time use)
        Card is removed from player's hand
        Note: Card goes to GLOBAL intrigue discard, handled by GameState

        Args:
            intrigue: IntrigueCard to play
        """
        if intrigue not in self.intrigue_cards:
            raise ValueError(f"Player {self.id} does not have intrigue {intrigue.name}")

        self.intrigue_cards.remove(intrigue)

    # ===== AGENT METHODS =====

    def unlock_third_agent(self):
        """
        Unlock third agent (happens when reaching Swordmaster)
        Permanently increases agents_total to 3
        """
        if not self.has_swordmaster:
            self.has_swordmaster = True
            self.agents_total = 3
            self.agents_available = min(self.agents_available + 1, 3)  # Gain the agent immediately

    def reset_agents(self):
        """
        Reset agents at start of round
        Called during recall phase
        """
        self.agents_available = self.agents_total

    def uplift_agent(self):
        """
        Uplift an agent from a board space
        Gained through Steersman card or Imperial Privilege board space
        This gives the player an additional agent action
        """
        self.agents_available += 1

    # ===== SPY METHODS (BLOODLINES) =====

    def place_spy(self, spy_space_name: str, allow_occupied: bool = False):
        """
        Place a spy on a spy placement

        Normal placement (allow_occupied=False): Can only place where no OTHER player has a spy
        Special placement (allow_occupied=True): Can place even where others have spies

        Note: Player can NEVER place where they already have a spy
        Note: If player has no spies available (all 3 placed), this will fail

        Args:
            spy_space_name: Name of spy placement (e.g., "Spacing Guild Spy")
            allow_occupied: Whether this placement can be made even if another player has a spy there

        Raises:
            ValueError: If player has no spies available or already has spy there
        """
        if self.spies_available < 1:
            raise ValueError(f"Player {self.id} has no spies available (all 3 placed)")

        if self.has_spy_at(spy_space_name):
            raise ValueError(f"Player {self.id} already has a spy at {spy_space_name}")

        self.spies_available -= 1
        self.spies_on_board[spy_space_name] = self.spies_on_board.get(spy_space_name, 0) + 1

    def recall_spy(self, spy_space_name: str) -> bool:
        """
        Recall a spy from a spy placement
        Spies persist across rounds until recalled

        Args:
            spy_space_name: Name of spy placement

        Returns:
            True if spy was recalled, False if no spy there
        """
        if spy_space_name not in self.spies_on_board:
            return False

        spies_on_space = self.spies_on_board[spy_space_name]
        if spies_on_space < 1:
            return False

        self.spies_on_board[spy_space_name] -= 1
        if self.spies_on_board[spy_space_name] == 0:
            del self.spies_on_board[spy_space_name]

        self.spies_available += 1
        return True

    def move_spy(self, from_spy_space: str, to_spy_space: str, allow_occupied: bool = False):
        """
        Move a spy from one spy placement to another
        Used when player has all 3 spies placed and gets a "place spy" effect
        Does NOT trigger spy recall effect (no card draw)

        Args:
            from_spy_space: Spy placement to recall spy from
            to_spy_space: Spy placement to place spy at
            allow_occupied: Whether placement can be made even if another player has spy there

        Raises:
            ValueError: If player has no spy at from_spy_space or already has spy at to_spy_space
        """
        if from_spy_space not in self.spies_on_board:
            raise ValueError(f"Player {self.id} has no spy at {from_spy_space}")

        if self.has_spy_at(to_spy_space):
            raise ValueError(f"Player {self.id} already has a spy at {to_spy_space}")

        # Move spy (temporarily recall, then place)
        self.recall_spy(from_spy_space)
        self.place_spy(to_spy_space, allow_occupied)

    def has_spy_at(self, spy_space_name: str) -> bool:
        """
        Check if player has at least one spy at a spy placement

        Args:
            spy_space_name: Name of spy placement

        Returns:
            True if player has a spy there
        """
        return self.spies_on_board.get(spy_space_name, 0) > 0

    def get_spy_count_at(self, spy_space_name: str) -> int:
        """
        Get number of spies at a spy placement
        (Should always be 0 or 1 per player per space)

        Args:
            spy_space_name: Name of spy placement

        Returns:
            Number of spies at that spy placement
        """
        return self.spies_on_board.get(spy_space_name, 0)

    def get_total_spies(self) -> int:
        """
        Get total number of spies player has
        Always returns 3 (spies are never gained or lost permanently)

        Returns:
            Total spies (always 3)
        """
        return 3

    def get_placed_spy_locations(self) -> List[str]:
        """
        Get list of all spy placements where this player has spies

        Returns:
            List of spy placement names
        """
        return list(self.spies_on_board.keys())


    # ===== TROOP METHODS =====

    def gain_troops(self, count: int):
        """
        Recruit troops from supply to garrison

        Args:
            count: Number of troops to recruit
        """
        recruited = min(count, self.troops_supply)
        self.troops_supply -= recruited
        self.troops_garrison += recruited

    def return_troops_to_supply_from_combat(self, count: int):
        """
        Return troops from combat to supply (NOT garrison)
        Called after combat resolution
        All troops in combat go back to supply, not garrison

        Args:
            count: Number of troops to return
        """
        self.troops_supply += count

    # ===== INFLUENCE METHODS =====

    def gain_influence(self, faction: str, amount: int, game_state: 'GameState'):
        """
        Gain influence with a faction
        Handles faction friendships (influence 2) and alliances (influence 4)

        Args:
            faction: Faction name
            amount: Amount of influence to gain
            game_state: Current game state (for checking alliances)
        """
        if faction not in self.influence:
            raise ValueError(f"Invalid faction: {faction}")

        old_influence = self.influence[faction]
        new_influence = min(old_influence + amount, 6)  # Max influence is 6
        self.influence[faction] = new_influence

        # Check for faction friendship (at influence 2)
        if old_influence < 2 and new_influence >= 2:
            if not self.faction_friendships[faction]:
                self.faction_friendships[faction] = True
                self.gain_vp(1)  # Friendship worth 1 VP

        # Check for alliance (at influence 4)
        # Alliance checking is handled by GameState to check all players
        if old_influence < 4 and new_influence >= 4:
            pass  # GameState will handle alliance grants

    def lose_influence(self, faction: str, amount: int, game_state: 'GameState'):
        """
        Lose influence with a faction
        Can cause loss of faction friendship or alliance

        Args:
            faction: Faction name
            amount: Amount of influence to lose
            game_state: Current game state
        """
        if faction not in self.influence:
            raise ValueError(f"Invalid faction: {faction}")

        old_influence = self.influence[faction]
        new_influence = max(old_influence - amount, 0)
        self.influence[faction] = new_influence

        # Check if lost faction friendship (drops below 2)
        if old_influence >= 2 and new_influence < 2:
            if self.faction_friendships[faction]:
                self.faction_friendships[faction] = False
                self.lose_vp(1)  # Lose 1 VP

        # Check if lost alliance (drops below 4)
        if old_influence >= 4 and new_influence < 4:
            if self.alliances[faction]:
                self.alliances[faction] = False
                self.lose_vp(1)  # Lose 1 VP
                # GameState will handle granting alliance to next highest player

    def grant_alliance(self, faction: str):
        """
        Grant alliance to this player
        Worth 1 VP

        Args:
            faction: Faction name
        """
        if not self.alliances[faction]:
            self.alliances[faction] = True
            self.gain_vp(1)

    def lose_alliance(self, faction: str):
        """
        Lose alliance (taken by another player or influence dropped)
        Lose 1 VP

        Args:
            faction: Faction name
        """
        if self.alliances[faction]:
            self.alliances[faction] = False
            self.lose_vp(1)

    def has_alliance(self, faction: str) -> bool:
        """
        Check if player has alliance with faction

        Args:
            faction: Faction name

        Returns:
            True if player has alliance
        """
        return self.alliances.get(faction, False)

    # ===== CARD TAG CHECKING =====

    def count_cards_with_tag_in_play(self, tag: 'CardTag') -> int:
        """
        Count how many cards in play have a specific tag
        Used for conditional effects like "if you have a Bene Gesserit card in play"

        Args:
            tag: CardTag to count

        Returns:
            Number of cards in play with this tag
        """
        count = 0
        for card in self.in_play:
            if card.has_tag(tag):
                count += 1
        return count

    def has_card_with_tag_in_play(self, tag: 'CardTag') -> bool:
        """
        Check if player has at least one card with specific tag in play

        Args:
            tag: CardTag to check

        Returns:
            True if at least one card in play has this tag
        """
        return self.count_cards_with_tag_in_play(tag) > 0

    # ===== CONTRACT METHODS =====

    def take_contract(self, contract: 'Contract'):
        """
        Player takes a contract (not yet completed)

        Args:
            contract: Contract to take
        """
        self.contracts_active.append(contract)

    def complete_contract(self, contract: 'Contract'):
        """
        Player completes a contract
        Move from active to completed

        Args:
            contract: Contract to complete
        """
        if contract in self.contracts_active:
            self.contracts_active.remove(contract)
            self.contracts_completed.append(contract)

    # ===== TECH METHODS =====

    def gain_tech(self, tech: 'Tech'):
        """
        Player acquires a tech

        Args:
            tech: Tech tile to acquire
        """
        if tech not in self.techs:
            self.techs.append(tech)

    def has_tech(self, tech_name: str) -> bool:
        """
        Check if player has a specific tech by name

        Args:
            tech_name: Name of the tech

        Returns:
            True if player has the tech
        """
        return any(tech.name == tech_name for tech in self.techs)

    def get_tech(self, tech_name: str) -> Optional['Tech']:
        """
        Get a tech by name

        Args:
            tech_name: Name of the tech

        Returns:
            Tech object or None if not found
        """
        for tech in self.techs:
            if tech.name == tech_name:
                return tech
        return None

    def get_passive_reveal_bonuses(self) -> Dict[str, int]:
        """
        Get all passive reveal bonuses from techs
        Combined into a single dictionary

        Returns:
            Dictionary of all passive reveal bonuses
            Example: {"solari": 2, "swords": 1}
        """
        bonuses = {}
        for tech in self.techs:
            if tech.has_passive_reveal_bonus():
                for resource, amount in tech.passive_reveal_bonus.items():
                    bonuses[resource] = bonuses.get(resource, 0) + amount
        return bonuses

    def trigger_tech_effects(self, condition: str, game_state: 'GameState'):
        """
        Trigger all tech effects for a specific condition

        Args:
            condition: Trigger condition (e.g., "complete_contract", "gain_intrigue")
            game_state: Current game state
        """
        from src.game.effects import EffectResolver
        for tech in self.techs:
            if tech.has_trigger(condition):
                EffectResolver.resolve_single_effect(tech.trigger_effect, self, game_state)

    # ===== STATE QUERIES =====

    def get_total_cards(self) -> int:
        """Get total number of cards in player's deck cycle"""
        return len(self.deck) + len(self.hand) + len(self.discard) + len(self.in_play)

    def get_visible_state(self, perspective_player_id: int) -> Dict:
        """
        Return state visible to a specific player
        Used for partial information in AI training

        Args:
            perspective_player_id: Which player is observing

        Returns:
            Dictionary of visible information
        """
        visible = {
            "id": self.id,
            "deck_size": len(self.deck),
            "hand_size": len(self.hand),
            "intrigue_count": len(self.intrigue_cards),
            "discard": [c.name for c in self.discard],  # Discard is public
            "in_play": [c.name for c in self.in_play],  # In-play is public
            "trash": [c.name for c in self.trash],  # Trash is public

            # Resources (all public)
            "solari": self.solari,
            "spice": self.spice,
            "water": self.water,
            "victory_points": self.victory_points,

            # Agents and troops (all public)
            "agents_total": self.agents_total,
            "agents_available": self.agents_available,
            "has_swordmaster": self.has_swordmaster,
            "troops_supply": self.troops_supply,
            "troops_garrison": self.troops_garrison,

            # Spies (all public)
            "spies_available": self.spies_available,
            "spies_on_board": self.spies_on_board.copy(),
            "total_spies": self.get_total_spies(),

            # Influence, friendships, and alliances (all public)
            "influence": self.influence.copy(),
            "faction_friendships": self.faction_friendships.copy(),
            "alliances": self.alliances.copy(),

            # Sandworm (public)
            "has_maker_hooks": self.has_maker_hooks,

            # Battle-icon tokens, face-up & unmatched (public)
            "battle_icons": list(self.battle_icons),

            # Contracts (public count)
            "contracts_active": len(self.contracts_active),
            "contracts_completed": len(self.contracts_completed),

            # Techs (public)
            "techs": [tech.name for tech in self.techs],
            "leader": (self.leader.name if self.leader is not None else None),
            "has_councilor": self.has_councilor,
        }

        # Players can see their own hand and intrigue cards
        if self.id == perspective_player_id:
            visible["hand"] = [c.name for c in self.hand]
            visible["intrigue_cards"] = [i.name for i in self.intrigue_cards]

        return visible

    def __repr__(self):
        return (f"Player {self.id} (VP: {self.victory_points}, "
                f"Solari: {self.solari}, Spice: {self.spice}, Water: {self.water}, "
                f"Cards: {self.get_total_cards()}, Intrigues: {len(self.intrigue_cards)}, "
                f"Agents: {self.agents_available}/{self.agents_total}, "
                f"Spies: {self.spies_available}/{self.get_total_spies()})")