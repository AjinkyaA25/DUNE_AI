"""
GameState for Dune Imperium Uprising.

Central state object. Tracks all shared game state, enforces rules, and
provides the interface used by EffectResolver, Player influence methods,
and the RL environment wrapper.

Effect ordering (per FAQ):
  Agent turn: card played → agent placed → optional spy recall (Gather
  Intelligence, BEFORE any effects) → board space + agent box effects in
  any order the player chooses.
  Atomic effects (double faction bumps, cost→reward arrows, Choose Two)
  cannot be interrupted.
  Mandatory effects (no "may") must execute; the engine raises
  MandatoryEffectSkipped if an agent tries to skip one.
  Spy placement is MANDATORY when the icon appears and supply > 0 (errata).
  Sandworms CANNOT retreat (Desert Ambush FAQ).
  Victory is checked end-of-round only; VP can exceed 12.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.game.cards.card import Card, CardType
from src.game.player.player import Player
from src.game.contract.contract import Contract, ContractType
from src.game.effects import EffectResolver
from src.game.combat.conflict import Conflict, CombatResult, BattleIcon
from src.game.combat.combat import Combat
from src.game.board.board import (
    UPRISING_BOARD,
    ALL_BOARD_SPACES,
    SPACE_ICONS as BOARD_SPACE_ICONS,
    SPACE_EFFECTS as BOARD_SPACE_EFFECTS,
    SPACE_MANDATORY_COSTS as BOARD_SPACE_MANDATORY_COSTS,
    SPACE_FACTION_INFLUENCE,
    COMBAT_SPACES,
    MAKER_SPACES,
    CONTROLLED_LOCATIONS,
    SHIELD_WALL_PROTECTED,
    SPECIAL_SPACES,
    INFLUENCE_GATED_SPACES,
    CONTROL_BONUS,
    MAKER_SPACE_OPTIONS,
    SWORDMASTER_COST_FIRST,
    SWORDMASTER_COST_AFTER,
    HIGH_COUNCIL_COST,
    OBSERVATION_POST_CONNECTIONS,
    ALL_OBSERVATION_POSTS,
    SPACE_TO_OBSERVATION_POSTS,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MandatoryEffectSkipped(Exception):
    """Raised when the agent skips a non-optional game effect."""


class AtomicBlockViolation(Exception):
    """Raised when an atomic effect group is interrupted."""


# ---------------------------------------------------------------------------
# Phase / Turn enums
# ---------------------------------------------------------------------------

class Phase(Enum):
    ROUND_START  = "round_start"
    PLAYER_TURNS = "player_turns"
    COMBAT       = "combat"
    MAKERS       = "makers"
    RECALL       = "recall"
    GAME_OVER    = "game_over"


class TurnType(Enum):
    AGENT  = "agent"
    REVEAL = "reveal"


# ---------------------------------------------------------------------------
# RL action types
# ---------------------------------------------------------------------------

class ActionType(Enum):
    AGENT_TURN           = "agent_turn"
    REVEAL_TURN          = "reveal_turn"
    ACQUIRE_CARD         = "acquire_card"
    ACQUIRE_RESERVE      = "acquire_reserve"
    END_REVEAL           = "end_reveal"
    PAY_SWORDMASTER      = "pay_swordmaster"     # legacy (CLI helper)
    PAY_HIGH_COUNCIL     = "pay_high_council"    # legacy (CLI helper)
    GATHER_SUPPORT       = "gather_support"      # legacy (CLI helper)
    RESOLVE_SPY          = "resolve_spy"
    RESOLVE_UPLIFT       = "resolve_uplift"
    RESOLVE_INTRIGUE_TRASH = "resolve_intrigue_trash"
    RESOLVE_DEPLOY       = "resolve_deploy"
    RESOLVE_TRASH        = "resolve_trash"
    RESOLVE_INFLUENCE    = "resolve_influence"
    RESOLVE_CONTRACT     = "resolve_contract"
    RESOLVE_OPTIONAL     = "resolve_optional"   # accept/decline a cost->reward arrow
    PLAY_INTRIGUE        = "play_intrigue"
    COMBAT_PASS          = "combat_pass"
    NO_OP                = "no_op"


@dataclass
class GameAction:
    """Fully specified action. Set only the fields relevant to action_type."""
    action_type: ActionType
    player_id: int
    # AGENT_TURN
    card_name: Optional[str]          = None
    space_name: Optional[str]         = None
    use_gather_intelligence: bool     = False
    use_infiltrate: bool              = False
    # Spy / uplift resolution
    spy_post_name: Optional[str]      = None
    uplift_space_name: Optional[str]  = None
    intrigue_card_name: Optional[str] = None
    # Acquiring cards
    acquire_card_name: Optional[str]  = None
    reserve_type: Optional[str]       = None   # "prepare_the_way" | "spice_must_flow"
    # Special-space choice (Spice Refinery / Sietch Tabr / Gather Support / Maker spaces)
    space_option: Optional[str]       = None
    # Combat deployment
    deploy_count: int                 = 0
    # Trash resolution
    trash_card_name: Optional[str]    = None
    # Influence-any resolution
    influence_faction: Optional[str]  = None
    # Contract choice (0 or 1 = which face-up contract)
    contract_index: int               = 0
    # Optional cost->reward arrow: accept and pay, or decline
    accept_optional: bool             = False
    # Heighliner (legacy)
    spice_cost: int                   = 0
    troop_count: int                  = 0

    def __repr__(self) -> str:
        extra = ""
        if self.space_option:
            extra += f", opt={self.space_option}"
        if self.action_type == ActionType.RESOLVE_DEPLOY:
            extra += f", n={self.deploy_count}"
        if self.intrigue_card_name:
            extra += f", intrigue={self.intrigue_card_name}"
        return (f"GameAction({self.action_type.value}, p={self.player_id}, "
                f"card={self.card_name}, space={self.space_name}{extra})")


# ---------------------------------------------------------------------------
# Effect queue entry
# ---------------------------------------------------------------------------

@dataclass
class QueuedEffect:
    effect_dict: Dict
    player_id: int
    atomic_group: int = 0     # 0 = non-atomic; same int > 0 = same atomic group
    mandatory: bool   = True
    source: str       = ""


# ---------------------------------------------------------------------------
# Game constants (board layout lives in src/game/board/board.py)
# ---------------------------------------------------------------------------

VP_TO_WIN   = 10
MAX_ROUNDS  = 10
FACTIONS    = ("emperor", "spacing_guild", "bene_gesserit", "fremen")


# ---------------------------------------------------------------------------
# Pending-choice containers
# ---------------------------------------------------------------------------

class PendingSpyPlacement:
    def __init__(self, player_id: int, count: int, allow_occupied: bool,
                 allowed_posts=None):
        self.player_id     = player_id
        self.count         = count
        self.allow_occupied = allow_occupied
        # Optional restriction (e.g. Reliable Informant: faction posts only).
        self.allowed_posts = set(allowed_posts) if allowed_posts else None


class PendingUplift:
    def __init__(self, player_id: int, count: int):
        self.player_id = player_id
        self.count     = count


class PendingIntrigueTrash:
    def __init__(self, player_id: int, benefit: Dict):
        self.player_id = player_id
        self.benefit   = benefit


class PendingDeployment:
    """After sending an Agent to a Combat space: choose how many troops to deploy."""
    def __init__(self, player_id: int, max_deploy: int):
        self.player_id  = player_id
        self.max_deploy = max_deploy


class PendingTrash:
    """Optional 'trash a card' (Desert Tactics, card effects). May be skipped."""
    def __init__(self, player_id: int, count: int = 1, on_trash=None):
        self.player_id = player_id
        self.count     = count
        # Effect dict resolved once per card actually trashed (e.g. Shishakli:
        # "if you trash a card, draw a card").
        self.on_trash  = on_trash


class PendingOptionalPayment:
    """
    A cost -> reward "arrow" effect.  Per the rulebook every arrow is optional:
    the player MAY pay the cost to gain the reward, or decline.  `cost` is a
    resource dict (spice / solari / water, plus the pseudo-cost "recall_spy" =
    recall N of your Spies); `discard` is how many cards must be discarded as
    part of the cost; `reward` is the effect dict gained on accept.
    `discard_tag` / `tag_bonus`: if a discarded card carries `discard_tag`,
    also resolve `tag_bonus` (Space-Time Folding: +1 draw for a Guild card).
    """
    def __init__(self, player_id: int, cost: Dict, reward: Dict,
                 discard: int = 0, label: str = "",
                 discard_tag: str = None, tag_bonus: Dict = None):
        self.player_id   = player_id
        self.cost        = dict(cost or {})
        self.reward      = dict(reward or {})
        self.discard     = int(discard)
        self.label       = label
        self.discard_tag = discard_tag
        self.tag_bonus   = dict(tag_bonus) if tag_bonus else None


class PendingInfluenceChoice:
    """Mandatory 'gain 1 Influence with any Faction' (Shipping)."""
    def __init__(self, player_id: int, count: int = 1):
        self.player_id = player_id
        self.count     = count


class PendingContractChoice:
    """Choose which of the (up to 2) face-up CHOAM contracts to take."""
    def __init__(self, player_id: int, count: int = 1):
        self.player_id = player_id
        self.count     = count


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

class GameState:
    """
    Central game state for Dune Imperium Uprising (2–4 players).

    Responsibilities:
      - Own all shared mutable state (board, decks, conflict, alliances, …)
      - Enforce rules: legal-move checking, phase transitions, victory detection
      - Effect queue with atomicity constraints
      - Persuasion pool per turn
      - RL interface: reset(), get_state_dict(), get_valid_actions(), step()
    """

    def __init__(
        self,
        num_players: int = 4,
        seed: Optional[int] = None,
        use_choam: bool = True,
    ):
        if not (2 <= num_players <= 4):
            raise ValueError("GameState supports 2–4 players.")

        self.num_players = num_players
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._seed      = seed
        self.use_choam  = use_choam

        # ===== PLAYERS =====
        self.players: List[Player] = [Player(i) for i in range(num_players)]
        # All players start at 1 VP in Uprising regardless of player count
        for p in self.players:
            p.victory_points = 1
        # Deal one starting battle-icon token per player from the fixed pool
        # of 2 Desert Mouse + 2 Crysknife (a 4-player game).
        _icon_pool = ["desert_mouse", "desert_mouse", "crysknife", "crysknife"]
        _order = list(range(4))
        self.rng.shuffle(_order)
        for i in range(num_players):
            self.players[i].battle_icons = [_icon_pool[_order[i]]]

        # ===== TURN ORDER =====
        self.first_player: int       = 0
        self.turn_order: List[int]   = list(range(num_players))

        # ===== PHASE / TURN TRACKING =====
        self.round: int                        = 0
        self.phase: Phase                      = Phase.ROUND_START
        self.current_turn_idx: int             = 0
        self.players_revealed: Set[int]        = set()
        self.current_turn_type: Optional[TurnType] = None
        # Tracks which player is in their post-reveal buying sub-phase
        self.player_in_reveal_buy: Optional[int]   = None

        # Combat intrigue phase
        self.combat_intrigue_pass_count: int   = 0
        self._combat_advance_after_pending: Optional[int] = None
        self.last_combat_actor: Optional[int]  = None

        # ===== CONFLICT =====
        self.conflict_deck: List[Conflict]               = []
        self.current_conflict: Optional[Conflict]        = None
        self.won_conflicts: Dict[int, List[Conflict]]    = {i: [] for i in range(num_players)}
        self._flipped_conflicts: Set[int]                = set()

        # ===== COMBAT TRACKING =====
        self.troops_in_conflict:   Dict[int, int] = {i: 0 for i in range(num_players)}
        self.sandworms_in_conflict: Dict[int, int] = {i: 0 for i in range(num_players)}
        self.combat_strength:       Dict[int, int] = {i: 0 for i in range(num_players)}
        self.swords_this_reveal:    Dict[int, int] = {i: 0 for i in range(num_players)}

        # ===== BOARD STATE =====
        self.agent_on_space:   Dict[str, Optional[int]] = {s: None for s in ALL_BOARD_SPACES}
        self.controlled_by:    Dict[str, Optional[int]] = {loc: None for loc in CONTROLLED_LOCATIONS}
        self.shield_wall_intact: bool                   = True
        self.maker_bonus_spice: Dict[str, int]          = {s: 0 for s in MAKER_SPACES}

        # ===== ALLIANCES =====
        self.alliance_holder: Dict[str, Optional[int]] = {f: None for f in FACTIONS}

        # ===== CARD POOLS =====
        self.imperium_deck:              List[Card] = []
        self.imperium_row:               List[Card] = []
        self.reserve_prepare_the_way:    List[Card] = []
        self.reserve_spice_must_flow:    List[Card] = []

        # ===== INTRIGUE =====
        self.intrigue_deck:    List = []
        self.intrigue_discard: List = []

        # ===== CHOAM CONTRACTS =====
        self.contracts_on_board: List[Contract] = []
        self.contract_bank:      List[Contract] = []

        # ===== PENDING CHOICES =====
        self.pending_spy_placements:    List[PendingSpyPlacement]  = []
        self.pending_uplifts:           List[PendingUplift]        = []
        self.pending_intrigue_trashes:  List[PendingIntrigueTrash] = []
        self.pending_deployments:       List[PendingDeployment]    = []
        self.pending_trashes:           List[PendingTrash]         = []
        self.pending_influence_choices: List[PendingInfluenceChoice] = []
        self.pending_contract_choices:  List[PendingContractChoice] = []
        self.pending_optional_payments: List[PendingOptionalPayment] = []

        # ===== COMBAT PHASE TURN TRACKING =====
        self.combat_turn_idx: int = 0
        # Set True while an Agent turn is resolving so Player.gain_troops() knows
        # to increment troops_recruited_this_turn.
        self._agent_turn_active: bool = False
        # player_id whose Agent turn is mid-resolution (awaiting follow-up choices);
        # play passes to the next player once it clears.
        self._agent_turn_open: Optional[int] = None

        # ===== PERSUASION POOL (per-turn) =====
        # Accumulated from revealed cards; spent to acquire Imperium row cards.
        self.persuasion_pool: Dict[int, int] = {i: 0 for i in range(num_players)}

        # ===== EFFECT QUEUE WITH ATOMICITY =====
        self._effect_queue:    List[QueuedEffect] = []
        self._atomic_depth:    int                = 0   # >0 = inside atomic block
        self._atomic_counter:  int                = 0   # monotonically increasing group IDs

        # ===== RESEARCH TRACK (Immortality expansion stub) =====
        self.research_track: Dict[int, int] = {i: 0 for i in range(num_players)}

        # ===== MISC =====
        self.spice_on_sandworm: int = 0
        self.game_over: bool        = False
        self.winner: Optional[int]  = None
        self._last_combat_result    = None
        # conditional reveal effects deferred until the player's pending choices
        # (spy placements etc.) resolve — so "any order" reward rules work.
        self._pending_reveal_conditionals: List[Tuple[Dict, int]] = []
        # Scratch list for trash_self / return_self_to_hand requests collected
        # during effect resolution (drained by _resolve_self_referential).
        self._self_ref_pending: List[str] = []
        self._combat_intrigue_players: Set[int] = set()   # who played a Combat Intrigue this Conflict
        self._prev_vp: Dict[int, int] = {i: 1 for i in range(num_players)}

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def setup_conflict_deck(self, conflict_cards: List[Conflict]) -> None:
        """Ordered: 1×Level-I on top, 5×Level-II, 4×Level-III on bottom."""
        self.conflict_deck = list(conflict_cards)

    def setup_imperium_deck(
        self,
        imperium_cards: List[Card],
        reserve_prepare: List[Card],
        reserve_flow: List[Card],
    ) -> None:
        shuffled = list(imperium_cards)
        self.rng.shuffle(shuffled)
        self.imperium_deck           = shuffled
        self.reserve_prepare_the_way = list(reserve_prepare)
        self.reserve_spice_must_flow = list(reserve_flow)
        self.refill_imperium_row()

    def setup_intrigue_deck(self, intrigue_cards: List) -> None:
        shuffled = list(intrigue_cards)
        self.rng.shuffle(shuffled)
        self.intrigue_deck = shuffled

    def setup_choam_contracts(self, contracts: List[Contract]) -> None:
        if not self.use_choam:
            return
        shuffled = list(contracts)
        self.rng.shuffle(shuffled)
        self.contract_bank = shuffled
        self._refill_contracts_on_board()
        self._refill_contracts_on_board()

    def setup_player_starting_decks(self, starting_cards: List[Card]) -> None:
        for player in self.players:
            deck = list(starting_cards)
            self.rng.shuffle(deck)
            player.deck = deck

    def deal_starting_hands(self) -> None:
        for i in range(self.num_players):
            self.draw_cards_for_player(i, 5)

    # -----------------------------------------------------------------------
    # RL interface
    # -----------------------------------------------------------------------

    def reset(self) -> Dict:
        """Re-initialise to a fresh game and return the initial observation."""
        self.__init__(
            num_players=self.num_players,
            seed=self._seed,
            use_choam=self.use_choam,
        )
        return self.get_state_dict()

    def get_state_dict(self) -> Dict:
        """
        Return a fully observable flat game-state dict suitable for RL.
        Discard piles are public information (per FAQ).
        """
        return {
            # --- meta ---
            "round":              self.round,
            "max_rounds":         MAX_ROUNDS,
            "phase":              self.phase.value,
            "current_player":     self.get_current_player_id(),
            "first_player":       self.first_player,
            "turn_order":         list(self.turn_order),
            "players_revealed":   list(self.players_revealed),
            "player_in_reveal_buy": self.player_in_reveal_buy,
            "game_over":          self.game_over,
            "winner":             self.winner,
            # --- board ---
            "shield_wall_intact": self.shield_wall_intact,
            "maker_bonus_spice":  dict(self.maker_bonus_spice),
            "controlled_by":      dict(self.controlled_by),
            "alliance_holder":    dict(self.alliance_holder),
            "agent_on_space":     {k: v for k, v in self.agent_on_space.items() if v is not None},
            # --- conflict ---
            "current_conflict":      (self.current_conflict.name if self.current_conflict else None),
            "conflict_deck_remaining": len(self.conflict_deck),
            "troops_in_conflict":    dict(self.troops_in_conflict),
            "sandworms_in_conflict": dict(self.sandworms_in_conflict),
            "combat_strength":       dict(self.combat_strength),
            # --- card pools ---
            "imperium_row":           [c.to_dict() for c in self.imperium_row],
            "imperium_deck_size":     len(self.imperium_deck),
            "intrigue_deck_size":     len(self.intrigue_deck),
            "intrigue_discard_size":  len(self.intrigue_discard),
            # public info: which Intrigues have already been played this game
            "intrigue_discard":       [c.name for c in self.intrigue_discard],
            "reserve_prepare_count":  len(self.reserve_prepare_the_way),
            "reserve_flow_count":     len(self.reserve_spice_must_flow),
            # --- choam ---
            "use_choam":           self.use_choam,
            "contracts_on_board":  [c.to_dict() for c in self.contracts_on_board],
            "contract_bank_size":  len(self.contract_bank),
            # --- persuasion ---
            "persuasion_pool":     dict(self.persuasion_pool),
            # --- pending ---
            "pending_spy_placements":   len(self.pending_spy_placements),
            "pending_uplifts":          len(self.pending_uplifts),
            "pending_intrigue_trashes": len(self.pending_intrigue_trashes),
            "pending_deployments":      len(self.pending_deployments),
            "pending_trashes":          len(self.pending_trashes),
            "pending_influence_choices": len(self.pending_influence_choices),
            "pending_contract_choices": len(self.pending_contract_choices),
            "pending_optional_payments": len(self.pending_optional_payments),
            # --- players (all public per rules) ---
            "players": [p.get_visible_state(p.id) for p in self.players],
            # --- research track ---
            "research_track": dict(self.research_track),
        }

    def get_valid_actions(self, player_id: int) -> List[GameAction]:
        """Return all legal GameActions for player_id given the current state."""
        actions: List[GameAction] = []

        # Pending mandatory effects always come first
        self._prune_dead_pendings(player_id)
        if self._has_mandatory_pending_for(player_id):
            return self._valid_pending_actions(player_id)

        if self.game_over:
            return [GameAction(ActionType.NO_OP, player_id)]

        phase = self.phase

        # --- Post-reveal buying sub-phase ---
        if self.player_in_reveal_buy == player_id:
            p = self.players[player_id]
            persuasion = self.persuasion_pool[player_id]
            for card in self.imperium_row:
                if card.cost <= persuasion:
                    actions.append(GameAction(ActionType.ACQUIRE_CARD,
                                              player_id,
                                              acquire_card_name=card.name))
            # Manipulate: a Row card set aside for this player at a discount.
            if p.reserved_card is not None and \
                    max(0, p.reserved_card.cost - p.reserved_discount) <= persuasion:
                actions.append(GameAction(ActionType.ACQUIRE_CARD, player_id,
                                          acquire_card_name=p.reserved_card.name))
            if self.reserve_prepare_the_way:
                ptw = self.reserve_prepare_the_way[-1]
                if ptw.cost <= persuasion:
                    actions.append(GameAction(ActionType.ACQUIRE_RESERVE,
                                              player_id,
                                              reserve_type="prepare_the_way"))
            for ic in p.intrigue_cards:
                ok, _ = ic.can_play(p, self, is_reveal_turn=True)
                if ok:
                    actions.append(GameAction(ActionType.PLAY_INTRIGUE, player_id,
                                              intrigue_card_name=ic.name))
            if self.reserve_spice_must_flow:
                smf = self.reserve_spice_must_flow[-1]
                if smf.cost <= persuasion:
                    actions.append(GameAction(ActionType.ACQUIRE_RESERVE,
                                              player_id,
                                              reserve_type="spice_must_flow"))
            actions.append(GameAction(ActionType.END_REVEAL, player_id))
            return actions

        # --- PLAYER_TURNS phase ---
        if phase == Phase.PLAYER_TURNS:
            current = self.get_current_player_id()
            if current != player_id:
                return [GameAction(ActionType.NO_OP, player_id)]

            p = self.players[player_id]
            not_revealed = player_id not in self.players_revealed

            # Agent turn: one action per valid (card, space, option, spy-mod) combo
            if p.agents_available > 0 and not_revealed:
                seen_cards = set()
                for card in p.hand:
                    if card.name in seen_cards:       # identical duplicates → one entry
                        continue
                    seen_cards.add(card.name)
                    for space in self.get_legal_agent_spaces(player_id, card):
                        options = self.get_space_options(player_id, space)
                        spy_mods = [(False, False)]
                        can_gi  = self.can_gather_intelligence(player_id, space)
                        can_inf = self.can_infiltrate(player_id, space)
                        if can_gi:
                            spy_mods.append((True, False))
                        if can_inf:
                            spy_mods.append((False, True))
                        # A Spy on BOTH posts bordering the space: one may
                        # Infiltrate while the other Gathers Intelligence.
                        # (Never two Gather Intelligence — no double card draw.)
                        if can_inf and self._connected_spy_count(player_id, space) >= 2:
                            spy_mods.append((True, True))
                        for opt in options:
                            for gi, inf in spy_mods:
                                actions.append(GameAction(
                                    ActionType.AGENT_TURN, player_id,
                                    card_name=card.name, space_name=space,
                                    space_option=opt,
                                    use_gather_intelligence=gi, use_infiltrate=inf,
                                ))

            # Plot Intrigue cards may be played on your own Agent/Reveal turn
            if not_revealed and not self.is_in_atomic_block():
                for ic in p.intrigue_cards:
                    ok, _ = ic.can_play(p, self, is_agent_turn=True)
                    if ok:
                        actions.append(GameAction(ActionType.PLAY_INTRIGUE, player_id,
                                                  intrigue_card_name=ic.name))

            # Reveal turn is always available if not yet revealed
            if not_revealed:
                actions.append(GameAction(ActionType.REVEAL_TURN, player_id))

            if not actions:
                actions.append(GameAction(ActionType.NO_OP, player_id))
            return actions

        # --- COMBAT phase ---
        if phase == Phase.COMBAT:
            if self._current_combat_actor() == player_id:
                p = self.players[player_id]
                for ic in p.intrigue_cards:
                    ok, _ = ic.can_play(p, self, in_combat=True)
                    if ok:
                        actions.append(GameAction(ActionType.PLAY_INTRIGUE, player_id,
                                                  intrigue_card_name=ic.name))
                actions.append(GameAction(ActionType.COMBAT_PASS, player_id))
            else:
                actions.append(GameAction(ActionType.NO_OP, player_id))
            return actions

        return [GameAction(ActionType.NO_OP, player_id)]

    def step(self, action: GameAction) -> Tuple[Dict, float, bool, Dict]:
        """
        Apply action, auto-advance automated phases, return
        (state_dict, reward, done, info).

        Reward is the VP delta for action.player_id this step.
        The 'done' flag is set only after end-of-round victory check.
        """
        pid = action.player_id
        old_vp = self.players[pid].victory_points
        info: Dict = {"action": repr(action), "error": None}

        try:
            self._dispatch_action(action)
        except (MandatoryEffectSkipped, AtomicBlockViolation, ValueError) as exc:
            info["error"] = str(exc)

        # Deferred conditional reveal effects fire once the revealing player has
        # cleared their pending choices (so e.g. a Spy placed on the Reveal turn
        # is counted by a "2+ Spies" reveal condition).
        self._flush_reveal_conditionals()

        # An Agent turn = place ONE agent, then play passes to the next player
        # (once any follow-up choices — deploy, spy, ... — are resolved).
        if self._agent_turn_open is not None:
            self._prune_dead_pendings(self._agent_turn_open)
        self._maybe_end_agent_turn()

        # A Combat intrigue that queued a choice defers the combat-turn advance
        # until that player has resolved it.
        ap = getattr(self, "_combat_advance_after_pending", None)
        if ap is not None:
            self._prune_dead_pendings(ap)
            if not self._has_mandatory_pending_for(ap):
                self._combat_advance_after_pending = None
                if self.phase == Phase.COMBAT:
                    self._advance_combat_turn()

        # Auto-advance phases that need no player decisions
        self._auto_advance()

        reward = float(self.players[pid].victory_points - old_vp)
        return self.get_state_dict(), reward, self.game_over, info

    def _flush_reveal_conditionals(self) -> None:
        """Resolve deferred conditional reveal effects for any player who has no
        outstanding mandatory pending choices."""
        if not self._pending_reveal_conditionals:
            return
        remaining: List[Tuple[Dict, int]] = []
        affected: Set[int] = set()
        for conds, pid in self._pending_reveal_conditionals:
            if self._has_mandatory_pending_for(pid):
                remaining.append((conds, pid))
                continue
            EffectResolver.resolve_single_effect(conds, self.players[pid], self)
            affected.add(pid)
            # a conditional could itself have queued a pending — stop for that pid
            if self._has_mandatory_pending_for(pid):
                pass
        self._pending_reveal_conditionals = remaining
        for pid in affected:
            self.update_combat_strength(pid)

    def _maybe_end_agent_turn(self) -> None:
        if self._agent_turn_open is None:
            return
        pid = self._agent_turn_open
        if self._has_mandatory_pending_for(pid):
            return
        self._agent_turn_open = None
        if self.phase == Phase.PLAYER_TURNS and self.player_in_reveal_buy is None:
            self.advance_to_next_player_turn()

    def _dispatch_action(self, action: GameAction) -> None:
        """Route action to the appropriate handler."""
        at = action.action_type
        pid = action.player_id

        if at == ActionType.AGENT_TURN:
            self._step_agent_turn(action)
        elif at == ActionType.REVEAL_TURN:
            self._step_reveal_turn(action)
        elif at == ActionType.ACQUIRE_CARD:
            self._step_acquire_card(action)
        elif at == ActionType.ACQUIRE_RESERVE:
            self._step_acquire_reserve(action)
        elif at == ActionType.END_REVEAL:
            self._step_end_reveal(action)
        elif at == ActionType.PAY_SWORDMASTER:
            self.pay_for_swordmaster(pid)
        elif at == ActionType.PAY_HIGH_COUNCIL:
            self.pay_for_high_council(pid)
        elif at == ActionType.GATHER_SUPPORT:
            self.apply_gather_support(pid)
        elif at == ActionType.RESOLVE_SPY:
            self._step_resolve_spy(action)
        elif at == ActionType.RESOLVE_UPLIFT:
            self._step_resolve_uplift(action)
        elif at == ActionType.RESOLVE_INTRIGUE_TRASH:
            self._step_resolve_intrigue_trash(action)
        elif at == ActionType.RESOLVE_DEPLOY:
            self._step_resolve_deploy(action)
        elif at == ActionType.RESOLVE_TRASH:
            self._step_resolve_trash(action)
        elif at == ActionType.RESOLVE_INFLUENCE:
            self._step_resolve_influence(action)
        elif at == ActionType.RESOLVE_CONTRACT:
            self._step_resolve_contract(action)
        elif at == ActionType.RESOLVE_OPTIONAL:
            self._step_resolve_optional(action)
        elif at == ActionType.PLAY_INTRIGUE:
            self._step_play_intrigue(action)
        elif at == ActionType.COMBAT_PASS:
            self._step_combat_pass(action)
        # NO_OP: do nothing

    def _step_agent_turn(self, action: GameAction) -> None:
        pid = action.player_id
        p   = self.players[pid]
        card = next((c for c in p.hand if c.name == action.card_name), None)
        if card is None:
            raise ValueError(f"Card '{action.card_name}' not in player {pid}'s hand")
        ok, reason = self.can_send_agent(pid, action.space_name, card)
        if not ok:
            raise ValueError(reason)
        self.apply_agent_turn(
            pid, card, action.space_name,
            infiltrate=action.use_infiltrate,
            gather_intelligence=action.use_gather_intelligence,
            space_option=action.space_option,
        )
        self.current_turn_type = TurnType.AGENT
        self._agent_turn_open  = pid

    def _step_reveal_turn(self, action: GameAction) -> None:
        pid = action.player_id
        # apply_reveal_turn() already accumulates persuasion into the pool:
        #   - each revealed card's .persuasion field and {"persuasion": n} effects
        #   - leader on_reveal hooks
        # plus anything a Signet Ring / agent effect added earlier this turn.
        self.apply_reveal_turn(pid)
        p = self.players[pid]
        # High Council Councilor bonus: +2 Persuasion each Reveal turn
        if p.has_councilor:
            self.gain_persuasion(pid, 2)
        # Assembly Hall bonus: +1 Persuasion if player has an Agent there
        if self.agent_on_space.get("Assembly Hall") == pid:
            self.gain_persuasion(pid, 1)
        self.player_in_reveal_buy = pid
        self.players_revealed.add(pid)
        self.current_turn_type = TurnType.REVEAL

    def _step_end_reveal(self, action: GameAction) -> None:
        pid = action.player_id
        if self.player_in_reveal_buy != pid:
            return
        # Force any still-deferred conditional reveal effects to resolve now.
        keep = []
        for conds, p in self._pending_reveal_conditionals:
            if p == pid:
                EffectResolver.resolve_single_effect(conds, self.players[p], self)
            else:
                keep.append((conds, p))
        self._pending_reveal_conditionals = keep
        self.update_combat_strength(pid)
        # Recall in-play cards and reset swords (mark_reveal_complete does both)
        self.mark_reveal_complete(pid)
        self.persuasion_pool[pid]  = 0
        self.player_in_reveal_buy  = None
        # Advance to next player or combat
        self.advance_to_next_player_turn()

    def _step_acquire_card(self, action: GameAction) -> None:
        pid  = action.player_id
        p    = self.players[pid]
        # Manipulate's reserved card (set aside for this player, cost reduced).
        if p.reserved_card is not None and \
                p.reserved_card.name == action.acquire_card_name:
            card = p.reserved_card
            cost = max(0, card.cost - p.reserved_discount)
            if cost > self.persuasion_pool[pid]:
                raise ValueError(
                    f"Insufficient persuasion for reserved '{card.name}' (cost {cost})")
            self.persuasion_pool[pid] -= cost
            p.reserved_card = None
            p.reserved_discount = 0
            p.discard.append(card)
            p.cards_acquired_this_turn += 1
            self._trigger_acquire_effects(pid, card)
            if self.use_choam:
                self.check_acquire_contracts(pid, card.name)
            return
        card = next((c for c in self.imperium_row if c.name == action.acquire_card_name), None)
        if card is None:
            raise ValueError(f"Card '{action.acquire_card_name}' not in Imperium row")
        if card.cost > self.persuasion_pool[pid]:
            raise ValueError(f"Insufficient persuasion ({self.persuasion_pool[pid]}) for '{card.name}' (cost {card.cost})")
        self.persuasion_pool[pid] -= card.cost
        self.acquire_card(pid, card)

    def _step_acquire_reserve(self, action: GameAction) -> None:
        pid = action.player_id
        rtype = action.reserve_type
        stack = (self.reserve_prepare_the_way if rtype == "prepare_the_way"
                 else self.reserve_spice_must_flow)
        if not stack:
            raise ValueError(f"Reserve '{rtype}' is empty")
        card = stack[-1]
        if card.cost > self.persuasion_pool[pid]:
            raise ValueError(f"Insufficient persuasion for reserve card")
        self.persuasion_pool[pid] -= card.cost
        self.acquire_reserve_card(pid, rtype)

    def _step_resolve_spy(self, action: GameAction) -> None:
        pid = action.player_id
        post = action.spy_post_name
        candidates = [p for p in self.pending_spy_placements if p.player_id == pid]
        if not candidates:
            raise ValueError("No pending spy placement for player")
        # Resolve against a queued placement that permits this post — restricted
        # placements (allowed_posts set) are matched first so they don't get
        # starved by a co-pending unrestricted one.
        candidates.sort(key=lambda p: p.allowed_posts is None)
        pending = next(
            (p for p in candidates
             if (p.allowed_posts is None or post in p.allowed_posts)
             and self.can_place_spy(pid, post, p.allow_occupied)),
            None)
        if pending is None:
            raise ValueError(f"Cannot place spy at '{post}'")
        player = self.players[pid]
        if player.spies_available < 1:
            # All 3 Spies are already on the board — Uprising errata makes
            # placement mandatory anyway, so recall one to make room, then
            # place the newly-gained Spy (regular/special/restricted, per the
            # icon that triggered this) as normal.
            if player.spies_on_board:
                self.players[pid].recall_spy(next(iter(player.spies_on_board)))
        if not self.place_spy(pid, post, pending.allow_occupied):
            raise ValueError(f"Cannot place spy at '{post}'")
        pending.count -= 1
        if pending.count <= 0:
            self.pending_spy_placements.remove(pending)

    def _step_resolve_uplift(self, action: GameAction) -> None:
        pid = action.player_id
        pending = next((p for p in self.pending_uplifts if p.player_id == pid), None)
        if pending is None:
            raise ValueError("No pending uplift for player")
        if not self.uplift_agent(pid, action.uplift_space_name):
            raise ValueError(f"Cannot uplift from '{action.uplift_space_name}'")
        pending.count -= 1
        if pending.count <= 0:
            self.pending_uplifts.remove(pending)

    def _step_resolve_intrigue_trash(self, action: GameAction) -> None:
        pid  = action.player_id
        p    = self.players[pid]
        card = next((c for c in p.intrigue_cards if c.name == action.intrigue_card_name), None)
        if card is None:
            raise ValueError(f"Intrigue card '{action.intrigue_card_name}' not in hand")
        self.resolve_intrigue_trash(pid, card)

    def _step_resolve_deploy(self, action: GameAction) -> None:
        pid = action.player_id
        pending = next((d for d in self.pending_deployments if d.player_id == pid), None)
        if pending is None:
            raise ValueError("No pending deployment for player")
        p = self.players[pid]
        budget_left = max(0, getattr(p, "deploy_budget_this_turn", 0)
                          - getattr(p, "deployed_this_turn", 0))
        n = max(0, min(action.deploy_count, pending.max_deploy,
                       p.troops_garrison, budget_left))
        if n > 0:
            self._do_deploy(pid, n)
        self.pending_deployments.remove(pending)

    def _to_trash(self, player, card) -> None:
        """Trash a card.  The Spice Must Flow / Prepare the Way return to their
        reserve stack instead of the player's permanent trash pile."""
        if getattr(card, "cost", 0) >= 1:
            player.trashed_costly_card_this_round = True   # Tenuous Bond
        if card.name == "The Spice Must Flow":
            self.reserve_spice_must_flow.append(card)
        elif card.name == "Prepare the Way":
            self.reserve_prepare_the_way.append(card)
        else:
            player.trash.append(card)

    def _step_resolve_trash(self, action: GameAction) -> None:
        pid = action.player_id
        pending = next((t for t in self.pending_trashes if t.player_id == pid), None)
        if pending is None:
            raise ValueError("No pending trash for player")
        name = action.trash_card_name
        if name:                                   # None / "" == decline
            p = self.players[pid]
            # You may trash only from HAND or DISCARD (never the deck).
            card = (next((c for c in p.hand    if c.name == name), None) or
                    next((c for c in p.discard if c.name == name), None))
            if card is None:
                raise ValueError(f"Card '{name}' not in hand or discard")
            (p.hand if card in p.hand else p.discard).remove(card)
            self._to_trash(p, card)
            # "When this card is trashed" effects (e.g. Sardaukar Soldier).
            for eff in getattr(card, "trash_effects", []):
                EffectResolver.resolve_single_effect(eff, p, self)
            # "If you trash a card, ..." reward on the pending (e.g. Shishakli).
            if pending.on_trash:
                EffectResolver.resolve_single_effect(pending.on_trash, p, self)
        pending.count -= 1
        if pending.count <= 0 or not name:
            self.pending_trashes.remove(pending)

    def _step_resolve_contract(self, action: GameAction) -> None:
        pid = action.player_id
        pending = next((c for c in self.pending_contract_choices
                        if c.player_id == pid), None)
        if pending is None:
            raise ValueError("No pending contract choice for player")
        if not self.contracts_on_board:
            self.players[pid].gain_solari(2)
        else:
            i = max(0, min(action.contract_index, len(self.contracts_on_board) - 1))
            contract = self.contracts_on_board.pop(i)
            self.players[pid].take_contract(contract)
            self._refill_contracts_on_board()
            if contract.is_immediate() and (
                    not contract.requires_intrigue() or self.players[pid].intrigue_cards):
                self.complete_contract(pid, contract)
        pending.count -= 1
        if pending.count <= 0:
            self.pending_contract_choices.remove(pending)

    def _step_resolve_optional(self, action: GameAction) -> None:
        pid = action.player_id
        pending = next((op for op in self.pending_optional_payments
                        if op.player_id == pid), None)
        if pending is None:
            raise ValueError("No pending optional payment for player")
        self.pending_optional_payments.remove(pending)
        if not action.accept_optional:
            return
        p = self.players[pid]
        if not self._can_pay_optional(pid, pending):
            return                                  # can't afford -> treat as decline
        for k, v in pending.cost.items():
            if k == "recall_spy":
                for post in list(p.spies_on_board)[:v]:
                    p.recall_spy(post)
                p.recalled_spy_this_turn = True
            else:
                setattr(p, k, getattr(p, k) - v)
        discarded_tag = False
        for _ in range(pending.discard):
            if pending.discard_tag:
                from src.game.cards.card import CardTag
                tag = CardTag(pending.discard_tag)
                tagged = [c for c in p.hand if c.has_tag(tag)]
                if tagged:
                    tgt = min(tagged, key=lambda c: c.persuasion + c.swords)
                    p.hand.remove(tgt)
                    p.discard.append(tgt)
                    discarded_tag = True
                    continue
            EffectResolver._discard_worst(p)
        EffectResolver.resolve_single_effect(pending.reward, p, self)
        if discarded_tag and pending.tag_bonus:
            EffectResolver.resolve_single_effect(pending.tag_bonus, p, self)

    def _step_resolve_influence(self, action: GameAction) -> None:
        pid = action.player_id
        pending = next((c for c in self.pending_influence_choices
                        if c.player_id == pid), None)
        if pending is None:
            raise ValueError("No pending influence choice for player")
        faction = action.influence_faction
        if faction not in FACTIONS:
            raise ValueError(f"Invalid faction '{faction}'")
        self.gain_influence_with_check(pid, faction, 1)
        pending.count -= 1
        if pending.count <= 0:
            self.pending_influence_choices.remove(pending)

    def _step_play_intrigue(self, action: GameAction) -> None:
        pid  = action.player_id
        p    = self.players[pid]
        card = next((c for c in p.intrigue_cards
                     if c.name == action.intrigue_card_name), None)
        if card is None:
            raise ValueError(f"Intrigue '{action.intrigue_card_name}' not held by player {pid}")
        in_combat  = self.phase == Phase.COMBAT
        is_reveal  = self.player_in_reveal_buy == pid
        # A Plot Intrigue may be played any time during your own turn window
        # (before OR after you place an agent / reveal).
        is_agent   = (self.phase == Phase.PLAYER_TURNS
                      and not is_reveal
                      and pid not in self.players_revealed
                      and self.get_current_player_id() == pid)
        ok, reason = card.can_play(
            p, self,
            is_agent_turn=is_agent, is_reveal_turn=is_reveal, in_combat=in_combat,
        )
        if not ok:
            raise ValueError(reason)
        p.play_intrigue(card)                        # remove from player's hand
        if in_combat:
            self._combat_intrigue_players.add(pid)
        card.resolve(p, self)                        # resolve + global discard
        if in_combat:
            for part in self.get_combat_participants():
                self.update_combat_strength(part)
            # A non-pass action breaks the consecutive-pass chain; play then
            # continues with the next participant.
            self.combat_intrigue_pass_count = 0
            self.last_combat_actor = pid
            # If the intrigue queued a mandatory choice (spy / contract / ...),
            # let this player resolve it before combat play moves on.
            if self._has_mandatory_pending_for(pid):
                self._combat_advance_after_pending = pid
            else:
                self._advance_combat_turn()

    def _step_combat_pass(self, action: GameAction) -> None:
        pid = action.player_id
        all_passed = self.record_combat_intrigue_pass(pid)
        if all_passed:
            self.resolve_combat()
        else:
            self._advance_combat_turn()

    def _auto_advance(self) -> None:
        """Silently move through phases that need no player decisions."""
        # Makers and Recall auto-resolve; ROUND_START auto-triggers next round.
        safety = 0
        while safety < 20:
            safety += 1
            if self.game_over:
                break
            if (self.phase == Phase.COMBAT
                    and self._combat_advance_after_pending is None
                    and not self.get_combat_participants()
                    and not any(self._has_mandatory_pending_for(q)
                                for q in range(self.num_players))):
                # Everyone retreated / no one deployed -> resolve the empty Conflict.
                self.resolve_combat()
            elif self.phase == Phase.MAKERS:
                self.resolve_makers_phase()          # -> RECALL
            elif self.phase == Phase.RECALL:
                # The round always completes (Makers + Recall) before the
                # end-of-round Victory check.  A player may cross 10 VP during
                # Combat but the game only ends here — and an Alliance stolen in
                # that same Combat can drop them back below 10, forcing another
                # round.
                self.resolve_recall_phase()          # -> ROUND_START
                if self.check_victory_conditions():
                    break
                self.start_new_round()
            elif self.phase == Phase.ROUND_START:
                self.start_new_round()
            else:
                break

    # -----------------------------------------------------------------------
    # Pending-effects helpers
    # -----------------------------------------------------------------------

    def _prune_dead_pendings(self, player_id: int) -> None:
        """Drop pending choices that can no longer be satisfied (they fizzle)."""
        p = self.players[player_id]
        self.pending_spy_placements = [
            psp for psp in self.pending_spy_placements
            if psp.player_id != player_id
            or any(self.can_place_spy(player_id, post, psp.allow_occupied)
                   for post in (psp.allowed_posts or ALL_OBSERVATION_POSTS))
        ]
        self.pending_uplifts = [
            pu for pu in self.pending_uplifts
            if pu.player_id != player_id or self._uplift_targets(player_id)
        ]
        self.pending_trashes = [
            pt for pt in self.pending_trashes
            if pt.player_id != player_id or p.hand or p.discard or p.in_play
        ]
        self.pending_intrigue_trashes = [
            pit for pit in self.pending_intrigue_trashes
            if pit.player_id != player_id or p.intrigue_cards
        ]
        self.pending_optional_payments = [
            op for op in self.pending_optional_payments
            if op.player_id != player_id or self._can_pay_optional(player_id, op)
        ]

    def _has_mandatory_pending_for(self, player_id: int) -> bool:
        return (
            any(p.player_id == player_id for p in self.pending_spy_placements) or
            any(p.player_id == player_id for p in self.pending_uplifts) or
            any(p.player_id == player_id for p in self.pending_intrigue_trashes) or
            any(p.player_id == player_id for p in self.pending_deployments) or
            any(p.player_id == player_id for p in self.pending_trashes) or
            any(p.player_id == player_id for p in self.pending_influence_choices) or
            any(p.player_id == player_id for p in self.pending_contract_choices) or
            any(p.player_id == player_id for p in self.pending_optional_payments)
        )

    def _can_pay_optional(self, player_id: int, op: "PendingOptionalPayment") -> bool:
        p = self.players[player_id]
        if len(p.hand) < op.discard:
            return False
        for k, v in op.cost.items():
            if k == "recall_spy":
                if sum(p.spies_on_board.values()) < v:
                    return False
            elif getattr(p, k, 0) < v:
                return False
        return True

    def _valid_pending_actions(self, player_id: int) -> List[GameAction]:
        actions: List[GameAction] = []
        p = self.players[player_id]

        for psp in self.pending_spy_placements:
            if psp.player_id != player_id:
                continue
            for post in (psp.allowed_posts or ALL_OBSERVATION_POSTS):
                if self.can_place_spy(player_id, post, psp.allow_occupied):
                    actions.append(GameAction(ActionType.RESOLVE_SPY,
                                              player_id, spy_post_name=post))

        for pu in self.pending_uplifts:
            if pu.player_id != player_id:
                continue
            for space in self._uplift_targets(player_id):
                actions.append(GameAction(ActionType.RESOLVE_UPLIFT,
                                          player_id, uplift_space_name=space))

        for pit in self.pending_intrigue_trashes:
            if pit.player_id != player_id:
                continue
            for ic in p.intrigue_cards:
                actions.append(GameAction(ActionType.RESOLVE_INTRIGUE_TRASH,
                                          player_id, intrigue_card_name=ic.name))

        for pd in self.pending_deployments:
            if pd.player_id != player_id:
                continue
            budget_left = max(0, getattr(p, "deploy_budget_this_turn", 0)
                              - getattr(p, "deployed_this_turn", 0))
            hi = min(pd.max_deploy, p.troops_garrison, budget_left)
            for n in range(0, hi + 1):
                actions.append(GameAction(ActionType.RESOLVE_DEPLOY,
                                          player_id, deploy_count=n))

        for pt in self.pending_trashes:
            if pt.player_id != player_id:
                continue
            names = {c.name for c in p.hand} | {c.name for c in p.discard}
            for nm in sorted(names):
                actions.append(GameAction(ActionType.RESOLVE_TRASH,
                                          player_id, trash_card_name=nm))
            actions.append(GameAction(ActionType.RESOLVE_TRASH,
                                      player_id, trash_card_name=None))  # decline

        for op in self.pending_optional_payments:
            if op.player_id != player_id:
                continue
            actions.append(GameAction(ActionType.RESOLVE_OPTIONAL, player_id,
                                      accept_optional=True))
            actions.append(GameAction(ActionType.RESOLVE_OPTIONAL, player_id,
                                      accept_optional=False))   # decline

        for pic in self.pending_influence_choices:
            if pic.player_id != player_id:
                continue
            for f in FACTIONS:
                actions.append(GameAction(ActionType.RESOLVE_INFLUENCE,
                                          player_id, influence_faction=f))

        for pcc in self.pending_contract_choices:
            if pcc.player_id != player_id:
                continue
            n = max(1, len(self.contracts_on_board))
            for i in range(n):
                actions.append(GameAction(ActionType.RESOLVE_CONTRACT,
                                          player_id, contract_index=i))

        return actions or [GameAction(ActionType.NO_OP, player_id)]

    # -----------------------------------------------------------------------
    # Effect queue
    # -----------------------------------------------------------------------

    def begin_atomic(self) -> int:
        """Start an atomic block. Returns the group ID."""
        self._atomic_depth   += 1
        self._atomic_counter += 1
        return self._atomic_counter

    def end_atomic(self) -> None:
        if self._atomic_depth > 0:
            self._atomic_depth -= 1

    def is_in_atomic_block(self) -> bool:
        return self._atomic_depth > 0

    def queue_effect(
        self,
        effect_dict: Dict,
        player_id: int,
        atomic: bool   = False,
        mandatory: bool = True,
        source: str    = "",
    ) -> None:
        group = self._atomic_counter if atomic else 0
        self._effect_queue.append(
            QueuedEffect(effect_dict, player_id, group, mandatory, source)
        )

    def flush_effect_queue(self) -> None:
        """Resolve all queued effects in FIFO order via EffectResolver."""
        while self._effect_queue:
            item = self._effect_queue.pop(0)
            player = self.players[item.player_id]
            EffectResolver.resolve_single_effect(item.effect_dict, player, self)

    # -----------------------------------------------------------------------
    # Persuasion pool
    # -----------------------------------------------------------------------

    def gain_persuasion(self, player_id: int, amount: int) -> None:
        self.persuasion_pool[player_id] = self.persuasion_pool.get(player_id, 0) + amount

    def spend_persuasion(self, player_id: int, amount: int) -> bool:
        if self.persuasion_pool.get(player_id, 0) < amount:
            return False
        self.persuasion_pool[player_id] -= amount
        return True

    def get_persuasion(self, player_id: int) -> int:
        return self.persuasion_pool.get(player_id, 0)

    def reset_persuasion_pool(self, player_id: int) -> None:
        self.persuasion_pool[player_id] = 0

    # -----------------------------------------------------------------------
    # Round management
    # -----------------------------------------------------------------------

    def start_new_round(self) -> None:
        self.round += 1

        idx = self.turn_order.index(self.first_player) if self.first_player in self.turn_order else 0
        self.turn_order = (
            list(range(self.num_players))[idx:] +
            list(range(self.num_players))[:idx]
        )

        if self.conflict_deck:
            self.current_conflict = self.conflict_deck.pop(0)
        else:
            self.current_conflict = None

        # Controlling player may deploy 1 troop when their location's conflict reveals
        if self.current_conflict and self.current_conflict.location:
            loc        = self.current_conflict.location
            controller = self.controlled_by.get(loc)
            if controller is not None:
                p = self.players[controller]
                if p.troops_supply > 0:
                    p.troops_supply -= 1
                    self.troops_in_conflict[controller] = (
                        self.troops_in_conflict.get(controller, 0) + 1
                    )

        for i in range(self.num_players):
            self.draw_cards_for_player(i, 5)

        self.phase                = Phase.PLAYER_TURNS
        self.current_turn_idx     = 0
        self.players_revealed     = set()
        self.player_in_reveal_buy = None
        self.swords_this_reveal   = {i: 0 for i in range(self.num_players)}
        self._prev_vp             = {i: self.players[i].victory_points for i in range(self.num_players)}

    def get_current_player_id(self) -> int:
        if self.phase == Phase.COMBAT:
            # A player with an unresolved mandatory choice (from a Combat
            # intrigue's spy/contract reward) acts before combat play resumes.
            for pid in self.turn_order:
                if self._has_mandatory_pending_for(pid):
                    return pid
            actor = self._current_combat_actor()
            return actor if actor is not None else self.turn_order[0]
        if self.current_turn_idx >= len(self.turn_order):
            return self.turn_order[0]
        return self.turn_order[self.current_turn_idx]

    def get_current_player(self) -> Player:
        return self.players[self.get_current_player_id()]

    def get_player(self, player_id: int) -> Player:
        return self.players[player_id]

    def advance_to_next_player_turn(self) -> None:
        if len(self.players_revealed) == self.num_players:
            self._begin_combat_phase()
            return

        self.current_turn_idx = (self.current_turn_idx + 1) % self.num_players
        checked = 0
        while self.turn_order[self.current_turn_idx] in self.players_revealed:
            self.current_turn_idx = (self.current_turn_idx + 1) % self.num_players
            checked += 1
            if checked > self.num_players:
                self._begin_combat_phase()
                return

    def _begin_combat_phase(self) -> None:
        self.phase                      = Phase.COMBAT
        self.combat_intrigue_pass_count = 0
        self.last_combat_actor          = None
        self.current_turn_idx           = 0
        self.combat_turn_idx            = 0
        self._combat_intrigue_players   = set()
        # If nobody has units, resolve straight through to Makers.
        if not self.get_combat_participants():
            self.resolve_combat()

    # -----------------------------------------------------------------------
    # Card draw (uses seeded RNG)
    # -----------------------------------------------------------------------

    def draw_cards_for_player(self, player_id: int, count: int) -> None:
        player = self.players[player_id]
        for _ in range(count):
            if not player.deck:
                if not player.discard:
                    break
                new_deck = list(player.discard)
                self.rng.shuffle(new_deck)
                player.deck    = new_deck
                player.discard = []
            if player.deck:
                player.hand.append(player.deck.pop())

    # -----------------------------------------------------------------------
    # Agent turn
    # -----------------------------------------------------------------------

    def can_send_agent(
        self,
        player_id: int,
        space_name: str,
        card: Card,
    ) -> Tuple[bool, str]:
        if space_name not in ALL_BOARD_SPACES:
            return False, f"Unknown board space: {space_name}"

        player = self.players[player_id]

        if player.agents_available < 1:
            return False, "No agents available"

        required_icon = BOARD_SPACE_ICONS[space_name]
        card_icons    = {s.value for s in card.access_symbols}

        has_matching_icon = required_icon in card_icons
        if (not has_matching_icon and required_icon == "emperor"
                and card.access_symbols            # must be an agent-playable card
                and getattr(player, "grant_emperor_access_this_turn", False)):
            has_matching_icon = True                 # Emperor's Invitation
        has_spy_icon      = (
            "spy" in card_icons and
            self._player_has_spy_connected_to(player_id, space_name)
        )

        if not has_matching_icon and not has_spy_icon:
            return False, f"Card has no icon matching '{required_icon}' (or no Spy connected)"

        if self.agent_on_space.get(space_name) is not None:
            if not self.can_infiltrate(player_id, space_name):
                return False, f"{space_name} is occupied and player cannot Infiltrate"

        _ignore_gates = (
            getattr(player, "ignore_influence_gates_this_turn", False)
            or any("ignore_influence_gates" in e for e in card.agent_effects))
        if space_name in INFLUENCE_GATED_SPACES and not _ignore_gates:
            faction, required_inf = INFLUENCE_GATED_SPACES[space_name]
            if player.influence[faction] < required_inf:
                return False, (
                    f"{space_name} requires {required_inf} {faction} influence "
                    f"(player has {player.influence[faction]})"
                )

        if space_name in BOARD_SPACE_MANDATORY_COSTS:
            cost = BOARD_SPACE_MANDATORY_COSTS[space_name]
            if not player.can_afford(cost):
                return False, f"Cannot afford {cost} to visit {space_name}"

        if space_name == "Swordmaster":
            if player.has_swordmaster:
                return False, "Player already has a Swordmaster"
            if player.solari < self._swordmaster_cost():
                return False, f"Cannot afford Swordmaster ({self._swordmaster_cost()} solari)"

        return True, ""

    # ------------------------------------------------------------------

    def get_space_options(self, player_id: int, space_name: str) -> List[Optional[str]]:
        """Legal `space_option` values for a special space (['default'] otherwise)."""
        p = self.players[player_id]
        if space_name == "Spice Refinery":
            return ["free"] + (["pay_spice"] if p.spice >= 1 else [])
        if space_name == "Gather Support":
            return ["free"] + (["paid"] if p.solari >= 2 else [])
        if space_name == "Sietch Tabr":
            opts = ["hooks"]
            if self.shield_wall_intact:
                opts.append("shield_wall")
            return opts
        if space_name in MAKER_SPACE_OPTIONS:          # Hagga Basin / Deep Desert
            opts = ["spice"]
            base_spice, worms = MAKER_SPACE_OPTIONS[space_name]
            loc = self.current_conflict.location if self.current_conflict else None
            worm_blocked = self.shield_wall_intact and loc in SHIELD_WALL_PROTECTED
            if p.has_maker_hooks and self.current_conflict and not worm_blocked:
                opts.append("sandworm")
            return opts
        return [None]

    def apply_agent_turn(
        self,
        player_id: int,
        card: Card,
        space_name: str,
        infiltrate: bool            = False,
        gather_intelligence: bool   = False,
        space_option: Optional[str] = None,
    ) -> None:
        """
        Execute an Agent turn in FAQ-mandated order:
          1. Play card (hand → in_play)
          2. Place agent; pay mandatory cost
          3. [Optional] Spy recall (Infiltrate / Gather Intelligence) — BEFORE effects
          4. Control bonus to space owner (if any)
          5. Board space effects (incl. special-space choice)
          6. Card Agent box effects
          7. Faction influence (for faction board spaces)
          8. CHOAM contract checks
          9. Combat deployment (queue PendingDeployment)
        """
        player = self.players[player_id]
        self._agent_turn_active = True
        self._current_agent_space = space_name       # can't uplift THIS agent
        player.troops_recruited_this_turn = 0
        player.recalled_spy_this_turn = False
        player.gained_spice_this_turn = False
        player.deploy_budget_this_turn = 0
        player.deployed_this_turn = 0
        player.maker_bonus_this_turn = 0
        player.agent_to_maker_this_turn = space_name in MAKER_SPACES
        player.agent_to_faction_this_turn = self._faction_for_space(space_name) is not None
        spice_before = player.spice
        if player.leader is not None:
            player.leader.reset_per_turn()
            player.leader.trigger("on_agent_turn_start", player, self)
        try:
            # 1. Play card
            player.play_card_as_agent(card)
            if player.leader is not None:
                player.leader.trigger("on_card_played", player, self, card)
                if card.name == "Signet Ring" and player.leader.signet is not None:
                    player.leader.signet(player, self)

            # 2. Pay mandatory cost + place agent
            if space_name in BOARD_SPACE_MANDATORY_COSTS:
                player.pay_cost(BOARD_SPACE_MANDATORY_COSTS[space_name])
            self.agent_on_space[space_name] = player_id
            player.agents_available -= 1

            # 3. Spy recall — MUST happen before effects (FAQ)
            if infiltrate:
                self._execute_infiltrate(player_id, space_name)
            if gather_intelligence:
                self._execute_gather_intelligence(player_id, space_name)

            # 4. Control bonus (to the controller, not the visitor)
            if space_name in CONTROLLED_LOCATIONS:
                controller = self.controlled_by.get(space_name)
                if controller is not None:
                    EffectResolver.resolve_single_effect(
                        CONTROL_BONUS[space_name], self.players[controller], self)

            # 5. Board space effects
            self._apply_board_space_effects(player_id, space_name, space_option)

            # 6. Card Agent box effects
            EffectResolver.resolve_agent_effects(card, player, self)

            # 7. Faction influence for faction board spaces
            faction = self._faction_for_space(space_name)
            if faction:
                self.gain_influence_with_check(player_id, faction, 1)

            # 8. CHOAM contract checks
            if self.use_choam:
                self.check_board_space_contracts(player_id, space_name)
                if space_name in MAKER_SPACES:
                    self.check_harvest_contracts(
                        player_id, space_name, player.spice - spice_before)

            # 8b. Leader: on_agent_placed
            if player.leader is not None:
                player.leader.trigger("on_agent_placed", player, self, space_name)

            # 9. Combat deployment — a Combat space is a deploy icon: it opens a
            #    per-turn budget of 2 (once) + every troop recruited this turn.
            space = UPRISING_BOARD[space_name]
            if space.is_combat_space and self.current_conflict is not None:
                player.deploy_budget_this_turn = max(
                    player.deploy_budget_this_turn,
                    2 + player.troops_recruited_this_turn)
                if (player.deploy_budget_this_turn - player.deployed_this_turn > 0
                        and not any(d.player_id == player_id
                                    for d in self.pending_deployments)):
                    self.pending_deployments.append(
                        PendingDeployment(player_id, player.troops_garrison))
        finally:
            self._agent_turn_active = False
            # One-shot access relaxers are consumed by this Agent turn.
            player.ignore_influence_gates_this_turn = False
            player.grant_emperor_access_this_turn = False

    def _apply_board_space_effects(
        self, player_id: int, space_name: str, space_option: Optional[str] = None
    ) -> None:
        player = self.players[player_id]

        # Maker bonus spice is always collected (for every Maker space).
        if space_name in MAKER_SPACES:
            bonus = self.maker_bonus_spice.get(space_name, 0)
            self.maker_bonus_spice[space_name] = 0
            if bonus > 0:
                player.gain_spice(bonus)
                player.maker_bonus_this_turn = bonus     # Thumper: double it

        if UPRISING_BOARD[space_name].special:
            self._apply_special_space(player_id, space_name, space_option)
            return

        for effect in BOARD_SPACE_EFFECTS.get(space_name, []):
            if "contract" in effect:
                if self.use_choam:
                    self.take_contract_from_board(player_id)
                else:
                    player.gain_solari(2)
                remainder = {k: v for k, v in effect.items() if k != "contract"}
                if remainder:
                    EffectResolver.resolve_single_effect(remainder, player, self)
            else:
                EffectResolver.resolve_single_effect(effect, player, self)

    def _apply_special_space(
        self, player_id: int, space_name: str, option: Optional[str]
    ) -> None:
        p = self.players[player_id]
        legal = self.get_space_options(player_id, space_name)
        if option not in legal:
            option = legal[0]

        if space_name == "Swordmaster":
            p.solari -= self._swordmaster_cost()
            p.unlock_third_agent()

        elif space_name == "High Council":
            # 5 solari mandatory cost already paid in apply_agent_turn step 2.
            if not p.has_councilor:
                p.has_councilor = True
            else:
                EffectResolver.resolve_single_effect(
                    {"spice": 2, "intrigue": 1, "troops": 3}, p, self)

        elif space_name == "Gather Support":
            if option == "paid":
                p.solari -= 2
                EffectResolver.resolve_single_effect({"troops": 2, "water": 1}, p, self)
            else:
                EffectResolver.resolve_single_effect({"troops": 2}, p, self)

        elif space_name == "Spice Refinery":
            if option == "pay_spice":
                p.spice -= 1
                p.gain_solari(4)
            else:
                p.gain_solari(2)

        elif space_name == "Sietch Tabr":
            if option == "shield_wall":
                p.gain_water(1)
                self.shield_wall_intact = False
            else:
                p.has_maker_hooks = True
                EffectResolver.resolve_single_effect({"troops": 1, "water": 1}, p, self)

        elif space_name in MAKER_SPACE_OPTIONS:        # Hagga Basin / Deep Desert
            base_spice, worms = MAKER_SPACE_OPTIONS[space_name]
            if option == "sandworm":
                self.spawn_sandworms(worms, source="maker_space")
            else:
                p.gain_spice(base_spice)

    # -----------------------------------------------------------------------
    # Special board space resolutions
    # -----------------------------------------------------------------------

    def _swordmaster_cost(self) -> int:
        any_has_sm = any(pl.has_swordmaster for pl in self.players)
        return SWORDMASTER_COST_AFTER if any_has_sm else SWORDMASTER_COST_FIRST

    def pay_for_swordmaster(self, player_id: int) -> bool:
        player = self.players[player_id]
        cost = self._swordmaster_cost()
        if player.has_swordmaster or player.solari < cost:
            return False
        player.solari -= cost
        player.unlock_third_agent()
        return True

    def pay_for_high_council(self, player_id: int) -> bool:
        player = self.players[player_id]
        if player.solari < HIGH_COUNCIL_COST:
            return False
        player.solari -= HIGH_COUNCIL_COST
        if not player.has_councilor:
            player.has_councilor = True
        else:
            EffectResolver.resolve_single_effect(
                {"spice": 2, "intrigue": 1, "troops": 3}, player, self)
        return True

    def apply_gather_support(self, player_id: int, paid: bool = False) -> bool:
        player = self.players[player_id]
        if paid and player.solari >= 2:
            player.solari -= 2
            EffectResolver.resolve_single_effect({"troops": 2, "water": 1}, player, self)
        else:
            EffectResolver.resolve_single_effect({"troops": 2}, player, self)
        return True

    def add_pending_deployment(self, player_id: int, max_deploy: int) -> None:
        """Queue an optional 'deploy up to N troops to the Conflict' choice
        (Sardaukar Coordination: recruited troops only, no +2 icon bonus)."""
        if max_deploy > 0 and self.current_conflict is not None:
            p = self.players[player_id]
            p.deploy_budget_this_turn = max(
                p.deploy_budget_this_turn,
                getattr(p, "deployed_this_turn", 0) + max_deploy)
            self.pending_deployments.append(PendingDeployment(player_id, max_deploy))

    def _do_deploy(self, player_id: int, n: int) -> None:
        """Move `n` troops from garrison straight into the current Conflict."""
        p = self.players[player_id]
        n = max(0, min(n, p.troops_garrison))
        p.troops_garrison -= n
        p.deployed_this_turn = getattr(p, "deployed_this_turn", 0) + n
        self.troops_in_conflict[player_id] = self.troops_in_conflict.get(player_id, 0) + n

    # -----------------------------------------------------------------------
    # Troops and sandworms
    # -----------------------------------------------------------------------

    def deploy_to_conflict(
        self,
        player_id: int,
        troops_from_garrison: int = 0,
        troops_from_this_turn: int = 0,
        sandworms: int = 0,
    ) -> bool:
        player = self.players[player_id]

        if troops_from_garrison > 2:
            return False
        if player.troops_garrison < troops_from_garrison:
            return False

        total_troops      = troops_from_garrison + troops_from_this_turn
        conflict_location = (self.current_conflict.location if self.current_conflict else None)

        if sandworms > 0:
            if not player.has_maker_hooks:
                return False
            if self.shield_wall_intact and conflict_location in SHIELD_WALL_PROTECTED:
                return False

        player.troops_garrison -= troops_from_garrison
        self.troops_in_conflict[player_id] = (
            self.troops_in_conflict.get(player_id, 0) + total_troops
        )
        self.sandworms_in_conflict[player_id] = (
            self.sandworms_in_conflict.get(player_id, 0) + sandworms
        )
        return True

    def spawn_sandworms(self, count: int, source: str = "card_effect") -> None:
        """
        Summon sandworms for the active player.
        Unlike board-space deployment, card effects (e.g. Unexpected Allies)
        can bypass the Maker Hooks requirement — the caller is responsible for
        checking that precondition if applicable.
        Shield Wall still blocks summons to protected locations.
        """
        if not self.current_conflict:
            return
        if (
            self.shield_wall_intact
            and self.current_conflict.location in SHIELD_WALL_PROTECTED
        ):
            return
        active = self.get_current_player_id()
        self.sandworms_in_conflict[active] = (
            self.sandworms_in_conflict.get(active, 0) + count
        )

    def destroy_shield_wall(self) -> None:
        self.shield_wall_intact = False

    # -----------------------------------------------------------------------
    # Reveal turn
    # -----------------------------------------------------------------------

    def apply_reveal_turn(self, player_id: int) -> None:
        """
        Reveal remaining hand, accumulate swords, resolve reveal effects,
        compute combat strength.
        Note: persuasion accumulation is handled in _step_reveal_turn so that
        the persuasion pool is ready for the buy sub-phase.
        """
        player         = self.players[player_id]
        if player.leader is not None:
            player.leader.reset_per_turn()
            player.leader.trigger("on_reveal", player, self)   # sees pre-reveal hand
        cards_to_reveal = list(player.hand)
        player.reveal_cards(cards_to_reveal)
        player._revealed_this_turn = len(cards_to_reveal)
        player.cards_acquired_this_turn = 0
        player.gained_spice_this_turn = False
        player.deploy_budget_this_turn = 0
        player.deployed_this_turn = 0

        swords = sum(c.swords for c in cards_to_reveal)
        self.swords_this_reveal[player_id] = (
            self.swords_this_reveal.get(player_id, 0) + swords
        )

        for card in cards_to_reveal:
            EffectResolver.resolve_reveal_effects(card, player, self)

        troops    = self.troops_in_conflict.get(player_id, 0)
        sandworms = self.sandworms_in_conflict.get(player_id, 0)
        strength  = Combat.calculate_player_strength(
            player, troops, sandworms,
            self.swords_this_reveal.get(player_id, 0), self
        )
        self.combat_strength[player_id] = strength

    def update_combat_strength(self, player_id: int) -> None:
        player    = self.players[player_id]
        troops    = self.troops_in_conflict.get(player_id, 0)
        sandworms = self.sandworms_in_conflict.get(player_id, 0)
        swords    = self.swords_this_reveal.get(player_id, 0)
        self.combat_strength[player_id] = Combat.calculate_player_strength(
            player, troops, sandworms, swords, self
        )

    def acquire_card(self, player_id: int, card: Card) -> bool:
        """
        Spend persuasion to acquire a card from the Imperium Row.
        FIX (Bug 1): Row refills BEFORE acquire effects resolve (FAQ).
        """
        if card not in self.imperium_row:
            return False
        self.imperium_row.remove(card)
        self.players[player_id].discard.append(card)
        self.players[player_id].cards_acquired_this_turn += 1
        # Refill row FIRST, then trigger acquire effects (FAQ requirement)
        self.refill_imperium_row()
        self._trigger_acquire_effects(player_id, card)
        if self.use_choam:
            self.check_acquire_contracts(player_id, card.name)
        return True

    def acquire_reserve_card(self, player_id: int, card_type: str) -> bool:
        if card_type == "prepare_the_way":
            stack = self.reserve_prepare_the_way
        elif card_type == "spice_must_flow":
            stack = self.reserve_spice_must_flow
        else:
            return False
        if not stack:
            return False
        card = stack.pop()
        self.players[player_id].discard.append(card)
        self.players[player_id].cards_acquired_this_turn += 1
        self._trigger_acquire_effects(player_id, card)
        if self.use_choam:
            self.check_acquire_contracts(player_id, card.name)
        return True

    def refill_imperium_row(self) -> None:
        while len(self.imperium_row) < 5 and self.imperium_deck:
            self.imperium_row.append(self.imperium_deck.pop(0))

    def _trigger_acquire_effects(self, player_id: int, card: Card) -> None:
        if hasattr(card, "acquire_effects") and card.acquire_effects:
            player = self.players[player_id]
            for effect in card.acquire_effects:
                EffectResolver.resolve_single_effect(effect, player, self)

    def mark_reveal_complete(self, player_id: int) -> None:
        """Move in-play cards to discard; reset swords. Called internally."""
        self.players[player_id].recall_cards()
        self.swords_this_reveal[player_id] = 0

    # -----------------------------------------------------------------------
    # Phase 3: Combat
    # -----------------------------------------------------------------------

    def get_combat_participants(self) -> List[int]:
        return [
            pid for pid in range(self.num_players)
            if (self.troops_in_conflict.get(pid, 0) +
                self.sandworms_in_conflict.get(pid, 0)) > 0
        ]

    def _current_combat_actor(self) -> Optional[int]:
        parts = self.get_combat_participants()
        if not parts:
            return None
        order = [pid for pid in self.turn_order if pid in parts]
        if not order:
            return None
        return order[self.combat_turn_idx % len(order)]

    def _advance_combat_turn(self) -> None:
        self.combat_turn_idx += 1

    def record_combat_intrigue_pass(self, player_id: int) -> bool:
        """Return True once every combat participant has passed consecutively."""
        participants = self.get_combat_participants()
        if not participants:
            return True
        self.combat_intrigue_pass_count += 1
        self.last_combat_actor = player_id
        return self.combat_intrigue_pass_count >= len(participants)

    def resolve_combat(self) -> Optional[CombatResult]:
        """
        Resolve the current Conflict.
        FIX (Bug 3): Ties below first place are handled — all tied players
          get the same reward tier rather than being split by turn order.
        FIX (Bug 6): Rewards are applied in player turn order when order matters.
        """
        if not self.current_conflict:
            return None

        conflict     = self.current_conflict
        participants = self.get_combat_participants()

        if not participants:
            result = CombatResult(
                conflict_name=conflict.name, rankings=[], rewards_awarded={},
                has_sandworm={}, winner=None, conflict_card_goes_to=None,
            )
            self._last_combat_result = result
            self._cleanup_after_combat()
            return result

        # Sort descending by strength; turn-order tiebreak only for first place
        ranked = sorted(
            [(pid, self.combat_strength.get(pid, 0)) for pid in participants],
            key=lambda x: (x[1], -self.turn_order.index(x[0])),
            reverse=True,
        )

        has_sandworm = {
            pid: self.sandworms_in_conflict.get(pid, 0) > 0
            for pid in participants
        }

        rewards_awarded: Dict[int, Dict] = {}
        winner:                Optional[int] = None
        conflict_card_goes_to: Optional[int] = None

        R1 = dict(conflict.first_place_reward)
        R2 = dict(conflict.second_place_reward)
        R3 = dict(conflict.third_place_reward) if conflict.third_place_reward else None
        has_third = R3 is not None and self.num_players >= 4

        def award(pid, raw):
            if raw:
                rewards_awarded[pid] = self._scale_reward(
                    dict(raw), has_sandworm.get(pid, False))

        # Group participants into strength tiers, highest first (strength > 0).
        tiers: List[List[int]] = []
        for st in sorted({s for _, s in ranked if s > 0}, reverse=True):
            tiers.append([pid for pid, s in ranked if s == st])

        if not tiers:
            pass                                    # everyone at 0 strength
        elif len(tiers[0]) >= 2:
            # ── Tie for 1st ──────────────────────────────────────────────
            # Each tied player gets the 2nd-place reward; no one takes the card.
            for pid in tiers[0]:
                award(pid, R2)
            # 4p, exactly 2 tied for 1st: the next tier competes for 3rd.
            if has_third and len(tiers[0]) == 2 and len(tiers) > 1:
                if len(tiers[1]) == 1:
                    award(tiers[1][0], R3)
                # tied for 3rd -> no reward
        else:
            # ── Clear winner ────────────────────────────────────────────
            winner = tiers[0][0]
            conflict_card_goes_to = winner
            award(winner, R1)
            if len(tiers) > 1:
                second = tiers[1]
                if len(second) == 1:
                    award(second[0], R2)
                    # 3rd place: only if a single distinct tier below exists
                    if has_third and len(tiers) > 2 and len(tiers[2]) == 1:
                        award(tiers[2][0], R3)
                    # tied for 3rd -> no reward
                else:
                    # Tie for 2nd: each tied player gets the 3rd-place reward;
                    # no separate 3rd place is awarded.
                    for pid in second:
                        award(pid, R3)

        # Apply rewards in player turn order (FAQ: resolve in turn order when order matters)
        for pid in self.turn_order:
            if pid in rewards_awarded:
                self._apply_combat_reward(pid, rewards_awarded[pid], conflict)

        if conflict_card_goes_to is not None:
            self.won_conflicts[conflict_card_goes_to].append(conflict)
            self._award_battle_icon(conflict_card_goes_to, conflict.battle_icon)
            winner_p = self.players[conflict_card_goes_to]
            if winner_p.leader is not None:
                winner_p.leader.trigger("on_combat_win", winner_p, self, conflict)

        result = CombatResult(
            conflict_name=conflict.name,
            rankings=ranked,
            rewards_awarded=rewards_awarded,
            has_sandworm=has_sandworm,
            winner=winner,
            conflict_card_goes_to=conflict_card_goes_to,
        )

        self._last_combat_result = result
        self._cleanup_after_combat()
        return result

    def _scale_reward(self, reward: Dict, with_sandworm: bool) -> Dict:
        """A sandworm in the Conflict doubles EVERY reward — flat resources and
        VP, and also the 'spend X -> gain a VP' conversion (both cost and VP
        double, e.g. Spice Freighters: pay 3 spice for 1 VP -> pay 6 for 2).
        Only `control` (and the internal `battle_icon` marker) can't scale."""
        if not with_sandworm:
            return reward
        _nodouble = ("control", "battle_icon")
        _vp_convert = ("may_pay_spice_for_vp", "may_pay_solari_for_vp",
                       "may_pay_troops_for_vp")
        scaled = {}
        for k, v in reward.items():
            if k in _nodouble:
                scaled[k] = v
            elif k in _vp_convert and isinstance(v, dict):
                scaled[k] = {"cost": v.get("cost", 0) * 2, "vp": v.get("vp", 1) * 2}
            elif k == "may_recall_spies_for_vp" and isinstance(v, dict):
                # Spy count can't scale past your 3 Spies — only the VP doubles.
                scaled[k] = {"count": v.get("count", 2), "vp": v.get("vp", 1) * 2}
            elif isinstance(v, (int, float)):
                scaled[k] = v * 2
            else:
                scaled[k] = v
        return scaled

    def _apply_combat_reward(self, player_id: int, reward: Dict, conflict: Conflict) -> None:
        if not reward:
            return
        player = self.players[player_id]
        for key, value in reward.items():
            if key == "control":
                if conflict.location:
                    self.gain_control_of_location(player_id, conflict.location)
            elif key == "influence_any":
                # auto-choose: the faction closest to (but below) a threshold,
                # else the one where the player has the least influence
                for _ in range(int(value)):
                    best = min(FACTIONS, key=lambda f: (
                        0 if player.influence[f] in (1, 3) else 1,
                        player.influence[f]))
                    self.gain_influence_with_check(player_id, best, 1)
            elif key == "trash":
                weak = ("Reconnaissance", "Diplomacy", "Dune, the Desert Planet",
                        "Dagger")
                pool = player.hand + player.discard
                tgt = next((c for c in pool if c.name in weak), None)
                if tgt is not None:
                    player.trash_card(tgt, from_location="any")
            elif key in ("may_pay_spice_for_vp", "may_pay_solari_for_vp",
                         "may_pay_troops_for_vp"):
                # Optional "spend X to gain a VP" — auto-take when affordable
                # (a VP is always worth these amounts by the time it appears).
                cost, vp = value.get("cost", 0), value.get("vp", 1)
                res = {"may_pay_spice_for_vp": ("spice", player.spice),
                       "may_pay_solari_for_vp": ("solari", player.solari),
                       "may_pay_troops_for_vp": ("troops_in_conflict",
                                                 self.troops_in_conflict.get(player_id, 0))
                       }[key]
                have = res[1]
                if have >= cost:
                    if key == "may_pay_troops_for_vp":
                        self.troops_in_conflict[player_id] -= cost
                        player.troops_supply += cost
                    else:
                        setattr(player, res[0], have - cost)
                    player.gain_vp(vp)
            elif key == "may_recall_spies_for_vp":
                # Battle for Arrakeen: recall N of your Spies from the board to
                # gain a VP.  Auto-taken when you have enough Spies out.
                count, vp = int(value.get("count", 2)), int(value.get("vp", 1))
                if sum(player.spies_on_board.values()) >= count:
                    for post in list(player.spies_on_board)[:count]:
                        player.recall_spy(post)
                    player.gain_vp(vp)
            else:
                EffectResolver.resolve_single_effect({key: value}, player, self)

    def _award_battle_icon(self, player_id: int, icon) -> bool:
        """
        Give the winner this Conflict's battle icon.  Two matching NON-wild
        icons immediately convert to 1 VP (the pair is consumed).  A "wild"
        icon never matches immediately — it waits for Endgame.
        """
        if icon is None:
            return False
        name = icon.value if hasattr(icon, "value") else str(icon)
        p = self.players[player_id]
        if name != "wild" and name in p.battle_icons:
            p.battle_icons.remove(name)          # consume the matched pair
            p.gain_vp(1)
            return True
        p.battle_icons.append(name)
        return False

        return False

    def _cleanup_after_combat(self) -> None:
        for pid in range(self.num_players):
            troop_count = self.troops_in_conflict.get(pid, 0)
            if troop_count > 0:
                self.players[pid].troops_supply += troop_count
            self.troops_in_conflict[pid]    = 0
            self.sandworms_in_conflict[pid] = 0
            self.combat_strength[pid]       = 0

        self.current_conflict = None
        self.phase            = Phase.MAKERS

    # -----------------------------------------------------------------------
    # Phase 4: Makers
    # -----------------------------------------------------------------------

    def resolve_makers_phase(self) -> None:
        for space in MAKER_SPACES:
            if self.agent_on_space.get(space) is None:
                self.maker_bonus_spice[space] = self.maker_bonus_spice.get(space, 0) + 1
        self.phase = Phase.RECALL

    # -----------------------------------------------------------------------
    # Phase 5: Recall
    # -----------------------------------------------------------------------

    def resolve_recall_phase(self) -> None:
        for player in self.players:
            player.reset_agents()
            player.ignore_influence_gates_this_turn = False
            player.grant_emperor_access_this_turn = False
            player.trashed_costly_card_this_round = False
            # An un-bought Manipulate card returns to the Imperium deck.
            if player.reserved_card is not None:
                self.imperium_deck.append(player.reserved_card)
                player.reserved_card = None
                player.reserved_discount = 0

        for space in ALL_BOARD_SPACES:
            self.agent_on_space[space] = None

        idx = self.turn_order.index(self.first_player)
        self.first_player = self.turn_order[(idx + 1) % self.num_players]

        self.phase = Phase.ROUND_START

    # -----------------------------------------------------------------------
    # Influence and alliances
    # -----------------------------------------------------------------------

    def gain_influence_with_check(
        self, player_id: int, faction: str, amount: int
    ) -> None:
        player  = self.players[player_id]
        old_inf = player.influence[faction]
        player.gain_influence(faction, amount, self)
        new_inf = player.influence[faction]

        # Re-check the Alliance whenever the player is at 4+ afterwards — gaining
        # influence at or above 4 can also OVERTAKE the current holder (e.g. 4->5
        # past a holder sitting on 4), not only cross the threshold from below.
        if new_inf >= 4:
            self._check_and_update_alliance(faction)

    def lose_influence_with_check(
        self, player_id: int, faction: str, amount: int
    ) -> None:
        player = self.players[player_id]
        player.lose_influence(faction, amount, self)
        self._check_and_update_alliance(faction)

    def _check_and_update_alliance(self, faction: str) -> None:
        """
        Determine who (if anyone) holds the alliance.
        Tie rule (FAQ): if the current holder is now tied, they KEEP the token.
        The holder only loses it when: (a) they drop to ≤3, or (b) another
        player overtakes them (strictly higher influence).
        FIX (Bug 4): alliance contract is triggered only once, inside here.
        """
        best_player: Optional[int] = None
        best_inf = 0

        for pid, player in enumerate(self.players):
            inf = player.influence[faction]
            if inf > best_inf:
                best_inf    = inf
                best_player = pid
            elif inf == best_inf and best_player is not None:
                # Tie: current holder keeps the token
                if self.alliance_holder.get(faction) == pid:
                    best_player = pid

        current_holder = self.alliance_holder.get(faction)

        if best_inf < 4:
            if current_holder is not None:
                self.players[current_holder].lose_alliance(faction)
                self.alliance_holder[faction] = None
            return

        if best_player != current_holder:
            if current_holder is not None:
                self.players[current_holder].lose_alliance(faction)
            self.alliance_holder[faction] = best_player
            self.players[best_player].grant_alliance(faction)
            # Trigger alliance contract once, here (not again in gain_influence_with_check)
            if self.use_choam:
                self.check_alliance_contracts(best_player, faction)

    # -----------------------------------------------------------------------
    # Spies
    # -----------------------------------------------------------------------

    def get_spies_at_post(self, post_name: str) -> Dict[int, int]:
        result = {}
        for pid, player in enumerate(self.players):
            count = player.spies_on_board.get(post_name, 0)
            if count > 0:
                result[pid] = count
        return result

    def post_occupied_by_other(self, player_id: int, post_name: str) -> bool:
        for pid, player in enumerate(self.players):
            if pid != player_id and player.spies_on_board.get(post_name, 0) > 0:
                return True
        return False

    def can_place_spy(
        self, player_id: int, post_name: str, allow_occupied: bool
    ) -> bool:
        if post_name not in ALL_OBSERVATION_POSTS:
            return False
        player = self.players[player_id]
        if player.has_spy_at(post_name):
            return False
        if not allow_occupied and self.post_occupied_by_other(player_id, post_name):
            return False
        return True

    def place_spy(self, player_id: int, post_name: str, allow_occupied: bool) -> bool:
        """
        Place a spy. If no spy in supply, auto-recall is required first
        (the RL env must request a recall before calling this).
        Spy placement is MANDATORY when the icon appears (Uprising errata).
        """
        if not self.can_place_spy(player_id, post_name, allow_occupied):
            return False
        player = self.players[player_id]
        if player.spies_available < 1:
            return False  # Caller must recall a spy first
        player.place_spy(post_name, allow_occupied)
        if player.leader is not None:
            player.leader.trigger("on_spy_placed", player, self)
        return True

    def add_pending_spy_placement(
        self, player_id: int, count: int, allow_occupied: bool,
        allowed_posts=None,
    ) -> None:
        """
        Queue a mandatory spy placement.
        Per Uprising errata, spy placement is NOT optional.
        The RL env must resolve this before other actions.
        `allowed_posts` optionally restricts which observation posts are legal.
        """
        self.pending_spy_placements.append(
            PendingSpyPlacement(player_id, count, allow_occupied, allowed_posts)
        )

    def can_infiltrate(self, player_id: int, space_name: str) -> bool:
        if self.agent_on_space.get(space_name) is None:
            return False
        if self.agent_on_space.get(space_name) == player_id:
            return False
        for post in SPACE_TO_OBSERVATION_POSTS.get(space_name, set()):
            if self.players[player_id].has_spy_at(post):
                return True
        return False

    def can_gather_intelligence(self, player_id: int, space_name: str) -> bool:
        for post in SPACE_TO_OBSERVATION_POSTS.get(space_name, set()):
            if self.players[player_id].has_spy_at(post):
                return True
        return False

    def _player_has_spy_connected_to(self, player_id: int, space_name: str) -> bool:
        return self.can_gather_intelligence(player_id, space_name)

    def _connected_spy_count(self, player_id: int, space_name: str) -> int:
        """How many distinct posts bordering `space_name` hold this player's Spy."""
        player = self.players[player_id]
        return sum(
            1 for post in SPACE_TO_OBSERVATION_POSTS.get(space_name, set())
            if player.has_spy_at(post)
        )

    def _execute_infiltrate(self, player_id: int, space_name: str) -> None:
        player = self.players[player_id]
        for post in SPACE_TO_OBSERVATION_POSTS.get(space_name, set()):
            if player.has_spy_at(post):
                player.recall_spy(post)
                player.recalled_spy_this_turn = True
                break

    def _execute_gather_intelligence(self, player_id: int, space_name: str) -> None:
        """Recall the spy and draw 1 card — MUST happen before any board effects."""
        player = self.players[player_id]
        for post in SPACE_TO_OBSERVATION_POSTS.get(space_name, set()):
            if player.has_spy_at(post):
                player.recall_spy(post)
                player.recalled_spy_this_turn = True
                self.draw_cards_for_player(player_id, 1)
                break

    # -----------------------------------------------------------------------
    # Intrigue
    # -----------------------------------------------------------------------

    def draw_intrigue_for_player(self, player_id: int, count: int) -> None:
        player = self.players[player_id]
        for _ in range(count):
            if not self.intrigue_deck:
                if not self.intrigue_discard:
                    break
                shuffled = list(self.intrigue_discard)
                self.rng.shuffle(shuffled)
                self.intrigue_deck    = shuffled
                self.intrigue_discard = []
            if self.intrigue_deck:
                card = self.intrigue_deck.pop(0)
                player.add_intrigue(card)
                if player.leader is not None:
                    player.leader.trigger("on_intrigue_gained", player, self)

    def discard_intrigue_card(self, card) -> None:
        self.intrigue_discard.append(card)

    # -----------------------------------------------------------------------
    # CHOAM Contracts
    # -----------------------------------------------------------------------

    def _refill_contracts_on_board(self) -> None:
        if len(self.contracts_on_board) < 2 and self.contract_bank:
            self.contracts_on_board.append(self.contract_bank.pop(0))

    def take_contract_from_board(self, player_id: int) -> Optional[Contract]:
        # If no contracts remain (or CHOAM is off) -> 2 Solari instead.
        if not self.use_choam or not self.contracts_on_board:
            self.players[player_id].gain_solari(2)
            return None
        if len(self.contracts_on_board) == 1:
            contract = self.contracts_on_board.pop(0)
            self.players[player_id].take_contract(contract)
            self._refill_contracts_on_board()
            if contract.is_immediate() and (
                    not contract.requires_intrigue() or self.players[player_id].intrigue_cards):
                self.complete_contract(player_id, contract)
            return contract
        # 2 face-up contracts -> the player chooses which one.
        self.pending_contract_choices.append(PendingContractChoice(player_id))
        return None

    def complete_contract(self, player_id: int, contract: Contract) -> None:
        player = self.players[player_id]
        player.complete_contract(contract)
        for key, value in contract.rewards.items():
            EffectResolver.resolve_single_effect({key: value}, player, self)

    def check_board_space_contracts(self, player_id: int, space_name: str) -> None:
        player = self.players[player_id]
        for contract in list(player.contracts_active):
            if contract.check_completion_on_board_visit(space_name, player, self):
                self.complete_contract(player_id, contract)

    def check_harvest_contracts(
        self, player_id: int, space_name: str, spice_gained: int
    ) -> None:
        player = self.players[player_id]
        for contract in list(player.contracts_active):
            if contract.check_completion_on_harvest(space_name, spice_gained, player, self):
                self.complete_contract(player_id, contract)

    def check_acquire_contracts(self, player_id: int, card_name: str) -> None:
        player = self.players[player_id]
        for contract in list(player.contracts_active):
            if contract.check_completion_on_acquire(card_name, player, self):
                self.complete_contract(player_id, contract)

    def check_alliance_contracts(self, player_id: int, faction: str) -> None:
        player = self.players[player_id]
        for contract in list(player.contracts_active):
            if contract.check_completion_on_alliance(faction, player, self):
                self.complete_contract(player_id, contract)

    # -----------------------------------------------------------------------
    # Control markers
    # -----------------------------------------------------------------------

    def gain_control_of_location(self, player_id: int, location: str) -> None:
        if location not in CONTROLLED_LOCATIONS:
            return
        if self.controlled_by.get(location) == player_id:
            return
        self.controlled_by[location] = player_id

    # -----------------------------------------------------------------------
    # Pending effects
    # -----------------------------------------------------------------------

    def add_pending_uplift(self, player_id: int, count: int) -> None:
        self.pending_uplifts.append(PendingUplift(player_id, count))

    def add_pending_trash(self, player_id: int, count: int = 1, on_trash=None) -> None:
        # Only queue if the player actually has something to trash.
        p = self.players[player_id]
        if p.hand or p.discard or p.in_play:
            self.pending_trashes.append(PendingTrash(player_id, count, on_trash))

    def add_pending_influence_choice(self, player_id: int, count: int = 1) -> None:
        self.pending_influence_choices.append(PendingInfluenceChoice(player_id, count))

    def add_pending_optional_payment(self, player_id: int, cost: Dict, reward: Dict,
                                     discard: int = 0, label: str = "",
                                     discard_tag: str = None,
                                     tag_bonus: Dict = None) -> None:
        """Queue a cost->reward arrow the player MAY take. Skipped silently if
        the cost can't be met (indistinguishable from declining)."""
        op = PendingOptionalPayment(player_id, cost, reward, discard, label,
                                    discard_tag, tag_bonus)
        if self._can_pay_optional(player_id, op):
            self.pending_optional_payments.append(op)

    def _uplift_targets(self, player_id: int) -> List[str]:
        """Spaces holding one of this player's OTHER agents (not the one just
        placed — e.g. not Imperial Privilege / not where Steersman went)."""
        return [sp for sp, occ in self.agent_on_space.items()
                if occ == player_id and sp != getattr(self, "_current_agent_space", None)]

    def uplift_agent(self, player_id: int, space_name: str) -> bool:
        if self.agent_on_space.get(space_name) != player_id:
            return False
        if space_name == getattr(self, "_current_agent_space", None):
            return False                       # cannot recall the agent you just placed
        self.agent_on_space[space_name] = None
        self.players[player_id].uplift_agent()
        return True

    def add_pending_intrigue_trash(self, player_id: int, benefit: Dict) -> None:
        self.pending_intrigue_trashes.append(
            PendingIntrigueTrash(player_id, benefit)
        )

    def resolve_intrigue_trash(self, player_id: int, intrigue_card) -> bool:
        player = self.players[player_id]
        if intrigue_card not in player.intrigue_cards:
            return False
        player.play_intrigue(intrigue_card)
        self.intrigue_discard.append(intrigue_card)
        for pending in list(self.pending_intrigue_trashes):
            if pending.player_id == player_id:
                EffectResolver.resolve_single_effect(pending.benefit, player, self)
                self.pending_intrigue_trashes.remove(pending)
                return True
        return False

    # -----------------------------------------------------------------------
    # Faction helpers
    # -----------------------------------------------------------------------

    def _faction_for_space(self, space_name: str) -> Optional[str]:
        # Which Influence track advances +1 when an Agent is placed here.
        # (Sietch Tabr has a City icon but grants no influence; Shipping grants
        #  a player-chosen influence handled via the "influence_any" effect.)
        return SPACE_FACTION_INFLUENCE.get(space_name)

    # -----------------------------------------------------------------------
    # Victory
    # -----------------------------------------------------------------------

    def check_victory_conditions(self) -> bool:
        """
        End-of-round check. Game ends when:
          - Any player has reached VP_TO_WIN (10) VP, OR
          - The conflict deck is exhausted (all 10 rounds played).
        VP can legitimately exceed 12; we track the true integer.
        """
        if self.game_over:
            return True
        deck_empty    = not self.conflict_deck and self.current_conflict is None
        rounds_done   = self.round >= MAX_ROUNDS
        vp_threshold  = any(p.victory_points >= VP_TO_WIN for p in self.players)
        if deck_empty or rounds_done or vp_threshold:
            self.game_over = True
            self.phase     = Phase.GAME_OVER
            _vp_before = {p.id: p.victory_points for p in self.players}
            self._resolve_endgame_intrigues()
            self._resolve_endgame_battle_icons()
            self._endgame_vp_gained = {
                p.id: p.victory_points - _vp_before[p.id] for p in self.players}
            self.winner    = self._determine_winner()
            return True
        return False

    def _resolve_endgame_intrigues(self) -> None:
        """Auto-play every held Endgame Intrigue (they only ever grant VP/resources)."""
        from src.game.intrigue.intrigue import IntrigueTiming
        for p in self.players:
            for ic in list(p.intrigue_cards):
                ts = ic.timing if isinstance(ic.timing, (set, tuple, list, frozenset)) \
                    else (ic.timing,)
                if IntrigueTiming.ENDGAME not in ts:
                    continue
                ok, _ = ic.can_play(p, self)
                if ok:
                    p.play_intrigue(ic)
                    ic.resolve(p, self)

    def _resolve_endgame_battle_icons(self) -> None:
        """
        Endgame only: a Wild battle icon matches ANY other icon (including
        another Wild).  Non-wild pairs already scored immediately when won.
        """
        for pid in range(self.num_players):
            p = self.players[pid]
            wilds = sum(1 for i in p.battle_icons if i == "wild")
            reals = sum(1 for i in p.battle_icons if i != "wild")
            paired = min(wilds, reals)            # wild + real
            wilds -= paired
            p.gain_vp(paired + wilds // 2)        # leftover wilds pair with each other
            p.battle_icons = []

    def _determine_winner(self) -> int:
        def sort_key(p: Player):
            return (p.victory_points, p.spice, p.solari, p.water, p.troops_garrison)
        return max(range(self.num_players), key=lambda i: sort_key(self.players[i]))

    def trigger_endgame_intrigues(self) -> None:
        pass  # Resolved by RL env / action space

    # -----------------------------------------------------------------------
    # Legacy observation (kept for compatibility; prefer get_state_dict)
    # -----------------------------------------------------------------------

    def get_observation(self, perspective_player_id: int) -> Dict:
        obs = self.get_state_dict()
        # Mask opponent hands (not in get_state_dict by default)
        obs["perspective_player"] = perspective_player_id
        return obs

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def get_legal_agent_spaces(self, player_id: int, card: Card) -> List[str]:
        legal = []
        for space in ALL_BOARD_SPACES:
            ok, _ = self.can_send_agent(player_id, space, card)
            if ok:
                legal.append(space)
        return legal

    def get_space_effects_preview(self, space_name: str) -> List[Dict]:
        """Approximate effect dicts a space yields — for heuristic scoring only."""
        if space_name not in UPRISING_BOARD:
            return []
        sp = UPRISING_BOARD[space_name]
        if not sp.special:
            return list(BOARD_SPACE_EFFECTS.get(space_name, []))
        return {
            "Swordmaster":   [{"agents": 1}],
            "High Council":  [{"persuasion": 2}],
            "Gather Support": [{"troops": 2}],
            "Spice Refinery": [{"solari": 2}],
            "Sietch Tabr":   [{"maker_hooks": 1, "troops": 1, "water": 1}],
            "Hagga Basin":   [{"spice": 2}],
            "Deep Desert":   [{"spice": 4}],
        }.get(space_name, [])

    def clone(self) -> "GameState":
        """
        Deep copy for lookahead search.  Card / Conflict / Leader / Contract
        objects are value-immutable (compared by name) so they are shared, not
        copied — only the mutable containers and primitive state are duplicated.
        """
        import copy as _copy
        memo: Dict[int, object] = {}
        # Share the flyweight game objects.
        for card in self.imperium_deck + self.imperium_row:
            memo[id(card)] = card
        for pl in self.players:
            for pile in (pl.deck, pl.hand, pl.discard, pl.in_play, pl.trash,
                         pl.intrigue_cards):
                for obj in pile:
                    memo[id(obj)] = obj
            if pl.leader is not None:
                # copy the leader (per-turn flags are mutable) but keep hooks
                pass
        for c in self.conflict_deck:
            memo[id(c)] = c
        if self.current_conflict is not None:
            memo[id(self.current_conflict)] = self.current_conflict
        for lst in self.won_conflicts.values():
            for c in lst:
                memo[id(c)] = c
        for ic in self.intrigue_deck + self.intrigue_discard:
            memo[id(ic)] = ic
        for ct in self.contract_bank + self.contracts_on_board:
            memo[id(ct)] = ct
        for pl in self.players:
            for ct in pl.contracts_active + pl.contracts_completed:
                memo[id(ct)] = ct
        return _copy.deepcopy(self, memo)

    def __repr__(self) -> str:
        return (
            f"GameState(round={self.round}/{MAX_ROUNDS}, phase={self.phase.value}, "
            f"players={self.num_players}, "
            f"conflict={self.current_conflict.name if self.current_conflict else None})"
        )
