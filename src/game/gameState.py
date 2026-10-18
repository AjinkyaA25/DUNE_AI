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
    LEADER_SPACES,
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
    SETUP        = "setup"          # Leader setup choices (Steersman, Kota) before round 1
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
    END_TURN             = "end_turn"      # close an Agent turn after its Plot window
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
    # Bloodlines
    RESOLVE_BL_CHOICE    = "resolve_bl_choice"  # tech buy / commander / skill choices
    RESOLVE_CHOICE       = "resolve_choice"     # a "choose" inside a card / intrigue (choices.py)
    ACTIVATE_TECH        = "activate_tech"      # flip a once-per-round tech tile


# Bonus printed on the 4th space of each influence track
INFLUENCE_4_BONUS = {"fremen": {"water": 1}, "bene_gesserit": {"intrigue": 1},
                     "spacing_guild": {"solari": 3}, "emperor": {"spy": 1}}

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
    # Bloodlines: how many of the deployed units are Sardaukar commanders
    # (None = no choice offered: commanders go first)
    deploy_cmd: Optional[int]         = None
    # Trash resolution
    trash_card_name: Optional[str]    = None
    trash_from: Optional[str]         = None   # "hand" | "discard" | "in_play" (None: first found)
    # Influence-any resolution
    influence_faction: Optional[str]  = None
    # Contract choice (0 or 1 = which face-up contract)
    contract_index: int               = 0
    # Optional cost->reward arrow: accept and pay, or decline
    accept_optional: bool             = False
    # Heighliner (legacy)
    spice_cost: int                   = 0
    troop_count: int                  = 0
    # Bloodlines: RESOLVE_BL_CHOICE option string / ACTIVATE_TECH tile name
    choice: Optional[str]             = None

    def __repr__(self) -> str:
        extra = ""
        if self.space_option:
            extra += f", opt={self.space_option}"
        if self.action_type == ActionType.RESOLVE_DEPLOY:
            extra += f", n={self.deploy_count}"
            if self.deploy_cmd is not None:
                extra += f", cmd={self.deploy_cmd}"
        if self.intrigue_card_name:
            extra += f", intrigue={self.intrigue_card_name}"
        if self.choice:
            extra += f", choice={self.choice}"
        # every field that tells two legal moves apart must show here: search,
        # the UI and the replay match moves across game copies by repr
        if self.use_infiltrate:
            extra += ", infiltrate"
        if self.use_gather_intelligence:
            extra += ", gather"
        for f in ("spy_post_name", "uplift_space_name", "acquire_card_name",
                  "reserve_type", "trash_card_name", "influence_faction"):
            v = getattr(self, f)
            if v is not None:
                extra += f", {f.split('_')[0]}={v}"
        if self.trash_from:
            extra += f", from={self.trash_from}"
        if self.action_type == ActionType.RESOLVE_CONTRACT:
            extra += f", contract={self.contract_index}"
        if self.accept_optional:
            extra += ", accept"
        if self.spice_cost or self.troop_count:
            extra += f", spice_cost={self.spice_cost}, troops={self.troop_count}"
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
    """Optional 'trash a card' (Desert Tactics, card effects). May be skipped -
    unless it is the cost of an Intrigue being played (`mandatory`)."""
    def __init__(self, player_id: int, count: int = 1, on_trash=None, mandatory: bool = False):
        self.player_id = player_id
        self.count     = count
        self.mandatory = mandatory
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
        use_bloodlines: bool = False,
        ix_techs: bool = False,
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
        self.agent_on_space:   Dict[str, Optional[int]] = {s: None for s in UPRISING_BOARD}
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
        self.pending_choices: List = []        # src.game.choices.PendingChoice

        # ===== COMBAT PHASE TURN TRACKING =====
        self.combat_turn_idx: int = 0
        # Set True while an Agent turn is resolving so Player.gain_troops() knows
        # to increment troops_recruited_this_turn.
        self._agent_turn_active: bool = False
        # player_id whose Agent turn is mid-resolution (awaiting follow-up choices);
        # play passes to the next player once it clears.
        self._agent_turn_open: Optional[int] = None
        # After an Agent is placed, a player holding a playable Plot Intrigue
        # may still play it before ending the turn (END_TURN).
        self._post_agent_window: Optional[int] = None
        # Seats (humans in the UI) that always get the post-Agent window and
        # must END_TURN themselves, so an Intrigue drawn by the Agent's space
        # (Assembly Hall, Secrets, ...) can still be played this turn.
        self.manual_end_turn: Set[int] = set()

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

        # ===== BLOODLINES (techs, Sardaukar Commanders, Command) =====
        self.use_bloodlines = use_bloodlines
        self.bl = None
        if use_bloodlines:
            from src.game.bloodlines.rules import Bloodlines
            self.bl = Bloodlines(self, ix_techs=ix_techs)

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
            "pending_choices":    len(getattr(self, "pending_choices", [])),
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
                    if self.intrigue_cost_payable(player_id, ic):
                        if self.intrigue_cost_payable(player_id, ic):
                            actions.append(GameAction(ActionType.PLAY_INTRIGUE, player_id,
                                                      intrigue_card_name=ic.name))
            if self.reserve_spice_must_flow:
                smf = self.reserve_spice_must_flow[-1]
                if smf.cost <= persuasion:
                    actions.append(GameAction(ActionType.ACQUIRE_RESERVE,
                                              player_id,
                                              reserve_type="spice_must_flow"))
            if self.bl:
                for name in self.bl.activatable(player_id, agent_turn=False):
                    actions.append(GameAction(ActionType.ACTIVATE_TECH, player_id,
                                              choice=name))
                # a commander from your supply may be re-recruited on the Reveal turn too
                if "supply" in self.bl.recruit_options(player_id, None):
                    actions.append(GameAction(ActionType.RESOLVE_CHOICE, player_id,
                                              choice="recruit:supply"))
            # a deploy from the Reveal turn (Holy War, Ruthless Leadership, ...):
            # any time, several times, troops gained this Reveal turn count
            actions += self._open_deploy_actions(player_id)
            actions.append(GameAction(ActionType.END_REVEAL, player_id))
            return actions

        # --- PLAYER_TURNS phase ---
        if phase == Phase.PLAYER_TURNS:
            current = self.get_current_player_id()
            if current != player_id:
                return [GameAction(ActionType.NO_OP, player_id)]

            p = self.players[player_id]
            not_revealed = player_id not in self.players_revealed

            # Agent already placed: only Plot Intrigues, or end the turn.
            if self._post_agent_window == player_id:
                if not self.is_in_atomic_block():
                    for ic in p.intrigue_cards:
                        ts = ic.timing if isinstance(ic.timing, (set, frozenset, tuple, list))                             else (ic.timing,)
                        if any(t.value == "plot" for t in ts) and                                 ic.can_play(p, self, is_agent_turn=True)[0]:
                            if self.intrigue_cost_payable(player_id, ic):
                                actions.append(GameAction(ActionType.PLAY_INTRIGUE, player_id,
                                                          intrigue_card_name=ic.name))
                actions += self._claim_actions(player_id)
                actions += self._open_deploy_actions(player_id)
                actions += self._late_flip_actions(player_id)
                actions += self._late_bl_actions(player_id)
                actions.append(GameAction(ActionType.END_TURN, player_id))
                return actions

            # Agent turn: one action per valid (card, space, option, spy-mod) combo
            if p.agents_available > 0 and not_revealed:
                seen_cards = set()
                for card in p.hand:
                    if card.name in seen_cards:       # identical duplicates → one entry
                        continue
                    seen_cards.add(card.name)
                    for space in self.get_legal_agent_spaces(player_id, card):
                        options = self.get_space_options(player_id, space)
                        can_gi  = self.can_gather_intelligence(player_id, space)
                        can_inf = self.can_infiltrate(player_id, space)
                        if self._blocked_for(player_id, space):
                            # an opponent is there: the only way in is to
                            # Infiltrate (which spends the bordering Spy)
                            spy_mods = []
                        else:
                            spy_mods = [(False, False)]
                            if can_gi:
                                spy_mods.append((True, False))
                                if p.leader is not None and p.leader.must_gather:
                                    spy_mods.remove((False, False))     # Gaius: you must
                        if can_inf:
                            spy_mods.append((False, True))
                        # A Spy on BOTH posts bordering the space: one may
                        # Infiltrate while the other Gathers Intelligence.
                        # (Never two Gather Intelligence — no double card draw.)
                        if can_inf and self._connected_spy_count(player_id, space) >= 2:
                            spy_mods.append((True, True))
                        if (card.name == "Signet Ring" and p.leader is not None
                                and p.leader.name == "Shaddam Corrino IV"):
                            # his ring forbids deploying units this turn: no worms
                            options = [o for o in options if o != "sandworm"] or options
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
                        if self.intrigue_cost_payable(player_id, ic):
                            actions.append(GameAction(ActionType.PLAY_INTRIGUE, player_id,
                                                      intrigue_card_name=ic.name))

            # Bloodlines: flip a once-per-round tech before this turn's action
            if self.bl and not_revealed and not self.is_in_atomic_block():
                for name in self.bl.activatable(
                        player_id, agent_turn=p.agents_available > 0):
                    actions.append(GameAction(ActionType.ACTIVATE_TECH, player_id,
                                              choice=name))

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
                        if self.intrigue_cost_payable(player_id, ic):
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

        bl_before = self.bl.snapshot() if self.bl else None
        try:
            self._dispatch_action(action)
        except (MandatoryEffectSkipped, AtomicBlockViolation, ValueError) as exc:
            info["error"] = str(exc)
        if self.bl:
            self.bl.sync(bl_before)

        # Combat strength is derived (units + revealed swords), so refresh it
        # after anything done during the Combat phase: a choice inside a
        # combat Intrigue (e.g. Questionable Methods' lose-influence -> +4
        # swords) adds swords after the card itself was played.
        if self.phase == Phase.COMBAT and self.current_conflict is not None:
            for part in self.get_combat_participants():
                self.update_combat_strength(part)

        for p in self.players:
            if p.leader is not None and p.leader.on_after_step is not None:
                p.leader.on_after_step(p, self)

        # Deferred conditional reveal effects fire once the revealing player has
        # cleared their pending choices (so e.g. a Spy placed on the Reveal turn
        # is counted by a "2+ Spies" reveal condition).
        self._flush_reveal_conditionals()

        # Withdrawn (Twisted Intrigue): pass this turn - play moves on
        passer = getattr(self, "_pass_requested", None)
        if passer is not None:
            self._pass_requested = None
            if self.phase == Phase.PLAYER_TURNS and self.get_current_player_id() == passer                     and not self._has_mandatory_pending_for(passer):
                self.advance_to_next_player_turn()

        # An Agent turn = place ONE agent, then play passes to the next player
        # (once any follow-up choices — deploy, spy, ... — are resolved).
        if self._agent_turn_open is not None:
            self._prune_dead_pendings(self._agent_turn_open)
        self._maybe_end_agent_turn()
        if action.action_type != ActionType.END_TURN:
            self._maybe_close_post_agent_window()

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
        if self.use_choam:
            self.check_turn_harvest_contracts(pid)
        self._agent_turn_open = None
        if self.phase == Phase.PLAYER_TURNS and (
                pid in self.manual_end_turn or self._has_playable_plot(pid) or self._turn_open_for(pid)):
            self._post_agent_window = pid
            return
        self.players[pid].deferred_rewards = []      # nothing claimable: forfeited
        if self.phase == Phase.PLAYER_TURNS and self.player_in_reveal_buy is None:
            self.advance_to_next_player_turn()

    # -- rewards taken when the player chooses ----------------------------
    def defer_reward_for(self, pid: int) -> bool:
        return getattr(self, "_defer_rewards_pid", None) == pid

    def defer_reward(self, pid: int, kind: str, n: int, effect: Optional[Dict] = None,
                     cond: Optional[str] = None) -> None:
        """One waiting reward: its kind (draw / intrigue, or "effect" - a card's
        conditional reward whose condition isn't met yet), amount, source
        (space or card) and the card conditions it needs when claimed."""
        items = self.players[pid].__dict__.setdefault("deferred_rewards", [])
        if isinstance(items, dict):                   # older form
            items = self.players[pid].deferred_rewards = []
        conds = list(getattr(self, "_defer_conds", [])) + ([cond] if cond else [])
        items.append({"kind": kind, "n": int(n), "src": getattr(self, "_defer_source", None),
                      "conds": conds, "effect": dict(effect) if effect else None})

    def _reward_items(self, pid: int) -> list:
        items = getattr(self.players[pid], "deferred_rewards", [])
        return items if isinstance(items, list) else []

    def reward_qualifies(self, pid: int, item: dict) -> bool:
        from src.game.effects import EffectResolver
        p = self.players[pid]
        return all(EffectResolver.cond_holds(c, p, self) for c in item.get("conds", []))

    def deferred(self, pid: int) -> list:
        """(index, item) for each waiting reward the player qualifies for now."""
        return [(i, it) for i, it in enumerate(self._reward_items(pid)) if self.reward_qualifies(pid, it)]

    def claim_reward(self, pid: int, idx: int) -> None:
        items = self._reward_items(pid)
        if not 0 <= idx < len(items):
            return
        it = items[idx]
        if not self.reward_qualifies(pid, it):
            return
        items.pop(idx)
        if it["kind"] == "draw":
            self.draw_cards_for_player(pid, it["n"])
        elif it["kind"] == "intrigue":
            self.draw_intrigue_for_player(pid, it["n"])
        elif it["kind"] == "nav":                    # Steersman's Navigation card, played when he likes
            from src.game.leader.navigation import play_navigation
            e = it["effect"]
            play_navigation(self.players[pid], self, e["card"], slot=e["slot"], faction=e["faction"])
        elif it["kind"] == "effect" and it.get("effect"):
            from src.game.effects import EffectResolver
            EffectResolver.resolve_single_effect(dict(it["effect"]), self.players[pid], self)

    def claim_all_rewards(self, pid: int) -> None:
        """End of turn: take every reward still qualified for; the rest are lost."""
        while True:
            q = self.deferred(pid)
            if not q:
                break
            self.claim_reward(pid, q[0][0])
        self.players[pid].deferred_rewards = []

    def in_own_turn(self, pid: int) -> bool:
        """pid is taking its Agent turn (incl. its follow-ups / window) or its
        Reveal turn - troops gained now are 'recruited this turn'."""
        return (getattr(self, "_agent_turn_active", False) and getattr(self, "_agent_turn_owner", None) in (None, pid)) \
            or pid in (self._agent_turn_open, self._post_agent_window, self.player_in_reveal_buy,
                       getattr(self, "_revealing_pid", None))

    def deploy_left(self, pid: int) -> int:
        """Troops the player may still move to the Conflict this turn from a
        combat space: 2 (once) + every troop recruited this turn - deployed."""
        p = self.players[pid]
        icon = getattr(p, "deploy_icon_this_turn", False)
        recruited = getattr(p, "deploy_recruited_this_turn", False)   # Sardaukar Coordination
        if not (icon or recruited) or self.current_conflict is None:
            return 0
        if getattr(p, "no_deploy_this_turn", False):
            return 0                                   # Shaddam's Signet Ring
        rec = getattr(p, "troops_recruited_this_turn", 0)
        budget = max(getattr(p, "deploy_budget_this_turn", 0),
                     2 + rec + getattr(p, "deploy_extra_this_turn", 0) if icon else rec)
        usable = p.troops_garrison - getattr(p, "undeployable_this_turn", 0)
        return max(0, min(usable, budget - getattr(p, "deployed_this_turn", 0)))

    def _open_deploy_actions(self, pid: int) -> List["GameAction"]:
        hi = self.deploy_left(pid)
        out: List[GameAction] = []
        if hi <= 0 or any(d.player_id == pid for d in self.pending_deployments):
            return out
        p = self.players[pid]
        cmd = getattr(p, "commanders_garrison", 0) if self.bl else 0
        plain = p.troops_garrison - cmd
        for n in range(1, hi + 1):
            if cmd:
                for c in range(max(0, n - plain), min(n, cmd) + 1):
                    out.append(GameAction(ActionType.RESOLVE_DEPLOY, pid, deploy_count=n, deploy_cmd=c))
            else:
                out.append(GameAction(ActionType.RESOLVE_DEPLOY, pid, deploy_count=n))
        return out

    def _late_bl_actions(self, pid: int) -> List["GameAction"]:
        if not self.bl:
            return []
        return [GameAction(ActionType.RESOLVE_CHOICE, pid, choice=c) for c in self.bl.late_choices(pid)]

    def _late_flip_actions(self, pid: int) -> List["GameAction"]:
        """Once-per-round techs flipped later in the Agent turn (Rapid Dropships
        bought at Assembly Hall can be flipped the same turn)."""
        if not self.bl or self.players[pid].agents_available < 0:
            return []
        return [GameAction(ActionType.ACTIVATE_TECH, pid, choice=n)
                for n in self.bl.activatable_after_placement(pid)]

    def _turn_open_for(self, pid: int) -> bool:
        """Something optional is still available this turn: a reward to claim
        or troops to deploy."""
        return bool(self.deferred(pid)) or self.deploy_left(pid) > 0 \
            or bool(self.bl and self.bl.late_choices(pid))

    def _claim_actions(self, pid: int) -> List["GameAction"]:
        return [GameAction(ActionType.RESOLVE_CHOICE, pid, choice=f"claim:{i}")
                for i, _ in self.deferred(pid)]

    def _has_playable_plot(self, pid: int) -> bool:
        from src.game.intrigue.intrigue import IntrigueTiming
        p = self.players[pid]
        for ic in p.intrigue_cards:
            ts = ic.timing if isinstance(ic.timing, (set, frozenset, tuple, list))                 else (ic.timing,)
            if IntrigueTiming.PLOT in ts and ic.can_play(p, self, is_agent_turn=True)[0]:
                return True
        return False

    def _maybe_close_post_agent_window(self, force: bool = False) -> None:
        """End the Plot window once the player ends the turn (or has no
        playable Plot Intrigue left) and has no pending choices."""
        pid = self._post_agent_window
        if pid is None:
            return
        self._prune_dead_pendings(pid)
        if self._has_mandatory_pending_for(pid):
            return
        if self.use_choam:
            self.check_turn_harvest_contracts(pid)
        if not force and self.phase == Phase.PLAYER_TURNS and (
                pid in self.manual_end_turn or self._has_playable_plot(pid) or self._turn_open_for(pid)):
            return
        if self._reward_items(pid):
            self.claim_all_rewards(pid)          # ending the turn takes what is left
            self._prune_dead_pendings(pid)
            if self._has_mandatory_pending_for(pid):
                return
        self._post_agent_window = None
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
        elif at == ActionType.END_TURN:
            self._maybe_close_post_agent_window(force=True)
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
            self.update_combat_strength(action.player_id)
        elif at == ActionType.PLAY_INTRIGUE:
            self._step_play_intrigue(action)
        elif at == ActionType.COMBAT_PASS:
            self._step_combat_pass(action)
        elif at == ActionType.RESOLVE_CHOICE:
            self._step_resolve_choice(action)
        elif at == ActionType.RESOLVE_BL_CHOICE and self.bl:
            self.bl.resolve_choice(pid, action.choice)
        elif at == ActionType.ACTIVATE_TECH and self.bl:
            self.bl.activate(pid, action.choice)
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
        p0 = self.players[pid]                     # the Reveal turn is a new turn
        p0.troops_recruited_this_turn = 0
        p0.deploy_icon_this_turn = False
        p0.deploy_budget_this_turn = 0
        p0.deploy_extra_this_turn = 0
        p0.deployed_this_turn = 0
        p0.units_deployed_this_turn = 0
        self._revealing_pid = pid
        self._reveal_open = pid                   # until the Reveal turn ends
        # apply_reveal_turn() already accumulates persuasion into the pool:
        #   - each revealed card's .persuasion field and {"persuasion": n} effects
        #   - leader on_reveal hooks
        # plus anything a Signet Ring / agent effect added earlier this turn.
        try:
            self.apply_reveal_turn(pid)
        finally:
            self._revealing_pid = None
        p = self.players[pid]
        # High Council Councilor bonus: +2 Persuasion each Reveal turn
        if p.has_councilor:
            self.gain_persuasion(pid, 2)
        # Assembly Hall bonus: +1 Persuasion if player has an Agent there
        if self.agent_on_space.get("Assembly Hall") == pid or                 pid in getattr(self, "infiltrated_by", {}).get("Assembly Hall", []):
            self.gain_persuasion(pid, 1)
        self.player_in_reveal_buy = pid
        self.players_revealed.add(pid)
        self.current_turn_type = TurnType.REVEAL
        if self.bl:
            self.bl.open_command_window(pid)

    def _step_end_reveal(self, action: GameAction) -> None:
        pid = action.player_id
        if self.player_in_reveal_buy != pid:
            return
        self._reveal_open = None
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
        if self.bl:
            self.bl.close_command_window(pid)
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
            self._to_acquired_pile(pid, card)
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
            # All 3 Spies are already on the board: recall one (the player's
            # pick, action.choice = "recall:<post>") to make room, then place.
            if player.spies_on_board:
                src = (action.choice[7:] if action.choice and action.choice.startswith("recall:")
                       else sorted(k for k, v in player.spies_on_board.items() if v > 0)[0])
                self.players[pid].recall_spy(src)
                player.recalled_spy_this_turn = True
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
        p = self.players[pid]
        if pending is None:
            n = max(0, min(action.deploy_count, self.deploy_left(pid)))
            if n <= 0:
                raise ValueError("No deployment available for player")
            self._do_deploy(pid, n, action.deploy_cmd)
            return
        budget_left = max(0, getattr(p, "deploy_budget_this_turn", 0)
                          - getattr(p, "deployed_this_turn", 0))
        n = max(0, min(action.deploy_count, pending.max_deploy,
                       p.troops_garrison, budget_left))
        if n > 0:
            self._do_deploy(pid, n, action.deploy_cmd)
        self.pending_deployments.remove(pending)

    def _to_trash(self, player, card) -> None:
        """Trash a card.  The Spice Must Flow / Prepare the Way return to their
        reserve stack instead of the player's permanent trash pile."""
        if getattr(card, "cost", 0) >= 1:
            player.trashed_costly_card_this_round = True   # Tenuous Bond
        if player.leader is not None:
            player.leader.trigger("on_trash", player, self, card)
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
            # Trash from your HAND, DISCARD or PLAY AREA (FAQ: e.g. send Guild
            # Envoy to Desert Tactics and trash it there) - never the deck.
            zones = {"hand": p.hand, "discard": p.discard, "in_play": p.in_play}
            order = [action.trash_from] if action.trash_from else ["hand", "discard", "in_play"]
            card = zone = None
            for z in order:
                card = next((c for c in zones[z] if c.name == name), None)
                if card is not None:
                    zone = zones[z]
                    break
            if card is None:
                raise ValueError(f"Card '{name}' not in {' / '.join(order)}")
            del zone[next(i for i, c in enumerate(zone) if c is card)]
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
        aside = getattr(self, "shaddam_contracts", {}).get(pid, [])
        if action.contract_index >= 100 and action.contract_index - 100 < len(aside):
            contract = aside.pop(action.contract_index - 100)
            self.players[pid].take_contract(contract)
            if contract.is_immediate() and (
                    not contract.requires_intrigue() or self.players[pid].intrigue_cards):
                self.complete_contract(pid, contract)
        elif not self.contracts_on_board:
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
                if action.choice and action.choice.startswith("recall:"):
                    for post in action.choice[7:].split("|"):
                        p.recall_spy(post)
                    p.recalled_spy_this_turn = True
                else:                               # no pick given (API use): first posts
                    for post in sorted(k for k, n in p.spies_on_board.items() if n > 0)[:v]:
                        p.recall_spy(post)
                    p.recalled_spy_this_turn = True
            else:
                setattr(p, k, getattr(p, k) - v)
        if pending.discard == 1 and len({c.name for c in p.hand}) > 1:
            # which card to discard is the player's choice
            from src.game.cards.card import CardTag
            opts, seen = [], set()
            for c in p.hand:
                if c.name in seen:
                    continue
                seen.add(c.name)
                eff = {"_discard_named": c.name, **pending.reward}
                if pending.discard_tag and pending.tag_bonus and c.has_tag(CardTag(pending.discard_tag)):
                    for k, v in pending.tag_bonus.items():
                        eff[k] = eff.get(k, 0) + v if isinstance(v, (int, float)) else v
                opts.append((c.name, f"Discard {c.name}", eff))
            self.add_pending_choice(pid, "discard", "Choose a card to discard", opts)
            return
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
                    EffectResolver._note_discard(p, tgt)
                    discarded_tag = True
                    continue
            EffectResolver._discard_worst(p)
        EffectResolver.resolve_single_effect(pending.reward, p, self)
        if discarded_tag and pending.tag_bonus:
            EffectResolver.resolve_single_effect(pending.tag_bonus, p, self)

    def _step_resolve_choice(self, action: GameAction) -> None:
        pid = action.player_id
        if action.choice and action.choice.startswith("claim:"):
            self.claim_reward(pid, int(action.choice[6:]))
            return
        if action.choice and action.choice.startswith("recruit:") and self.bl:
            opt = action.choice[8:]
            space = None if opt == "supply" else getattr(self.players[pid], "bl_space_this_turn", None)
            self.bl.recruit(pid, space, opt)
            return
        if action.choice and action.choice.startswith("techbuy:") and self.bl:
            self.bl.buy_tech(pid, int(action.choice[8:]))
            return
        pc = next((c for c in getattr(self, "pending_choices", []) if c.player_id == pid), None)
        if pc is None:
            raise ValueError("No pending choice for player")
        opt = next((o for o in pc.options if o[0] == action.choice), None)
        if opt is None:
            raise ValueError(f"'{action.choice}' is not an option of {pc.kind}")
        from src.game.choices import affordable
        if not affordable(self, self.players[pid], opt[2]):
            raise ValueError(f"can't pay for '{action.choice}' any more")
        self.pending_choices.remove(pc)
        if opt[2]:
            EffectResolver.resolve_single_effect(dict(opt[2]), self.players[pid], self)

    def turn_start_for(self, player_id: int) -> bool:
        """The player's Agent turn has started and nothing was done yet."""
        return (self.phase == Phase.PLAYER_TURNS and self.player_in_reveal_buy is None
                and self._agent_turn_open is None and self._post_agent_window is None
                and self.get_current_player_id() == player_id
                and player_id not in self.players_revealed
                and self.players[player_id].agents_available > 0)

    def note_units_deployed(self, player_id: int, n: int) -> None:
        """Units (troops, commanders, sandworms) sent to the Conflict this turn
        (Distraction: 'when you deploy three or more units in a single turn')."""
        if n > 0:
            p = self.players[player_id]
            p.units_deployed_this_turn = getattr(p, "units_deployed_this_turn", 0) + int(n)

    def recall_spies_choice(self, player_id: int, n: int = 1) -> None:
        """Recall n of the player's Spies - which ones is the player's choice
        (a pending choice per Spy when more than one post is possible)."""
        p = self.players[player_id]
        p.recalled_spy_this_turn = True
        for _ in range(int(n)):
            posts = sorted(k for k, v in p.spies_on_board.items() if v > 0)
            if not posts:
                return
            if len(posts) <= int(n) - _:          # every Spy goes: nothing to choose
                p.recall_spy(posts[0])
                continue
            self.add_pending_choice(player_id, "recall_spy", "Choose which Spy to recall",
                                    [(f"recall:{x}", f"Recall your Spy from {x}", {"_recall_post": x})
                                     for x in posts])
            return_left = int(n) - 1 - _
            if return_left:
                # further recalls are asked after this one is resolved
                self._recall_queue = getattr(self, "_recall_queue", {})
                self._recall_queue[player_id] = self._recall_queue.get(player_id, 0) + return_left
            return

    def add_pending_choice(self, player_id: int, kind: str, prompt: str, options) -> None:
        """Queue a 'choose' decision (see src/game/choices.py)."""
        from src.game.choices import PendingChoice
        if not options:
            return
        if not hasattr(self, "pending_choices"):
            self.pending_choices = []
        self.pending_choices.append(PendingChoice(player_id, kind, prompt, list(options)))

    def pending_choice_for(self, player_id: int):
        return next((c for c in getattr(self, "pending_choices", [])
                     if c.player_id == player_id), None)

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
        # one ability: its cost must be paid; several: the player picks which
        self._intrigue_cost_for = action.player_id if self.intrigue_abilities_usable(action.player_id, card) <= 1 else None
        try:
            card.resolve(p, self)                    # resolve + global discard
        finally:
            self._intrigue_cost_for = None
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
        if self.phase == Phase.SETUP and not any(
                self._has_mandatory_pending_for(q) for q in range(self.num_players)):
            self.start_new_round()                 # every Leader setup choice is made
        safety = 0
        while safety < 20:
            safety += 1
            if self.game_over:
                break
            if self.phase == Phase.COMBAT and self.current_conflict is None:
                # Conflict resolved; its optional "pay X -> VP" rewards are
                # still being decided. Makers follow once they all are.
                if any(self._has_mandatory_pending_for(q) for q in range(self.num_players)):
                    break
                self.phase = Phase.MAKERS
            elif (self.phase == Phase.COMBAT
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
        from src.game.choices import affordable
        self.pending_choices = [            # a choice with no payable option fizzles
            pc for pc in getattr(self, "pending_choices", [])
            if pc.player_id != player_id or any(affordable(self, p, o[2]) for o in pc.options)
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
            any(p.player_id == player_id for p in self.pending_optional_payments) or
            any(p.player_id == player_id for p in getattr(self, "pending_choices", [])) or
            (self.bl is not None and self.bl.has_pending(player_id))
        )

    def _ability_usable(self, p, e) -> bool:
        """One ability (a top-level effect dict) of an Intrigue: usable now if
        any of its parts can be used - a failed condition only blocks its own
        reward (Shaddam's Favor: the troop with or without 3 Emperor
        influence), and a cost blocks what it pays for."""
        from src.game.effects import EffectResolver
        if not isinstance(e, dict):
            return True
        parts = []
        for k, v in e.items():
            if k == "choose_one" and isinstance(v, dict) and v.get("may") is False:
                from src.game.choices import affordable, _option_possible
                parts.append(any(affordable(self, p, o.get("effect", {}))
                                 and _option_possible(self, p, o.get("effect", {}))
                                 for o in v.get("options", [])))
            elif k in ("bl_choose", "choose_one", "choose_by_combat", "options"):
                parts.append(True)                # a choice: checked per option
            elif k == "_leader_fx":
                from src.game.leader.twisted import playable
                parts.append(playable(self, p, v))
            elif k.startswith("if_bl_"):
                try:
                    ok = bool(self.bl) and self.bl.cond(k[6:], p)
                except Exception:
                    ok = True
                parts.append(ok and self._ability_usable(p, v))
            elif k.startswith("if_"):
                try:
                    ok = EffectResolver.cond_holds(k[3:], p, self)
                except Exception:
                    ok = True
                parts.append(bool(ok) and self._ability_usable(p, v))
            elif k == "pay_then" and isinstance(v, dict):
                cost = v.get("cost", {})
                parts.append(not (any(getattr(p, r, 0) < n for r, n in cost.items() if r != "recall_spy")
                                  or cost.get("recall_spy", 0) > sum(p.spies_on_board.values())))
            elif k == "discard_then":
                parts.append(len(p.hand) >= 1)
            elif k == "trash_then":
                parts.append(bool(p.hand or p.discard or p.in_play))
            elif k == "influence_swap":
                parts.append(any(v2 > 0 for v2 in p.influence.values()))
            else:
                parts.append(True)
        return any(parts) if parts else True

    def intrigue_abilities_usable(self, player_id: int, ic) -> int:
        p = self.players[player_id]
        return sum(1 for e in (getattr(ic, "effects", []) or []) if self._ability_usable(p, e))

    def intrigue_cost_payable(self, player_id: int, ic) -> bool:
        """An Intrigue can be played if at least ONE of its abilities can be
        used now (user rule): e.g. Strategic Stockpiling with only the 5 spice,
        or only the 3 water + 3 Fremen influence; Change Allegiances with only
        an influence to give, or only the 3 spice."""
        return self.intrigue_abilities_usable(player_id, ic) > 0

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
        if self.phase == Phase.PLAYER_TURNS and player_id in (self._agent_turn_open,
                                                             self._post_agent_window):
            actions += self._claim_actions(player_id)     # rewards in any order
            actions += self._open_deploy_actions(player_id)
            actions += self._late_flip_actions(player_id)
            actions += self._late_bl_actions(player_id)

        for psp in self.pending_spy_placements:
            if psp.player_id != player_id:
                continue
            on_board = sorted(k for k, v in p.spies_on_board.items() if v > 0)
            for post in sorted(psp.allowed_posts or ALL_OBSERVATION_POSTS):
                if self.can_place_spy(player_id, post, psp.allow_occupied):
                    if p.spies_available > 0:
                        actions.append(GameAction(ActionType.RESOLVE_SPY,
                                                  player_id, spy_post_name=post))
                    else:                       # none in supply: choose which Spy to recall
                        for src in on_board:
                            actions.append(GameAction(ActionType.RESOLVE_SPY, player_id,
                                                      spy_post_name=post, choice=f"recall:{src}"))

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
            hi = min(pd.max_deploy, p.troops_garrison - getattr(p, "undeployable_this_turn", 0),
                     budget_left)
            if getattr(p, "no_deploy_this_turn", False):
                hi = 0
            cmd = getattr(p, "commanders_garrison", 0) if self.bl else 0
            plain = p.troops_garrison - cmd
            for n in range(0, hi + 1):
                if cmd and n:          # which units go: troops and/or commanders
                    for c in range(max(0, n - plain), min(n, cmd) + 1):
                        actions.append(GameAction(ActionType.RESOLVE_DEPLOY, player_id,
                                                  deploy_count=n, deploy_cmd=c))
                else:
                    actions.append(GameAction(ActionType.RESOLVE_DEPLOY,
                                              player_id, deploy_count=n))

        for pt in self.pending_trashes:
            if pt.player_id != player_id:
                continue
            # one option per card name per zone: a copy in the discard is a
            # different choice from one still in hand
            for zone, cards in (("hand", p.hand), ("discard", p.discard),
                                ("in_play", p.in_play)):
                for nm in sorted({c.name for c in cards}):
                    actions.append(GameAction(ActionType.RESOLVE_TRASH, player_id,
                                              trash_card_name=nm, trash_from=zone))
            actions.append(GameAction(ActionType.RESOLVE_TRASH,
                                      player_id, trash_card_name=None))  # decline
            if getattr(pt, "mandatory", False):    # an Intrigue's cost: no "trash nothing"
                actions = [a for a in actions if not (a.action_type == ActionType.RESOLVE_TRASH
                                                      and a.trash_card_name is None)]

        for op in self.pending_optional_payments:
            if op.player_id != player_id:
                continue
            need = int(op.cost.get("recall_spy", 0))
            posts = sorted(k for k, v in p.spies_on_board.items() if v > 0)
            if need and len(posts) > need:
                # recalling is part of the cost: one "accept" per choice of Spies
                import itertools
                for combo in itertools.combinations(posts, need):
                    actions.append(GameAction(ActionType.RESOLVE_OPTIONAL, player_id,
                                              accept_optional=True, choice="recall:" + "|".join(combo)))
            else:
                actions.append(GameAction(ActionType.RESOLVE_OPTIONAL, player_id,
                                          accept_optional=True))
            if not getattr(op, "mandatory", False):   # an Intrigue's cost must be paid
                actions.append(GameAction(ActionType.RESOLVE_OPTIONAL, player_id,
                                          accept_optional=False))   # decline
            break                                     # one arrow at a time (the next follows)

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
            for k, _c in enumerate(getattr(self, "shaddam_contracts", {}).get(player_id, [])):
                actions.append(GameAction(ActionType.RESOLVE_CONTRACT,       # Shaddam's
                                          player_id, contract_index=100 + k))

        pc = self.pending_choice_for(player_id)
        if pc is not None:
            from src.game.choices import affordable
            # costs are re-checked now: another pending step may have spent the resources
            for key, _label, _eff in pc.options:
                if affordable(self, p, _eff):
                    actions.append(GameAction(ActionType.RESOLVE_CHOICE, player_id, choice=key))

        if self.bl:
            for opt in self.bl.pending_options(player_id):
                actions.append(GameAction(ActionType.RESOLVE_BL_CHOICE, player_id,
                                          choice=opt))

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
        if self.bl:
            self.bl.on_persuasion(player_id, amount)

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

        # play goes clockwise from the First Player token (seat order 0..n-1)
        n = self.num_players
        self.turn_order = [(self.first_player + i) % n for i in range(n)]

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
        if self.bl:
            self.bl.on_round_start()
        for p in self.players:
            if p.leader is not None:
                p.leader.trigger("on_round_start", p, self)

        self.phase                = Phase.PLAYER_TURNS
        self.current_turn_idx     = 0
        self._reset_harvest_turn()
        self.players_revealed     = set()
        self.player_in_reveal_buy = None
        self.swords_this_reveal   = {i: 0 for i in range(self.num_players)}
        self._prev_vp             = {i: self.players[i].victory_points for i in range(self.num_players)}

    def get_current_player_id(self) -> int:
        if self.phase == Phase.SETUP:
            return next((q for q in range(self.num_players) if self._has_mandatory_pending_for(q)), 0)
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

    def _reset_harvest_turn(self) -> None:
        """A new player turn starts: Harvest contracts count only the spice
        gained during the turn in which the Agent visits a maker space."""
        for p in self.players:
            p.harvest_spice_this_turn = 0
            p.visited_maker_this_turn = False

    def advance_to_next_player_turn(self) -> None:
        self._reset_harvest_turn()
        # the turn is over: Shaddam's Signet (no deploying) and the deploy budget
        # end with it, so a Plot Intrigue played at the start of the next
        # (Reveal) turn deploys on that turn's own terms
        ended = self.players[self.turn_order[self.current_turn_idx]]
        ended.no_deploy_this_turn = False
        ended.troops_recruited_this_turn = 0
        ended.deploy_budget_this_turn = 0
        ended.deployed_this_turn = 0
        ended.deploy_recruited_this_turn = False
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
        drawn = []
        for _ in range(count):
            if not player.deck:
                if not player.discard:
                    break
                new_deck = list(player.discard)
                self.rng.shuffle(new_deck)
                player.deck    = new_deck
                player.discard = []
            if player.deck:
                drawn.append(player.deck.pop())
                player.hand.append(drawn[-1])
        if drawn and getattr(self, "_reveal_open", None) == player_id:
            # a card drawn during your Reveal turn is revealed too (its
            # persuasion, swords and Reveal effects count) - user, 2026-10-16
            player.reveal_cards(drawn)
            player._revealed_this_turn = getattr(player, "_revealed_this_turn", 0) + len(drawn)
            self.swords_this_reveal[player_id] = (self.swords_this_reveal.get(player_id, 0)
                                                  + sum(c.swords for c in drawn))
            for card in drawn:
                EffectResolver.resolve_reveal_effects(card, player, self)
            if self.bl and self.bl.has(player_id, "Artillery"):   # per revealed card
                self.swords_this_reveal[player_id] += sum(
                    1 for c in drawn if c.swords > 0 or any("swords" in e for e in c.reveal_effects))
            self.update_combat_strength(player_id)

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

        if space_name in LEADER_SPACES and not self._leader_space_active(space_name):
            return False, f"{space_name} is not in this game"
        required_icon = BOARD_SPACE_ICONS[space_name]
        card_icons    = {s.value for s in card.access_symbols}
        if self.bl:
            card_icons |= self.bl.extra_icons(player_id, card)
        extra = getattr(player, "extra_icons_this_turn", None)
        if extra and card.access_symbols:
            card_icons |= set(extra)                 # Twisted Intrigue: Resourceful
        if (player.leader is not None and player.leader.card_spy_icon
                and card.access_symbols):
            card_icons.add("spy")                    # Gaius Helen Mohiam

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

        if self.agent_on_space.get(space_name) is not None and not (
                self.bl and "ignore_blocking" in player.bl_flags   # Invasion Ships
                and self.agent_on_space.get(space_name) != player_id):
            if not self.can_infiltrate(player_id, space_name):
                return False, f"{space_name} is occupied and player cannot Infiltrate"

        _ignore_gates = (
            getattr(player, "ignore_influence_gates_this_turn", False)
            or any("ignore_influence_gates" in e for e in card.agent_effects))
        if player.leader is not None and space_name in player.leader.ignore_gates:
            _ignore_gates = True                     # Liet Kynes: Sietch Tabr
        if space_name in INFLUENCE_GATED_SPACES and not _ignore_gates:
            faction, required_inf = INFLUENCE_GATED_SPACES[space_name]
            if player.influence[faction] < required_inf:
                return False, (
                    f"{space_name} requires {required_inf} {faction} influence "
                    f"(player has {player.influence[faction]})"
                )

        if space_name in BOARD_SPACE_MANDATORY_COSTS:
            cost = self._space_cost(player_id, space_name)
            if not player.can_afford(cost):
                return False, f"Cannot afford {cost} to visit {space_name}"

        if space_name == "Swordmaster":
            if player.has_swordmaster:
                return False, "Player already has a Swordmaster"
            if player.solari < self._swordmaster_cost(player_id):
                return False, f"Cannot afford Swordmaster ({self._swordmaster_cost(player_id)} solari)"

        return True, ""

    # ------------------------------------------------------------------

    def get_space_options(self, player_id: int, space_name: str) -> List[Optional[str]]:
        """Legal `space_option` values for a special space (['default'] otherwise)."""
        p = self.players[player_id]
        if space_name == "Spice Refinery":
            return ["free"] + (["pay_spice"] if p.spice >= 1 - self._nav_discount(player_id, "spice", 1) else [])
        if space_name == "Gather Support":
            return ["free"] + (["paid"] if p.solari >= 2 - self._nav_discount(player_id, "solari", 2) else [])
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
            if p.leader is not None and p.leader.no_sandworms:
                worm_blocked = False                  # Liet: rewards even behind the Shield Wall
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
        if self._blocked_for(player_id, space_name) and not infiltrate:
            raise ValueError(f"{space_name} is occupied: you must Infiltrate to go there")
        self._agent_turn_active = True
        self._agent_turn_owner = player_id
        player.deferred_rewards = {}           # draws / Intrigues claimed when the player chooses
        self._current_agent_space = space_name       # can't uplift THIS agent
        self._current_space_option = space_option
        player.extra_icons_this_turn = None
        player.troops_recruited_this_turn = 0
        player.recalled_spy_this_turn = False
        player.gained_spice_this_turn = False
        player.spice_gained_this_turn = 0
        player.turn_serial = getattr(player, "turn_serial", 0) + 1
        player.deploy_budget_this_turn = 0
        player.deployed_this_turn = 0
        player.units_deployed_this_turn = 0
        player.deploy_icon_this_turn = False
        player.deploy_recruited_this_turn = False
        player.bl_space_this_turn = None
        player.bl_tech_bought_this_turn = False
        player.no_deploy_this_turn = False
        player.undeployable_this_turn = 0
        player.deploy_extra_this_turn = 0
        player.maker_bonus_this_turn = 0
        player.agent_to_maker_this_turn = space_name in MAKER_SPACES
        if space_name in MAKER_SPACES:
            player.visited_maker_this_turn = True
        player.agent_to_faction_this_turn = self._faction_for_space(space_name) is not None
        if player.leader is not None:
            player.leader.reset_per_turn()
            player.leader.trigger("on_agent_turn_start", player, self)
        try:
            # 1. Play card
            player.play_card_as_agent(card, icons_checked=True)
            if player.leader is not None:
                player.leader.trigger("on_card_played", player, self, card)
                if card.name == "Signet Ring" and player.leader.name == "Shaddam Corrino IV":
                    player.no_deploy_this_turn = True

            # 2. Pay mandatory cost + place agent
            if space_name in BOARD_SPACE_MANDATORY_COSTS:
                player.pay_cost(self._space_cost(player_id, space_name))
            if self.agent_on_space.get(space_name) is None:
                self.agent_on_space[space_name] = player_id
            else:                                   # Infiltrated: both Agents are there
                self.infiltrated_by = getattr(self, "infiltrated_by", {})
                self.infiltrated_by.setdefault(space_name, []).append(player_id)
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

            # 5. Board space effects / 6. Card Agent box effects
            # (their card draws and Intrigues wait to be claimed - see defer_reward)
            self._defer_rewards_pid = player_id
            try:
                self._defer_source = space_name
                self._apply_board_space_effects(player_id, space_name, space_option)
                self._defer_source = card.name
                EffectResolver.resolve_agent_effects(card, player, self)
                # Signet Ring: the Leader's ability is the card's Agent box,
                # resolved once the Agent is on its space
                if (card.name == "Signet Ring" and player.leader is not None
                        and player.leader.signet is not None):
                    self._defer_source = "Signet Ring"
                    if getattr(self, "leader_log", None) is not None:
                        self.leader_log.append((player_id, player.leader.name, "signet"))
                    player.leader.signet(player, self)
            finally:
                self._defer_rewards_pid = None
                self._defer_source = None

            # 7. Faction influence for faction board spaces
            faction = self._faction_for_space(space_name)
            if faction:
                self.gain_influence_with_check(player_id, faction, 1)

            # 8. CHOAM contract checks
            if self.use_choam:
                self.check_board_space_contracts(player_id, space_name)
                self.check_turn_harvest_contracts(player_id)

            # 8b. Leader: on_agent_placed (+ every other Leader that watches spaces)
            if player.leader is not None:
                player.leader.trigger("on_agent_placed", player, self, space_name)
            for other in self.players:
                if other.id != player_id and other.leader is not None:
                    other.leader.trigger("on_other_agent_placed", other, self, player_id, space_name)

            # 8c. Bloodlines: tech purchase (green space) / commander recruit
            if self.bl:
                self.bl.on_agent_placed(player_id, space_name, card)

            # 9. Combat deployment — a Combat space is a deploy icon: it opens a
            #    per-turn budget of 2 (once) + every troop recruited this turn.
            space = UPRISING_BOARD[space_name]
            dropships = self.bl is not None and "dropships" in player.bl_flags
            if (space.is_combat_space or dropships) and self.current_conflict is not None:
                # Deploying is open for the rest of this turn (user rule): any
                # time, more than once, and troops gained later this turn
                # (e.g. a Strike Fleet reward claimed after a Spy recall) count.
                player.deploy_icon_this_turn = True
        finally:
            self._agent_turn_active = False
            # One-shot access relaxers are consumed by this Agent turn.
            player.ignore_influence_gates_this_turn = False
            player.grant_emperor_access_this_turn = False
            if self.bl:
                self.bl.end_agent_turn(player_id)

    def _leader_space_active(self, space_name: str) -> bool:
        """Tuek's Sietch exists only when someone plays Esmar Tuek."""
        return space_name in getattr(self, "leader_spaces", set())

    def _space_cost(self, player_id: int, space_name: str) -> Dict:
        """Mandatory cost to visit a space (Navigation Chamber: 1 less)."""
        cost = BOARD_SPACE_MANDATORY_COSTS.get(space_name, {})
        return self.bl.space_cost(player_id, cost) if self.bl else cost

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
            p.solari -= self._swordmaster_cost(player_id)
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
                p.solari -= 2 - self._nav_discount(player_id, "solari", 2)
                EffectResolver.resolve_single_effect({"troops": 2, "water": 1}, p, self)
            else:
                EffectResolver.resolve_single_effect({"troops": 2}, p, self)

        elif space_name == "Spice Refinery":
            if option == "pay_spice":
                p.spice -= 1 - self._nav_discount(player_id, "spice", 1)
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

    def _swordmaster_cost(self, player_id: Optional[int] = None) -> int:
        any_has_sm = any(pl.has_swordmaster for pl in self.players)
        base = SWORDMASTER_COST_AFTER if any_has_sm else SWORDMASTER_COST_FIRST
        if player_id is not None and self.players[player_id].leader is not None:
            base = max(0, base - self.players[player_id].leader.swordmaster_discount)
        return base - self._nav_discount(player_id, "solari", base)

    def _nav_discount(self, player_id: Optional[int], res: str, amount: int) -> int:
        """Navigation Chamber: a board space costs 1 spice or 1 solari less -
        also the optional / special costs (Spice Refinery, Gather Support,
        Swordmaster)."""
        if player_id is None or not self.bl or amount <= 0:
            return 0
        return 1 if self.bl.has(player_id, "Navigation Chamber") else 0

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
        if getattr(self.players[player_id], "no_deploy_this_turn", False):
            return                                       # Shaddam's Signet Ring
        if max_deploy > 0 and self.current_conflict is not None:
            p = self.players[player_id]
            p.deploy_budget_this_turn = max(
                p.deploy_budget_this_turn,
                getattr(p, "deployed_this_turn", 0) + max_deploy)
            self.pending_deployments.append(PendingDeployment(player_id, max_deploy))

    def _do_deploy(self, player_id: int, n: int, cmd: Optional[int] = None) -> None:
        """Move `n` troops from garrison straight into the current Conflict
        (`cmd` of them commanders; None = commanders first)."""
        p = self.players[player_id]
        if getattr(p, "no_deploy_this_turn", False):
            return                                       # Shaddam's Signet Ring
        n = max(0, min(n, p.troops_garrison))
        if self.bl:
            self.bl.deploy_commanders(player_id, n, cmd)
        p.troops_garrison -= n
        p.deployed_this_turn = getattr(p, "deployed_this_turn", 0) + n
        self.note_units_deployed(player_id, n)
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

        if getattr(player, "no_deploy_this_turn", False):
            return False                                 # Shaddam's Signet Ring
        if sandworms > 0:
            if not player.has_maker_hooks:
                return False
            liet = player.leader is not None and player.leader.no_sandworms
            if self.shield_wall_intact and conflict_location in SHIELD_WALL_PROTECTED and not liet:
                return False
            if liet:                                     # rewards instead of worms
                player.leader.trigger("on_sandworm_replaced", player, self, sandworms)
                sandworms = 0

        player.troops_garrison -= troops_from_garrison
        self.note_units_deployed(player_id, total_troops + sandworms)
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
        active = self.get_current_player_id()
        lp = self.players[active]
        if getattr(lp, "no_deploy_this_turn", False):
            return                                                 # Shaddam's Signet Ring
        if lp.leader is not None and lp.leader.no_sandworms:      # Liet Kynes
            lp.leader.trigger("on_sandworm_replaced", lp, self, count)
            return
        if (
            self.shield_wall_intact
            and self.current_conflict.location in SHIELD_WALL_PROTECTED
        ):
            return
        self.sandworms_in_conflict[active] = (
            self.sandworms_in_conflict.get(active, 0) + count
        )
        self.note_units_deployed(active, count)

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
        player.spice_gained_this_turn = 0          # a Reveal turn is a new turn
        player.turn_serial = getattr(player, "turn_serial", 0) + 1
        player.cards_acquired_this_turn = 0
        player.gained_spice_this_turn = False
        player.deploy_budget_this_turn = 0
        player.deployed_this_turn = 0
        player.units_deployed_this_turn = 0
        player.deploy_icon_this_turn = False
        player.deploy_recruited_this_turn = False
        player.bl_space_this_turn = None
        player.bl_tech_bought_this_turn = False
        player.no_deploy_this_turn = False
        player.undeployable_this_turn = 0
        player.deploy_extra_this_turn = 0

        swords = sum(c.swords for c in cards_to_reveal)
        self.swords_this_reveal[player_id] = (
            self.swords_this_reveal.get(player_id, 0) + swords
        )

        for card in cards_to_reveal:
            EffectResolver.resolve_reveal_effects(card, player, self)
        if self.bl:
            self.bl.on_reveal(player_id, cards_to_reveal)

        troops    = self.troops_in_conflict.get(player_id, 0)
        sandworms = self.sandworms_in_conflict.get(player_id, 0)
        strength  = Combat.calculate_player_strength(
            player, troops, sandworms,
            self.swords_this_reveal.get(player_id, 0), self
        )
        self.combat_strength[player_id] = strength
        if player.leader is not None:
            player.leader.trigger("on_reveal_done", player, self)
            self.update_combat_strength(player_id)

    def update_combat_strength(self, player_id: int) -> None:
        player    = self.players[player_id]
        troops    = self.troops_in_conflict.get(player_id, 0)
        sandworms = self.sandworms_in_conflict.get(player_id, 0)
        swords    = self.swords_this_reveal.get(player_id, 0)
        self.combat_strength[player_id] = Combat.calculate_player_strength(
            player, troops, sandworms, swords, self
        )

    def retreat_units(self, player_id: int, n: int, cmd: Optional[int] = None) -> int:
        """Move n units from the Conflict to the garrison; `cmd` of them
        Sardaukar commanders (None = plain troops first). Strength is
        recalculated - revealed swords stay while any unit remains."""
        p = self.players[player_id]
        n = max(0, min(int(n), self.troops_in_conflict.get(player_id, 0)))
        if n <= 0:
            return 0
        if self.bl:
            cin = self.bl.commanders_in_conflict.get(player_id, 0)
            plain = self.troops_in_conflict.get(player_id, 0) - cin
            lo, hi = max(0, n - plain), min(n, cin)
            c = lo if cmd is None else max(lo, min(hi, int(cmd)))
            self.bl.commanders_in_conflict[player_id] = cin - c
            p.commanders_garrison += c
        self.troops_in_conflict[player_id] -= n
        p.troops_garrison += n
        if p.leader is not None:                        # commanders count (user)
            p.leader.trigger("on_units_left_conflict", p, self, n)
        self.update_combat_strength(player_id)
        return n

    def deploy_with_choice(self, player_id: int, n: int) -> None:
        """Deploy n units from the garrison; with plain troops and commanders
        both there the player picks the mix (a commander is a troop)."""
        p = self.players[player_id]
        if getattr(p, "no_deploy_this_turn", False) or self.current_conflict is None:
            return
        n = max(0, min(int(n), p.troops_garrison))
        cg = getattr(p, "commanders_garrison", 0) if self.bl else 0
        lo, hi = max(0, n - (p.troops_garrison - cg)), min(n, cg)
        if n > 0 and hi > lo:
            opts = []
            for c in range(lo, hi + 1):
                t = n - c
                what = " + ".join(x for x in (f"{t} troop{'s' if t != 1 else ''}" if t else "",
                                              f"{c} commander{'s' if c != 1 else ''}" if c else "") if x)
                opts.append((f"dmix{c}", f"Deploy {what}", {"_deploy_mix": [n, c]}))
            self.add_pending_choice(player_id, "deploy_mix", f"Deploy {n}: which units?", opts)
            return
        self.deploy_units(player_id, n, lo)

    def deploy_units(self, player_id: int, n: int, cmd: int = 0) -> None:
        """Move n units (cmd of them commanders) from garrison to the Conflict."""
        p = self.players[player_id]
        if getattr(p, "no_deploy_this_turn", False) or self.current_conflict is None:
            return
        n = max(0, min(int(n), p.troops_garrison))
        if n <= 0:
            return
        if self.bl is not None:
            self.bl.deploy_commanders(player_id, n, cmd=cmd)
        p.troops_garrison -= n
        self.troops_in_conflict[player_id] = self.troops_in_conflict.get(player_id, 0) + n
        self.note_units_deployed(player_id, n)
        self.update_combat_strength(player_id)

    def note(self, text: str) -> None:
        """A line for the UI game log (things that happen to other players)."""
        if not hasattr(self, "ui_notes") or self.ui_notes is None:
            self.ui_notes = []
        self.ui_notes.append(text)

    def retreat_with_choice(self, player_id: int, n: int, reward: Optional[Dict] = None) -> None:
        """Retreat n units; if both plain troops and commanders are in the
        Conflict the player picks the mix (troops and commanders are both
        "troops")."""
        n = max(0, min(int(n), self.troops_in_conflict.get(player_id, 0)))
        cin = self.bl.commanders_in_conflict.get(player_id, 0) if self.bl else 0
        plain = self.troops_in_conflict.get(player_id, 0) - cin
        lo, hi = max(0, n - plain), min(n, cin)
        if n > 0 and hi > lo:
            opts = []
            for c in range(lo, hi + 1):
                t = n - c
                what = " + ".join(x for x in (f"{t} troop{'s' if t != 1 else ''}" if t else "",
                                              f"{c} commander{'s' if c != 1 else ''}" if c else "") if x)
                opts.append((f"mix{c}", f"Retreat {what}", {"_retreat_mix": [n, c], **(reward or {})}))
            self.add_pending_choice(player_id, "retreat_mix", f"Retreat {n}: which units?", opts)
            return
        self.retreat_units(player_id, n)
        if reward:
            from src.game.effects import EffectResolver
            EffectResolver.resolve_single_effect(dict(reward), self.players[player_id], self)

    def acquire_card(self, player_id: int, card: Card) -> bool:
        """
        Spend persuasion to acquire a card from the Imperium Row.
        FIX (Bug 1): Row refills BEFORE acquire effects resolve (FAQ).
        """
        if card not in self.imperium_row:
            return False
        self.imperium_row.remove(card)
        self._to_acquired_pile(player_id, card)
        self.players[player_id].cards_acquired_this_turn += 1
        from src.game.choices import _call_to_arms
        _call_to_arms(self, self.players[player_id])
        # Refill row FIRST, then trigger acquire effects (FAQ requirement)
        self.refill_imperium_row()
        self._trigger_acquire_effects(player_id, card)
        if self.use_choam:
            self.check_acquire_contracts(player_id, card.name)
        return True

    def _to_acquired_pile(self, player_id: int, card: Card) -> None:
        """Acquired cards go to the discard pile - or on top of the deck with
        Spaceport (always taken: a fresh card is drawn sooner)."""
        p = self.players[player_id]
        if self.bl and (self.bl.has(player_id, "Spaceport")
                        or getattr(p, "bl_topdeck", False)):   # Recruitment Mission
            p.deck.append(card)             # deck.pop() draws from the end
        else:
            p.discard.append(card)

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
        self._to_acquired_pile(player_id, card)
        self.players[player_id].cards_acquired_this_turn += 1
        from src.game.choices import _call_to_arms
        _call_to_arms(self, self.players[player_id])
        self._trigger_acquire_effects(player_id, card)
        if self.use_choam:
            self.check_acquire_contracts(player_id, card.name)
        return True

    def refill_imperium_row(self) -> None:
        while len(self.imperium_row) < 5 and self.imperium_deck:
            self.imperium_row.append(self.imperium_deck.pop(0))

    def _trigger_acquire_effects(self, player_id: int, card: Card) -> None:
        if card.name == "The Spice Must Flow":      # its VP stays even once trashed
            pl = self.players[player_id]
            pl.tsmf_acquired = getattr(pl, "tsmf_acquired", 0) + 1
        if hasattr(card, "acquire_effects") and card.acquire_effects:
            player = self.players[player_id]
            for effect in card.acquire_effects:
                EffectResolver.resolve_single_effect(effect, player, self)

    def mark_reveal_complete(self, player_id: int) -> None:
        """Move in-play cards to discard. Swords revealed stay in the Conflict
        until it is resolved (user rule): a later retreat or Combat Intrigue
        recalculates strength with them - they only stop counting when the
        player has no units left there. Reset in _cleanup_after_combat."""
        self.players[player_id].recall_cards()

    # -----------------------------------------------------------------------
    # Phase 3: Combat
    # -----------------------------------------------------------------------

    def get_combat_participants(self) -> List[int]:
        duncan = getattr(self, "duncan_in_conflict", {})
        return [
            pid for pid in range(self.num_players)
            if (self.troops_in_conflict.get(pid, 0) +
                self.sandworms_in_conflict.get(pid, 0) + (1 if duncan.get(pid) else 0)) > 0
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
        for part in participants:              # never resolve on a stale strength
            self.update_combat_strength(part)

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
            if self.bl:
                self.bl.on_win_conflict(conflict_card_goes_to)
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
        if any(self._has_mandatory_pending_for(q) for q in range(self.num_players)):
            self.phase = Phase.COMBAT       # reward choices first (see _auto_advance)
        return result

    def _scale_reward(self, reward: Dict, with_sandworm: bool) -> Dict:
        """A sandworm in the Conflict doubles EVERY reward — flat resources and
        VP.  A 'spend X -> gain a VP' conversion doubles by letting the winner
        take it TWICE (`times`), each application still at the printed cost /
        spy count — so a winner who can only afford ONE application still
        banks that one VP (Battle for Imperial Basin with 5 spice: pay 4 -> 1
        VP; the doubled 2nd application needs 4 more they don't have).  Battle
        for Arrakeen's recall-2-Spies conversion therefore caps at one VP even
        with a worm, since a 2nd application would need a 4th Spy.
        Only `control` (and the internal `battle_icon` marker) can't scale."""
        if not with_sandworm:
            return reward
        _nodouble = ("control", "battle_icon")
        _vp_convert = ("may_pay_spice_for_vp", "may_pay_solari_for_vp",
                       "may_pay_troops_for_vp", "may_recall_spies_for_vp")
        scaled = {}
        for k, v in reward.items():
            if k in _nodouble:
                scaled[k] = v
            elif k in _vp_convert and isinstance(v, dict):
                scaled[k] = {**v, "times": 2 * int(v.get("times", 1))}
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
                # the winner picks the faction (RESOLVE_INFLUENCE), like a card's
                self.add_pending_influence_choice(player_id, int(value))
            elif key == "trash":
                # the winner picks the card to trash (or none) from hand,
                # discard or play (RESOLVE_TRASH), e.g. Skirmish (Wild) 1st place
                self.add_pending_trash(player_id, int(value))
            elif key in ("may_pay_spice_for_vp", "may_pay_solari_for_vp"):
                # Optional "spend X -> VP" arrow: the player chooses (accept /
                # decline), once per application (a sandworm grants a 2nd:
                # `times`). Each is queued only while it is affordable.
                cost, vp = value.get("cost", 0), int(value.get("vp", 1))
                attr = "spice" if key == "may_pay_spice_for_vp" else "solari"
                for _ in range(int(value.get("times", 1))):
                    self.add_pending_optional_payment(
                        player_id, {attr: cost}, {"vp": vp}, label="conflict_vp")
            elif key == "may_pay_troops_for_vp":
                # troops in the Conflict -> VP: auto-taken (they go home anyway)
                cost, vp = value.get("cost", 0), int(value.get("vp", 1))
                for _ in range(int(value.get("times", 1))):
                    have = self.troops_in_conflict.get(player_id, 0)
                    if have < cost:
                        break
                    self.troops_in_conflict[player_id] = have - cost
                    player.troops_supply += cost
                    player.gain_vp(vp)
            elif key == "may_recall_spies_for_vp":
                # Battle for Arrakeen: recall N of your Spies -> VP, the
                # player's choice; a sandworm grants a 2nd application
                count, vp = int(value.get("count", 2)), int(value.get("vp", 1))
                for _ in range(int(value.get("times", 1))):
                    self.add_pending_optional_payment(
                        player_id, {"recall_spy": count}, {"vp": vp}, label="conflict_vp")
            else:
                EffectResolver.resolve_single_effect({key: value}, player, self)

    def _award_battle_icon(self, player_id: int, icon) -> bool:
        """
        Give the winner this Conflict's battle icon.  Two matching NON-wild
        icons immediately convert to 1 VP (the pair is consumed).  A "wild"
        icon never matches immediately — it waits for Endgame. With
        Ornithopter Fleet every icon is an Ornithopter (wild ones too), so
        they pair at once.
        """
        if icon is None:
            return False
        name = icon.value if hasattr(icon, "value") else str(icon)
        if self.bl:
            name = self.bl.battle_icon(player_id, name)     # Ornithopter Fleet
        p = self.players[player_id]
        if name != "wild" and name in p.battle_icons:
            p.battle_icons.remove(name)          # consume the matched pair
            p.gain_vp(1)
            return True
        p.battle_icons.append(name)
        return False

        return False

    def _cleanup_after_combat(self) -> None:
        units = {pid: self.troops_in_conflict.get(pid, 0) for pid in range(self.num_players)}
        if self.bl:
            self.bl.after_combat()        # commanders -> commander supply
        for pid in range(self.num_players):
            troop_count = self.troops_in_conflict.get(pid, 0)
            if troop_count > 0:
                self.players[pid].troops_supply += troop_count
            self.troops_in_conflict[pid]    = 0
            if units[pid] > 0 and self.players[pid].leader is not None:
                # troops (commanders included) returned after combat count as lost
                self.players[pid].leader.trigger("on_units_left_conflict",
                                                 self.players[pid], self, units[pid])
            self.sandworms_in_conflict[pid] = 0
            self.combat_strength[pid]       = 0
            self.swords_this_reveal[pid]    = 0

        self.duncan_in_conflict = {}
        self.current_conflict = None
        self.phase            = Phase.MAKERS

    # -----------------------------------------------------------------------
    # Phase 4: Makers
    # -----------------------------------------------------------------------

    def resolve_makers_phase(self) -> None:
        for space in MAKER_SPACES:
            if space in LEADER_SPACES and not self._leader_space_active(space):
                continue
            if self.agent_on_space.get(space) is None:
                self.maker_bonus_spice[space] = self.maker_bonus_spice.get(space, 0) + 1
        self.phase = Phase.RECALL

    # -----------------------------------------------------------------------
    # Phase 5: Recall
    # -----------------------------------------------------------------------

    def resolve_recall_phase(self) -> None:
        if self.bl:
            self.bl.reset_round()
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
        self.infiltrated_by = {}

        # the First Player token passes to the next seat every round
        self.first_player = (self.first_player + 1) % self.num_players

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

        # "reach 2 influence" (rulebook p.17: passing 2 counts; only going up)
        if old_inf < 2 <= new_inf and player.leader is not None:
            player.leader.trigger("on_influence_2", player, self, faction)

        # The 4th space of each track carries a bonus (user, 2026-10-14):
        # Fremen 1 water, Bene Gesserit 1 Intrigue, Spacing Guild 3 solari,
        # Emperor a Spy - gained when you reach it.
        if old_inf < 4 <= new_inf:
            from src.game.effects import EffectResolver
            EffectResolver.resolve_single_effect(dict(INFLUENCE_4_BONUS[faction]), player, self)

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

    def _blocked_for(self, player_id: int, space_name: str) -> bool:
        """An opponent's Agent is on the space (and nothing lets you ignore it)."""
        occ = self.agent_on_space.get(space_name)
        if occ is None or occ == player_id:
            return False
        p = self.players[player_id]
        return not (self.bl and "ignore_blocking" in getattr(p, "bl_flags", set()))

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
        before = len(player.intrigue_cards)
        self._draw_intrigues(player, count)
        if self.bl:
            self.bl.on_intrigue_drawn(player_id, len(player.intrigue_cards) - before)

    def _draw_intrigues(self, player, count: int) -> None:
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
        if getattr(card, "twisted", False) and getattr(self, "twisted_owner", None) is not None:
            owner = self.players[self.twisted_owner]
            if owner.leader is not None and "twisted_discard" in owner.leader.state:
                owner.leader.state["twisted_discard"].append(card)
                return
        self.intrigue_discard.append(card)

    # -----------------------------------------------------------------------
    # CHOAM Contracts
    # -----------------------------------------------------------------------

    def _refill_contracts_on_board(self) -> None:
        if len(self.contracts_on_board) < 2 and self.contract_bank:
            self.contracts_on_board.append(self.contract_bank.pop(0))

    def take_contract_from_board(self, player_id: int) -> Optional[Contract]:
        # If no contracts remain (or CHOAM is off) -> 2 Solari instead.
        aside = getattr(self, "shaddam_contracts", {}).get(player_id, [])
        if not self.use_choam or (not self.contracts_on_board and not aside):
            self.players[player_id].gain_solari(2)
            return None
        if len(self.contracts_on_board) == 1 and not aside:
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
        # completed on your Agent turn: its card draws / Intrigues wait to be
        # claimed like any reward (e.g. Shishakli trash -> draw, then the
        # Spice Refinery contract's 2 draws)
        if getattr(self, "_agent_turn_active", False) and self._defer_reward_owner() == player_id:
            old = (getattr(self, "_defer_rewards_pid", None), getattr(self, "_defer_source", None))
            self._defer_rewards_pid, self._defer_source = player_id, f"contract {contract.name}"
            try:
                return self._complete_contract(player_id, contract)
            finally:
                self._defer_rewards_pid, self._defer_source = old
        return self._complete_contract(player_id, contract)

    def _defer_reward_owner(self) -> Optional[int]:
        return getattr(self, "_agent_turn_owner", None)

    def _complete_contract(self, player_id: int, contract: Contract) -> None:
        player = self.players[player_id]
        player.complete_contract(contract)
        for key, value in contract.rewards.items():
            EffectResolver.resolve_single_effect({key: value}, player, self)
        if self.bl:
            self.bl.on_complete_contract(player_id)

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

    def check_turn_harvest_contracts(self, player_id: int) -> None:
        """Harvest 3+/4+: the Agent visited a maker space this turn and the
        player has gained that much spice this turn by ANY means (board space,
        card, Intrigue, ...). Spice spent this turn does not reduce it."""
        player = self.players[player_id]
        if not getattr(player, "visited_maker_this_turn", False):
            return
        gained = getattr(player, "harvest_spice_this_turn", 0)
        for contract in list(player.contracts_active):
            if (contract.contract_type == ContractType.HARVEST
                    and gained >= contract.trigger_condition.get("min_spice", 3)):
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
            # trash -> reward on an Intrigue being played: the trash is its cost
            mandatory = on_trash is not None and getattr(self, "_intrigue_cost_for", None) == player_id
            self.pending_trashes.append(PendingTrash(player_id, count, on_trash, mandatory=mandatory))

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
        op.mandatory = getattr(self, "_intrigue_cost_for", None) == player_id
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
        self.discard_intrigue_card(intrigue_card)
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
            if self.bl:
                self.bl.endgame()
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
            chaumurky = bool(self.bl and self.bl.wins_ties(p.id))
            return (p.victory_points, chaumurky, p.spice, p.solari, p.water,
                    p.troops_garrison)
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
        for space in UPRISING_BOARD:              # fixed order (not the set)
            ok, _ = self.can_send_agent(player_id, space, card)
            if ok:
                legal.append(space)
        return legal

    def get_space_effects_preview(self, space_name: str,
                                  space_option: Optional[str] = None) -> List[Dict]:
        """Approximate effect dicts a space yields — for heuristic scoring only.

        `space_option` matters for the special spaces that branch: without it
        the preview always describes the "default" branch, which made the
        heuristic score a Hagga Basin / Deep Desert *sandworm* pick as if it
        were the spice pick, and a Sietch Tabr wall-break as if it were hooks.
        """
        if space_name not in UPRISING_BOARD:
            return []
        sp = UPRISING_BOARD[space_name]
        if not sp.special:
            effs = list(BOARD_SPACE_EFFECTS.get(space_name, []))
            if space_name == "Assembly Hall":       # +1 Persuasion on your Reveal turn
                effs.append({"persuasion": 1})
            return effs
        if space_name == "Sietch Tabr":
            return ([{"water": 1, "destroy_shield_wall": 1}]
                    if space_option == "shield_wall"
                    else [{"maker_hooks": 1, "troops": 1, "water": 1}])
        if space_name in MAKER_SPACE_OPTIONS:
            base_spice, worms = MAKER_SPACE_OPTIONS[space_name]
            if space_option == "sandworm":
                return [{"sandworm": worms}]
            return [{"spice": base_spice}]
        if space_name == "Spice Refinery":
            return [{"solari": 4, "spice": -1}] if space_option == "pay_spice" \
                else [{"solari": 2}]
        return {
            "Swordmaster":   [{"agents": 1}],
            "High Council":  [{"persuasion": 2}],
            "Gather Support": [{"troops": 2}],
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
