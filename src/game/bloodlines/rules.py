"""
Bloodlines rules: tech market + tile effects, Sardaukar Commanders + skills,
and the Command keyword. Attached to a GameState as `gs.bl` when the game is
created with use_bloodlines=True; GameState calls the hooks below at the
matching points of the turn (all hooks are no-ops for rules not in play).

Rules as explained by the user (see memory: bloodlines_rules):

Techs - 3 face-up stacks. When you send an Agent to a green (Landsraad)
  space you may buy the top tile of one stack with spice; a High Council seat
  makes it 1 spice cheaper.

Command - on your Reveal turn, if your total persuasion (cards, techs,
  skills: everything) reaches 6+, Command effects fire (once per reveal).

Sardaukar Commanders - one starts on each of 6 spaces. When you send an Agent
  to one of those spaces you may pay 2 solari (max one recruit per turn) to
  take the commander there into your garrison and choose any skill you don't
  hold; a recruited space stays empty for the rest of the game. A commander
  deploys like a troop (strength 2). After combat it goes to your commander
  supply; re-recruiting it (2 solari, no new skill) is only possible by
  sending an Agent to one of the 6 commander spaces.

Skills (held by the player, no duplicates):
  Hardy/Driven/Charismatic - Reveal turn, a commander in the Conflict:
      +1 troop / +1 spice / +1 persuasion.
  Desperate - same trigger: may trash the skill for +3 swords.
  Canny - Agent on a green space: +2 swords.   } need >=1 commander in the
  Fierce - +1 sword, +1 more if an opponent   } Conflict at combat time;
           has a sandworm in the Conflict.     } retreating the last one
  Loyal - 3+ Emperor influence: +2 swords.     } loses them.

Commanders are tracked as a subset of the player's troop counts
(troops_garrison / troops_in_conflict include them), so existing deploy,
retreat and strength code handles them unchanged; `sync()` re-balances the
subset after anything that moved troops without knowing about commanders.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from src.game.bloodlines.techs import TECHS, TECH_BY_NAME, TechTile
from src.game.effects import EffectResolver

if TYPE_CHECKING:
    from src.game.gameState import GameState
    from src.game.player.player import Player

COMMANDER_SPACES = ("Deliver Supplies", "Dutiful Service", "Sardaukar",
                    "High Council", "Gather Support", "Assembly Hall")
GREEN_SPACES = ("High Council", "Swordmaster", "Imperial Privilege",
                "Assembly Hall", "Gather Support")
SKILLS = ("Canny", "Charismatic", "Desperate", "Driven", "Fierce", "Hardy",
          "Loyal")
# AI default when a skill must be picked without a real choice (Plasteel
# Blades' bonus skill): strongest general-purpose skills first.
SKILL_PRIORITY = ("Canny", "Loyal", "Fierce", "Hardy", "Charismatic",
                  "Driven", "Desperate")
COMMANDER_COST = 2
COMMAND_THRESHOLD = 6
LITANY = "card:Litany Against Fear"
ALL_AGENT_ICONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen",
                   "landsraad", "city", "desert")
FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")


class PendingBLChoice:
    """A Bloodlines decision the player must make before play continues.
    `options` are the legal ActionType.RESOLVE_BL_CHOICE `choice` strings."""

    def __init__(self, player_id: int, kind: str, options: List[str],
                 space: Optional[str] = None, payloads: Optional[Dict] = None,
                 discount: int = 0, bonus: Optional[Dict] = None,
                 tag: Optional[str] = None):
        self.player_id = player_id
        self.kind = kind
        self.options = options
        self.space = space
        self.payloads = payloads or {}   # effect choices: option -> effect dict
        self.discount = discount         # tech offers: spice off the price
        self.bonus = bonus               # trash-from-hand: reward if tagged
        self.tag = tag

    def __repr__(self) -> str:
        return f"PendingBLChoice({self.kind}, p={self.player_id}, {self.options})"


class Bloodlines:
    def __init__(self, gs: "GameState"):
        self.gs = gs
        pool = [d for d in TECHS]
        order = list(range(len(pool)))
        gs.rng.shuffle(order)
        pool = [pool[i] for i in order]
        self.tech_stacks: List[List] = [pool[i::3] for i in range(3)]
        self.commander_on_space: Dict[str, bool] = {s: True for s in COMMANDER_SPACES}
        self.commanders_in_conflict: Dict[int, int] = {p.id: 0 for p in gs.players}
        self.pending: List[PendingBLChoice] = []
        for p in gs.players:
            p.techs = []
            p.skills = set()
            p.commanders_garrison = 0
            p.commanders_supply = 0
            p.bl_reveal_persuasion = None   # None = not in a reveal window
            p.bl_command_fired = False
            p.bl_recruited_this_turn = False
            p.bl_flags = set()              # per-turn tech flags
            p.bl_card_commands = []         # Command effects of revealed cards
            p.bl_after_turn = []            # effects resolved at end of turn
            p.bl_discarded = []             # cards discarded by effects (hooks)
            p.bl_commander_discount = 0     # Honor Guard (this turn)
            p.bl_shigawire = False          # Urgent Shigawire (this round)
            p.bl_completed_contract = False
            p.spice_gained_this_turn = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _p(self, pid: int) -> "Player":
        return self.gs.players[pid]

    def has(self, pid: int, tech: str) -> bool:
        return any(t.name == tech for t in self._p(pid).techs)

    def _tile(self, pid: int, tech: str) -> Optional[TechTile]:
        return next((t for t in self._p(pid).techs if t.name == tech), None)

    def _resolve(self, pid: int, effects: List[Dict]) -> None:
        for eff in effects:
            EffectResolver.resolve_single_effect(dict(eff), self._p(pid), self.gs)

    def _trash_tile(self, pid: int, tech: str) -> None:
        p = self._p(pid)
        p.techs = [t for t in p.techs if t.name != tech]

    # ------------------------------------------------------------------
    # pending choices
    # ------------------------------------------------------------------
    def has_pending(self, pid: int) -> bool:
        return any(c.player_id == pid for c in self.pending)

    def pending_options(self, pid: int) -> List[str]:
        """Legal choices right now: purchases are re-checked against current
        resources (another pending choice may have spent them meanwhile)."""
        c = next((c for c in self.pending if c.player_id == pid), None)
        if c is None:
            return []
        if c.kind == "tech_buy":
            live = {f"stack:{i}" for i in self.buyable_stacks(pid)}
            c.options = [o for o in c.options if o in live or o == "decline"]
        elif c.kind == "commander":
            live = set(self.recruit_options(pid, c.space))
            c.options = [o for o in c.options if o in live or o == "decline"]
        elif c.kind == "tech_offer":
            live = {f"stack:{i}" for i in self.buyable_stacks(pid, c.discount)}
            c.options = [o for o in c.options if o in live or o == "decline"]
        elif c.kind == "effect":
            c.options = [o for o in c.options
                         if self.option_legal(pid, c.payloads[o])]
            if not c.options:                  # nothing legal: fizzles
                self.pending.remove(c)
                return []
        return list(c.options)

    def resolve_choice(self, pid: int, choice: str) -> None:
        c = next((c for c in self.pending if c.player_id == pid), None)
        if c is None:
            raise ValueError("No pending Bloodlines choice")
        if choice not in self.pending_options(pid):
            raise ValueError(f"'{choice}' is not a legal {c.kind} choice")
        self.pending.remove(c)
        if choice == "decline":
            return
        if c.kind in ("tech_buy", "tech_offer"):
            self.buy_tech(pid, int(choice.split(":")[1]), c.discount)
        elif c.kind == "effect":
            EffectResolver.resolve_single_effect(
                dict(c.payloads[choice]), self._p(pid), self.gs)
        elif c.kind == "trash_hand":
            p = self._p(pid)
            card = next((x for x in p.hand if x.name == choice), None)
            if card is not None:
                p.hand.remove(card)
                self.gs._to_trash(p, card)
                for eff in getattr(card, "trash_effects", []):
                    EffectResolver.resolve_single_effect(eff, p, self.gs)
                from src.game.cards.card import CardTag
                if c.tag and card.has_tag(CardTag(c.tag)):
                    EffectResolver.resolve_single_effect(dict(c.bonus), p, self.gs)
        elif c.kind == "commander":
            self.recruit(pid, c.space, choice)
        elif c.kind == "desperate":
            p = self._p(pid)
            p.skills.discard("Desperate")
            self.gs.swords_this_reveal[pid] = self.gs.swords_this_reveal.get(pid, 0) + 3
            self.gs.update_combat_strength(pid)
        elif c.kind == "disposal":
            p = self._p(pid)
            card = next((x for x in p.in_play if x.name == choice), None)
            if card is not None:
                p.in_play.remove(card)
                self.gs._to_trash(p, card)
                for eff in getattr(card, "trash_effects", []):
                    EffectResolver.resolve_single_effect(eff, p, self.gs)

    # ------------------------------------------------------------------
    # tech market
    # ------------------------------------------------------------------
    def tech_price(self, pid: int, d, discount: int = 0) -> int:
        return max(0, d.cost - discount - (1 if self._p(pid).has_councilor else 0))

    def buyable_stacks(self, pid: int, discount: int = 0) -> List[int]:
        p = self._p(pid)
        out = []
        for i, stack in enumerate(self.tech_stacks):
            if not stack:
                continue
            d = stack[0]
            if self.has(pid, d.name) or p.spice < self.tech_price(pid, d, discount):
                continue
            if d.name == "Advanced Data Analysis" and not any(p.spies_on_board.values()):
                continue
            out.append(i)
        return out

    def buy_tech(self, pid: int, stack_idx: int, discount: int = 0) -> None:
        p = self._p(pid)
        d = self.tech_stacks[stack_idx].pop(0)
        p.spice -= self.tech_price(pid, d, discount)
        if d.name == "Advanced Data Analysis":      # trash one of your Spies
            post = next(k for k, v in p.spies_on_board.items() if v > 0)
            p.spies_on_board[post] -= 1
            if p.spies_on_board[post] == 0:
                del p.spies_on_board[post]
        p.techs.append(TechTile(d))
        self._resolve(pid, d.acquire)
        if d.name == "Servo-Receivers" and p.leader is not None \
                and getattr(p.leader, "signet", None) is not None:
            p.leader.signet(p, self.gs)

    # ------------------------------------------------------------------
    # agent turn
    # ------------------------------------------------------------------
    def on_agent_placed(self, pid: int, space: str, card=None) -> None:
        """After the space and card effects: offer a tech (green space) and a
        commander recruit (commander space)."""
        p = self._p(pid)
        if card is not None and p.bl_shigawire and self._is_bene(card):
            p.bl_shigawire = False
            self.gs.draw_cards_for_player(pid, 1)
        if space in GREEN_SPACES:
            stacks = self.buyable_stacks(pid)
            if stacks:
                self.pending.append(PendingBLChoice(
                    pid, "tech_buy", [f"stack:{i}" for i in stacks] + ["decline"]))
        if space in COMMANDER_SPACES and not p.bl_recruited_this_turn:
            opts = self.recruit_options(pid, space)
            if opts:
                self.pending.append(PendingBLChoice(
                    pid, "commander", opts + ["decline"], space=space))

    def commander_cost(self, pid: int) -> int:
        return max(0, COMMANDER_COST
                   - (1 if self.has(pid, "Sardaukar High Command") else 0)
                   - self._p(pid).bl_commander_discount)

    def recruit_options(self, pid: int, space: str) -> List[str]:
        p = self._p(pid)
        if p.solari < self.commander_cost(pid):
            return []
        opts = []
        if self.commander_on_space.get(space):
            lacking = [s for s in SKILLS if s not in p.skills]
            opts += [f"board:{s}" for s in lacking] or ["board:"]
        if p.commanders_supply > 0:
            opts.append("supply")
        return opts

    def recruit(self, pid: int, space: str, choice: str) -> None:
        p = self._p(pid)
        p.solari -= self.commander_cost(pid)
        if choice == "supply":
            p.commanders_supply -= 1
        else:
            self.commander_on_space[space] = False
            skill = choice.split(":", 1)[1]
            if skill:
                p.skills.add(skill)
        p.commanders_garrison += 1
        p.troops_garrison += 1
        p.bl_recruited_this_turn = True
        # Plasteel Blades: trash it -> an additional skill
        if self.has(pid, "Plasteel Blades"):
            extra = next((s for s in SKILL_PRIORITY if s not in p.skills), None)
            if extra:
                self._trash_tile(pid, "Plasteel Blades")
                p.skills.add(extra)

    def deploy_commanders(self, pid: int, n: int) -> None:
        """Called when n units leave the garrison for the Conflict: commanders
        go first (their skills only work while one is in the Conflict)."""
        p = self._p(pid)
        c = min(n, p.commanders_garrison)
        p.commanders_garrison -= c
        self.commanders_in_conflict[pid] += c

    def retreat_to_garrison(self, pid: int, n: int) -> None:
        """n units pulled back from the Conflict to the garrison: troops go
        first, commanders only once no plain troops are left there."""
        troops_only = self.gs.troops_in_conflict.get(pid, 0) + n             - self.commanders_in_conflict[pid]
        c = max(0, n - troops_only)
        self.commanders_in_conflict[pid] -= c
        self._p(pid).commanders_garrison += c

    def end_agent_turn(self, pid: int) -> None:
        p = self._p(pid)
        after, p.bl_after_turn = p.bl_after_turn, []
        for eff in after:
            EffectResolver.resolve_single_effect(dict(eff), p, self.gs)
        p.bl_completed_contract = False
        p.bl_commander_discount = 0
        p.bl_envoy = False
        p.bl_recruited_this_turn = False
        p.bl_flags.discard("ignore_blocking")
        p.bl_flags.discard("dropships")

    # ------------------------------------------------------------------
    # activations (flip icon)
    # ------------------------------------------------------------------
    def activatable(self, pid: int, agent_turn: bool) -> List[str]:
        p = self._p(pid)
        out = []
        if agent_turn and p.agents_available > 0 and \
                any(c.name == "Litany Against Fear" for c in p.hand):
            out.append(LITANY)
        for t in p.techs:
            a = t.d.activation
            if a is None or t.used_this_round:
                continue
            if a.get("turn") == "agent" and not agent_turn:
                continue
            if any(getattr(p, k, 0) < v for k, v in a.get("cost", {}).items()):
                continue
            if len(p.hand) < a.get("discard", 0):
                continue
            if a.get("flag") == "cycle_intrigues" and not p.intrigue_cards:
                continue
            out.append(t.name)
        return out

    def activate(self, pid: int, name: str) -> None:
        p = self._p(pid)
        if name == LITANY:
            # put it into play -> draw a card, and pass the turn
            card = next(c for c in p.hand if c.name == "Litany Against Fear")
            p.hand.remove(card)
            p.in_play.append(card)
            self.gs.draw_cards_for_player(pid, 1)
            self.gs._agent_turn_open = pid
            return
        t = self._tile(pid, name)
        if t is None or t.used_this_round or t.d.activation is None:
            raise ValueError(f"{name} can't be activated")
        a = t.d.activation
        for k, v in a.get("cost", {}).items():
            setattr(p, k, getattr(p, k) - v)
        for _ in range(a.get("discard", 0)):
            EffectResolver._discard_worst(p)
        self._resolve(pid, a.get("effects", []))
        flag = a.get("flag")
        if flag in ("dropships", "ignore_blocking"):
            p.bl_flags.add(flag)
        elif flag == "spy_drones" and getattr(p, "recalled_spy_this_turn", False):
            self.gs.add_pending_trash(pid, 1)
        elif flag == "cycle_intrigues":
            cards = list(p.intrigue_cards)
            p.intrigue_cards.clear()
            self.gs.intrigue_deck.extend(cards)          # bottom of the deck
            self.gs.draw_intrigue_for_player(pid, len(cards))
        t.used_this_round = True
        if a.get("trash_self"):
            self._trash_tile(pid, name)

    # ------------------------------------------------------------------
    # reveal turn + Command
    # ------------------------------------------------------------------
    def on_reveal(self, pid: int, revealed: List) -> None:
        """Inside apply_reveal_turn, after the cards' own reveal effects and
        before combat strength is computed."""
        gs, p = self.gs, self._p(pid)
        for t in list(p.techs):
            self._resolve(pid, t.d.reveal)
            if t.name == "Artillery":
                n = sum(1 for c in revealed
                        if c.swords > 0 or any("swords" in e for e in c.reveal_effects))
                gs.swords_this_reveal[pid] = gs.swords_this_reveal.get(pid, 0) + n
            elif t.name == "Restricted Ordnance" and p.has_councilor:
                gs.swords_this_reveal[pid] = gs.swords_this_reveal.get(pid, 0) + 4
            elif t.name == "Forbidden Weapons":
                # "you must choose": swords + lose an influence while you have
                # one to lose, else lose all spice and trash the tile
                if any(p.influence[f] > 0 for f in FACTIONS):
                    gs.swords_this_reveal[pid] = gs.swords_this_reveal.get(pid, 0) + 3
                    EffectResolver.resolve_single_effect(
                        {"lose_influence_any": 1}, p, gs)
                else:
                    p.spice = 0
                    self._trash_tile(pid, t.name)
        if getattr(p, "bl_reveal_bonus", 0):            # Recruitment Mission
            gs.gain_persuasion(pid, p.bl_reveal_bonus)
            p.bl_reveal_bonus = 0
        if self.commanders_in_conflict[pid] > 0:
            if "Hardy" in p.skills:
                p.gain_troops(1)
            if "Driven" in p.skills:
                p.gain_spice(1)
            if "Charismatic" in p.skills:
                gs.gain_persuasion(pid, 1)
            if "Desperate" in p.skills:
                self.pending.append(PendingBLChoice(pid, "desperate",
                                                    ["use", "decline"]))

    def open_command_window(self, pid: int) -> None:
        """Reveal done, buying starts: count persuasion from here on."""
        p = self._p(pid)
        p.bl_reveal_persuasion = self.gs.persuasion_pool[pid]
        p.bl_command_fired = False
        self._check_command(pid)

    def on_persuasion(self, pid: int, amount: int) -> None:
        p = self._p(pid)
        if p.bl_reveal_persuasion is not None and amount > 0:
            p.bl_reveal_persuasion += amount
            self._check_command(pid)

    def _check_command(self, pid: int) -> None:
        p = self._p(pid)
        if p.bl_command_fired or (p.bl_reveal_persuasion or 0) < COMMAND_THRESHOLD:
            return
        p.bl_command_fired = True
        swords_before = self.gs.swords_this_reveal.get(pid, 0)
        cmds, p.bl_card_commands = p.bl_card_commands, []
        for eff in cmds:
            EffectResolver.resolve_single_effect(dict(eff), p, self.gs)
        for t in p.techs:
            self._resolve(pid, t.d.command)
            if t.name == "Disposal Facility" and p.in_play:
                self.pending.append(PendingBLChoice(
                    pid, "disposal",
                    sorted({c.name for c in p.in_play}) + ["decline"]))
        if self.gs.swords_this_reveal.get(pid, 0) != swords_before:
            self.gs.update_combat_strength(pid)

    def close_command_window(self, pid: int) -> None:
        p = self._p(pid)
        p.bl_reveal_persuasion = None
        p.bl_card_commands = []

    # ------------------------------------------------------------------
    # round / combat / misc triggers
    # ------------------------------------------------------------------
    def on_round_start(self) -> None:
        for p in self.gs.players:
            for t in list(p.techs):
                t.reset_for_round()
                self._resolve(p.id, t.d.round_start)

    def strength_bonus(self, pid: int) -> int:
        """Sword skills: only while >=1 commander is in the Conflict."""
        if self.commanders_in_conflict.get(pid, 0) <= 0:
            return 0
        gs, p = self.gs, self._p(pid)
        bonus = 0
        if "Canny" in p.skills and any(gs.agent_on_space.get(s) == pid
                                       for s in GREEN_SPACES):
            bonus += 2
        if "Fierce" in p.skills:
            bonus += 1
            if any(gs.sandworms_in_conflict.get(o.id, 0) > 0
                   for o in gs.players if o.id != pid):
                bonus += 1
        if "Loyal" in p.skills and p.influence["emperor"] >= 3:
            bonus += 2
        return bonus

    def on_win_conflict(self, pid: int) -> None:
        for t in list(self._p(pid).techs):
            self._resolve(pid, t.d.on_win_conflict)

    def on_complete_contract(self, pid: int) -> None:
        self._p(pid).bl_completed_contract = True
        for t in list(self._p(pid).techs):
            self._resolve(pid, t.d.on_complete_contract)

    def on_intrigue_drawn(self, pid: int, n: int) -> None:
        """Suspensor Suits: per Intrigue drawn during your own turn, a troop
        straight into the Conflict."""
        gs, p = self.gs, self._p(pid)
        if n <= 0 or not self.has(pid, "Suspensor Suits") or gs.current_conflict is None:
            return
        if gs.phase.value != "player_turns" or gs.get_current_player_id() != pid:
            return
        k = min(n, p.troops_supply)
        p.troops_supply -= k
        gs.troops_in_conflict[pid] = gs.troops_in_conflict.get(pid, 0) + k

    def battle_icon(self, pid: int, name: str) -> str:
        return "wild" if self.has(pid, "Ornithopter Fleet") else name

    def steal_protected(self, pid: int) -> bool:
        return self.has(pid, "Gene-Locked Vault") and len(self._p(pid).intrigue_cards) < 5

    def space_cost(self, pid: int, cost: Dict[str, int]) -> Dict[str, int]:
        """Navigation Chamber: board spaces cost 1 spice or 1 solari less."""
        if not cost or not self.has(pid, "Navigation Chamber"):
            return cost
        cost = dict(cost)
        for res in ("spice", "solari"):
            if cost.get(res, 0) > 0:
                cost[res] -= 1
                break
        return cost

    def signet_icons(self, pid: int) -> set:
        return set(FACTIONS) if self.has(pid, "Servo-Receivers") else set()

    def after_combat(self) -> None:
        """Before troops go home: commanders go to their owner's commander
        supply (not the troop supply)."""
        gs = self.gs
        for pid, c in self.commanders_in_conflict.items():
            if c > 0:
                gs.troops_in_conflict[pid] = max(0, gs.troops_in_conflict.get(pid, 0) - c)
                self._p(pid).commanders_supply += c
                self.commanders_in_conflict[pid] = 0

    def snapshot(self) -> Dict[int, tuple]:
        return {p.id: (p.troops_garrison, p.troops_supply) for p in self.gs.players}

    def sync(self, before: Optional[Dict[int, tuple]] = None) -> None:
        """Keep the commander subset within the troop counts after moves made
        by code that doesn't know about commanders (retreats, losses).
        `before` = snapshot() from before the action: units that left the
        Conflict went back to the garrison if it grew, else to the supply."""
        gs = self.gs
        for p in gs.players:
            done, p.bl_discarded = p.bl_discarded, []
            for card in done:                       # "when this is discarded"
                eff = getattr(card, "bl_on_discard", None)
                if eff:
                    EffectResolver.resolve_single_effect(dict(eff), p, gs)
        for p in gs.players:
            g0, _ = (before or {}).get(p.id, (p.troops_garrison, p.troops_supply))
            extra = self.commanders_in_conflict[p.id] - gs.troops_in_conflict.get(p.id, 0)
            if extra > 0:
                self.commanders_in_conflict[p.id] -= extra
                to_garrison = min(extra, max(0, p.troops_garrison - g0))
                p.commanders_garrison += to_garrison
                rest = extra - to_garrison          # retreated to troop supply
                p.troops_supply -= rest
                p.commanders_supply += rest
            extra = p.commanders_garrison - p.troops_garrison
            if extra > 0:              # lost from garrison (to troop supply)
                p.commanders_garrison -= extra
                p.troops_supply -= extra
                p.commanders_supply += extra


    # ------------------------------------------------------------------
    # Bloodlines card effects (keys documented in bloodlines/cards.py)
    # ------------------------------------------------------------------
    def cond(self, name: str, p: "Player") -> bool:
        gs = self.gs
        if name == "commander_in_conflict":
            return self.commanders_in_conflict.get(p.id, 0) > 0
        if name == "gained_spice_2":
            return p.spice_gained_this_turn >= 2
        if name == "completed_contract":
            return p.bl_completed_contract
        if name.startswith("techs_"):
            return len(p.techs) >= int(name[6:])
        if name == "garrison_4":
            return p.troops_garrison >= 4
        if name == "other_bene":
            from src.game.cards.card import CardTag
            return p.count_cards_with_tag_in_play(CardTag.BENE_GESSERIT) >= 2
        if name == "agent_on_emperor":
            from src.game.board.board import SPACE_ICONS
            return any(gs.agent_on_space.get(s) == p.id
                       for s, ic in SPACE_ICONS.items() if ic == "emperor")
        if name == "endgame_solari_10":
            return gs.game_over and p.solari >= 10
        if name == "not_endgame":
            return not gs.game_over
        if name == "endgame_techs_3":
            return gs.game_over and len(p.techs) >= 3
        if name == "endgame_water_3":
            return gs.game_over and p.water >= 3
        raise ValueError(f"unknown Bloodlines condition {name}")

    def option_legal(self, pid: int, opt: Dict) -> bool:
        p = self._p(pid)
        if "bl_retreat" in opt and \
                self.gs.troops_in_conflict.get(pid, 0) < opt["bl_retreat"]:
            return False
        if "discard_then" in opt and not p.hand:
            return False
        if "bl_trash_card" in opt and not any(
                c.name == opt["bl_trash_card"] for c in p.in_play + p.hand):
            return False
        conds = [k for k in opt if k.startswith("if_bl_")]
        if conds and len(opt) == 1:
            return self.cond(conds[0][6:], p)
        return True

    def resolve_effect(self, effect: Dict, p: "Player") -> None:
        """Called by EffectResolver for effect dicts with Bloodlines keys."""
        gs = self.gs
        for k, v in effect.items():
            if k.startswith("if_bl_"):
                if self.cond(k[6:], p):
                    EffectResolver.resolve_single_effect(dict(v), p, gs)
            elif k == "command":
                p.bl_card_commands.append(v)
            elif k == "bl_choose":
                opts = {f"opt:{i}": o for i, o in enumerate(v["options"])}
                both = v.get("both_if")
                if both and self.cond(both, p):
                    for o in opts.values():
                        EffectResolver.resolve_single_effect(dict(o), p, gs)
                    continue
                self.pending.append(PendingBLChoice(
                    p.id, "effect", list(opts), payloads=opts))
            elif k == "bl_trash_card":
                card = next((c for c in p.in_play + p.hand if c.name == v), None)
                if card is not None:
                    (p.in_play if card in p.in_play else p.hand).remove(card)
                    gs._to_trash(p, card)
                    for eff in getattr(card, "trash_effects", []):
                        EffectResolver.resolve_single_effect(eff, p, gs)
            elif k == "bl_retreat":
                n = min(int(v), gs.troops_in_conflict.get(p.id, 0))
                gs.troops_in_conflict[p.id] -= n
                p.troops_garrison += n
                self.retreat_to_garrison(p.id, n)
                gs.update_combat_strength(p.id)
            elif k == "bl_discard_for":
                gs.add_pending_optional_payment(
                    p.id, {}, v["reward"], discard=v.get("n", 1), label="bl_discard_for",
                    discard_tag=v.get("tag"), tag_bonus=v.get("bonus"))
            elif k == "bl_complete_contract":
                if p.contracts_active:
                    gs.complete_contract(p.id, p.contracts_active[0])
            elif k == "bl_force_retreat":
                self._force_retreat(p.id)
            elif k == "bl_trash_from_hand":
                if p.hand:
                    self.pending.append(PendingBLChoice(
                        p.id, "trash_hand",
                        sorted({c.name for c in p.hand}) + ["decline"],
                        bonus=v["bonus"], tag=v["tag"]))
            elif k == "bl_opponents_lose_troop":
                for o in gs.players:
                    if o.id == p.id:
                        continue
                    if o.troops_garrison > 0:
                        o.troops_garrison -= 1
                        o.troops_supply += 1
                    elif gs.troops_in_conflict.get(o.id, 0) > 0:
                        gs.troops_in_conflict[o.id] -= 1
                        o.troops_supply += 1
                        gs.update_combat_strength(o.id)
            elif k == "bl_after_turn":
                p.bl_after_turn.append(v)
            elif k == "bl_conflict_bonus":
                if gs.current_conflict is not None:
                    # Conflict cards are shared between game copies (search):
                    # change a private copy, never the shared card
                    import copy
                    gs.current_conflict = copy.copy(gs.current_conflict)
                    r = dict(gs.current_conflict.first_place_reward)
                    for rk, rv in v.items():
                        r[rk] = r.get(rk, 0) + rv
                    gs.current_conflict.first_place_reward = r
            elif k == "bl_free_commander":
                p.commanders_garrison += 1
                p.troops_garrison += 1
            elif k == "bl_recover_bene":             # Other Memory
                card = next((c for c in reversed(p.discard) if self._is_bene(c)), None)
                if card is not None:
                    p.discard.remove(card)
                    p.hand.append(card)
            elif k == "bl_reveal_persuasion":        # Recruitment Mission
                p.bl_reveal_bonus = getattr(p, "bl_reveal_bonus", 0) + int(v)
            elif k == "bl_topdeck_round":
                p.bl_topdeck = True
            elif k == "bl_envoy":                    # Dispatch an Envoy
                p.bl_envoy = True
            elif k == "bl_shigawire":
                p.bl_shigawire = True
            elif k == "bl_commander_discount":
                p.bl_commander_discount += int(v)
            elif k == "bl_tech_offer":
                stacks = self.buyable_stacks(p.id, int(v))
                if stacks:
                    self.pending.append(PendingBLChoice(
                        p.id, "tech_offer", [f"stack:{i}" for i in stacks] + ["decline"],
                        discount=int(v)))

    def _force_retreat(self, pid: int) -> None:
        """Force one enemy troop to retreat: from the strongest opponent."""
        gs = self.gs
        opps = [o.id for o in gs.players
                if o.id != pid and gs.troops_in_conflict.get(o.id, 0) > 0]
        if not opps:
            return
        tgt = max(opps, key=lambda q: gs.combat_strength.get(q, 0)
                  or gs.troops_in_conflict.get(q, 0) * 2)
        gs.troops_in_conflict[tgt] -= 1
        gs.players[tgt].troops_supply += 1
        gs.update_combat_strength(tgt)

    @staticmethod
    def _is_bene(card) -> bool:
        from src.game.cards.card import CardTag
        return card.has_tag(CardTag.BENE_GESSERIT)

    def extra_icons(self, pid: int, card) -> set:
        """Agent icons a card gains from Bloodlines effects."""
        p = self._p(pid)
        icons = set()
        if card.name == "Signet Ring":
            icons |= self.signet_icons(pid)                 # Servo-Receivers
        if p.bl_shigawire and self._is_bene(card):
            icons |= set(ALL_AGENT_ICONS)                   # Urgent Shigawire
        if card.name == "Delivery Logistics":
            icons |= self.contract_icons(pid)
        if getattr(p, "bl_envoy", False):
            icons |= set(FACTIONS)                          # Dispatch an Envoy
        return icons

    def contract_icons(self, pid: int) -> set:
        """Agent icons of the spaces your incomplete contracts point at."""
        from src.game.board.board import SPACE_ICONS
        from src.game.contract.contract import ContractType
        out = set()
        for c in self._p(pid).contracts_active:
            if c.contract_type == ContractType.BOARD_SPACE:
                sp = c.trigger_condition.get("board_space")
                if sp in SPACE_ICONS:
                    out.add(SPACE_ICONS[sp])
            elif c.contract_type == ContractType.HARVEST:
                out.add("desert")
        return out

    def reset_round(self) -> None:
        for p in self.gs.players:
            p.bl_shigawire = False
            p.bl_reveal_bonus = 0
            p.bl_topdeck = False

    # ------------------------------------------------------------------
    # endgame
    # ------------------------------------------------------------------
    def endgame(self) -> None:
        gs = self.gs
        for p in gs.players:
            names = {t.name for t in p.techs}
            if "Panopticon" in names:           # before influence-based scoring
                for f in FACTIONS:
                    if p.influence[f] <= 1:
                        gs.gain_influence_with_check(p.id, f, 1)
            if "CHOAM Transports" in names and len(p.contracts_completed) >= 4:
                p.gain_vp(1)
            if "Holtzman Engine" in names:
                tsmf = sum(1 for c in p.get_all_cards()
                           if c.name == "The Spice Must Flow") \
                    if hasattr(p, "get_all_cards") else sum(
                        1 for c in p.deck + p.hand + p.discard + p.in_play
                        if c.name == "The Spice Must Flow")
                if tsmf >= 2:
                    p.gain_vp(1)
            if "Memocorders" in names and all(p.influence[f] >= 3 for f in FACTIONS):
                p.gain_vp(1)
            if "Spy Satellites" in names:
                p.gain_vp(sum(1 for f in FACTIONS if p.influence[f] <= 1))

    def wins_ties(self, pid: int) -> bool:
        return self.has(pid, "Chaumurky")
