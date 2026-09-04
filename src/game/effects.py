"""
Effect resolution system for Dune Imperium Uprising.

Mandatory vs optional (FAQ):
  Effects are mandatory unless the word "may" appears, there is a cost→reward
  arrow, or the black-X trash icon is used.  The mandatory/optional distinction
  is enforced at the action-space level (GameState.get_valid_actions); this
  resolver executes every effect it is given.

  "You cannot execute & fizzle a conditional effect while the precondition is
  false." Conditionals (if_hooks, if_tag_*, etc.) are evaluated at resolution
  time.  For end-of-turn conditionals (e.g. Prepare the Way), the RL env must
  re-check conditions before ending the turn.

Spy placement (Uprising errata):
  Spy placement is MANDATORY when the icon appears and supply > 0.  This
  resolver queues a PendingSpyPlacement; GameState enforces that pending
  spy placements must be resolved before other actions.

Sandworm card effects (FAQ):
  Card effects that deploy sandworms (e.g. Unexpected Allies, Arrakis Revolt)
  do NOT require Maker Hooks — the Maker Hooks requirement applies only to
  the board-space mechanic (visiting Hagga Basin / Deep Desert / Sietch Tabr).
  Use the "sandworm" key for card-granted sandworms (no hooks needed).
  Use "sandworm_maker_space" key when the board-space hooks requirement applies.

Card draw:
  Always routed through GameState.draw_cards_for_player() to guarantee use of
  the game's seeded RNG.  Player.draw_cards() must NOT be called from here.
"""
from typing import Dict, List, Optional
from src.game.cards.card import CardTag


class EffectResolver:
    """Resolves effect dicts produced by cards and board spaces."""

    @staticmethod
    def resolve_agent_effects(card: 'Card', player: 'Player', game_state: 'GameState') -> None:
        for effect in card.agent_effects:
            game_state._self_ref_pending = []
            EffectResolver.resolve_single_effect(effect, player, game_state)
            EffectResolver._resolve_self_referential(effect, card, player, game_state)

    @staticmethod
    def _discard_worst(player) -> None:
        """Discard the least useful card from hand (approximation for 'discard a card')."""
        if not player.hand:
            return
        weak = ("Reconnaissance", "Diplomacy", "Dune, the Desert Planet", "Dagger",
                "Convincing Argument")
        tgt = next((c for c in player.hand if c.name in weak), None)
        if tgt is None:
            tgt = min(player.hand,
                      key=lambda c: c.persuasion + c.swords
                      + len(getattr(c, "agent_effects", [])))
        player.hand.remove(tgt)
        player.discard.append(tgt)

    @staticmethod
    def _resolve_self_referential(effect: Dict, card, player, game_state) -> None:
        """
        Handle effect keys that act on the card being played itself.  These are
        collected during `resolve_single_effect` (which has no card ref) via
        `game_state._self_ref_pending`, so they fire correctly even when nested
        inside a conditional (e.g. Weirding Woman, Treacherous Maneuver).
        """
        reqs = set(getattr(game_state, "_self_ref_pending", []))
        if effect.get("trash_self"):
            reqs.add("trash_self")
        if effect.get("return_self_to_hand"):
            reqs.add("return_self_to_hand")
        game_state._self_ref_pending = []

        if "trash_self" in reqs:
            for pile in (player.in_play, player.hand, player.discard):
                if card in pile:
                    pile.remove(card)
                    player.trash.append(card)
                    for eff in getattr(card, "trash_effects", []):
                        EffectResolver.resolve_single_effect(eff, player, game_state)
                    break
        if "return_self_to_hand" in reqs and card in player.in_play:
            player.in_play.remove(card)
            player.hand.append(card)

    @staticmethod
    def resolve_reveal_effects(card: 'Card', player: 'Player', game_state: 'GameState') -> None:
        # Reveal resources (direct gains listed on the card face)
        for resource, amount in card.reveal_resources.items():
            if resource == "solari":
                player.gain_solari(amount)
            elif resource == "spice":
                player.gain_spice(amount)
            elif resource == "water":
                player.gain_water(amount)

        # Persuasion accumulates into the GameState persuasion pool for the
        # current player; it is NOT added directly to the player's resources.
        if hasattr(card, "persuasion") and card.persuasion:
            game_state.gain_persuasion(player.id, card.persuasion)

        # Other reveal effects.  Reveal rewards may be resolved in any order
        # (rulebook), so we resolve every UNCONDITIONAL part now and DEFER the
        # conditional parts (if_*) until the player has resolved any pending
        # choices this reveal triggered (e.g. placing a Spy first, so that a
        # "2+ Spies" condition then sees the new count).
        for effect in card.reveal_effects:
            plain = {k: v for k, v in effect.items() if not k.startswith("if_")}
            conds = {k: v for k, v in effect.items() if k.startswith("if_")}
            if plain:
                EffectResolver.resolve_single_effect(plain, player, game_state)
            EffectResolver._resolve_self_referential(effect, card, player, game_state)
            if conds:
                game_state._pending_reveal_conditionals.append((conds, player.id))

    @staticmethod
    def resolve_single_effect(effect: Dict, player: 'Player', game_state: 'GameState') -> None:
        """
        Execute a single effect dict.

        Automatic (mandatory) effects fire immediately.
        Optional effects (keys starting with "may_") do NOT resolve here —
        they surface as available actions in GameState.get_valid_actions().
        Pending choices (spy placement, uplift, intrigue trash) are queued and
        must be resolved by the RL agent before any other action.

        Effect keys:
          solari, spice, water, vp                  — direct resource gains
          troops                                     — recruit from supply to garrison
          draw                                       — draw cards (seeded RNG via GameState)
          intrigue                                   — draw intrigue cards
          influence_<faction>                        — gain faction influence
          spy                                        — queue mandatory spy placement (normal)
          spy_special                                — queue mandatory spy placement (occupied OK)
          uplift                                     — queue agent uplift choice
          sandworm                                   — deploy sandworm from card effect (no hooks needed)
          sandworm_maker_space                       — deploy sandworm via board space (hooks required)
          spice_from_sandworm                        — take spice from sandworm card token
          maker_hooks                                — grant Maker Hooks to player
          trash_intrigue_for                         — queue intrigue-trash-for-benefit choice
          if_hooks, if_tag_*, if_tag_*_count_*      — conditional sub-effects
          may_convert, may_trash                     — optional; surfaced as actions, NOT executed
          persuasion                                 — add to per-turn persuasion pool
          swords                                     — add to per-turn swords pool (reveal only)
        """
        # ===== SELF-REFERENTIAL requests (drained later with the card ref) =====
        # Recorded here so they fire even when nested in a conditional that
        # only resolved because its precondition was true.
        _sr = getattr(game_state, "_self_ref_pending", None)
        if _sr is not None:
            if effect.get("trash_self"):
                _sr.append("trash_self")
            if effect.get("return_self_to_hand"):
                _sr.append("return_self_to_hand")

        # ===== BASIC RESOURCES =====
        if "solari" in effect:
            player.gain_solari(effect["solari"])

        if "spice" in effect:
            player.gain_spice(effect["spice"])

        if "water" in effect:
            player.gain_water(effect["water"])

        if "vp" in effect:
            player.gain_vp(effect["vp"])

        # ===== PERSUASION (reveal pool, not a player resource) =====
        if "persuasion" in effect:
            game_state.gain_persuasion(player.id, effect["persuasion"])

        # ===== SWORDS (add to reveal swords pool) =====
        if "swords" in effect:
            game_state.swords_this_reveal[player.id] = (
                game_state.swords_this_reveal.get(player.id, 0) + effect["swords"]
            )

        # ===== TROOPS =====
        if "troops" in effect:
            _before = player.troops_garrison
            player.gain_troops(effect["troops"])
            if getattr(game_state, "_agent_turn_active", False):
                player.troops_recruited_this_turn += player.troops_garrison - _before

        # ===== TRASH A CARD (optional; surfaced as a pending choice) =====
        if "trash" in effect:
            game_state.add_pending_trash(player.id, effect["trash"])

        # ===== TRASH A CARD -> REWARD IF YOU DO (Shishakli: trash -> draw) =====
        if "trash_then" in effect:
            game_state.add_pending_trash(player.id, 1, on_trash=effect["trash_then"])

        # ===== GAIN INFLUENCE WITH ANY FACTION (player choice) =====
        if "influence_any" in effect:
            game_state.add_pending_influence_choice(player.id, effect["influence_any"])

        # ===== CARD DRAW — always through seeded GameState RNG (Bug 7 fix) =====
        if "draw" in effect:
            game_state.draw_cards_for_player(player.id, effect["draw"])

        # ===== INTRIGUE =====
        if "intrigue" in effect:
            game_state.draw_intrigue_for_player(player.id, effect["intrigue"])

        # ===== INFLUENCE =====
        if "influence_emperor" in effect:
            game_state.gain_influence_with_check(player.id, "emperor", effect["influence_emperor"])

        if "influence_spacing_guild" in effect:
            game_state.gain_influence_with_check(player.id, "spacing_guild", effect["influence_spacing_guild"])

        if "influence_bene_gesserit" in effect:
            game_state.gain_influence_with_check(player.id, "bene_gesserit", effect["influence_bene_gesserit"])

        if "influence_fremen" in effect:
            game_state.gain_influence_with_check(player.id, "fremen", effect["influence_fremen"])

        # ===== SPY PLACEMENT (mandatory per Uprising errata) =====
        if "spy" in effect:
            game_state.add_pending_spy_placement(player.id, effect["spy"], allow_occupied=False)

        if "spy_special" in effect:
            game_state.add_pending_spy_placement(player.id, effect["spy_special"], allow_occupied=True)

        # Restricted spy placement (Reliable Informant: faction posts only).
        if "spy_posts" in effect:
            spec = effect["spy_posts"]
            game_state.add_pending_spy_placement(
                player.id, int(spec.get("count", 1)), allow_occupied=False,
                allowed_posts=spec.get("posts"))

        # ===== UPLIFT =====
        if "uplift" in effect:
            game_state.add_pending_uplift(player.id, effect["uplift"])

        # =================================================================
        # Intrigue-card effect vocabulary (approximate; see intrigue notes)
        # =================================================================
        gs = game_state
        cc = gs.current_conflict
        _FACS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")

        def _best_gain_faction():
            return (next((f for f in _FACS if player.influence[f] == 3), None)
                    or next((f for f in _FACS if player.influence[f] == 1), None)
                    or min(_FACS, key=lambda f: player.influence[f]))

        def _safe_lose_faction(exclude=()):
            # a Faction we can lose 1 from without dropping under a threshold
            for f in _FACS:
                if f in exclude:
                    continue
                v = player.influence[f]
                if v in (1, 3) or v >= 5:
                    return f
            return None

        def _units_in_conflict():
            return (gs.troops_in_conflict.get(player.id, 0)
                    + gs.sandworms_in_conflict.get(player.id, 0))

        def _losing_combat():
            mine = gs.combat_strength.get(player.id, 0)
            return any(gs.combat_strength.get(q, 0) > mine
                       for q in range(gs.num_players) if q != player.id)

        # -- gain influence with a choice of Faction --------------------
        if "influence_choice" in effect:
            for _ in range(int(effect["influence_choice"])):
                gs.gain_influence_with_check(player.id, _best_gain_faction(), 1)

        # -- gain influence, choosing between a RESTRICTED pair of Factions --
        def _pick_from(cands):
            return (next((f for f in cands if player.influence[f] == 3), None)
                    or next((f for f in cands if player.influence[f] == 1), None)
                    or min(cands, key=lambda f: player.influence[f]))

        if "influence_bene_or_fremen" in effect:               # Sietch Ritual
            for _ in range(int(effect["influence_bene_or_fremen"])):
                gs.gain_influence_with_check(
                    player.id, _pick_from(("bene_gesserit", "fremen")), 1)

        if "influence_emperor_or_spacing" in effect:           # Imperium Politics
            for _ in range(int(effect["influence_emperor_or_spacing"])):
                gs.gain_influence_with_check(
                    player.id, _pick_from(("emperor", "spacing_guild")), 1)

        # -- Call to Arms: troops per card acquired this Reveal turn ----
        if "troops_per_card_acquired_this_turn" in effect:
            k = getattr(player, "cards_acquired_this_turn", 0)
            if k:
                EffectResolver.resolve_single_effect(
                    {"troops": k * int(effect["troops_per_card_acquired_this_turn"])},
                    player, gs)

        # -- lose influence from a safe Faction ------------------------
        if "lose_influence_any" in effect:
            for _ in range(int(effect["lose_influence_any"])):
                f = _safe_lose_faction() or next(
                    (x for x in _FACS if player.influence[x] > 0), None)
                if f:
                    gs.lose_influence_with_check(player.id, f, 1)

        # -- pay a cost, then gain rewards.  Every cost->reward arrow is a MAY:
        #    it is surfaced as an accept/decline choice, never auto-paid. -----
        if "pay_then" in effect:
            spec = effect["pay_then"]
            rew = {k: v for k, v in spec.items() if k != "cost"}
            gs.add_pending_optional_payment(player.id, spec.get("cost", {}), rew,
                                            label="pay_then")

        # -- MAY recall a placed Spy, then gain rewards (an arrow) ----
        if "recall_spy_then" in effect:
            gs.add_pending_optional_payment(
                player.id, {"recall_spy": 1}, effect["recall_spy_then"],
                label="recall_spy_then")

        if "recall_spies_swords" in effect:
            spec = effect["recall_spies_swords"]
            posts = list(player.spies_on_board)
            n = min(spec.get("count", 2), len(posts))
            for post in posts[:n]:
                player.recall_spy(post)
            player.recalled_spy_this_turn = n > 0
            if n >= spec.get("count", 2):
                gs.swords_this_reveal[player.id] = gs.swords_this_reveal.get(player.id, 0) \
                    + spec.get("swords", 0)

        # -- retreat troops from the Conflict (only when it helps) ----
        if "retreat" in effect and _units_in_conflict() and _losing_combat():
            n = min(int(effect["retreat"]), gs.troops_in_conflict.get(player.id, 0))
            gs.troops_in_conflict[player.id] -= n
            player.troops_garrison += n

        # -- retreat min (or max, if losing) troops -> gain a reward ---------
        # (Go to Ground -> Spy, Reach Agreement -> contract, Spice is Power
        #  option A -> 3 spice.)  `only_if_losing` skips it in a winning fight.
        if "retreat_for" in effect:
            spec = effect["retreat_for"]
            in_conf = gs.troops_in_conflict.get(player.id, 0)
            lo = int(spec.get("min", 1))
            if in_conf >= lo and not (spec.get("only_if_losing") and not _losing_combat()):
                n = min(int(spec.get("max", lo)) if _losing_combat() else lo, in_conf)
                gs.troops_in_conflict[player.id] -= n
                player.troops_garrison += n
                EffectResolver.resolve_single_effect(spec.get("reward", {}), player, gs)

        # -- Chani reveal: MAY retreat 2 troops from the Conflict for 4 swords —
        #    only when a unit (troop or worm) would still remain, otherwise the
        #    swords would fizzle (0 units => 0 combat strength from swords). ---
        if "chani_retreat" in effect:
            in_conf = gs.troops_in_conflict.get(player.id, 0)
            worms = gs.sandworms_in_conflict.get(player.id, 0)
            if in_conf >= 2 and (in_conf - 2 + worms) >= 1:
                gs.troops_in_conflict[player.id] -= 2
                player.troops_garrison += 2
                gs.swords_this_reveal[player.id] = \
                    gs.swords_this_reveal.get(player.id, 0) + 4

        # -- Tactical Option: 2 swords, OR pull ALL your troops out of a
        #    hopeless Conflict. ---------------------------------------------
        if "tactical_option" in effect:
            if _losing_combat() and gs.troops_in_conflict.get(player.id, 0) > 0:
                n = gs.troops_in_conflict.get(player.id, 0)
                gs.troops_in_conflict[player.id] = 0
                player.troops_garrison += n
            else:
                gs.swords_this_reveal[player.id] = \
                    gs.swords_this_reveal.get(player.id, 0) + 2

        # -- Crysknife / Desert Mouse / Ornithopter intrigue: 1 spice when
        #    played as a Plot; at Endgame, match one held icon of its type
        #    for 1 VP instead. --------------------------------------------
        if "spice_or_match_icon" in effect:
            spec = effect["spice_or_match_icon"]
            if getattr(gs, "game_over", False):
                icon = spec["icon"]
                if icon in player.battle_icons:
                    player.battle_icons.remove(icon)
                    player.gain_vp(1)
            else:
                player.gain_spice(int(spec.get("spice", 1)))

        # -- Grasp Arrakis Endgame: 1 VP if you hold 2+ unmatched real icons --
        if "grasp_arrakis_endgame" in effect and getattr(gs, "game_over", False):
            if sum(1 for i in player.battle_icons if i != "wild") >= 2:
                player.gain_vp(1)

        # -- swords per Faction friendship (2+ influence) -------------
        if "swords_per_friendship" in effect:
            k = sum(1 for f in _FACS if player.influence[f] >= 2)
            gs.swords_this_reveal[player.id] = gs.swords_this_reveal.get(player.id, 0) \
                + k * int(effect["swords_per_friendship"])

        # -- acquire a cheap card from the Row for free --------------
        # Inspire Awe: if you have a sandworm in the Conflict, the acquired
        # card goes straight to hand instead of the discard pile.
        if "acquire_free" in effect:
            spec = effect["acquire_free"]
            mx = spec.get("max_cost", 3)
            opts = [c for c in gs.imperium_row if c.cost <= mx]
            if opts:
                best = max(opts, key=lambda c: c.persuasion + c.swords + c.cost * 0.3)
                gs.imperium_row.remove(best)
                to_hand = (spec.get("to_hand_if_sandworm")
                           and gs.sandworms_in_conflict.get(player.id, 0) > 0)
                (player.hand if to_hand else player.discard).append(best)
                gs.refill_imperium_row()

        # -- detonate the Shield Wall (unconditional) ---------------
        if "break_shield_wall" in effect:
            gs.shield_wall_intact = False

        # -- look at top of deck: draw it, or trash if it's a weak starter --
        if "peek_top" in effect:
            if not player.deck and player.discard:
                nd = list(player.discard); gs.rng.shuffle(nd)
                player.deck, player.discard = nd, []
            if player.deck:
                top = player.deck[-1]
                weak = ("Reconnaissance", "Diplomacy", "Dune, the Desert Planet",
                        "Dagger", "Convincing Argument")
                if top.name in weak:
                    player.trash.append(player.deck.pop())
                else:
                    player.hand.append(player.deck.pop())

        # -- churn the Imperium Row (remove weakest, refill) --------
        if "refresh_imperium_row" in effect and gs.imperium_row:
            worst = min(gs.imperium_row,
                        key=lambda c: c.persuasion + c.swords + c.cost * 0.3)
            gs.imperium_row.remove(worst)
            gs.imperium_deck.append(worst)
            gs.refill_imperium_row()

        # -- Manipulate: set aside the BEST Row card for yourself at a
        #    discount (only you may buy it, this round), then refill. --------
        if "manipulate" in effect and gs.imperium_row:
            best = max(gs.imperium_row,
                       key=lambda c: c.persuasion + c.swords + c.cost * 0.3)
            gs.imperium_row.remove(best)
            player.reserved_card = best
            player.reserved_discount = int(effect["manipulate"])
            gs.refill_imperium_row()

        # -- Market Opportunity: spice<->solari, whichever helps ----
        # Market Opportunity: EITHER 5 solari -> 5 spice, OR 2 spice -> 5 solari.
        # Auto-convert toward the resource the player currently has less of.
        if "market_convert" in effect:
            can_solari = player.solari >= 5        # -> 5 spice
            can_spice  = player.spice >= 2         # -> 5 solari
            if can_solari and can_spice:
                if player.spice <= player.solari:
                    player.solari -= 5; player.gain_spice(5)
                else:
                    player.spice -= 2; player.gain_solari(5)
            elif can_solari:
                player.solari -= 5; player.gain_spice(5)
            elif can_spice:
                player.spice -= 2; player.gain_solari(5)

        # -- Unexpected Allies: pay water -> sandworm + deploy troops --
        if "water_summon" in effect and cc is not None:
            spec = effect["water_summon"]
            if player.water >= spec.get("cost", 2):
                player.water -= spec.get("cost", 2)
                gs.spawn_sandworms(spec.get("sandworm", 1), source="card_effect")
                d = min(spec.get("deploy", 0), player.troops_garrison)
                player.troops_garrison -= d
                gs.troops_in_conflict[player.id] = gs.troops_in_conflict.get(player.id, 0) + d

        # -- ENDGAME: flip face-up Conflict cards for VP -------------
        # (Only meaningful during Endgame scoring — the plot half of a dual
        #  Plot/Endgame card does nothing here.)
        if "flip_conflicts_vp" in effect and getattr(gs, "game_over", False):
            spec = effect["flip_conflicts_vp"]
            icon = spec.get("icon")
            from src.game.combat.conflict import BattleIcon as _BI
            cards = [c for c in gs.won_conflicts.get(player.id, [])
                     if id(c) not in gs._flipped_conflicts and c.battle_icon is not None
                     and (icon is None or c.battle_icon.value == icon
                          or c.battle_icon == _BI.WILD)]
            need = spec.get("count", 1)
            if len(cards) >= need:
                for c in cards[:need]:
                    gs._flipped_conflicts.add(id(c))
                player.gain_vp(spec.get("vp", 1))

        # -- conditional wrappers unique to intrigues ----------------
        if "if_sandworm_in_conflict" in effect:
            if gs.sandworms_in_conflict.get(player.id, 0) > 0:
                EffectResolver.resolve_single_effect(
                    effect["if_sandworm_in_conflict"], player, gs)
        if "if_units_in_conflict" in effect:      # bare: 1+ unit
            if _units_in_conflict() > 0:
                EffectResolver.resolve_single_effect(
                    effect["if_units_in_conflict"], player, gs)
        if "if_tsmf_2" in effect:
            n = sum(1 for pile in (player.deck, player.hand, player.discard, player.in_play)
                    for c in pile if c.name == "The Spice Must Flow")
            if n >= 2:
                EffectResolver.resolve_single_effect(effect["if_tsmf_2"], player, gs)
        if "if_shadow_alliance" in effect:
            hit = any(player.influence[f] >= 4 and gs.alliance_holder.get(f) not in (None, player.id)
                      for f in _FACS)
            if hit:
                EffectResolver.resolve_single_effect(effect["if_shadow_alliance"], player, gs)

        # -- Opportunism: lose 2 Influence (any factions, may be the same) +
        #    pay 2 solari -> 1 VP.  Sheds from the highest track(s) first.
        if "opportunism_vp" in effect and player.solari >= 2:
            if sum(player.influence[f] for f in _FACS) >= 2:
                for _ in range(2):
                    f = max(_FACS, key=lambda x: player.influence[x])
                    if player.influence[f] > 0:
                        gs.lose_influence_with_check(player.id, f, 1)
                player.solari -= 2
                player.gain_vp(1)

        # -- pick a sub-effect based on whether you're in combat -----
        if "choose_by_combat" in effect:
            spec = effect["choose_by_combat"]
            key = "combat" if _units_in_conflict() > 0 and gs.phase.name == "COMBAT" \
                else "else"
            if spec.get(key):
                EffectResolver.resolve_single_effect(spec[key], player, gs)

        # -- pick a sub-effect based on completed-contract count -----
        # (Delivery Agreement reveal: 1 VP at 4+ contracts, else 1 spice.)
        if "choose_by_contracts" in effect:
            spec = effect["choose_by_contracts"]
            n = int(spec.get("n", 4))
            branch = "yes" if len(player.contracts_completed) >= n else "no"
            if spec.get(branch):
                EffectResolver.resolve_single_effect(spec[branch], player, gs)

        # -- Interstellar Trade reveal: 1 persuasion per completed contract --
        if "persuasion_per_contract" in effect:
            k = len(player.contracts_completed)
            if k:
                gs.gain_persuasion(player.id, k * int(effect["persuasion_per_contract"]))

        # -- Desert Power reveal: 2 persuasion, OR (with Maker Hooks) pay 1
        #    water to summon a sandworm into the current Conflict.  The worm
        #    needs the Shield Wall down (or a Conflict not behind it); the
        #    worm is auto-taken only when already committed to that Conflict.
        if "worm_or_persuasion" in effect:
            spec = effect["worm_or_persuasion"]
            took_worm = False
            if (cc is not None and player.has_maker_hooks
                    and player.water >= spec.get("water", 1)):
                from src.game.board.board import SHIELD_WALL_PROTECTED as _SWP
                deployable = not (gs.shield_wall_intact and cc.location in _SWP)
                committed = (gs.troops_in_conflict.get(player.id, 0) > 0
                             or gs.sandworms_in_conflict.get(player.id, 0) > 0)
                if deployable and committed:
                    player.water -= spec.get("water", 1)
                    gs.spawn_sandworms(spec.get("sandworm", 1), source="card_effect")
                    took_worm = True
            if not took_worm:
                gs.gain_persuasion(player.id, spec.get("persuasion", 2))

        # -- In High Places reveal: MAY recall N Spies for +M persuasion ----
        if "recall_spies_persuasion" in effect:
            spec = effect["recall_spies_persuasion"]
            posts = list(player.spies_on_board)
            need = int(spec.get("count", 2))
            if len(posts) >= need:
                for post in posts[:need]:
                    player.recall_spy(post)
                player.recalled_spy_this_turn = True
                gs.gain_persuasion(player.id, int(spec.get("persuasion", 0)))

        # -- Leadership agent: draw 1 per sandworm you have in the Conflict --
        if "draw_per_sandworm_in_conflict" in effect:
            k = gs.sandworms_in_conflict.get(player.id, 0)
            if k:
                gs.draw_cards_for_player(player.id, k * int(effect["draw_per_sandworm_in_conflict"]))

        # -- Leadership reveal: +1 sword per OTHER card revealed this turn --
        if "swords_per_other_revealed_card" in effect:
            k = max(0, getattr(player, "_revealed_this_turn", 0) - 1)
            if k:
                gs.swords_this_reveal[player.id] = gs.swords_this_reveal.get(player.id, 0) \
                    + k * int(effect["swords_per_other_revealed_card"])

        # -- Sardaukar Coordination reveal: +1 sword per Emperor card in play
        #    (only if you have another Emperor card besides this one) ---------
        if "swords_per_emperor_card" in effect:
            k = player.count_cards_with_tag_in_play(CardTag.EMPEROR)
            if k >= 2:
                gs.swords_this_reveal[player.id] = gs.swords_this_reveal.get(player.id, 0) \
                    + k * int(effect["swords_per_emperor_card"])

        # -- Overthrow agent: gain N influence with the Faction you visited --
        if "influence_faction_visited" in effect:
            fac = gs._faction_for_space(getattr(gs, "_current_agent_space", None))
            if fac:
                gs.gain_influence_with_check(player.id, fac, int(effect["influence_faction_visited"]))

        # -- The Beast's Spoils agent: one reward per DISTINCT face-up battle
        #    icon among the Conflict cards this player has won ----------------
        if "beast_spoils" in effect:
            from src.game.combat.conflict import BattleIcon as _BI
            icons = set()
            for c in gs.won_conflicts.get(player.id, []):
                if id(c) in gs._flipped_conflicts:
                    continue
                bi = getattr(c, "battle_icon", None)
                if bi is None:
                    continue
                if bi == _BI.WILD:
                    icons |= {_BI.DESERT_MOUSE, _BI.CRYSKNIFE, _BI.ORNITHOPTER}
                else:
                    icons.add(bi)
            if _BI.DESERT_MOUSE in icons:
                player.gain_spice(1)
            if _BI.ORNITHOPTER in icons:
                EffectResolver.resolve_single_effect({"troops": 1}, player, gs)
            if _BI.CRYSKNIFE in icons:
                gs.add_pending_trash(player.id, 1)

        # -- Treacherous Maneuver agent: trash this card + another Emperor card
        #    in play -> gain 1 influence with the Faction you visited ---------
        if "trash_pair_emperor_influence" in effect:
            tm = next((c for c in player.in_play
                       if c.name == "Treacherous Maneuver"), None)
            others = [c for c in player.in_play
                      if c.has_tag(CardTag.EMPEROR) and c is not tm]
            if tm is not None and others:
                others.sort(key=lambda c: c.persuasion + c.swords
                            + len(getattr(c, "agent_effects", [])))
                for c in (tm, others[0]):
                    player.in_play.remove(c)
                    player.trash.append(c)
                    for eff in getattr(c, "trash_effects", []):
                        EffectResolver.resolve_single_effect(eff, player, gs)
                fac = gs._faction_for_space(getattr(gs, "_current_agent_space", None))
                if fac:
                    gs.gain_influence_with_check(player.id, fac, 1)

        # -- Undercover Asset reveal: choose 1 Spy OR N swords --------------
        if "spy_or_swords" in effect:
            spec = effect["spy_or_swords"]
            contesting = (gs.troops_in_conflict.get(player.id, 0) > 0
                          or gs.sandworms_in_conflict.get(player.id, 0) > 0)
            if contesting:
                gs.swords_this_reveal[player.id] = gs.swords_this_reveal.get(player.id, 0) \
                    + int(spec.get("swords", 2))
            else:
                gs.add_pending_spy_placement(player.id, int(spec.get("spy", 1)),
                                             allow_occupied=False)

        # -- Unswerving Loyalty reveal (Fremen bond): deploy 1 OR retreat 1 --
        if "deploy_or_retreat" in effect and cc is not None:
            n = int(effect["deploy_or_retreat"])
            in_conf = gs.troops_in_conflict.get(player.id, 0)
            if in_conf > 0 and _losing_combat():
                k = min(n, in_conf)
                gs.troops_in_conflict[player.id] -= k
                player.troops_garrison += k
            elif player.troops_garrison > 0:
                k = min(n, player.troops_garrison)
                player.troops_garrison -= k
                gs.troops_in_conflict[player.id] = in_conf + k

        # -- Long Live the Fighters agent: look at top 3, draw 1 / discard 1
        #    / trash 1 (all mandatory).  Auto: keep the strongest, trash the
        #    weakest starter-ish card, discard the middle. ------------------
        if "look_top3_draw_discard_trash" in effect:
            if len(player.deck) < 3 and player.discard:
                nd = list(player.discard); gs.rng.shuffle(nd)
                player.deck = nd + player.deck
                player.discard = []
            top = [player.deck.pop() for _ in range(min(3, len(player.deck)))]
            if top:
                def _val(c):
                    return c.persuasion + c.swords + len(getattr(c, "agent_effects", [])) \
                        + len(getattr(c, "access_symbols", []))
                top.sort(key=_val, reverse=True)
                player.hand.append(top[0])                       # draw the best
                if len(top) >= 3:
                    player.discard.append(top[1])                # discard the middle
                    gs._to_trash(player, top[2])                 # trash the worst
                elif len(top) == 2:
                    gs._to_trash(player, top[1])

        # -- Sardaukar Coordination agent: MAY deploy troops you recruited this
        #    turn to the Conflict (any amount, no cap beyond what you recruited) -
        if "deploy_recruited" in effect:
            cap = min(getattr(player, "troops_recruited_this_turn", 0),
                      player.troops_garrison)
            if not any(d.player_id == player.id for d in gs.pending_deployments):
                gs.add_pending_deployment(player.id, cap)

        # -- Adaptive Tactics: grant a deploy action.  The per-turn deploy budget
        #    is 2 (once) + every troop recruited this turn + `extra` (this card's
        #    own troop, which isn't counted as "recruited").  Icons don't stack. --
        if "grant_deploy" in effect and gs.current_conflict is not None:
            extra = int(effect["grant_deploy"])
            player.deploy_budget_this_turn = max(
                getattr(player, "deploy_budget_this_turn", 0),
                2 + getattr(player, "troops_recruited_this_turn", 0) + extra)
            left = player.deploy_budget_this_turn - getattr(player, "deployed_this_turn", 0)
            if left > 0 and not any(d.player_id == player.id
                                    for d in gs.pending_deployments):
                from src.game.gameState import PendingDeployment as _PD
                gs.pending_deployments.append(_PD(player.id, player.troops_garrison))

        # -- Special Mission: either place a Spy on a City observation post, OR
        #    (if you have Spies to spare) recall one for 2 spice + optional
        #    wall-break.  Approximated: recall when you have 2+ Spies out. --------
        if "special_mission" in effect:
            if sum(player.spies_on_board.values()) >= 2:
                gs.add_pending_optional_payment(
                    player.id, {"recall_spy": 1},
                    {"spice": 2, "may_break_shield_wall": 1}, label="special_mission")
            else:
                gs.add_pending_spy_placement(
                    player.id, 1, allow_occupied=False,
                    allowed_posts=["Arrakeen Post", "Research Station Left Post",
                                   "Research Station Right Post"])

        # -- ignore a board space's Influence requirement this turn ---------
        # (Undercover Asset agent box, Insider Information intrigue.)
        if "ignore_influence_gates" in effect:
            player.ignore_influence_gates_this_turn = True

        # -- False Orders: spy on a post bordering a space you have an Agent on;
        #    bounce every OTHER player's Spy there (they must re-place it).
        #    You may not target a post you already occupy. -------------------
        if "false_orders" in effect and player.spies_available > 0:
            from src.game.board.board import SPACE_TO_OBSERVATION_POSTS as _S2P
            my_spaces = [s for s, o in gs.agent_on_space.items() if o == player.id]
            cands = set()
            for s in my_spaces:
                cands |= _S2P.get(s, set())
            cands = [pp for pp in cands if not player.has_spy_at(pp)]
            if cands:
                target = max(cands, key=lambda pp: sum(
                    1 for q in gs.players
                    if q.id != player.id and q.has_spy_at(pp)))
                for q in gs.players:
                    if q.id != player.id and q.has_spy_at(target):
                        q.recall_spy(target)
                        gs.add_pending_spy_placement(q.id, 1, allow_occupied=False)
                player.place_spy(target, allow_occupied=False)

        # -- Emperor's Invitation: draw a card, OR let the card you play this
        #    round reach an Emperor space regardless of its icons.  Auto: take
        #    the access only when you can't otherwise reach an Emperor space. --
        if "emperor_access_or_draw" in effect:
            from src.game.cards.card import AccessSymbol as _AS
            has_emp = any(_AS.EMPEROR in c.access_symbols for c in player.hand)
            if not has_emp and player.agents_available > 0:
                player.grant_emperor_access_this_turn = True
            else:
                gs.draw_cards_for_player(player.id, 1)

        # -- Coercive Negotiation: look at the top 3 reserve contracts and take
        #    the best one (only with 3+ units in the Conflict — gated by the card).
        if "take_contract_from_reserve" in effect and getattr(gs, "use_choam", False):
            n = int(effect["take_contract_from_reserve"])
            pool = gs.contract_bank[:n]
            if pool:
                def _cval(ct):
                    base = sum(v for v in ct.rewards.values()
                               if isinstance(v, (int, float)))
                    return base + (6 if "vp" in ct.rewards else 0)
                best = max(pool, key=_cval)
                gs.contract_bank.remove(best)
                player.take_contract(best)
                if best.is_immediate() and (
                        not best.requires_intrigue() or player.intrigue_cards):
                    gs.complete_contract(player.id, best)

        # -- Smuggler's Haven reveal: +2 spice if a Spy sits on a post that
        #    borders a Maker board space (Imperial Basin / Hagga / Deep Desert) -
        if "if_spy_at_maker_post" in effect:
            from src.game.board.board import (MAKER_SPACES as _MK,
                                              SPACE_TO_OBSERVATION_POSTS as _S2P)
            maker_posts = set()
            for _s in _MK:
                maker_posts |= _S2P.get(_s, set())
            if any(player.has_spy_at(pp) for pp in maker_posts):
                EffectResolver.resolve_single_effect(
                    effect["if_spy_at_maker_post"], player, gs)

        # -- Price is No Object agent: MAY acquire a card by paying Solari equal
        #    to its cost instead of Persuasion — the Row OR either reserve stack
        #    (9 solari for The Spice Must Flow, 2 for Prepare the Way). Only
        #    auto-buys when a card clears a quality bar; often no card is fine. -
        if "acquire_with_solari" in effect:
            def _pno_score(c):
                v = c.persuasion + c.swords + c.cost * 0.3
                if any("vp" in e for e in getattr(c, "acquire_effects", [])):
                    v += 6                            # e.g. TSMF's acquire VP
                return v

            mx = int(effect["acquire_with_solari"].get("max_cost", 99))
            opts = [(c, "row") for c in gs.imperium_row if c.cost <= mx]
            if gs.reserve_spice_must_flow and gs.reserve_spice_must_flow[-1].cost <= mx:
                opts.append((gs.reserve_spice_must_flow[-1], "tsmf"))
            if gs.reserve_prepare_the_way and gs.reserve_prepare_the_way[-1].cost <= mx:
                opts.append((gs.reserve_prepare_the_way[-1], "ptw"))
            afford = [(c, src) for c, src in opts if c.cost <= player.solari]
            if afford:
                best, src = max(afford, key=lambda t: _pno_score(t[0]))
                if _pno_score(best) >= 3.0:           # quality bar
                    player.solari -= best.cost
                    if src == "row":
                        gs.imperium_row.remove(best)
                        gs.refill_imperium_row()
                    elif src == "tsmf":
                        gs.reserve_spice_must_flow.pop()
                    else:
                        gs.reserve_prepare_the_way.pop()
                    player.discard.append(best)
                    player.cards_acquired_this_turn += 1
                    gs._trigger_acquire_effects(player.id, best)

        if "if_opp_combat_intrigue" in effect:
            if any(q != player.id for q in getattr(gs, "_combat_intrigue_players", set())):
                EffectResolver.resolve_single_effect(
                    effect["if_opp_combat_intrigue"], player, gs)

        # ===== SANDWORM (Bug 8 fix) =====
        # Card-granted sandworms (Unexpected Allies, Arrakis Revolt, etc.) do NOT
        # require Maker Hooks — that restriction applies only to the board-space
        # mechanic.  spawn_sandworms() still respects the Shield Wall check.
        if "sandworm" in effect:
            game_state.spawn_sandworms(effect["sandworm"], source="card_effect")

        # Board-space sandworm activation: Maker Hooks ARE required here.
        if "sandworm_maker_space" in effect:
            if player.has_maker_hooks:
                game_state.spawn_sandworms(effect["sandworm_maker_space"], source="maker_space")

        if "spice_from_sandworm" in effect:
            spice_to_take = min(effect["spice_from_sandworm"], game_state.spice_on_sandworm)
            player.gain_spice(spice_to_take)
            game_state.spice_on_sandworm -= spice_to_take

        # ===== MAKER HOOKS =====
        if "maker_hooks" in effect:
            player.has_maker_hooks = True

        # ===== TAKE A CHOAM CONTRACT =====
        if "contract" in effect:
            if getattr(game_state, "use_choam", False):
                for _ in range(effect["contract"]):
                    game_state.take_contract_from_board(player.id)
            else:
                player.gain_solari(2 * effect["contract"])

        # ===== DEPLOY TROOPS TO THE CONFLICT (escalate) =====
        if "deploy" in effect and game_state.current_conflict is not None:
            n = min(effect["deploy"], player.troops_garrison)
            if n > 0:
                player.troops_garrison -= n
                game_state.troops_in_conflict[player.id] = (
                    game_state.troops_in_conflict.get(player.id, 0) + n)

        # ===== SECRETS: steal a random Intrigue from every opponent 4+ =====
        if "secrets_steal" in effect:
            for opp in game_state.players:
                if opp.id != player.id and len(opp.intrigue_cards) >= 4:
                    idx = int(game_state.rng.integers(len(opp.intrigue_cards)))
                    stolen = opp.intrigue_cards.pop(idx)
                    player.intrigue_cards.append(stolen)

        # ===== EACH OPPONENT DISCARDS A CARD =====
        if "opponents_discard" in effect:
            for opp in game_state.players:
                if opp.id != player.id and opp.hand:
                    idx = int(game_state.rng.integers(len(opp.hand)))
                    opp.discard.append(opp.hand.pop(idx))

        # ===== BREAK THE SHIELD WALL (optional) =====
        # Approximated: break it if the current Conflict is at a protected
        # location (so sandworms become deployable there); otherwise leave it.
        if "may_break_shield_wall" in effect and game_state.shield_wall_intact:
            from src.game.board.board import SHIELD_WALL_PROTECTED
            loc = (game_state.current_conflict.location
                   if game_state.current_conflict else None)
            if loc in SHIELD_WALL_PROTECTED:
                game_state.shield_wall_intact = False

        # ===== PAY SPICE -> SUMMON A SANDWORM (+ may break the Shield Wall) =====
        # Arrakis Revolt's agent box (gated behind if_hooks by the card).
        if "pay_spice_spawn_sandworm" in effect:
            cost = effect["pay_spice_spawn_sandworm"]
            if player.spice >= cost and game_state.current_conflict is not None:
                from src.game.board.board import SHIELD_WALL_PROTECTED
                loc = game_state.current_conflict.location
                if game_state.shield_wall_intact and loc in SHIELD_WALL_PROTECTED:
                    game_state.shield_wall_intact = False
                player.spice -= cost
                game_state.spawn_sandworms(1, source="card_effect")

        # ===== DISCARD A CARD -> RESOLVE A SUB-EFFECT (cost -> reward) =====
        # An arrow effect: the player MAY discard a card for the reward, or
        # decline.  Surfaced as an accept/decline choice.
        if "discard_then" in effect:
            game_state.add_pending_optional_payment(
                player.id, {}, effect["discard_then"], discard=1, label="discard_then")

        # ===== GUILD ENVOY: mandatory discard 1; bonus if it was a Spacing =====
        # Guild card.  Auto-discards a spare Spacing Guild card to earn the
        # bonus when one is in hand, otherwise the least useful card.
        if "discard_then_if_sg" in effect and player.hand:
            sg = [c for c in player.hand if c.has_tag(CardTag.SPACING_GUILD)]
            if sg and len(player.hand) > 1:
                tgt = min(sg, key=lambda c: c.persuasion + c.swords
                          + len(getattr(c, "agent_effects", [])))
                player.hand.remove(tgt)
                player.discard.append(tgt)
                EffectResolver.resolve_single_effect(
                    effect["discard_then_if_sg"], player, game_state)
            else:
                EffectResolver._discard_worst(player)

        # ===== SPACE-TIME FOLDING: MAY discard a card -> draw; +1 draw if the =====
        # discarded card was a Spacing Guild card.
        if "discard_then_sg" in effect:
            spec = effect["discard_then_sg"]
            game_state.add_pending_optional_payment(
                player.id, {}, spec.get("base", {}), discard=1,
                label="discard_then_sg", discard_tag="spacing_guild",
                tag_bonus=spec.get("sg_bonus"))

        # ===== DISCARD N + PAY SOLARI -> GAIN VP (Corrinth City agent) =====
        # Arrow effect -> optional accept/decline.
        if "discard_pay_vp" in effect:
            d = effect["discard_pay_vp"]
            game_state.add_pending_optional_payment(
                player.id, {"solari": d.get("solari", 0)}, {"vp": d.get("vp", 1)},
                discard=d.get("discard", 0), label="discard_pay_vp")

        # ===== SWAP INFLUENCE: -1 on one Faction, +1 on another =====
        # (Captured Mentat) The player MAY do this, and the two Factions may be
        # the same (a no-op).  Auto-choice: only swap when it strictly helps —
        # move a cube that isn't buying anything to a Faction where +1 crosses a
        # friendship (2) or alliance (4) threshold.
        if "influence_swap" in effect:
            factions = ("emperor", "spacing_guild", "bene_gesserit", "fremen")
            inf = player.influence
            gain = next((f for f in factions if inf[f] == 3), None) \
                or next((f for f in factions if inf[f] == 1), None)
            if gain is not None:
                # safe to lose: won't drop below a threshold we hold and isn't 0
                give = next((f for f in factions
                             if f != gain and (inf[f] in (1, 3) or inf[f] >= 5)), None)
                if give is not None and inf[give] > 0:
                    game_state.lose_influence_with_check(player.id, give, 1)
                    game_state.gain_influence_with_check(player.id, gain, 1)

        # ===== HIGH COUNCIL SEAT, OR SOLARI (Corrinth City reveal choice) =====
        if "hc_seat_or_solari" in effect:
            amt = effect["hc_seat_or_solari"]
            if not player.has_councilor and player.solari >= amt:
                player.solari -= amt
                player.has_councilor = True
            else:
                player.gain_solari(amt)

        # ===== TRASH INTRIGUE FOR BENEFIT =====
        # This is a cost→reward (arrow) effect; the player may decline by not
        # paying the cost (trashing an intrigue).  Only queue if they have one.
        if "trash_intrigue_for" in effect:
            if player.intrigue_cards:
                game_state.add_pending_intrigue_trash(player.id, effect["trash_intrigue_for"])

        # ===== CONDITIONAL EFFECTS =====
        # FAQ: "You cannot execute & fizzle a conditional effect while the
        # precondition is false."  The conditions are checked here at resolution
        # time.  For end-of-turn conditionals, the env re-checks before end_turn.

        if "if_hooks" in effect:
            if player.has_maker_hooks:
                EffectResolver.resolve_single_effect(effect["if_hooks"], player, game_state)

        if "if_tag_bene" in effect:
            if player.has_card_with_tag_in_play(CardTag.BENE_GESSERIT):
                EffectResolver.resolve_single_effect(effect["if_tag_bene"], player, game_state)

        if "if_tag_fremen" in effect:
            if player.has_card_with_tag_in_play(CardTag.FREMEN):
                EffectResolver.resolve_single_effect(effect["if_tag_fremen"], player, game_state)

        if "if_tag_fremen_count_2" in effect:
            if player.count_cards_with_tag_in_play(CardTag.FREMEN) >= 2:
                EffectResolver.resolve_single_effect(effect["if_tag_fremen_count_2"], player, game_state)

        if "if_tag_emperor" in effect:
            if player.has_card_with_tag_in_play(CardTag.EMPEROR):
                EffectResolver.resolve_single_effect(effect["if_tag_emperor"], player, game_state)

        if "if_tag_spacing" in effect:
            if player.has_card_with_tag_in_play(CardTag.SPACING_GUILD):
                EffectResolver.resolve_single_effect(effect["if_tag_spacing"], player, game_state)

        # ----- Uprising card conditions (approximate; see card notes) -----
        _tag = {"bene": CardTag.BENE_GESSERIT, "fremen": CardTag.FREMEN,
                "emperor": CardTag.EMPEROR, "spacing": CardTag.SPACING_GUILD}

        def _cond(key: str) -> bool:
            if key == "fremen_bond":
                return player.influence["fremen"] >= 2
            if key == "councilor":
                return player.has_councilor
            if key == "swordmaster":
                return player.has_swordmaster
            if key == "spy_recalled":
                return getattr(player, "recalled_spy_this_turn", False)
            if key == "agent_to_maker":
                return getattr(player, "agent_to_maker_this_turn", False)
            if key == "faction_agent":
                return getattr(player, "agent_to_faction_this_turn", False)
            if key == "trashed_costly":              # trashed a 1+ persuasion card
                return getattr(player, "trashed_costly_card_this_round", False)
            if key == "any_alliance":
                return any(player.alliances.values())
            if key.startswith("alliance_"):
                return player.alliances.get(key[9:], False)
            if key.startswith("influence_"):        # influence_fremen_2
                fac, _, n = key[10:].rpartition("_")
                return player.influence.get(fac, 0) >= int(n)
            if key.startswith("contracts_"):
                return len(player.contracts_completed) >= int(key[10:])
            if key.startswith("tag_other_"):         # another <tag> card besides self
                return player.count_cards_with_tag_in_play(_tag[key[10:]]) >= 2
            if key.startswith("spies_"):             # spies_2 == 2+ spies on board
                return sum(player.spies_on_board.values()) >= int(key[6:])
            if key.startswith("units_in_conflict_"):  # troops + sandworms in Conflict
                n = int(key.rsplit("_", 1)[1])
                return (game_state.troops_in_conflict.get(player.id, 0)
                        + game_state.sandworms_in_conflict.get(player.id, 0)) >= n
            return False

        for k, sub in list(effect.items()):
            if k.startswith("if_") and k[3:] in (
                    "fremen_bond", "councilor", "swordmaster", "spy_recalled",
                    "agent_to_maker", "faction_agent", "any_alliance", "trashed_costly"):
                if _cond(k[3:]):
                    EffectResolver.resolve_single_effect(sub, player, game_state)
            elif (k.startswith("if_alliance_") or k.startswith("if_influence_")
                  or k.startswith("if_contracts_") or k.startswith("if_tag_other_")
                  or k.startswith("if_spies_") or k.startswith("if_units_in_conflict_")):
                if _cond(k[3:]):
                    EffectResolver.resolve_single_effect(sub, player, game_state)

        # ===== OPTIONAL EFFECTS (surfaced as RL actions; NOT executed here) =====
        # "may_convert": becomes a ConversionAction in get_valid_actions()
        # "may_trash":   becomes a TrashAction in get_valid_actions()
