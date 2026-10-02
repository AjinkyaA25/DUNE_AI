"""
Player choices inside card / intrigue effects.

Many effects say "choose": draw OR trash the card you looked at, which Row
card to acquire, which Faction, how many troops to retreat, this OR that.
These used to be resolved by a fixed rule inside the effect resolver, so
neither a human nor the AI ever made the decision. Now each one becomes a
PendingChoice the deciding player resolves with ActionType.RESOLVE_CHOICE
(`choice` = option key) before play continues.

`build(gs, player, key, spec)` turns one choice-effect into
(kind, prompt, options) WITHOUT changing the game, so the AI can value an
effect as its best option. Each option is (key, label, effect dict); the
effect uses the normal vocabulary plus the internal "_" keys executed by
`apply_internal` (draw/trash the top card, acquire a named Row card, lose a
named influence, retreat N troops, pay a cost, ...).
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.game.gameState import GameState
    from src.game.player.player import Player

FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")
FNAME = {"emperor": "Emperor", "spacing_guild": "Spacing Guild",
         "bene_gesserit": "Bene Gesserit", "fremen": "Fremen"}
Option = Tuple[str, str, Dict]

# effect keys that open a choice (handled here instead of in EffectResolver)
CHOICE_KEYS = (
    "peek_top", "acquire_free", "influence_choice", "influence_bene_or_fremen",
    "influence_emperor_or_spacing", "lose_influence_any", "lose_influence_for",
    "retreat_for", "tactical_option", "manipulate", "market_convert",
    "opportunism_vp", "influence_swap", "take_contract_from_reserve",
    "emperor_access_or_draw", "special_mission", "choose_one", "trash_discard_for",
    "deploy_up_to",
)


class PendingChoice:
    def __init__(self, player_id: int, kind: str, prompt: str, options: List[Option]):
        self.player_id = player_id
        self.kind = kind
        self.prompt = prompt
        self.options = options

    def __repr__(self) -> str:
        return f"PendingChoice({self.kind}, p={self.player_id}, {[o[0] for o in self.options]})"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fx(eff: Dict) -> str:
    """Short readable text for a reward dict."""
    parts = []
    for k, v in eff.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float)):
            parts.append(f"{v} {k.replace('_', ' ')}" if v != 1 else k.replace("_", " "))
        elif isinstance(v, dict):
            parts.append(k.replace("_", " "))
    return ", ".join(parts)


def _units_in_conflict(gs, pid) -> int:
    return gs.troops_in_conflict.get(pid, 0) + gs.sandworms_in_conflict.get(pid, 0)


def affordable(gs: "GameState", p: "Player", eff: Dict) -> bool:
    """Can this option's own costs be paid right now?"""
    for k, v in (eff.get("_pay") or {}).items():
        if getattr(p, k, 0) < v:
            return False
    if eff.get("_recall_spy") and sum(p.spies_on_board.values()) < eff["_recall_spy"]:
        return False
    if eff.get("_retreat_n") and gs.troops_in_conflict.get(p.id, 0) < eff["_retreat_n"]:
        return False
    for f in eff.get("_lose_influences", []) or ([eff["_lose_influence"]] if "_lose_influence" in eff else []):
        need = (eff.get("_lose_influences") or []).count(f) or 1
        if p.influence.get(f, 0) < need:
            return False
    if ("deploy" in eff or "deploy_up_to" in eff) and p.troops_garrison <= 0:
        return False
    return True


# ---------------------------------------------------------------------------
# building choices
# ---------------------------------------------------------------------------

def build(gs: "GameState", p: "Player", key: str, spec) -> Optional[Tuple[str, str, List[Option]]]:
    pid = p.id
    opts: List[Option] = []

    if key == "peek_top":                                     # Poison Snooper
        if not p.deck and p.discard:
            return ("peek_top", "Look at the top card of your deck (after shuffling)",
                    [("draw", "Shuffle your discard, then draw or trash the top card",
                      {"_peek_after_shuffle": 1})])
        if not p.deck:
            return None
        top = p.deck[-1].name
        return ("peek_top", f"Top card of your deck: {top}",
                [("draw", f"Draw {top}", {"_draw_top": 1}),
                 ("trash", f"Trash {top}", {"_trash_top": 1})])

    if key == "acquire_free":                                 # Inspire Awe, Impress
        mx = spec.get("max_cost", 3)
        to_hand = bool(spec.get("to_hand_if_sandworm")) and gs.sandworms_in_conflict.get(pid, 0) > 0
        seen = set()
        for c in gs.imperium_row:
            if c.cost <= mx and c.name not in seen:
                seen.add(c.name)
                opts.append((c.name, f"Acquire {c.name} ({c.cost})" + (" to your hand" if to_hand else ""),
                             {"_acquire_named": {"card": c.name, "to_hand": to_hand}}))
        return ("acquire_free", f"Acquire a card that costs {mx} or less", opts) if opts else None

    if key == "influence_choice":                             # Bribery / Buy Access
        n = int(spec)
        if n >= 2:                                            # "choose two" Factions
            for a, b in itertools.combinations(FACTIONS, 2):
                opts.append((f"{a}+{b}", f"+1 {FNAME[a]} and +1 {FNAME[b]}",
                             {"_gain_influences": [a, b]}))
            return ("influence_choice", "Choose two Factions", opts)
        return ("influence_choice", "Gain 1 influence with a Faction",
                [(f, f"+1 {FNAME[f]}", {"_gain_influences": [f]}) for f in FACTIONS])

    if key in ("influence_bene_or_fremen", "influence_emperor_or_spacing"):
        pair = ("bene_gesserit", "fremen") if "bene" in key else ("emperor", "spacing_guild")
        return (key, f"Gain 1 influence: {FNAME[pair[0]]} or {FNAME[pair[1]]}",
                [(f, f"+1 {FNAME[f]}", {"_gain_influences": [f] * int(spec)}) for f in pair])

    if key in ("lose_influence_any", "lose_influence_for"):
        reward = spec if isinstance(spec, dict) else {}
        for f in FACTIONS:
            if p.influence.get(f, 0) > 0:
                opts.append((f, f"Lose 1 {FNAME[f]} influence" + (f" → {_fx(reward)}" if reward else ""),
                             {"_lose_influence": f, **reward}))
        if key == "lose_influence_for":
            opts.append(("decline", "Don't", {}))
            return ("lose_influence_for", f"You may lose 1 influence for {_fx(reward)}", opts)
        return ("lose_influence_any", "Lose 1 influence (your choice of Faction)", opts) if opts else None

    if key == "retreat_for":                                  # Go to Ground, Reach Agreement
        in_conf = gs.troops_in_conflict.get(pid, 0)
        lo, hi = int(spec.get("min", 1)), int(spec.get("max", 1))
        reward = spec.get("reward", {})
        for n in range(lo, min(hi, in_conf) + 1):
            opts.append((f"retreat{n}", f"Retreat {n} troop{'s' if n > 1 else ''} → {_fx(reward)}",
                         {"_retreat_n": n, **reward}))
        if not opts:
            return None
        opts.append(("decline", "Don't retreat", {}))
        return ("retreat_for", "Retreat troops from the Conflict", opts)

    if key == "tactical_option":
        opts.append(("swords", "+2 swords", {"swords": 2}))
        for n in range(1, gs.troops_in_conflict.get(pid, 0) + 1):
            opts.append((f"retreat{n}", f"Retreat {n} troop{'s' if n > 1 else ''}", {"_retreat_n": n}))
        return ("tactical_option", "Tactical Option: 2 swords OR retreat any number of troops", opts)

    if key == "manipulate":
        for c in gs.imperium_row:
            opts.append((c.name, f"Set aside {c.name} (buy it for 1 less this round)",
                         {"_manipulate": c.name, "_discount": int(spec)}))
        return ("manipulate", "Remove and replace a card in the Imperium Row", opts) if opts else None

    if key == "market_convert":                               # Market Opportunity
        if p.spice >= 2:
            opts.append(("spice", "Pay 2 spice → 5 solari", {"_pay": {"spice": 2}, "solari": 5}))
        if p.solari >= 5:
            opts.append(("solari", "Pay 5 solari → 5 spice", {"_pay": {"solari": 5}, "spice": 5}))
        if not opts:
            return None
        opts.append(("decline", "Don't trade", {}))
        return ("market_convert", "Market Opportunity", opts)

    if key == "opportunism_vp":
        if p.solari >= 2:
            for a, b in itertools.combinations_with_replacement(FACTIONS, 2):
                need = {a: 1, b: 1} if a != b else {a: 2}
                if all(p.influence.get(f, 0) >= n for f, n in need.items()):
                    lab = (f"{FNAME[a]} and {FNAME[b]}" if a != b else f"2 {FNAME[a]}")
                    opts.append((f"{a}+{b}", f"Lose {lab} influence + pay 2 solari → 1 VP",
                                 {"_lose_influences": [a, b], "_pay": {"solari": 2}, "vp": 1}))
        if not opts:
            return None
        opts.append(("decline", "Don't", {}))
        return ("opportunism_vp", "Opportunism: lose 2 influence + 2 solari → 1 VP", opts)

    if key == "influence_swap":                               # Change Allegiances etc.
        for give in FACTIONS:
            if p.influence.get(give, 0) <= 0:
                continue
            for gain in FACTIONS:
                if gain != give:
                    opts.append((f"{give}>{gain}", f"-1 {FNAME[give]}, +1 {FNAME[gain]}",
                                 {"_lose_influence": give, "_gain_influences": [gain]}))
        if not opts:
            return None
        opts.append(("decline", "Don't swap", {}))
        return ("influence_swap", "Move 1 influence to another Faction", opts)

    if key == "take_contract_from_reserve":                   # Coercive Negotiation
        if not getattr(gs, "use_choam", False):
            return None
        for i, ct in enumerate(gs.contract_bank[:int(spec)]):
            opts.append((f"c{i}", f"Take contract {ct.name} → {_fx(ct.rewards)}", {"_take_contract": i}))
        return ("take_contract_from_reserve", "Take one of the revealed contracts", opts) if opts else None

    if key == "emperor_access_or_draw":                       # Emperor's Invitation
        opts.append(("draw", "Draw a card", {"draw": 1}))
        if p.agents_available > 0:
            opts.append(("access", "The card you play this turn has the Emperor icon",
                         {"_grant_emperor_access": 1}))
        return ("emperor_access_or_draw", "Emperor's Invitation", opts)

    if key == "special_mission":
        if p.spies_available > 0:
            opts.append(("spy", "Place a Spy on a City observation post", {"_city_spy": 1}))
        if sum(p.spies_on_board.values()) >= 1:
            opts.append(("recall", "Recall a Spy → 2 spice (+ may break the Shield Wall)",
                         {"_recall_spy": 1, "spice": 2, "may_break_shield_wall": 1}))
        return ("special_mission", "Special Mission", opts) if opts else None

    if key == "choose_one":                                   # generic OR card
        for i, o in enumerate(spec.get("options", [])):
            eff = o.get("effect", {})
            if affordable(gs, p, eff) and _option_possible(gs, p, eff):
                opts.append((f"o{i}", o.get("label", _fx(eff)), eff))
        if spec.get("may", True):
            opts.append(("decline", "Neither", {}))
        real = [o for o in opts if o[0] != "decline"]
        return ("choose_one", spec.get("prompt", "Choose one"), opts) if real else None

    if key == "deploy_up_to":                                 # Detonation, Counterattack
        if gs.current_conflict is None:
            return None
        hi = min(int(spec), p.troops_garrison)
        if hi <= 0:
            return None
        for n in range(hi, -1, -1):
            opts.append((f"deploy{n}", f"Deploy {n} troop{'s' if n != 1 else ''} from your garrison",
                         {"_deploy_n": n} if n else {}))
        return ("deploy_up_to", f"Deploy up to {spec} troops to the Conflict", opts)

    if key == "trash_discard_for":                            # Tenuous Bond (combat)
        mn = int(spec.get("min_cost", 1))
        reward = spec.get("reward", {})
        seen = set()
        for c in p.discard:
            if (c.cost or 0) >= mn and c.name not in seen:
                seen.add(c.name)
                opts.append((c.name, f"Trash {c.name} from your discard → {_fx(reward)}",
                             {"_trash_discard": c.name, **reward}))
        if not opts:
            return None
        opts.append(("decline", "Don't trash", {}))
        return ("trash_discard_for", f"Trash a card costing {mn}+ from your discard pile", opts)
    return None


def _option_possible(gs, p, eff: Dict) -> bool:
    if eff.get("_recall_spy_any") and not p.spies_on_board:
        return False
    if "spy" in eff and p.spies_available <= 0 and not eff.get("_pay"):
        return p.spies_available > 0
    if "spy" in eff and p.spies_available <= 0:
        return False
    if eff.get("if_units_in_conflict_any") and _units_in_conflict(gs, p.id) == 0:
        return False
    return True


# ---------------------------------------------------------------------------
# executing a chosen option
# ---------------------------------------------------------------------------

def apply_internal(gs: "GameState", p: "Player", eff: Dict) -> None:
    """Run the '_' keys of an option (the normal keys go through the resolver)."""
    from src.game.effects import EffectResolver
    pid = p.id
    for k, v in (eff.get("_pay") or {}).items():
        setattr(p, k, getattr(p, k) - v)
    if eff.get("_recall_spy"):
        for post in list(p.spies_on_board)[:eff["_recall_spy"]]:
            p.recall_spy(post)
        p.recalled_spy_this_turn = True
    if eff.get("_peek_after_shuffle"):
        nd = list(p.discard)
        gs.rng.shuffle(nd)
        p.deck, p.discard = nd + p.deck, []
        built = build(gs, p, "peek_top", 1)
        if built and len(built[2]) > 1:
            gs.add_pending_choice(pid, *built)
    if eff.get("_draw_top") and p.deck:
        p.hand.append(p.deck.pop())
    if eff.get("_trash_top") and p.deck:
        gs._to_trash(p, p.deck.pop())
    if "_acquire_named" in eff:
        spec = eff["_acquire_named"]
        card = next((c for c in gs.imperium_row if c.name == spec["card"]), None)
        if card is not None:
            gs.imperium_row.remove(card)
            (p.hand if spec.get("to_hand") else p.discard).append(card)
            p.cards_acquired_this_turn += 1
            gs.refill_imperium_row()
            gs._trigger_acquire_effects(pid, card)
            _call_to_arms(gs, p)
    for f in eff.get("_lose_influences", []):
        gs.lose_influence_with_check(pid, f, 1)
    if "_lose_influence" in eff:
        gs.lose_influence_with_check(pid, eff["_lose_influence"], 1)
    for f in eff.get("_gain_influences", []):
        gs.gain_influence_with_check(pid, f, 1)
    if "_manipulate" in eff:
        card = next((c for c in gs.imperium_row if c.name == eff["_manipulate"]), None)
        if card is not None:
            gs.imperium_row.remove(card)
            p.reserved_card = card
            p.reserved_discount = int(eff.get("_discount", 1))
            gs.refill_imperium_row()
    if eff.get("_retreat_n"):
        n = min(int(eff["_retreat_n"]), gs.troops_in_conflict.get(pid, 0))
        gs.troops_in_conflict[pid] -= n
        p.troops_garrison += n
        gs.update_combat_strength(pid)           # step() re-syncs commanders
    if "_trash_discard" in eff:
        card = next((c for c in p.discard if c.name == eff["_trash_discard"]), None)
        if card is not None:
            p.discard.remove(card)
            gs._to_trash(p, card)
    if "_discard_named" in eff:
        card = next((c for c in p.hand if c.name == eff["_discard_named"]), None)
        if card is not None:
            p.hand.remove(card)
            p.discard.append(card)
            EffectResolver._note_discard(p, card)
    if "_take_contract" in eff:
        i = int(eff["_take_contract"])
        pool = gs.contract_bank[:3]
        if i < len(pool):
            best = pool[i]
            for other in pool:                 # take one, trash the rest
                gs.contract_bank.remove(other)
            p.take_contract(best)
            if best.is_immediate() and (not best.requires_intrigue() or p.intrigue_cards):
                gs.complete_contract(pid, best)
    if eff.get("_deploy_n"):
        n = min(int(eff["_deploy_n"]), p.troops_garrison)
        if n > 0 and gs.current_conflict is not None:
            if getattr(gs, "bl", None) is not None:
                gs.bl.deploy_commanders(pid, n)
            p.troops_garrison -= n
            gs.troops_in_conflict[pid] = gs.troops_in_conflict.get(pid, 0) + n
            gs.update_combat_strength(pid)
    if eff.get("_grant_emperor_access"):
        p.grant_emperor_access_this_turn = True
    if eff.get("_city_spy"):
        gs.add_pending_spy_placement(pid, 1, allow_occupied=False,
                                     allowed_posts=["Arrakeen Post", "Research Station Left Post",
                                                    "Research Station Right Post"])


def _call_to_arms(gs, p) -> None:
    """Call to Arms: +1 troop per card acquired during your Reveal turn this
    round, after the card was played."""
    if getattr(p, "call_to_arms_round", None) == gs.round:
        p.gain_troops(1)
