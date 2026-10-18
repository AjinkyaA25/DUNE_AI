"""
All Dune Imperium: Uprising + Bloodlines Leaders (19 cards).

Card text: src/game/leader/LEADERS.md, transcribed from the printed cards
(dunecardshub.com images) and the rulings listed there. Every hook is a
module-level function (so game states with Leaders can be pickled), and every
decision a Leader offers is a pending choice the player resolves
(`ActionType.RESOLVE_CHOICE`), so the AI - and a human in the UI - decides.

Choice options may carry `"_leader_fx": [name, *args]`, executed by
`run_fx` (registered in src/game/choices.apply_internal), and `"_v"`, a plain
resource dict the AI uses to value an option (ignored by the engine).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.game.effects import EffectResolver
from src.game.leader.leader import Leader

FACTIONS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")
CITY_POSTS = ["Arrakeen Post", "Research Station Left Post", "Research Station Right Post"]
LANDSRAAD_POSTS = ["Landsraad Post", "Green Post"]
FACTION_POSTS = ["Emperor Post", "Spacing Guild Post", "Bene Gesserit Post", "Fremen Post"]
TUEKS_SIETCH = "Tuek's Sietch"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fx(player, gs, effect: Dict) -> None:
    EffectResolver.resolve_single_effect(dict(effect), player, gs)


def _choose(player, gs, prompt: str, options: List, may: bool = True, kind: str = "leader") -> None:
    """options: [(key, label, effect)]; unaffordable / impossible ones are dropped."""
    from src.game.choices import affordable, _option_possible
    opts = [o for o in options if affordable(gs, player, o[2]) and _option_possible(gs, player, o[2])]
    if may:
        opts.append(("decline", "Decline", {}))
    if not [o for o in opts if o[0] != "decline"]:
        return
    if len(opts) == 1:                               # a single forced option: just do it
        _fx(player, gs, opts[0][2])
        return
    gs.add_pending_choice(player.id, kind, prompt, opts)


def _note(gs, p, what: str) -> None:
    """Record a passive ability that paid out (leader audit / study logs)."""
    log = getattr(gs, "leader_log", None)
    if log is not None:
        log.append((p.id, p.leader.name, what))


def can_spy(player, gs, posts: Optional[List[str]] = None, allow_occupied: bool = False) -> bool:
    """Is there a post (in `posts`) a Spy of this player could go to - counting
    recalling one first when none is left in supply?"""
    from src.game.board.board import ALL_OBSERVATION_POSTS
    if player.spies_available <= 0 and not player.spies_on_board:
        return False
    for post in (posts or ALL_OBSERVATION_POSTS):
        if player.has_spy_at(post):
            continue
        if player.spies_available > 0 and gs.can_place_spy(player.id, post, allow_occupied):
            return True
        if player.spies_available <= 0:
            occupied = any(q.has_spy_at(post) for q in gs.players if q.id != player.id)
            if allow_occupied or not occupied:
                return True
    return False


def _spy(player, gs, posts: Optional[List[str]] = None, allow_occupied: bool = False) -> None:
    if not can_spy(player, gs, posts, allow_occupied):
        return
    # none in supply: the placement offers recalling each of the player's
    # Spies first (GameState._valid_pending_actions, "recall:<post>")
    gs.add_pending_spy_placement(player.id, 1, allow_occupied=allow_occupied, allowed_posts=posts)


def _space_icon(space: Optional[str]) -> Optional[str]:
    from src.game.board.board import SPACE_ICONS
    return SPACE_ICONS.get(space) if space else None


def _has_alliance(player) -> bool:
    return any(player.alliances.values())


# effects run from choice options ("_leader_fx": [name, *args]) -------------

FX = {}


def fx(name):
    def deco(fn):
        FX[name] = fn
        return fn
    return deco


def run_fx(gs, player, spec) -> None:
    name, *args = spec
    if name not in FX:                     # effects defined with their card decks
        import src.game.leader.twisted     # noqa: F401
        import src.game.leader.navigation  # noqa: F401
    FX[name](gs, player, *args)


@fx("trash_hand")
def _fx_trash_hand(gs, p, card_name: str, reward_if_costly: Optional[Dict] = None) -> None:
    card = next((c for c in p.hand if c.name == card_name), None)
    if card is None:
        return
    p.hand.remove(card)
    gs._to_trash(p, card)
    if reward_if_costly and getattr(card, "cost", 0) >= 1:
        _fx(p, gs, reward_if_costly)


def _trash_any_options(p, reward_if_costly: Optional[Dict] = None, label_extra: str = ""):
    seen, out = set(), []
    for zone, label in (("hand", "hand"), ("discard", "discard pile"), ("in_play", "play area")):
        for c in getattr(p, zone):
            if (zone, c.name) in seen:
                continue
            seen.add((zone, c.name))
            bonus = reward_if_costly if (reward_if_costly and getattr(c, "cost", 0) >= 1) else None
            out.append((f"trash:{zone}:{c.name}", f"Trash {c.name} from your {label}" + (label_extra if bonus else ""),
                        {"_leader_fx": ["trash_zone", zone, c.name, reward_if_costly], "_v": dict(bonus or {})}))
    return out


@fx("trash_zone")
def _fx_trash_zone(gs, p, zone: str, card_name: str, reward_if_costly: Optional[Dict] = None) -> None:
    pile = getattr(p, zone)
    card = next((c for c in pile if c.name == card_name), None)
    if card is None:
        return
    pile.remove(card)
    gs._to_trash(p, card)
    if reward_if_costly and getattr(card, "cost", 0) >= 1:
        _fx(p, gs, reward_if_costly)


@fx("trash_in_play")
def _fx_trash_in_play(gs, p, card_name: str) -> None:
    card = next((c for c in p.in_play if c.name == card_name), None)
    if card is not None:
        p.in_play.remove(card)
        gs._to_trash(p, card)


@fx("spy_posts")
def _fx_spy_posts(gs, p, posts, allow_occupied=False) -> None:
    _spy(p, gs, list(posts) if posts else None, allow_occupied)


def _trash_hand_options(p, reward_if_costly: Optional[Dict] = None, label_extra: str = ""):
    seen, out = set(), []
    for c in p.hand:
        if c.name in seen:
            continue
        seen.add(c.name)
        bonus = reward_if_costly if (reward_if_costly and getattr(c, "cost", 0) >= 1) else None
        out.append((f"trash:{c.name}", f"Trash {c.name} from your hand" + (label_extra if bonus else ""),
                    {"_leader_fx": ["trash_hand", c.name, reward_if_costly], "_v": dict(bonus or {})}))
    return out


# ---------------------------------------------------------------------------
# Uprising
# ---------------------------------------------------------------------------

# Gurney Halleck - Always Smiling / Warmaster
def gurney_reveal_done(p, gs):
    if gs.current_conflict is not None and gs.combat_strength.get(p.id, 0) >= 6:
        gs.gain_persuasion(p.id, 1)


def gurney_signet(p, gs):
    _fx(p, gs, {"troops": 1})


# Feyd-Rautha - Devious Strength / Personal Training
FEYD_TRACK = {                       # node -> (next nodes, reward)
    "start": (["A", "B"], None),
    "A": (["C"], {"pay_then": {"cost": {"solari": 1}, "trash": 1}}),
    "B": (["C"], {"spy": 1}),
    "C": (["D", "E"], {"trash": 1}),
    "D": (["end"], {"trash": 1}),
    "E": (["F"], {"spy": 1}),
    "F": (["end"], {"spice": 2}),
    "end": ([], {"troops": 1, "spy": 1}),
}
FEYD_LABEL = {"A": "pay 1 solari -> trash a card", "B": "place a Spy", "C": "trash a card",
              "D": "trash a card", "E": "place a Spy", "F": "2 spice", "end": "1 troop + a Spy"}
FEYD_VALUE = {"A": {"trash": 1}, "B": {"spy": 1}, "C": {"trash": 1}, "D": {"trash": 1},
              "E": {"spy": 1}, "F": {"spice": 2}, "end": {"troops": 1, "spy": 1}}


def feyd_reveal(p, gs):
    # Devious Strength: recall a Spy -> 2 swords (optional, once per Reveal
    # turn, any time in it: a Spy placed during the turn can be recalled)
    p.leader.state["devious_offered"] = False
    feyd_offer(p, gs)


def feyd_spy_placed(p, gs):
    if getattr(gs, "_reveal_open", None) == p.id or gs.player_in_reveal_buy == p.id:
        feyd_offer(p, gs)


def feyd_offer(p, gs):
    if p.leader.state.get("devious_offered"):
        return
    units = gs.troops_in_conflict.get(p.id, 0) + gs.sandworms_in_conflict.get(p.id, 0)
    if p.spies_on_board and gs.current_conflict is not None and units > 0:
        p.leader.state["devious_offered"] = True
        gs.add_pending_optional_payment(p.id, {"recall_spy": 1}, {"swords": 2},
                                        label="Devious Strength")


@fx("feyd_move")
def _fx_feyd_move(gs, p, node: str) -> None:
    p.leader.state["feyd"] = node
    _fx(p, gs, FEYD_TRACK[node][1])


def feyd_signet(p, gs):
    node = p.leader.state.get("feyd", "start")
    nxt = FEYD_TRACK[node][0]
    if not nxt:                       # ruling: at the end of the track the signet gives nothing
        return
    _choose(p, gs, "Personal Training: move your Feyd token",
            [(f"feyd:{n}", f"Training: {FEYD_LABEL[n]}",
              {"_leader_fx": ["feyd_move", n],
               "_v": FEYD_VALUE[n] if (n != "A" or p.solari >= 1) else {}}) for n in nxt], may=False)


# Lady Amber Metulli - Desert Scouts / Fill Coffers
def amber_reveal_done(p, gs):
    if gs.troops_in_conflict.get(p.id, 0) > 0:
        _fx(p, gs, {"retreat": 1})


def amber_signet(p, gs):
    _fx(p, gs, {"solari": 1})
    if _has_alliance(p):
        _fx(p, gs, {"spice": 1})


# Lady Jessica / Reverend Mother Jessica
def jessica_agent_placed(p, gs, space):
    st = p.leader.state
    icon = _space_icon(space)
    if not st.get("flipped"):
        if icon == "bene_gesserit" and st.get("memories", 0) > 0:
            n = st["memories"]
            _choose(p, gs, "Other Memories: return your memories, draw a card for each, then flip",
                    [("memories", f"Return {n} memories -> draw {n}, become Reverend Mother",
                      {"_leader_fx": ["jessica_flip"], "_v": {"draw": n, "troops": n}})])
        return
    # Reverend Mother: once per turn, pay 1 water to repeat the space's printed effects
    if icon in ("bene_gesserit", "fremen") and p.water >= 1 and p.leader.once_per_turn("rm"):
        _choose(p, gs, f"Reverend Mother: pay 1 water to repeat {space}",
                [("repeat", f"Pay 1 water: repeat {space}",
                  {"_pay": {"water": 1}, "_leader_fx": ["rm_repeat", space,
                                                         getattr(gs, "_current_space_option", None)]})])


@fx("jessica_flip")
def _fx_jessica_flip(gs, p) -> None:
    st = p.leader.state
    n = st.get("memories", 0)
    p.troops_supply += n
    st["memories"] = 0
    st["flipped"] = True
    p.leader.name = "Reverend Mother Jessica"
    _fx(p, gs, {"draw": n})


@fx("rm_repeat")
def _fx_rm_repeat(gs, p, space: str, option=None) -> None:
    gs._apply_board_space_effects(p.id, space, option)


@fx("jessica_memory")
def _fx_jessica_memory(gs, p) -> None:
    if p.troops_supply > 0:
        p.troops_supply -= 1
        p.leader.state["memories"] = p.leader.state.get("memories", 0) + 1


def jessica_signet(p, gs):
    if not p.leader.state.get("flipped"):
        # Spice Agony: 1 spice -> an Intrigue + a memory (no spice, no memory)
        gs.add_pending_optional_payment(p.id, {"spice": 1},
                                        {"intrigue": 1, "_leader_fx": ["jessica_memory"]},
                                        label="Spice Agony")
    else:
        # Water of Life: 1 spice -> 1 water
        gs.add_pending_optional_payment(p.id, {"spice": 1}, {"water": 1}, label="Water of Life")


# Lady Margot Fenring - Loyalty / Arrakis Informant
def margot_influence_2(p, gs, faction):
    if faction == "bene_gesserit":
        _fx(p, gs, {"spice": 2})


def margot_signet(p, gs):
    _spy(p, gs, CITY_POSTS)


# Muad'Dib - Unpredictable Foe / Lead the Way
def muaddib_reveal_done(p, gs):
    if gs.sandworms_in_conflict.get(p.id, 0) > 0:
        _fx(p, gs, {"intrigue": 1})


def muaddib_signet(p, gs):
    _fx(p, gs, {"draw": 1})


# Princess Irulan - Imperial Birthright / Chronicler's Insight
def irulan_influence_2(p, gs, faction):
    if faction == "emperor":
        _fx(p, gs, {"intrigue": 1})


def irulan_signet(p, gs):
    opts, seen = [], set()
    for c in gs.imperium_row:
        if getattr(c, "cost", 0) == 1 and c.name not in seen:      # exactly 1 (user)
            seen.add(c.name)
            opts.append((f"acquire:{c.name}", f"Acquire {c.name} to your hand",
                         {"_acquire_named": {"card": c.name, "to_hand": True}}))
    opts += _trash_hand_options(p, {"spice": 2}, " (+2 spice)")
    _choose(p, gs, "Chronicler's Insight: acquire a 1-cost card to your hand, or trash a card", opts)


# Shaddam Corrino IV - Sardaukar Commander / Emperor of the Known Universe
SARDAUKAR_CONTRACTS = ("Sardaukar (Draw)", "Sardaukar (Uplift)")


def shaddam_setup(p, gs):
    """Set aside both Sardaukar contracts: only Shaddam may take them."""
    aside = [c for c in gs.contract_bank + gs.contracts_on_board if c.name in SARDAUKAR_CONTRACTS]
    for c in aside:
        if c in gs.contract_bank:
            gs.contract_bank.remove(c)
        if c in gs.contracts_on_board:
            gs.contracts_on_board.remove(c)
    if hasattr(gs, "_refill_contracts_on_board"):
        gs._refill_contracts_on_board()
    gs.shaddam_contracts = {p.id: aside}


def shaddam_signet(p, gs):
    p.no_deploy_this_turn = True            # units can't be deployed this turn
    _choose(p, gs, "Emperor of the Known Universe",
            [("troop", "1 solari + 1 troop", {"solari": 1, "troops": 1}),
             ("influence", "Pay 3 solari: +1 influence on any track", {"_pay": {"solari": 3}, "influence_any": 1})],
            may=False)


# Staban Tuek - Limited Allies + Smuggle Spice / Unseen Network
def staban_deck_setup(p, gs):
    card = next((c for c in p.deck + p.hand if c.name == "Diplomacy"), None)
    if card is not None:
        (p.deck if card in p.deck else p.hand).remove(card)


def staban_other_agent_placed(p, gs, placer, space):
    from src.game.board.board import MAKER_SPACES, SPACE_TO_OBSERVATION_POSTS
    if space in MAKER_SPACES and any(p.has_spy_at(post) for post in SPACE_TO_OBSERVATION_POSTS.get(space, ())):
        _fx(p, gs, {"spice": 1})


@fx("staban_spy")
def _fx_staban_spy(gs, p, group: str) -> None:
    if group == "landsraad":
        _spy(p, gs, LANDSRAAD_POSTS)
        gs.add_pending_optional_payment(p.id, {"spice": 1}, {"solari": 3}, label="Unseen Network")
    elif group == "faction":
        _spy(p, gs, FACTION_POSTS)
        gs.add_pending_optional_payment(p.id, {"solari": 2}, {"intrigue": 1}, label="Unseen Network")
    else:
        _spy(p, gs, None)


def staban_signet(p, gs):
    opts = []
    if can_spy(p, gs, LANDSRAAD_POSTS):
        opts.append(("landsraad", "Spy on a Landsraad post (then 1 spice -> 3 solari)",
                     {"_leader_fx": ["staban_spy", "landsraad"],
                      "_v": {"spy": 1, **({"solari": 3, "spice": -1} if p.spice >= 1 else {})}}))
    if can_spy(p, gs, FACTION_POSTS):
        opts.append(("faction", "Spy on a Faction post (then 2 solari -> an Intrigue)",
                     {"_leader_fx": ["staban_spy", "faction"],
                      "_v": {"spy": 1, **({"intrigue": 1, "solari": -2} if p.solari >= 2 else {})}}))
    if can_spy(p, gs, None):
        opts.append(("any", "Spy on any post", {"_leader_fx": ["staban_spy", "any"], "_v": {"spy": 1}}))
    _choose(p, gs, "Unseen Network: place a Spy", opts, may=False)


# ---------------------------------------------------------------------------
# Bloodlines
# ---------------------------------------------------------------------------

# Chani - Tactician / Fedaykin Maneuver
CHANI_START, CHANI_END = 2, 10                    # 1-4 players start on the 3rd space
CHANI_REWARDS = {5: {"spice": 1}, 10: {"water": 1}}


def chani_units_left(p, gs, n):
    st = p.leader.state
    pos = st.get("tactics", CHANI_START)
    for _ in range(int(n)):
        pos += 1
        if pos in CHANI_REWARDS:
            _fx(p, gs, CHANI_REWARDS[pos])
        if pos >= CHANI_END:                      # reset; extra troops don't advance
            pos = CHANI_START
            break
    st["tactics"] = pos


def chani_signet(p, gs):
    opts = []
    n = gs.troops_in_conflict.get(p.id, 0)
    for k in range(1, n + 1):
        opts.append((f"retreat{k}", f"Retreat {k} troop{'s' if k > 1 else ''}", {"_retreat_n": k}))
    if p.influence.get("fremen", 0) >= 2:
        opts.append(("water", "Pay 1 water: draw 2 cards", {"_pay": {"water": 1}, "draw": 2}))
    _choose(p, gs, "Fedaykin Maneuver", opts)


# Count Hasimir Fenring - Assassin / Corrino Liaison
def fenring_trash(p, gs, card):
    from src.game.intrigue.intrigue import IntrigueCard
    if not isinstance(card, IntrigueCard):
        p.gain_solari(1)


def fenring_signet(p, gs):
    seen, opts = set(), []
    for c in p.in_play:
        if c.name not in seen:
            seen.add(c.name)
            opts.append((f"trash:{c.name}", f"Trash {c.name} from your play area",
                         {"_leader_fx": ["trash_in_play", c.name], "_v": {"solari": 1}}))
    if can_spy(p, gs, ["Emperor Post"], True):
        opts.append(("spy", "Place a Spy on the Emperor post (even if occupied)",
                     {"_leader_fx": ["spy_posts", ["Emperor Post"], True], "_v": {"spy": 1}}))
    _choose(p, gs, "Corrino Liaison", opts)


# Duncan Idaho - Ginaz Swordmaster / Into the Fray
@fx("duncan_fray")
def _fx_duncan_fray(gs, p) -> None:
    space = getattr(gs, "_current_agent_space", None)
    if space and gs.agent_on_space.get(space) == p.id:
        gs.agent_on_space[space] = None          # the Agent leaves its space
    gs.duncan_in_conflict = dict(getattr(gs, "duncan_in_conflict", {}))
    gs.duncan_in_conflict[p.id] = 3 if p.has_swordmaster else 2
    gs.update_combat_strength(p.id)


def duncan_signet(p, gs):
    if gs.current_conflict is None or getattr(gs, "duncan_in_conflict", {}).get(p.id):
        return
    s = 3 if p.has_swordmaster else 2
    _choose(p, gs, "Into the Fray: deploy this Agent to the Conflict?",
            [("fray", f"Deploy the Agent as a {s}-strength unit (can't retreat)",
              {"_leader_fx": ["duncan_fray"], "_v": {"swords": s}})])


# Esmar Tuek - Tuek's Sietch / Smuggle Spice
def esmar_setup(p, gs):
    gs.leader_spaces = set(getattr(gs, "leader_spaces", set())) | {TUEKS_SIETCH}
    gs.maker_bonus_spice.setdefault(TUEKS_SIETCH, 0)


def esmar_agent_placed(p, gs, space):
    if space == TUEKS_SIETCH:
        _note(gs, p, "own_sietch_visit")
        _fx(p, gs, {"solari": 1})


def esmar_other_agent_placed(p, gs, placer, space):
    if space == TUEKS_SIETCH:
        _note(gs, p, "opponent_sietch_visit")
        _fx(p, gs, {"intrigue": 1})


@fx("esmar_place")
def _fx_esmar_place(gs, p) -> None:
    if getattr(gs, "_current_agent_space", None) == TUEKS_SIETCH:
        _fx(p, gs, {"spice": 1})                 # placed and gathered on the same turn (ruling)
    else:
        gs.maker_bonus_spice[TUEKS_SIETCH] = gs.maker_bonus_spice.get(TUEKS_SIETCH, 0) + 1


@fx("esmar_take")
def _fx_esmar_take(gs, p, space: str) -> None:
    if gs.maker_bonus_spice.get(space, 0) > 0:
        gs.maker_bonus_spice[space] -= 1
        _fx(p, gs, {"spice": 1})


def esmar_signet(p, gs):
    opts = [("place", "Place 1 bonus spice on Tuek's Sietch", {"_leader_fx": ["esmar_place"],
                                                              "_v": {"spice": 0.6}})]
    for space, n in sorted(gs.maker_bonus_spice.items()):
        if n > 0 and (space != TUEKS_SIETCH or gs._leader_space_active(space)):
            opts.append((f"take:{space}", f"Take 1 bonus spice from {space}",
                         {"_leader_fx": ["esmar_take", space], "_v": {"spice": 1}}))
    _choose(p, gs, "Smuggle Spice", opts, may=False)


# Gaius Helen Mohiam - Clandestine / Listeners
def gaius_signet(p, gs):
    opts = []
    if can_spy(p, gs, LANDSRAAD_POSTS):
        opts.append(("landsraad", "Place a Spy on a Landsraad post",
                     {"_leader_fx": ["spy_posts", LANDSRAAD_POSTS], "_v": {"spy": 1}}))
    if can_spy(p, gs, None):
        opts.append(("any", "Pay 1 spice: place a Spy", {"_pay": {"spice": 1},
                                                         "_leader_fx": ["spy_posts", None], "_v": {"spy": 1}}))
    _choose(p, gs, "Listeners", opts, may=False)


# Kota Odax of Ix - Secret Project / Reverse Engineering
def kota_setup(p, gs):
    bl = getattr(gs, "bl", None)
    if bl is None:
        return
    bottoms = [(i, s[-1]) for i, s in enumerate(bl.tech_stacks) if s]
    if not bottoms:
        return
    _choose(p, gs, "Secret Project: keep one bottom Tech tile face down",
            [(f"secret:{i}", f"Keep {t.name} (cost {t.cost})", {"_leader_fx": ["kota_secret", i]})
             for i, t in bottoms], may=False)


@fx("kota_secret")
def _fx_kota_secret(gs, p, i: int) -> None:
    stack = gs.bl.tech_stacks[i]
    if stack:
        gs.bl.kota_secret = {p.id: stack.pop()}


@fx("kota_trash_tech")
def _fx_kota_trash_tech(gs, p, name: str) -> None:
    gs.bl._trash_tile(p.id, name)
    _fx(p, gs, {"intrigue": 1, "draw": 1})


def kota_signet(p, gs):
    opts = [("spice", "1 spice", {"spice": 1})]
    for t in getattr(p, "techs", []) or []:
        nm = getattr(t, "name", None)
        if nm:
            opts.append((f"trash:{nm}", f"Trash {nm} -> an Intrigue + draw a card",
                         {"_leader_fx": ["kota_trash_tech", nm], "_v": {"intrigue": 1, "draw": 1}}))
    _choose(p, gs, "Reverse Engineering", opts, may=False)


# Liet Kynes - Arrakis Planetologist / Judge of the Change
def liet_sandworm_replaced(p, gs, count):
    for _ in range(int(count)):
        _fx(p, gs, {"trash": 1, "spice": 1, "intrigue": 1})


def liet_signet(p, gs):
    icon = _space_icon(getattr(gs, "_current_agent_space", None))
    if icon == "landsraad" and p.influence.get("emperor", 0) >= 2:
        _fx(p, gs, {"water": 1})
    elif icon == "city":
        _fx(p, gs, {"solari": 1})
    elif icon == "desert":
        _fx(p, gs, {"spice": 1})


# Piter De Vries - Twisted Genius / Harkonnen Advisor
def piter_setup(p, gs):
    from src.game.leader.twisted import create_twisted_intrigues
    deck = create_twisted_intrigues()
    gs.rng.shuffle(deck)
    p.leader.state["twisted_deck"] = deck
    p.leader.state["twisted_discard"] = []
    gs.twisted_owner = p.id


def piter_round_start(p, gs):
    st = p.leader.state
    if not st.get("twisted_deck") and st.get("twisted_discard"):
        st["twisted_deck"] = list(st["twisted_discard"])
        st["twisted_discard"] = []
        gs.rng.shuffle(st["twisted_deck"])
    if st.get("twisted_deck"):
        p.add_intrigue(st["twisted_deck"].pop(0))


def piter_signet(p, gs):
    _fx(p, gs, {"troops": 1})
    p.undeployable_this_turn = getattr(p, "undeployable_this_turn", 0) + 1
    # the recruited troop can't go to the Conflict this turn
    p.troops_recruited_this_turn = max(0, getattr(p, "troops_recruited_this_turn", 0) - 1)


# Steersman Y'rkoon - Strange Form + Hungry for Spice / Plot Course
def steersman_deck_setup(p, gs):
    card = next((c for c in p.deck + p.hand if c.name == "Signet Ring"), None)
    if card is not None:
        (p.deck if card in p.deck else p.hand).remove(card)


def steersman_setup(p, gs):
    from src.game.leader.navigation import NAV_IDS
    p.water = 0                                    # starts without water
    ids = list(NAV_IDS)
    gs.rng.shuffle(ids)
    hand = ids[:5]
    p.leader.state["nav_hand"] = hand
    p.leader.state["nav"] = []
    p.leader.state["nav_next"] = 0
    _choose(p, gs, "Plot Course: choose the Navigation card to return to the box",
            [(f"drop:{i}", f"Return Navigation card {i}; keep the others",
              {"_leader_fx": ["nav_place", i]}) for i in hand], may=False)


@fx("nav_place")
def _fx_nav_place(gs, p, drop: int) -> None:
    """Keep the other 4, then lay them face down in any order: one slot at a time."""
    st = p.leader.state
    st["nav_keep"] = [i for i in st["nav_hand"] if i != drop]
    st["nav"] = []
    st["nav_next"] = 0
    _nav_next_slot(p, gs)


def _nav_suggested(keep):
    """Suggested order: slot-dependent #4 wants slot 1, #3 wants slot 4."""
    best = [i for i in keep if i not in (3, 4)]
    if 4 in keep:
        best.insert(0, 4)
    if 3 in keep:
        best.append(3)
    return best


def _nav_next_slot(p, gs) -> None:
    st = p.leader.state
    left = [i for i in st["nav_keep"] if i not in st["nav"]]
    if len(left) <= 1:
        st["nav"] += left                         # the last card takes the last slot
        return
    slot = len(st["nav"]) + 1
    hint = next(i for i in _nav_suggested(st["nav_keep"]) if i in left)
    _choose(p, gs, f"Plot Course: which Navigation card goes in slot {slot}?",
            [(f"nav:{i}", f"Navigation card {i} in slot {slot}",
              {"_leader_fx": ["nav_slot", i], "_v": ({"spice": 0.5} if i == hint else {})})
             for i in left], may=False)


@fx("nav_slot")
def _fx_nav_slot(gs, p, card: int) -> None:
    p.leader.state["nav"].append(card)
    _nav_next_slot(p, gs)


@fx("nav_order")
def _fx_nav_order(gs, p, order) -> None:         # older saved games: the whole order at once
    p.leader.state["nav"] = list(order)
    p.leader.state["nav_next"] = 0


def steersman_influence_2(p, gs, faction):
    from src.game.leader.navigation import play_navigation
    st = p.leader.state
    k = st.get("nav_next", 0)
    if k < len(st.get("nav", [])):
        st["nav_next"] = k + 1
        agent_turn = (getattr(gs, "_agent_turn_active", False)
                      and getattr(gs, "_agent_turn_owner", None) in (None, p.id))
        if agent_turn or p.id in (gs._agent_turn_open, gs._post_agent_window):
            # it may resolve at any point this turn (user, 2026-10-17): e.g. take
            # Espionage's Spy first and then recall it with card 2, or gain the
            # resources a card asks for; ending the turn plays it at the latest
            gs.defer_reward(p.id, "nav", 1, effect={"card": st["nav"][k], "slot": k + 1,
                                                     "faction": faction})
        else:
            play_navigation(p, gs, st["nav"][k], slot=k + 1, faction=faction)


def steersman_reveal_done(p, gs):
    if p.leader.state.get("nav_reveal_persuasion"):         # Navigation card 3 in slot 4
        gs.gain_persuasion(p.id, 1)
        gs.note(f"{p.leader.name}: +1 persuasion (Navigation card 3 in slot 4)")


def steersman_after_step(p, gs):
    """Hungry for Spice: 3+ spice gained in one of HIS turns (not in Combat)."""
    st = p.leader.state
    serial = getattr(p, "turn_serial", 0)          # bumped by the engine at each turn start
    got = getattr(p, "spice_gained_this_turn", 0)
    if st.get("hungry_serial") != serial:
        st.update(hungry_serial=serial, hungry_acc=0, hungry_done=False, hungry_seen=0)
    delta = got - st.get("hungry_seen", 0)
    st["hungry_seen"] = got
    if delta > 0 and gs.in_own_turn(p.id):
        st["hungry_acc"] = st.get("hungry_acc", 0) + delta
    if st["hungry_acc"] >= 3 and not st.get("hungry_done"):
        st["hungry_done"] = True
        _note(gs, p, "hungry_for_spice")
        _fx(p, gs, {"draw": 1})


# ---------------------------------------------------------------------------
# Leader factory
# ---------------------------------------------------------------------------

def create_leaders(include_bloodlines: bool = True) -> List[Leader]:
    L = Leader
    up = [
        L("Gurney Halleck", 1, on_reveal_done=gurney_reveal_done, signet=gurney_signet),
        L("Feyd-Rautha Harkonnen", 1, on_reveal=feyd_reveal, on_reveal_done=feyd_spy_placed,
          on_spy_placed=feyd_spy_placed, signet=feyd_signet),
        L("Lady Amber Metulli", 1, on_reveal_done=amber_reveal_done, signet=amber_signet),
        L("Lady Jessica", 2, on_agent_placed=jessica_agent_placed, signet=jessica_signet),
        L("Lady Margot Fenring", 1, on_influence_2=margot_influence_2, signet=margot_signet),
        L("Muad'Dib", 1, on_reveal_done=muaddib_reveal_done, signet=muaddib_signet),
        L("Princess Irulan", 2, on_influence_2=irulan_influence_2, signet=irulan_signet),
        L("Shaddam Corrino IV", 2, on_setup=shaddam_setup, signet=shaddam_signet),
        L("Staban Tuek", 3, on_deck_setup=staban_deck_setup,
          on_other_agent_placed=staban_other_agent_placed, signet=staban_signet),
    ]
    bl = [
        L("Chani", 2, on_units_left_conflict=chani_units_left, signet=chani_signet),
        L("Count Hasimir Fenring", 2, on_trash=fenring_trash, signet=fenring_signet),
        L("Duncan Idaho", 1, swordmaster_discount=2, signet=duncan_signet),
        L("Esmar Tuek", 1, on_setup=esmar_setup, on_agent_placed=esmar_agent_placed,
          on_other_agent_placed=esmar_other_agent_placed, signet=esmar_signet),
        L("Gaius Helen Mohiam", 3, card_spy_icon=True, must_gather=True, signet=gaius_signet),
        L("Kota Odax of Ix", 3, on_setup=kota_setup, signet=kota_signet),
        L("Liet Kynes", 3, ignore_gates={"Sietch Tabr"}, no_sandworms=True,
          on_sandworm_replaced=liet_sandworm_replaced, signet=liet_signet),
        L("Piter De Vries", 2, on_setup=piter_setup, on_round_start=piter_round_start,
          signet=piter_signet),
        L("Steersman Y'rkoon", 3, on_deck_setup=steersman_deck_setup, on_setup=steersman_setup,
          on_influence_2=steersman_influence_2, on_after_step=steersman_after_step,
          on_reveal_done=steersman_reveal_done),
    ]
    for ld in bl:
        ld.notes = "Bloodlines"
    return up + (bl if include_bloodlines else [])


LEADER_NAMES = [ld.name for ld in create_leaders()]
ALL_LEADERS = LEADER_NAMES


def get_leader(name: str) -> Leader:
    fresh = {ld.name: ld for ld in create_leaders()}
    if name not in fresh:
        raise KeyError(f"Unknown leader '{name}'. Options: {sorted(fresh)}")
    return fresh[name]


def create_neutral_leader() -> Leader:
    """A Leader with no abilities (Signet Ring becomes a plain card)."""
    return Leader("Neutral", complexity=0, notes="No abilities. Playing with Leaders disabled.")


def available_leaders(gs) -> List[Leader]:
    """Kota needs the Bloodlines tech market, Shaddam the CHOAM contracts."""
    pool = create_leaders(include_bloodlines=getattr(gs, "bl", None) is not None)
    if not gs.use_choam:
        pool = [ld for ld in pool if ld.name != "Shaddam Corrino IV"]
    return pool


def assign_leaders(gs, leaders: Optional[List[str]] = None, neutral: bool = False) -> None:
    """Attach a Leader to each player: `neutral=True` -> no-op Leaders;
    `leaders` = explicit names; otherwise distinct random Leaders."""
    if neutral:
        for player in gs.players:
            player.leader = create_neutral_leader()
        return
    pool = available_leaders(gs)
    if leaders is None:
        gs.rng.shuffle(pool)
        chosen = pool[: gs.num_players]
    else:
        by_name = {ld.name: ld for ld in create_leaders()}
        chosen = [by_name[n] for n in leaders]
    for player, leader in zip(gs.players, chosen):
        player.leader = leader
        if leader.on_deck_setup is not None:
            leader.on_deck_setup(player, gs)
