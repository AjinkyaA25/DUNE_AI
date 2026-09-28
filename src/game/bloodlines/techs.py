"""
Bloodlines tech tiles (plus the Rise of Ix tiles the user's TTS games also
use), transcribed from the tile images in `Tech tiles/`.

Market: the pool is shuffled into 3 face-up stacks; buying a tile reveals the
next one of that stack. A tile may be bought (with spice) when you send an
Agent to a green Landsraad space; a High Council seat makes it 1 spice
cheaper.

Each tile is data: effect dicts in the engine's usual vocabulary (see
effects.py) keyed by *when* they fire. Hooks in GameState call
`fire(player, tech, when)`; the few tiles that need custom logic (endgame
scoring, cost/icon modifiers) are handled by name in bloodlines/rules.py.

Icon readings (confirmed by the user): black "?" diamond = +1 influence with
any faction (red base = lose 1); cylinder = place a Spy; plain green card =
draw a card; flip icon = once per round.

Left out of the pool (user decision): Detonation Devices (dreadnoughts) and
Troop Transports (Shipping track) - Rise of Ix pieces Uprising doesn't have.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TechDef:
    name: str
    cost: int                                   # spice
    acquire: List[Dict] = field(default_factory=list)
    reveal: List[Dict] = field(default_factory=list)
    command: List[Dict] = field(default_factory=list)   # reveal, 6+ persuasion
    round_start: List[Dict] = field(default_factory=list)
    on_win_conflict: List[Dict] = field(default_factory=list)
    on_complete_contract: List[Dict] = field(default_factory=list)
    # Once-per-round activation (flip icon): {"cost": {...}, "discard": n,
    # "effects": [...], "turn": "agent"|"any", "trash_self": bool}
    activation: Optional[Dict] = None
    note: str = ""          # rules text handled in code / display only
    unsure: str = ""        # icon reading that still needs confirming


TECHS: List[TechDef] = [
    TechDef("Advanced Data Analysis", 3,
            activation={"effects": [{"intrigue": 1}], "turn": "any"},
            note="To acquire this you must trash one of your Spies from the "
                 "board (return it to the box)."),
    TechDef("Artillery", 1,
            note="Reveal Turn: +1 sword for each revealed card that provides "
                 "1 or more swords this turn."),
    TechDef("Chaumurky", 4, acquire=[{"intrigue": 2}],
            note="Endgame: you win tiebreakers."),
    TechDef("CHOAM Transports", 6,
            on_complete_contract=[{"draw": 1}],
            note="Endgame: worth 1 VP if you have completed 4+ contracts.",
            unsure="acquire icon (small eye/spy-post box) not transcribed"),
    TechDef("Delivery Bay", 3, acquire=[{"draw": 1}],
            command=[{"solari": 2}]),
    TechDef("Disposal Facility", 3, acquire=[{"trash": 1}],
            note="Reveal Turn: if you have 6+ persuasion, you may trash one "
                 "of your cards in play."),
    TechDef("Rapid Dropships", 4, acquire=[{"troops": 2}],
            activation={"effects": [], "turn": "agent", "flag": "dropships"},
            note="Agent Turn (flip): this Agent turn counts as sending your "
                 "Agent to a Combat space (you may deploy troops).",
            unsure="read the crossed-blades icon as 'combat space / deploy'"),
    TechDef("Flagship", 8, acquire=[{"vp": 1}],
            activation={"cost": {"solari": 4}, "effects": [{"troops": 3}],
                        "turn": "any"}),
    TechDef("Forbidden Weapons", 2, acquire=[{"troops": 1}],
            note="Reveal Turn: you must choose: 3 swords and lose 1 influence "
                 "with any faction, OR lose all your spice and trash this.",
            unsure="red acquire icon (above the troop) not transcribed"),
    TechDef("Gene-Locked Vault", 2, acquire=[{"intrigue": 1}],
            note="Acquire: an Intrigue OR draw a card (engine takes the "
                 "Intrigue). Your Intrigue cards can't be stolen unless you "
                 "have five or more."),
    TechDef("Glowglobes", 2, acquire=[{"influence_any": 1}],
            note="You may look at the top card of your deck at any time."),
    TechDef("Holoprojectors", 3,
            activation={"discard": 1, "effects": [{"draw": 1}], "turn": "any"}),
    TechDef("Holtzman Engine", 6, round_start=[{"draw": 1}],
            note="Endgame: worth 1 VP if you have at least two The Spice "
                 "Must Flow."),
    TechDef("Invasion Ships", 5, acquire=[{"troops": 4}],
            activation={"discard": 1, "effects": [], "turn": "agent",
                        "flag": "ignore_blocking"},
            note="Flip + discard a card: enemy Agents don't block your Agent "
                 "this turn."),
    TechDef("Memocorders", 2, acquire=[{"influence_any": 1}],
            note="Endgame: 1 VP if you have 3+ influence on all four tracks."),
    TechDef("Minimic Film", 2, reveal=[{"persuasion": 1}]),
    TechDef("Navigation Chamber", 5, acquire=[{"influence_any": 1}],
            note="Board spaces cost you 1 spice or 1 solari less."),
    TechDef("Ornithopter Fleet", 4, acquire=[{"troops": 2}],
            note="All of your battle icons are wild (ornithopter)."),
    TechDef("Panopticon", 5, reveal=[{"spy": 1, "troops": 1}],
            note="Endgame: gain 1 influence with each faction where you have "
                 "1 or less influence."),
    TechDef("Planetary Array", 2, acquire=[{"trash": 1}],
            on_win_conflict=[{"draw": 1}]),
    TechDef("Plasteel Blades", 3, acquire=[{"solari": 4}],
            note="Whenever you recruit a Sardaukar Commander: trash this -> "
                 "gain an additional Sardaukar Commander skill."),
    TechDef("Restricted Ordnance", 4,
            note="Reveal Turn: if you have a seat on the High Council: "
                 "4 swords."),
    TechDef("Sardaukar High Command", 7, acquire=[{"vp": 1}],
            note="Recruiting a Sardaukar Commander (including when you "
                 "acquire one) costs you 1 solari less."),
    TechDef("Self-Destroying Messages", 4, acquire=[{"intrigue": 2}],
            reveal=[{"persuasion": 1}]),
    TechDef("Servo-Receivers", 2,
            note="Your Signet Ring has the Emperor, Spacing Guild, Bene "
                 "Gesserit and Fremen icons.",
            unsure="acquire icon (a signet ring) read as: resolve your "
                   "leader's Signet Ring ability once"),
    TechDef("Shuttle Fleet", 6, acquire=[{"influence_any": 2}],
            round_start=[{"solari": 2}],
            note="Acquire: choose two different factions, +1 influence each "
                 "(engine: two any-faction choices)."),
    TechDef("Sonic Snoopers", 2, acquire=[{"intrigue": 1}],
            activation={"effects": [], "turn": "any", "trash_self": True,
                        "flag": "cycle_intrigues"},
            note="Trash this -> put any number of your Intrigue cards on the "
                 "bottom of the Intrigue deck, then draw that many."),
    TechDef("Spaceport", 5, acquire=[{"draw": 2}],
            note="You may put cards you acquire on top of your deck."),
    TechDef("Spy Drones", 5, acquire=[{"spy": 2}],
            activation={"effects": [{"solari": 1}], "turn": "any",
                        "flag": "spy_drones"},
            note="Flip: 1 solari AND if you recalled a Spy this turn: trash "
                 "a card."),
    TechDef("Spy Satellites", 4,
            activation={"cost": {"spice": 3}, "effects": [{"vp": 1}],
                        "turn": "any", "trash_self": True},
            note="Endgame: worth 1 VP for each faction where you have 1 or "
                 "less influence."),
    TechDef("Suspensor Suits", 3,
            note="For each Intrigue card you draw or steal during your turn: "
                 "1 troop, deploy it to the Conflict."),
    TechDef("Training Depot", 1, command=[{"swords": 2}]),
    TechDef("Training Drones", 3,
            activation={"effects": [{"troops": 1}], "turn": "any"}),
    TechDef("Windtraps", 2, acquire=[{"water": 1}],
            on_win_conflict=[{"water": 1}]),
]

# In the user's tile folder but not in the engine pool (Rise of Ix pieces).
EXCLUDED = {
    "Detonation Devices": "dreadnoughts (Rise of Ix only)",
    "Troop Transports": "Shipping track (Rise of Ix only)",
}

TECH_BY_NAME: Dict[str, TechDef] = {t.name: t for t in TECHS}


class TechTile:
    """A tile owned by a player (tracks its once-per-round flip)."""

    def __init__(self, d: TechDef):
        self.d = d
        self.name = d.name
        self.cost = d.cost
        self.used_this_round = False

    # Player/Combat code written for the Rise-of-Ix Tech class asks these.
    def has_passive_reveal_bonus(self) -> bool:
        return False

    def reset_for_round(self) -> None:
        self.used_this_round = False

    def to_dict(self) -> Dict:
        return {"name": self.name, "cost": self.cost,
                "used_this_round": self.used_this_round}

    def __repr__(self) -> str:
        return f"TechTile({self.name})"
