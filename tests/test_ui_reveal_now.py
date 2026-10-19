"""The UI's 'persuasion if you revealed now' (ui/server.py Session._reveal_now)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui"))

import server as S  # noqa: E402
from tests.test_leader_audit_fixes import _card, _game  # noqa: E402


def _now(gs, pid=0):
    class F:
        log = []
    return S.Session._reveal_now(F(), gs, pid)["persuasion"]


def _hand(names):
    gs = _game(["Gurney Halleck", "Chani", "Lady Jessica", "Duncan Idaho"])
    p = gs.players[0]
    p.hand = [_card(gs, n) for n in names]
    p.troops_garrison = 0
    return gs, p


def test_conditional_cards_count_at_their_current_value():
    gs, p = _hand(["Bene Gesserit Operative", "Command Center", "Imperial Throneship"])
    assert _now(gs) == 1 + 1 + 2                      # <2 Spies, no retreat yet, garrison < 4
    p.spies_on_board = {"Emperor Post": 1, "Fremen Post": 1}
    p.troops_garrison = 4
    assert _now(gs) == 3 + 1 + 3                      # Operative 3, Throneship +1


def test_high_council_and_assembly_hall_count_but_not_once_the_agent_left():
    gs, p = _hand(["Convincing Argument"])
    p.has_councilor = True
    gs.agent_on_space["Assembly Hall"] = p.id
    assert _now(gs) == 2 + 2 + 1
    gs.agent_on_space["Assembly Hall"] = None         # e.g. Duncan's Into the Fray moved it
    assert _now(gs) == 2 + 2
