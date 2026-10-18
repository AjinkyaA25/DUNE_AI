"""Readable game transcripts + rule checks, for reviewing how a game was played
(by you, me, or a stronger LLM).

Replays a recorded game and writes, for every move: round / phase / player /
the move, and exactly what changed for each player (VP, resources, influence,
troops, spies, cards, techs, contracts, ...). Moves that break an invariant are
flagged with "!! CHECK:".

    python -m src.selfplay.transcript game_logs/ui_X.json            # a UI game
    python -m src.selfplay.transcript models_az_rl_scratch/games/g3_12.jsonl.gz --line 0
    python -m src.selfplay.transcript --latest models_az_rl_scratch --count 5

Writes reports/transcripts/<name>.txt and prints the path.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os

ZONES = ("hand", "deck", "discard", "in_play", "trash")


def snapshot(gs, p) -> dict:
    s = {"VP": p.victory_points, "solari": p.solari, "spice": p.spice, "water": p.water,
         "garrison": p.troops_garrison, "supply": p.troops_supply,
         "in_conflict": gs.troops_in_conflict.get(p.id, 0),
         "worms": gs.sandworms_in_conflict.get(p.id, 0),
         "strength": gs.combat_strength.get(p.id, 0),
         "spies_on_board": dict(p.spies_on_board), "spies_supply": p.spies_available,
         "influence": dict(p.influence), "intrigues": len(p.intrigue_cards),
         "hand": len(p.hand), "agents": f"{p.agents_available}/{p.agents_total}",
         "swordmaster": p.has_swordmaster, "high_council": p.has_councilor,
         "alliances": sorted(k for k, v in p.alliances.items() if v),
         "techs": [t.name for t in getattr(p, "techs", [])],
         "contracts": [c.name for c in p.contracts_active],
         "completed": len(p.contracts_completed),
         "trash": sorted(c.name for c in getattr(p, "trash", [])),
         "cards": len(p.hand) + len(p.deck) + len(p.discard) + len(p.in_play)}
    if gs.bl:
        s["commanders"] = (getattr(p, "commanders_garrison", 0), gs.bl.commanders_in_conflict.get(p.id, 0),
                           getattr(p, "commanders_supply", 0))
        s["skills"] = sorted(getattr(p, "skills", set()))
    return s


def checks(gs) -> list:
    """Invariants that must always hold."""
    out = []
    for p in gs.players:
        for r in ("solari", "spice", "water"):
            if getattr(p, r) < 0:
                out.append(f"P{p.id} has negative {r} ({getattr(p, r)})")
        if p.spies_available < 0 or sum(p.spies_on_board.values()) + p.spies_available > 3:
            out.append(f"P{p.id} spies: {p.spies_available} supply + {sum(p.spies_on_board.values())} on board")
        if p.troops_garrison < 0 or p.troops_supply < 0 or gs.troops_in_conflict.get(p.id, 0) < 0:
            out.append(f"P{p.id} negative troop count")
        if gs.bl:
            cg, cc, cs = (getattr(p, "commanders_garrison", 0), gs.bl.commanders_in_conflict.get(p.id, 0),
                          getattr(p, "commanders_supply", 0))
            if min(cg, cc, cs) < 0:
                out.append(f"P{p.id} negative commander count {cg, cc, cs}")
            if cg > p.troops_garrison:
                out.append(f"P{p.id} more commanders ({cg}) than units ({p.troops_garrison}) in garrison")
            if cc > gs.troops_in_conflict.get(p.id, 0):
                out.append(f"P{p.id} more commanders ({cc}) than units in the Conflict")
        units = p.troops_garrison + p.troops_supply + gs.troops_in_conflict.get(p.id, 0)
        cmd = (getattr(p, "commanders_garrison", 0) + gs.bl.commanders_in_conflict.get(p.id, 0)) if gs.bl else 0
        if units - cmd > 12:
            out.append(f"P{p.id} has {units - cmd} troops in total (max 12)")
        if p.agents_available > p.agents_total:
            out.append(f"P{p.id} agents {p.agents_available}/{p.agents_total}")
        for f, v in p.influence.items():
            if not 0 <= v <= 6:
                out.append(f"P{p.id} influence {f} = {v}")
    return out


def diff(a: dict, b: dict) -> str:
    parts = []
    for k in b:
        if a.get(k) != b[k]:
            if isinstance(b[k], (int, float)) and isinstance(a.get(k), (int, float)) and not isinstance(b[k], bool):
                d = b[k] - a[k]
                parts.append(f"{k} {a[k]}->{b[k]} ({d:+})")
            else:
                parts.append(f"{k} {a.get(k)}->{b[k]}")
    return ", ".join(parts)


def transcript(setup: dict, moves: list, title: str, seat_names=None) -> tuple:
    import sys
    sys.argv = sys.argv[:1]
    from src.data.card_definitions import setup_game
    from src.az.mcts import active
    import ui.server as srv
    gs = setup_game(num_players=4, seed=setup["seed"], neutral_leaders=True,
                    use_bloodlines=setup.get("bloodlines", True))
    gs.manual_end_turn = set(setup.get("manual", []))
    lines = [f"# {title}", f"seed {setup['seed']} | players: " +
             ", ".join(f"P{i}={n}" for i, n in enumerate(seat_names or ["?"] * 4)), ""]
    flags = 0
    last_round = None
    for n, vi in enumerate(moves):
        if gs.game_over:
            break
        pid = gs.player_in_reveal_buy if gs.player_in_reveal_buy is not None else active(gs)
        valid = gs.get_valid_actions(pid)
        if not 0 <= vi < len(valid):
            lines.append(f"!! CHECK: move {n} index {vi} not legal ({len(valid)} options) - replay stops")
            flags += 1
            break
        a = valid[vi]
        if gs.round != last_round:
            last_round = gs.round
            cf = gs.current_conflict
            lines.append(f"\n=== Round {gs.round} | Conflict: {cf.name if cf else '-'} | "
                         f"Shield Wall {'up' if gs.shield_wall_intact else 'down'} | "
                         f"Row: {', '.join(c.name for c in gs.imperium_row)}")
        before = [snapshot(gs, p) for p in gs.players]
        label = srv.label(gs, a)
        gs.step(a)
        after = [snapshot(gs, p) for p in gs.players]
        changes = [f"P{i}: {diff(before[i], after[i])}" for i in range(4) if before[i] != after[i]]
        lines.append(f"[{n:>3}] R{gs.round} {gs.phase.value:<12} P{pid} {label}"
                     + (f"\n        {' | '.join(changes)}" if changes else ""))
        for c in checks(gs):
            lines.append(f"        !! CHECK: {c}")
            flags += 1
    vp = [p.victory_points for p in gs.players]
    lines.append(f"\n# end: round {gs.round}, game over {gs.game_over}, VP {vp}, winner {gs.winner}, rule flags {flags}")
    return "\n".join(lines), flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--line", type=int, default=0)
    ap.add_argument("--latest", help="run folder: transcribe its newest recorded games")
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--out", default=os.path.join("reports", "transcripts"))
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    jobs = []
    if A.latest:
        files = sorted(glob.glob(os.path.join(A.latest, "games", "*.jsonl.gz")), key=os.path.getmtime, reverse=True)
        for f in files:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for i, l in enumerate(fh):
                    jobs.append((f, i, json.loads(l)))
                    if len(jobs) >= A.count:
                        break
            if len(jobs) >= A.count:
                break
    elif A.file and A.file.endswith(".jsonl.gz"):
        with gzip.open(A.file, "rt", encoding="utf-8") as fh:
            rec = json.loads(fh.readlines()[A.line])
        jobs.append((A.file, A.line, rec))
    elif A.file:
        d = json.load(open(A.file, encoding="utf-8"))
        rec = {"seed": d["seed"], "moves": [m["vi"] for m in d["moves"]], "ui": d}
        jobs.append((A.file, 0, rec))
    for f, i, rec in jobs:
        ui = rec.get("ui")
        setup = {"seed": rec["seed"], "bloodlines": True,
                 "manual": [ui["human_id"]] if ui and ui.get("human_id") is not None else []}
        names = rec.get("styles") or (["you" if ui and k == ui.get("human_id") else
                                      (ui["opponents"].get(str(k), "AI") if ui else "AI") for k in range(4)])
        name = os.path.basename(f).replace(".jsonl.gz", "").replace(".json", "") + (f"_{i}" if f.endswith(".gz") else "")
        text, flags = transcript(setup, rec["moves"], f"{f} game {i}", names)
        path = os.path.join(A.out, name + ".txt")
        open(path, "w", encoding="utf-8").write(text)
        print(f"{path}: {flags} rule flags")


if __name__ == "__main__":
    main()
