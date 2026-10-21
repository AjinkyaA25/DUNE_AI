"""Leader study: play many 4-player games with real Leaders, check the rules
live on every move, and report how each Leader plays and how fast games end.

    python -m src.selfplay.leader_study play --games 1800 [--spec look2] [--workers 16]
    python -m src.selfplay.leader_study report

play    rotates the 18 Leaders (each appears equally often, every seat) and
        writes one JSON line per game to reports/leader_study/games.jsonl:
        leaders, winner, VP per round, Leader abilities fired, and every
        rule-check flag with the move it happened on. Moves are stored as
        indices into get_valid_actions (replay with PYTHONHASHSEED=0).
report  per-Leader table (win share, VP, rounds of the games it wins, share
        of its wins by round 8, abilities fired per game) plus a flag
        summary -> reports/leader_study/report.md

Live checks per move (on top of transcript.checks: negative resources,
spy / troop / influence limits): an Intrigue played that was not held or in
the wrong phase, a Signet Ring with no Leader ability firing, a card object
in two places at once, Twisted Intrigue count, Leader track positions out of
range, a missing Leader, engine errors and exceptions, a game that never ends.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from multiprocessing import Pool

OUT = os.path.join("reports", "leader_study")
GAMES = os.path.join(OUT, "games.jsonl")
SPECS = {
    "look2": "look:D2:K3:M1:ev=models_eval_look/evaluator.npz:tuned=config/evolve_look/champion.json",
    "heur": "heuristic:tuned=config/evolve_look/champion.json",
}


def _leaders_for(g: int):
    from src.game.leader.leader_definitions import LEADER_NAMES
    import random
    # every block of 9 games seats each of the 18 Leaders exactly twice, with
    # random table-mates and seats
    perm = random.Random(g // 9).sample(list(LEADER_NAMES), len(LEADER_NAMES)) * 2
    i = (g % 9) * 4
    return perm[i:i + 4]


def _card_places(gs):
    """Yield (place, card) for every card object the game holds."""
    for p in gs.players:
        for zone in ("deck", "hand", "discard", "in_play", "trash", "intrigue_cards"):
            for c in getattr(p, zone, []) or []:
                yield f"P{p.id}.{zone}", c
        if p.leader is not None:
            for zone in ("twisted_deck", "twisted_discard"):
                for c in p.leader.state.get(zone, []) or []:
                    yield f"P{p.id}.{zone}", c
    for zone in ("imperium_row", "imperium_deck", "intrigue_deck", "intrigue_discard"):
        for c in getattr(gs, zone, []) or []:
            yield zone, c


_STARTERS = None


def _starter_names():
    global _STARTERS
    if _STARTERS is None:
        from src.data.card_definitions import create_starter_cards
        _STARTERS = {c.name for c in create_starter_cards()}
    return _STARTERS


def live_checks(gs, pid, a, fired, phase_before, held_before):
    from src.game.gameState import ActionType, Phase
    from src.game.intrigue.intrigue import IntrigueTiming
    from src.selfplay.transcript import checks
    out = list(checks(gs))
    if a.action_type == ActionType.PLAY_INTRIGUE:
        ic = held_before.get(a.intrigue_card_name)
        if ic is None:
            out.append(f"intrigue: P{pid} played {a.intrigue_card_name} without holding it")
        else:
            ts = ic.timing if isinstance(ic.timing, (tuple, list, set, frozenset)) else (ic.timing,)
            if phase_before == Phase.COMBAT and IntrigueTiming.COMBAT not in ts:
                out.append(f"intrigue: P{pid} played non-Combat {ic.name} during Combat")
            if phase_before != Phase.COMBAT and tuple(ts) == (IntrigueTiming.COMBAT,):
                out.append(f"intrigue: P{pid} played Combat-only {ic.name} outside Combat")
    if a.action_type == ActionType.AGENT_TURN and a.card_name == "Signet Ring" \
            and not any(h == "signet" for q, _, h in fired if q == pid):
        out.append(f"signet: P{pid} ({gs.players[pid].leader.name}) played the Signet Ring, no ability fired")
    seen = {}
    for place, c in _card_places(gs):
        if getattr(c, "name", None) in _starter_names():     # starter cards are shared definitions
            continue
        if id(c) in seen:
            out.append(f"duplicate card: {getattr(c, 'name', c)} in {seen[id(c)]} and {place}")
        seen.setdefault(id(c), place)
    for p in gs.players:
        ld = p.leader
        if ld is None or ld.name == "Neutral":
            out.append(f"leader: P{p.id} has no real Leader")
            continue
        st = ld.state
        if "twisted_deck" in st:
            n = len(st["twisted_deck"]) + len(st["twisted_discard"]) + sum(
                getattr(c, "twisted", False) for q in gs.players for c in q.intrigue_cards)
            if n > 12:
                out.append(f"twisted: {n} Twisted Intrigues in the game (max 12)")
        if "tactics" in st and not 2 <= st["tactics"] < 10:
            out.append(f"chani: tactics track at {st['tactics']}")
        if "nav" in st and len(set(st["nav"])) != len(st["nav"]):
            out.append(f"steersman: duplicate Navigation cards {st['nav']}")
    return out


def play_one(args):
    g, spec, seed0 = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.az.mcts import active
    from src.game.gameState import ActionType
    seed = seed0 + g
    leaders = _leaders_for(g)
    gs = setup_game(4, seed=seed, leaders=leaders, use_bloodlines=True)
    gs.leader_log = []
    ag = [make_agent(SPECS.get(spec, spec), seed=seed * 4 + i) for i in range(4)]
    moves, flags, n = [], [], 0
    vp_by_round, last_round = [], gs.round
    fired_all = collections.Counter()
    try:
        while not gs.game_over and n < 6000:
            pid = active(gs)
            valid = gs.get_valid_actions(pid)
            if not valid:
                flags.append((n, gs.round, f"stuck: P{pid} has no legal action in {gs.phase.value}"))
                break
            a = ag[pid].select_action(gs, pid, valid)
            k = next(i for i, v in enumerate(valid) if repr(v) == repr(a))
            moves.append(k)
            phase_before = gs.phase
            held = {c.name: c for c in gs.players[pid].intrigue_cards}
            k0 = len(gs.leader_log)
            _, _, _, info = gs.step(a)
            fired = gs.leader_log[k0:]
            for q, name, h in fired:
                fired_all[f"{name}|{h}"] += 1
            if info.get("error"):
                flags.append((n, gs.round, f"engine error: {str(info['error'])[:200]}"))
            for f in live_checks(gs, pid, a, fired, phase_before, held):
                flags.append((n, gs.round, f))
            if gs.round != last_round:
                vp_by_round.append([p.victory_points for p in gs.players])
                last_round = gs.round
            n += 1
        if not gs.game_over:
            flags.append((n, gs.round, f"unfinished: game not over after {n} moves"))
    except Exception:
        import traceback
        flags.append((n, gs.round, "EXCEPTION " + traceback.format_exc()[-700:]))
    return {"g": g, "seed": seed, "leaders": leaders, "spec": spec, "moves": moves,
            "rounds": gs.round, "game_over": gs.game_over, "winner": gs.winner,
            "vp": [p.victory_points for p in gs.players], "vp_by_round": vp_by_round,
            "fired": dict(fired_all), "flags": flags[:200]}


def play(n, spec, workers, seed0):
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(GAMES):
        done = {json.loads(l)["g"] for l in open(GAMES) if l.strip()}
    jobs = [(g, spec, seed0) for g in range(n) if g not in done]
    with Pool(workers, maxtasksperchild=25) as pool, open(GAMES, "a") as f:
        for k, r in enumerate(pool.imap_unordered(play_one, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)} games", flush=True)


def load():
    return [json.loads(l) for l in open(GAMES) if l.strip()]


def _flag_kind(msg: str) -> str:
    return msg.split(":")[0] if not msg.startswith("EXCEPTION") else "EXCEPTION"


def report():
    games = load()
    rows = collections.defaultdict(lambda: collections.Counter())
    win_rounds = collections.defaultdict(list)
    fired = collections.defaultdict(collections.Counter)
    flag_kinds, flag_examples = collections.Counter(), {}
    rounds = [g["rounds"] for g in games]
    for g in games:
        for i, l in enumerate(g["leaders"]):
            r = rows[l]
            r["games"] += 1
            r["wins"] += g["winner"] == i
            r["vp"] += g["vp"][i]
            if len(g["vp_by_round"]) >= 6:
                r["vp_r6"] += g["vp_by_round"][5][i]
                r["n_r6"] += 1
            if g["winner"] == i:
                win_rounds[l].append(g["rounds"])
        for key, c in g["fired"].items():
            name, h = key.split("|")
            fired[name][h] += c
        for n, rd, msg in g["flags"]:
            k = _flag_kind(msg)
            flag_kinds[k] += 1
            flag_examples.setdefault(k, []).append(f"game {g['g']} move {n} R{rd}: {msg[:300]}")
    lines = [f"# Leader study - {len(games)} games ({games[0]['spec'] if games else '-'})", "",
             f"Average rounds per game: **{sum(rounds) / max(1, len(rounds)):.2f}** | "
             f"ended by round 8: {sum(r <= 8 for r in rounds) / max(1, len(rounds)):.0%} | "
             f"unfinished: {sum(not g['game_over'] for g in games)}", "",
             "Round distribution: " + ", ".join(f"R{k}: {v}" for k, v in sorted(collections.Counter(rounds).items())),
             "", "| Leader | games | win % | avg VP | VP after R6 | rounds when it wins | its wins by R8 | abilities fired / game |",
             "|---|---|---|---|---|---|---|---|"]
    for l in sorted(rows, key=lambda x: -rows[x]["wins"] / rows[x]["games"]):
        r, wr = rows[l], win_rounds[l]
        ab = ", ".join(f"{h} {c / r['games']:.1f}" for h, c in fired[l].most_common())
        lines.append(f"| {l} | {r['games']} | {r['wins'] / r['games']:.1%} | {r['vp'] / r['games']:.2f} | "
                     f"{r['vp_r6'] / max(1, r['n_r6']):.2f} | {sum(wr) / max(1, len(wr)):.2f} | "
                     f"{sum(x <= 8 for x in wr) / max(1, len(wr)):.0%} | {ab or '-'} |")
    lines += ["", f"## Rule-check flags: {sum(flag_kinds.values())}", ""]
    for k, c in flag_kinds.most_common():
        lines.append(f"- **{k}** x{c}")
        lines += [f"    - {e}" for e in flag_examples[k][:5]]
    text = "\n".join(lines)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("play")
    a.add_argument("--games", type=int, default=1800)
    a.add_argument("--spec", default="look2")
    a.add_argument("--workers", type=int, default=16)
    a.add_argument("--seed", type=int, default=11_000_000)
    sub.add_parser("report")
    A = ap.parse_args()
    if A.cmd == "play":
        play(A.games, A.spec, A.workers, A.seed)
    else:
        report()


if __name__ == "__main__":
    main()
