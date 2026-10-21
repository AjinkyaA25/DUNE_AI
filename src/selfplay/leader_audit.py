"""Simulate games with real Leaders, record every game, write readable
transcripts and flag rule violations - for review by a stronger model.

    python -m src.selfplay.leader_audit play --games 60 [--spec heuristic] [--workers 4]
    python -m src.selfplay.leader_audit transcripts [--count 20]
    python -m src.selfplay.leader_audit summary

play         rotates the 18 Leaders so each appears equally often; each game is
             stored as setup seed + leaders + the index of every move among
             get_valid_actions (run with PYTHONHASHSEED=0 so it replays) in
             reports/leader_audit/games.jsonl.
transcripts  replays games into reports/leader_audit/transcripts/game_<n>.txt:
             per move the player (and Leader), the move, what changed, every
             Leader ability that fired, and automatic "!! CHECK" flags.
summary      ability-trigger counts per Leader, flag counts, rounds, win share.

Automatic checks (beyond transcript.checks: negative resources, spies,
troops, influence): Leader missing / neutral, Combat Intrigue played outside
Combat, Plot Intrigue played in Combat, an Intrigue played that the player
did not hold, a Signet Ring played with no Leader ability firing, a Leader
ability raising an exception, engine step errors.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool

OUT = os.path.join("reports", "leader_audit")
GAMES = os.path.join(OUT, "games.jsonl")


def _leaders_for(g: int):
    from src.game.leader.leader_definitions import LEADER_NAMES
    names = list(LEADER_NAMES)
    k = (g * 4) % len(names)
    picked = [names[(k + i) % len(names)] for i in range(4)]
    return picked


def play_one(args):
    g, spec, seed0 = args
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    from src.az.mcts import active
    seed = seed0 + g
    leaders = _leaders_for(g)
    gs = setup_game(4, seed=seed, leaders=leaders, use_bloodlines=True)
    ag = [make_agent(spec, seed=seed * 4 + i) for i in range(4)]
    moves, errors, n = [], [], 0
    try:
        while not gs.game_over and n < 5000:
            pid = active(gs)
            valid = gs.get_valid_actions(pid)
            a = ag[pid].select_action(gs, pid, valid)
            k = next(i for i, v in enumerate(valid) if repr(v) == repr(a))
            moves.append(k)
            _, _, _, info = gs.step(a)
            if info.get("error"):
                errors.append((n, info["error"]))
            n += 1
    except Exception as e:                       # a crash is a finding, not the end of the run
        import traceback
        errors.append((n, "EXCEPTION " + traceback.format_exc()[-600:]))
    return {"g": g, "seed": seed, "leaders": leaders, "spec": spec, "moves": moves,
            "rounds": gs.round, "game_over": gs.game_over,
            "vp": [p.victory_points for p in gs.players], "winner": gs.winner, "errors": errors}


def play(n, spec, workers, seed0):
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(GAMES):
        done = {json.loads(l)["g"] for l in open(GAMES) if l.strip()}
    jobs = [(g, spec, seed0) for g in range(n) if g not in done]
    with Pool(workers, maxtasksperchild=20) as pool, open(GAMES, "a") as f:
        for k, r in enumerate(pool.imap_unordered(play_one, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            if k % 10 == 0:
                print(f"  {k}/{len(jobs)} games", flush=True)


def load():
    return [json.loads(l) for l in open(GAMES) if l.strip()]


# ------------------------------------------------------------------ transcripts

def _leader_state(p):
    ld = p.leader
    if ld is None:
        return "NO LEADER"
    st = {k: v for k, v in ld.state.items() if k not in ("twisted_deck", "twisted_discard", "nav_hand")}
    if "twisted_deck" in ld.state:
        st["twisted_deck"] = len(ld.state["twisted_deck"])
        st["twisted_discard"] = len(ld.state["twisted_discard"])
    return f"{ld.name} {st}" if st else ld.name


def transcript(game: dict):
    sys.argv = sys.argv[:1]
    from src.data.card_definitions import setup_game
    from src.az.mcts import active
    from src.game.gameState import ActionType, Phase
    from src.game.intrigue.intrigue import IntrigueTiming
    from src.selfplay.transcript import snapshot, checks, diff
    import ui.server as srv
    gs = setup_game(4, seed=game["seed"], leaders=game["leaders"], use_bloodlines=True)
    gs.leader_log = []
    lines = [f"# Leader audit game {game['g']} (seed {game['seed']}, agents {game['spec']})",
             "players: " + ", ".join(f"P{i}={n}" for i, n in enumerate(game["leaders"])), ""]
    for p in gs.players:
        lines.append(f"P{p.id} start: {_leader_state(p)} | deck {sorted(c.name for c in p.deck + p.hand)} | "
                     f"water {p.water} solari {p.solari} spice {p.spice}")
    flags = collections.Counter()

    def flag(msg):
        flags[msg.split(":")[0]] += 1
        lines.append(f"        !! CHECK: {msg}")

    for p in gs.players:
        if p.leader is None or p.leader.name == "Neutral":
            flag(f"P{p.id} has no real Leader")
    last_round = None
    for n, vi in enumerate(game["moves"]):
        if gs.game_over:
            break
        pid = active(gs)
        valid = gs.get_valid_actions(pid)
        if not 0 <= vi < len(valid):
            flag(f"replay: move {n} index {vi} not legal ({len(valid)} options) - replay stops")
            break
        a = valid[vi]
        if gs.round != last_round:
            last_round = gs.round
            cf = gs.current_conflict
            lines.append(f"\n=== Round {gs.round} | Conflict: {cf.name if cf else '-'} | "
                         f"Row: {', '.join(c.name for c in gs.imperium_row)}")
            lines.append("    leaders: " + " | ".join(f"P{p.id} {_leader_state(p)}" for p in gs.players))
        pc = gs.pending_choice_for(pid)
        prompt = f"  [choice: {pc.prompt}]" if (pc is not None and a.action_type == ActionType.RESOLVE_CHOICE) else ""
        # intrigue timing checks
        if a.action_type == ActionType.PLAY_INTRIGUE:
            p = gs.players[pid]
            ic = next((c for c in p.intrigue_cards if c.name == a.intrigue_card_name), None)
            if ic is None:
                flag(f"intrigue: P{pid} played {a.intrigue_card_name} without holding it")
            else:
                ts = ic.timing if isinstance(ic.timing, (tuple, list, set, frozenset)) else (ic.timing,)
                if gs.phase == Phase.COMBAT and IntrigueTiming.COMBAT not in ts:
                    flag(f"intrigue: P{pid} played non-Combat {ic.name} during Combat")
                if gs.phase != Phase.COMBAT and ts == (IntrigueTiming.COMBAT,):
                    flag(f"intrigue: P{pid} played Combat-only {ic.name} outside Combat")
        before = [snapshot(gs, p) for p in gs.players]
        lstate = [_leader_state(p) for p in gs.players]
        k0 = len(gs.leader_log)
        label = srv.label(gs, a)
        _, _, _, info = gs.step(a)
        after = [snapshot(gs, p) for p in gs.players]
        changes = [f"P{i}: {diff(before[i], after[i])}" for i in range(4) if before[i] != after[i]]
        lchg = [f"P{i} leader {lstate[i]} -> {_leader_state(gs.players[i])}"
                for i in range(4) if lstate[i] != _leader_state(gs.players[i])]
        fired = gs.leader_log[k0:]
        lines.append(f"[{n:>3}] R{gs.round} {gs.phase.value:<12} P{pid}({game['leaders'][pid]}) {label}{prompt}"
                     + (f"\n        {' | '.join(changes)}" if changes else "")
                     + (f"\n        ability: {', '.join(f'P{q} {h}' for q, _, h in fired)}" if fired else "")
                     + (f"\n        {' | '.join(lchg)}" if lchg else ""))
        if info.get("error"):
            flag(f"engine error: {info['error'][:160]}")
        if a.action_type == ActionType.AGENT_TURN and a.card_name == "Signet Ring" \
                and not any(h == "signet" for _, _, h in fired):
            flag(f"signet: P{pid} played the Signet Ring but no Leader ability fired")
        for c in checks(gs):
            flag(f"state: {c}")
    vp = [p.victory_points for p in gs.players]
    lines.append(f"\n# end: round {gs.round}, game over {gs.game_over}, VP {vp}, winner {gs.winner}")
    for n, e in game.get("errors", []):
        flag(f"recorded error at move {n}: {e[:300]}")
    trig = collections.Counter((game["leaders"][q], h) for q, _, h in gs.leader_log)
    lines.append("# leader abilities fired: " + ", ".join(f"{l} {h} x{c}" for (l, h), c in sorted(trig.items())))
    lines.append(f"# flags: {dict(flags)}")
    return "\n".join(lines), flags, trig


def write_transcripts(count: int):
    d = os.path.join(OUT, "transcripts")
    os.makedirs(d, exist_ok=True)
    tot_flags, tot_trig = collections.Counter(), collections.Counter()
    for game in load()[:count]:
        text, flags, trig = transcript(game)
        open(os.path.join(d, f"game_{game['g']:03d}.txt"), "w", encoding="utf-8").write(text)
        tot_flags.update(flags)
        tot_trig.update(trig)
    print("flags:", dict(tot_flags))
    print("abilities fired:")
    for (l, h), c in sorted(tot_trig.items()):
        print(f"  {l:24} {h:24} {c}")


def summary():
    games = load()
    wins, seen, rounds, errs = collections.Counter(), collections.Counter(), [], 0
    for g in games:
        rounds.append(g["rounds"])
        errs += len(g["errors"])
        for i, l in enumerate(g["leaders"]):
            seen[l] += 1
            wins[l] += g["winner"] == i
    print(f"{len(games)} games, avg rounds {sum(rounds) / max(1, len(rounds)):.2f}, "
          f"not finished {sum(not g['game_over'] for g in games)}, recorded errors {errs}")
    for l in sorted(seen, key=lambda x: -wins[x] / seen[x]):
        print(f"  {l:24} {wins[l]:>3}/{seen[l]:<3} wins ({wins[l] / seen[l]:.0%})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("play")
    a.add_argument("--games", type=int, default=60)
    a.add_argument("--spec", default="heuristic")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--seed", type=int, default=9_000_000)
    b = sub.add_parser("transcripts")
    b.add_argument("--count", type=int, default=20)
    sub.add_parser("summary")
    A = ap.parse_args()
    if A.cmd == "play":
        play(A.games, A.spec, A.workers, A.seed)
    elif A.cmd == "transcripts":
        write_transcripts(A.count)
    else:
        summary()


if __name__ == "__main__":
    main()
