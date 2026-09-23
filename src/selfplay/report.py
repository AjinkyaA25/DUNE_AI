"""
Self-play diagnostic report — the standing "how is the AI evolving" readout.

Runs N games with a given agent spec (default: 4x value:models/value_best.npz)
and prints:

  * seat / turn-order win-rate (and first-player seat effect)
  * game-length distribution + endgame-trigger rate + winner VP by final round
  * most-bought cards: count, buys/game, median round, buyer win-rate
  * cards that win the most / least (buyer win-rate, min-sample gated)
  * intrigue usage + anomalies (combat cards dumped off-combat, endgame cards
    left unplayed, cards never drawn/played, intrigues stranded in hand)
  * round 7 & 8 combat scoring (reward VP, control rate, winner strength, VP swing)
  * final-VP distribution (all seats + winners) with a rough VP-source split
  * friendships / alliances per game (all seats + winners)

Usage:
  python -m src.selfplay.report --games 150
  python -m src.selfplay.report --games 150 --agent value:models/value_v08.npz
  python -m src.selfplay.report --games 120 --agent heuristic --tag baseline
"""
from __future__ import annotations

import argparse
import statistics as st
from collections import Counter, defaultdict
from typing import Dict, List

from src.ai.agents import make_agent
from src.selfplay.game_log import record_game

_RESERVE_NAME = {"spice_must_flow": "The Spice Must Flow",
                 "prepare_the_way": "Prepare the Way"}


def _rget(rewards: dict, pid: int) -> dict:
    """record_game keeps int keys in-process but str keys after a JSON round
    trip — accept either."""
    return rewards.get(pid) or rewards.get(str(pid)) or {}


def _guaranteed_vp(d: dict) -> int:
    """Top-level guaranteed 'vp' only (ignores optional may_pay_*_for_vp)."""
    v = d.get("vp", 0)
    return int(v) if isinstance(v, (int, float)) else 0


def _sum_vp(d) -> int:
    """Recursively total any 'vp' entries in a reward dict (incl. auto-taken
    conversions)."""
    tot = 0
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "vp" and isinstance(v, (int, float)):
                tot += int(v)
            elif isinstance(v, dict):
                tot += _sum_vp(v)
    return tot


def _intrigue_deck():
    from src.data.card_definitions import create_intrigue_deck
    return create_intrigue_deck()


def _load_intrigue_names() -> List[str]:
    try:
        return sorted({c.name for c in _intrigue_deck()})
    except Exception:
        return []


def run(n_games: int, agent_spec: str, base_seed: int = 900_000) -> Dict:
    games = []
    for i in range(n_games):
        agents = {s: make_agent(agent_spec, seed=(base_seed + i) * 4 + s)
                  for s in range(4)}
        games.append(record_game(agents, seed=base_seed + i, num_players=4))
    return {"games": games, "agent": agent_spec, "n": n_games}


# --------------------------------------------------------------------------

def _seat_winrate(games) -> None:
    seat_wins = Counter()
    fp_seat_wins = Counter()      # did the round-1 first player win?
    fp_count = Counter()
    for g in games:
        w = g["summary"]["winner"]
        seat_wins[w] += 1
        fp = None
        for r in g["rounds"]:
            if r.get("type") == "round_start" and r["round"] == 1:
                fp = r.get("first_player")
                break
        if fp is not None:
            fp_count[fp] += 1
            if fp == w:
                fp_seat_wins["won"] += 1
    n = len(games)
    print("SEAT / TURN-ORDER WIN-RATE")
    for s in range(4):
        print(f"  seat {s}: {seat_wins[s]:3d}/{n}  = {seat_wins[s]/n:5.1%}   (fair 25.0%)")
    fpw = fp_seat_wins["won"]
    fpn = sum(fp_count.values())
    if fpn:
        print(f"  round-1 first player wins the game: {fpw}/{fpn} = {fpw/fpn:.1%}")
    print()


def _game_length(games) -> None:
    n = len(games)
    ln = Counter(g["summary"]["rounds_played"] for g in games)
    winner_vp_by_round = defaultdict(list)
    endgame_hits = 0
    for g in games:
        s = g["summary"]
        winner_vp_by_round[s["rounds_played"]].append(s["final_vp"][s["winner"]])
        if any(v >= 10 for v in s["final_vp"]):
            endgame_hits += 1
    print("GAME LENGTH")
    for r in sorted(ln):
        vps = winner_vp_by_round[r]
        print(f"  R{r:<2d}: {ln[r]:3d} ({ln[r]/n:5.1%})   winner VP avg {st.mean(vps):4.1f}")
    med = st.median([g["summary"]["rounds_played"] for g in games])
    print(f"  median length: R{med:g}   10-VP threshold reached: {endgame_hits}/{n} = {endgame_hits/n:.1%}")
    print()


def _iter_buys(g):
    for ev in g["events"]:
        act = ev.get("action")
        if act == "acquire_card":
            yield ev.get("card"), ev.get("round", 0), ev.get("player")
        elif act == "acquire_reserve":
            yield (_RESERVE_NAME.get(ev.get("reserve"), ev.get("reserve")),
                   ev.get("round", 0), ev.get("player"))


def _cards(games) -> None:
    n = len(games)
    count = Counter()
    rounds = defaultdict(list)
    wl = defaultdict(lambda: [0, 0])       # card -> [by winner, by loser]
    per_round = Counter()
    for g in games:
        w = g["summary"]["winner"]
        for card, rnd, pid in _iter_buys(g):
            if card is None:
                continue
            count[card] += 1
            rounds[card].append(rnd)
            wl[card][0 if pid == w else 1] += 1
            per_round[rnd] += 1
    total = sum(count.values())
    print("MOST-BOUGHT CARDS")
    print(f"  {'card':30} {'n':>5} {'/game':>6} {'medRd':>6} {'buyerWin%':>10}")
    for card, c in count.most_common(25):
        win, los = wl[card]
        wr = win / (win + los) if (win + los) else 0.0
        print(f"  {card:30.30} {c:>5} {c/n:>6.2f} {st.median(rounds[card]):>6.1f} {wr:>9.1%}")
    print(f"  total buys/game: {total/n:.2f}  ({total/n/4:.2f}/player)")

    gated = [(card, wl[card][0] / (wl[card][0] + wl[card][1]), count[card])
             for card in count if (wl[card][0] + wl[card][1]) >= max(8, n // 12)]
    gated.sort(key=lambda t: t[1])
    print("\n  CARDS THAT WIN LEAST (buyer win-rate, >= %d buys):" % max(8, n // 12))
    for card, wr, c in gated[:8]:
        print(f"    {card:30.30} {wr:>6.1%}  (n={c})")
    print("  CARDS THAT WIN MOST:")
    for card, wr, c in gated[-8:][::-1]:
        print(f"    {card:30.30} {wr:>6.1%}  (n={c})")
    print()


def _intrigues(games) -> None:
    n = len(games)
    known = _load_intrigue_names()
    played = Counter()
    played_off_combat = Counter()      # a COMBAT-timed card played outside combat
    total_played = 0
    stranded = []                      # intrigues still in hand at game end
    from src.game.intrigue.intrigue import IntrigueTiming
    try:
        timing = {c.name: (c.timing if isinstance(c.timing, (set, tuple, list, frozenset))
                           else (c.timing,))
                  for c in _intrigue_deck()}
    except Exception:
        timing = {}
    # ENDGAME intrigues auto-resolve at game end (no play_intrigue event) — not a
    # sign the AI refuses them; separate them from the genuinely-unused list.
    endgame_cap = {nm for nm, ts in timing.items()
                   if IntrigueTiming.ENDGAME in tuple(ts)}

    for g in games:
        for ev in g["events"]:
            if ev.get("action") != "play_intrigue":
                continue
            name = ev.get("intrigue")
            played[name] += 1
            total_played += 1
            ts = tuple(timing.get(name, ()))
            # only a COMBAT-ONLY card is anomalous off-combat; dual PLOT/COMBAT
            # cards are legal to play as a plot
            if ts == (IntrigueTiming.COMBAT,) and ev.get("phase") != "combat":
                played_off_combat[name] += 1
        for r in g["rounds"][::-1]:
            if r.get("type") in ("round_end", "round_start"):
                snap = r["standings"]
                stranded.append(sum(p["intrigue_cards"] for p in snap))
                break

    print("INTRIGUE USAGE")
    print(f"  played/game: {total_played/n:.2f}   distinct cards seen: {len(played)}"
          + (f" / {len(known)} in deck" if known else ""))
    if stranded:
        print(f"  intrigues left unplayed at game end (all seats): {st.mean(stranded):.2f}/game")
    print("  most played:  " + ", ".join(f"{k} {v}" for k, v in played.most_common(6)))
    if known:
        never = [k for k in known if k not in played and k not in endgame_cap]
        auto_eg = [k for k in known if k not in played and k in endgame_cap]
        if auto_eg:
            print(f"  never played as an event ({len(auto_eg)}) - endgame intrigues, "
                  f"auto-resolved at game end if their condition holds: "
                  + ", ".join(auto_eg))
        if never:
            print(f"  GENUINELY never played ({len(never)}): " + ", ".join(never[:12])
                  + (" ..." if len(never) > 12 else ""))
        elif not auto_eg:
            print("  every intrigue in the deck gets played at least once")
    off = sum(played_off_combat.values())
    if off:
        print(f"  ANOMALY - combat-timed intrigue played outside combat: {off} times  "
              + ", ".join(f"{k}:{v}" for k, v in played_off_combat.most_common(5)))
    else:
        print("  no combat-timed intrigues played off-combat (good)")
    print()


def _r78_combat(games) -> None:
    for target in (7, 8):
        reward_vp, ctrl, strengths, swings, decided = [], 0, [], [], 0
        n_conf = 0
        for g in games:
            starts = {r["round"]: r for r in g["rounds"]
                      if r.get("type") == "round_start"}
            for c in g["combats"]:
                if c["round"] != target:
                    continue
                n_conf += 1
                w = c["winner"]
                if w is None:
                    continue
                decided += 1
                rw = _rget(c["rewards"], w)
                reward_vp.append(_sum_vp(rw))
                if rw.get("control"):
                    ctrl += 1
                if c["rankings"]:
                    strengths.append(c["rankings"][0]["strength"])
                a = starts.get(target)
                b = starts.get(target + 1)
                if a and b:
                    swings.append(b["standings"][w]["vp"] - a["standings"][w]["vp"])
        if not n_conf:
            print(f"ROUND {target} COMBAT: no games reached it")
            continue
        print(f"ROUND {target} COMBAT  ({decided}/{n_conf} conflicts decided)")
        if reward_vp:
            print(f"  first-place reward VP (incl. auto-conversions): avg {st.mean(reward_vp):.2f}  "
                  f"(control on {ctrl}/{decided} = {ctrl/decided:.0%})")
        if strengths:
            print(f"  winning strength: avg {st.mean(strengths):.1f}  "
                  f"max {max(strengths)}")
        if swings:
            print(f"  winner's total VP gained that round: avg {st.mean(swings):.2f}")
    print()


def _vp_distribution(games) -> None:
    n = len(games)
    all_vp, win_vp = [], []
    src = Counter()
    src_n = 0
    for g in games:
        s = g["summary"]
        w = s["winner"]
        all_vp += s["final_vp"]
        win_vp.append(s["final_vp"][w])
        for pid, pl in enumerate(s["players"]):
            fv = pl["final_vp"]
            combat_vp = 0
            for c in g["combats"]:
                combat_vp += _sum_vp(_rget(c["rewards"], pid))
            fr = pl["friendships"]
            al = pl["alliances"]
            eg = pl.get("endgame_vp", 0)
            tsmf = pl.get("tsmf_bought", 0)
            attributed = 1 + fr + al + eg + tsmf + combat_vp   # 1 = starting VP
            other = fv - attributed
            src["start"] += 1
            src["friendship"] += fr
            src["alliance"] += al
            src["endgame(icons/intrigue)"] += eg
            src["TSMF"] += tsmf
            src["combat reward"] += combat_vp
            src["cards / contracts / other"] += other
            src_n += 1
    qs = st.quantiles(all_vp, n=4)
    print("FINAL-VP DISTRIBUTION")
    print(f"  all seats : mean {st.mean(all_vp):.1f}  median {st.median(all_vp):.0f}  "
          f"q1/q3 {qs[0]:.0f}/{qs[2]:.0f}  range {min(all_vp)}-{max(all_vp)}")
    print(f"  winners   : mean {st.mean(win_vp):.1f}  median {st.median(win_vp):.0f}  "
          f"range {min(win_vp)}-{max(win_vp)}")
    hist = Counter(all_vp)
    print("  histogram (all seats):")
    for v in range(min(hist), max(hist) + 1):
        bar = "#" * round(40 * hist[v] / max(hist.values()))
        print(f"    {v:2d} | {bar} {hist[v]}")
    print(f"  approx VP source split (per seat-game, avg over {src_n}):")
    for k, tot in src.most_common():
        print(f"    {k:28} {tot/src_n:5.2f}")
    print()


def _friend_alliance(games) -> None:
    n = len(games)
    fr_all, al_all, fr_win, al_win = [], [], [], []
    for g in games:
        s = g["summary"]
        w = s["winner"]
        for pid, pl in enumerate(s["players"]):
            fr_all.append(pl["friendships"])
            al_all.append(pl["alliances"])
            if pid == w:
                fr_win.append(pl["friendships"])
                al_win.append(pl["alliances"])
    print("FRIENDSHIPS / ALLIANCES (end of game)")
    print(f"  all seats : friendships {st.mean(fr_all):.2f}   alliances {st.mean(al_all):.2f}")
    print(f"  winners   : friendships {st.mean(fr_win):.2f}   alliances {st.mean(al_win):.2f}")
    print(f"  alliance count distribution (all seats): "
          + "  ".join(f"{k}:{v}" for k, v in sorted(Counter(al_all).items())))
    print(f"  winners with >=1 alliance: {sum(1 for x in al_win if x)}/{len(al_win)} "
          f"= {sum(1 for x in al_win if x)/len(al_win):.0%}")
    print()


def report(games, agent_spec: str, tag: str = "") -> None:
    print("=" * 68)
    print(f"SELF-PLAY REPORT   agent=4x {agent_spec}   games={len(games)}"
          + (f"   [{tag}]" if tag else ""))
    print("=" * 68 + "\n")
    _seat_winrate(games)
    _game_length(games)
    _cards(games)
    _intrigues(games)
    _r78_combat(games)
    _vp_distribution(games)
    _friend_alliance(games)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--agent", type=str, default="value:models/value_best.npz")
    ap.add_argument("--seed", type=int, default=900_000)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    out = run(args.games, args.agent, base_seed=args.seed)
    report(out["games"], out["agent"], tag=args.tag)


if __name__ == "__main__":
    main()
