"""Where does PPO overrule the heuristic it sits on, and how does its play differ?
(user, 2026-10-20: "making sure we are accurately honing in on specific moves").

1. Decisions: games with PPO (greedy) in every seat. At each decision we log the
   heuristic's top move and the move PPO plays, by kind (Agent space, card buy,
   tech, contract, Intrigue, deploy, choice ...) and round.
   - overrule rate per kind / round band
   - preference shift per item: how often PPO plays it vs how often the heuristic
     alone would have (e.g. "buys Delivery Bay 2.3x as often")
   - the most common swaps "heuristic -> PPO"
2. Playstyle: the same seeds with PPO in one seat vs 3 heuristics, and the plain
   heuristic in that seat vs the same 3 (paired), comparing end-of-game state and
   where the VP came from.

    python -m src.selfplay.ppo_overrule --net models_ppo/best.npz --games 600 --style-games 2000
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import time
from multiprocessing import Pool

NET = None
BASE = "config/ppo/base.json"


def band(r: int) -> str:
    return "R1-3" if r <= 3 else ("R4-6" if r <= 6 else "R7+")


def describe(gs, pid, a):
    """-> (kind, item) of a move."""
    from src.game.gameState import ActionType as T
    at = a.action_type
    ch = getattr(a, "choice", None)
    if at == T.AGENT_TURN:
        return "agent space", a.space_name
    if at == T.REVEAL_TURN:
        return "agent space", "(Reveal turn)"
    if at == T.ACQUIRE_CARD:
        return "buy", a.acquire_card_name
    if at == T.ACQUIRE_RESERVE:
        return "buy", "The Spice Must Flow" if a.reserve_type == "spice_must_flow" else "Prepare the Way"
    if at == T.END_REVEAL:
        return "buy", "(stop buying)"
    if at == T.PLAY_INTRIGUE:
        return "intrigue", getattr(a, "intrigue_card_name", "?")
    if at == T.COMBAT_PASS:
        return "intrigue", "(pass in combat)"
    if at == T.RESOLVE_CONTRACT:
        board = getattr(gs, "contracts_on_board", []) or []
        i = a.contract_index
        return "contract", board[i].name if 0 <= i < len(board) else f"(set aside {i})"
    if at == T.ACTIVATE_TECH:
        return "activate tech", str(ch)
    if isinstance(ch, str) and ch:
        if ch.startswith("stack:") and getattr(gs, "bl", None) is not None:
            try:
                d, _ = gs.bl.tech_at(pid, int(ch.split(":")[1]))
                return "tech", d.name if d is not None else ch
            except (ValueError, IndexError):
                return "tech", ch
        if ch.startswith(("board:", "skill:")):
            return "sardaukar skill", ch.split(":", 1)[1]
        if ch.startswith("recruit:"):
            return "commander", ch
        head = ch.split(":", 1)[0]
        return f"choice {head}" if ":" in ch else "choice", ch[:60]
    keep = []
    for f in dataclasses.fields(a):
        if f.name in ("action_type", "player_id"):
            continue
        v = getattr(a, f.name)
        if v not in (None, False, 0, "", [], {}) and v != f.default:
            keep.append(f"{f.name}={v}")
    return at.value, ", ".join(keep)[:60] or at.value


KEYS = ("dec", "over", "detail", "dec_band", "over_band", "h", "p", "swap")


def situation(valid) -> str:
    """What kind of decision this is, from the moves on offer."""
    from src.game.gameState import ActionType as T
    types = {v.action_type for v in valid}
    if T.AGENT_TURN in types or T.REVEAL_TURN in types:
        return "agent turn"
    if T.ACQUIRE_CARD in types or T.ACQUIRE_RESERVE in types:
        return "buy"
    if T.PLAY_INTRIGUE in types or T.COMBAT_PASS in types:
        return "intrigue / combat"
    if T.RESOLVE_CONTRACT in types:
        return "contract"
    if T.RESOLVE_DEPLOY in types:
        return "deploy"
    if T.ACTIVATE_TECH in types:
        return "activate tech"
    return describe(None, 0, valid[0])[0] if valid[0].action_type not in (T.RESOLVE_BL_CHOICE,)         else _bl_kind(valid)


def _bl_kind(valid) -> str:
    chs = [str(getattr(v, "choice", "")) for v in valid]
    if any(c.startswith("stack:") for c in chs):
        return "tech"
    if any(c.startswith(("board:", "skill:")) for c in chs):
        return "sardaukar skill"
    if any(c.startswith("recruit:") for c in chs):
        return "commander"
    return "bloodlines choice"


def decision_game(g):
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    import random
    from src.game.leader.leader_definitions import LEADER_NAMES
    spec = f"ppo:tuned={BASE}:greedy:net={NET}"
    agents = [make_agent(spec, seed=g * 4 + s) for s in range(4)]
    lineup = random.Random(990000 + g).sample(list(LEADER_NAMES), 4)
    gs = setup_game(num_players=4, seed=990000 + g, leaders=lineup, use_bloodlines=True)
    C = {k: collections.Counter() for k in KEYS}
    n = 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        a = agents[pid].select_action(gs, pid, valid)
        if len(valid) > 1 and agents[pid].last:
            ht = agents[pid].last["heur_top"]
            pk, item = describe(gs, pid, a)
            hk, hitem = describe(gs, pid, ht)
            kind = situation(valid)
            b = band(gs.round)
            C["dec"][kind] += 1
            C["dec_band"][(kind, b)] += 1
            C["p"][(pk, item)] += 1
            C["h"][(hk, hitem)] += 1
            if repr(ht) != repr(a):
                C["detail"][kind] += 1                   # any difference (e.g. another card at the same space)
            if (pk, item) != (hk, hitem):
                C["over"][kind] += 1
                C["over_band"][(kind, b)] += 1
                C["swap"][(kind, f"{hitem}" if hk == pk else f"[{hk}] {hitem}",
                           f"{item}" if hk == pk else f"[{pk}] {item}")] += 1
        gs.step(a)
        n += 1
    return C


def style_game(g):
    import src.selfplay.evolve as E
    ppo = f"ppo:tuned={BASE}:greedy:net={NET}"
    h = f"heuristic:tuned={BASE}"
    out = {}
    for name, first in (("ppo", ppo), ("heuristic", h)):
        _, rnd, o = E.play((g, [first, h, h, h], True))
        out[name] = {"rounds": rnd, **o[0]}
    return out


def init(net, base):
    global NET, BASE
    NET, BASE = net, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="models_ppo/best.npz")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--style-games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--out", default="reports/ppo_overrule")
    A = ap.parse_args()
    t0 = time.time()
    tot = {k: collections.Counter() for k in KEYS}
    styles = []
    with Pool(A.workers, initializer=init, initargs=(A.net, A.base), maxtasksperchild=50) as pool:
        for C in pool.imap_unordered(decision_game, range(A.games)):
            for k in tot:
                tot[k].update(C[k])
        print(f"decision games done {time.time() - t0:.0f}s", flush=True)
        styles = list(pool.imap_unordered(style_game, range(70_000_000, 70_000_000 + A.style_games)))
        print(f"style games done {time.time() - t0:.0f}s", flush=True)
    lines = report(tot, styles, A)
    open(A.out + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    json.dump({"net": A.net, "games": A.games, "style_games": A.style_games,
               "counts": {k: [[list(x) if isinstance(x, tuple) else x, n] for x, n in v.items()]
                          for k, v in tot.items()},
               "style": styles}, open(A.out + ".json", "w"))


def report(T, styles, A):
    L = [f"PPO overrule analysis - net {A.net}, heuristic base {A.base}",
         f"{A.games} games with PPO in all 4 seats; {sum(T['dec'].values())} decisions with 2+ moves", ""]
    L.append("1. How often PPO plays something other than the heuristic's top move")
    L.append("   overruled = a different space / card / tech / Intrigue ...; any = also counts e.g. another")
    L.append("   card played at the same space, or another number of troops")
    L.append(f"   {'decision':<22}{'decisions':>10}{'overruled':>11}{'any':>8}   R1-3    R4-6    R7+")
    for kind, n in T["dec"].most_common():
        if n < 200:
            continue
        bands = "  ".join(f"{T['over_band'][(kind, b)] / max(1, T['dec_band'][(kind, b)]):6.1%}"
                          for b in ("R1-3", "R4-6", "R7+"))
        L.append(f"   {kind:<22}{n:>10}{T['over'][kind] / n:>10.1%}{T['detail'][kind] / n:>8.1%}   {bands}")
    all_n = sum(T["dec"].values())
    L.append(f"   {'all':<22}{all_n:>10}{sum(T['over'].values()) / all_n:>10.1%}"
             f"{sum(T['detail'].values()) / all_n:>8.1%}")
    L.append("")
    L.append("2. Preference shifts: times PPO plays it vs times the heuristic alone would have")
    L.append("   (per 100 games with 4 PPO players; 'x' = PPO / heuristic)")
    per = 100.0 / A.games
    for kind in ("agent space", "buy", "tech", "contract", "intrigue", "sardaukar skill", "activate tech"):
        items = {it for (k, it) in list(T["p"]) + list(T["h"]) if k == kind}
        rows = []
        for it in items:
            p, h = T["p"][(kind, it)], T["h"][(kind, it)]
            if p + h < 40:
                continue
            rows.append((p - h, it, p, h))
        if not rows:
            continue
        rows.sort()
        L.append(f"   {kind}:")
        show = rows[:8] + [r for r in rows[-8:] if r not in rows[:8]]
        for d, it, p, h in sorted(show, key=lambda r: -r[0]):
            ratio = f"{p / h:5.2f}x" if h else "  new"
            L.append(f"     {it[:38]:<38} PPO {p * per:7.1f}  heuristic {h * per:7.1f}  {ratio}")
    L.append("")
    L.append("3. Most common swaps (heuristic's move -> PPO's move), per 100 games")
    for kind in ("agent turn", "buy", "tech", "contract", "intrigue / combat", "deploy", "sardaukar skill"):
        sw = [(n, h, p) for (k, h, p), n in T["swap"].items() if k == kind]
        if not sw:
            continue
        L.append(f"   {kind}:")
        for n, h, p in sorted(sw, reverse=True)[:10]:
            L.append(f"     {n * per:6.1f}  {h[:34]:<34} -> {p}")
    L.append("")
    if styles:
        n = len(styles)
        L.append(f"4. Playstyle: the same {n} games, one seat played by PPO vs by the plain heuristic")
        L.append("   (the other 3 seats are the plain heuristic in both)")
        for who in ("ppo", "heuristic"):
            w = sum(s[who]["win"] for s in styles) / n
            r = sum(s[who]["rounds"] for s in styles) / n
            vp = sum(s[who]["vp"] for s in styles) / n
            r6 = sum(s[who]["vp_r6"] for s in styles) / n
            L.append(f"   {who:<10} win {w:6.1%}  VP {vp:5.2f}  VP after round 6 {r6:5.2f}  avg rounds {r:4.2f}")
        L.append("   end-of-game state (average)         PPO   heuristic")
        keys = sorted({k for s in styles for k in s["ppo"]["stats"]})
        for k in keys:
            a = sum(s["ppo"]["stats"].get(k, 0) for s in styles) / n
            b = sum(s["heuristic"]["stats"].get(k, 0) for s in styles) / n
            L.append(f"     {k:<32}{a:7.2f} {b:9.2f}")
        L.append("   where the VP came from (average per game)")
        srcs = collections.Counter()
        for s in styles:
            for who in ("ppo", "heuristic"):
                for k, v in s[who]["src"].items():
                    srcs[k] += abs(v)
        for k, _ in srcs.most_common(14):
            a = sum(s["ppo"]["src"].get(k, 0) for s in styles) / n
            b = sum(s["heuristic"]["src"].get(k, 0) for s in styles) / n
            L.append(f"     {k[:32]:<32}{a:7.2f} {b:9.2f}")
    return L


if __name__ == "__main__":
    main()
