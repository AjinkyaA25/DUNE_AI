"""Four-way arena with VP-source accounting per agent.
usage: python -m src.selfplay.gap_arena GAMES SPEC1 SPEC2 SPEC3 SPEC4 [--out F.jsonl] [--workers N]
With --out each game is appended as it finishes and a rerun resumes."""
import sys, collections, itertools, json
from multiprocessing import Pool
PERMS = list(itertools.permutations(range(4)))


def cat(names):
    s = " < ".join(names)
    if "alliance" in s: return "alliance"
    if s.startswith("gain_influence"): return "friendship"
    if "battle_icon" in s: return "icons"
    if "combat_reward" in s: return "conflict"
    if "acquire_reserve" in s: return "tsmf"
    return "other"


def _seat_stats(gs, s):
    """End-of-game resources and what the seat built (persona comparisons)."""
    p = gs.players[s]
    return {"solari": p.solari, "spice": p.spice, "water": p.water,
            "swordmaster": int(bool(getattr(p, "has_swordmaster", False))),
            "high_council": int(bool(getattr(p, "has_councilor", False))),
            "contracts_done": len(getattr(p, "contracts_completed", []) or []),
            "contracts_held": len(getattr(p, "contracts_active", []) or []),
            "techs": len(getattr(p, "techs", []) or []), "intrigues": len(p.intrigue_cards),
            "deck": len(p.deck) + len(p.hand) + len(p.discard) + len(p.in_play)}


def game_lineup(g, specs_by_seat, lineup, seat):
    """A game with a fixed Leader lineup; result for one seat (Leader A/B)."""
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    agents = [make_agent(specs_by_seat[s], seed=g * 4 + s) for s in range(4)]
    gs = setup_game(num_players=4, seed=700000 + g, leaders=lineup, use_bloodlines=True)
    n = 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        n += 1
    vp = [p.victory_points for p in gs.players]
    win = gs.winner if gs.winner is not None else max(range(4), key=lambda s: vp[s])
    return {"win": int(win == seat), "vp": vp[seat], "rounds": gs.round, **_seat_stats(gs, seat)}


def game(args):
    g, specs = args[0], args[1]
    with_leaders = len(args) > 2 and args[2]
    from src.game.player.player import Player
    from src.game.gameState import ActionType
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    log = []
    orig = Player.gain_vp

    def gv(self, amount, source=None):
        if id(self) not in real:               # a search playout's copy
            return orig(self, amount, source)
        f = sys._getframe(1); names = []
        while f and len(names) < 3:
            names.append(f.f_code.co_name); f = f.f_back
        log.append((self.id, amount, cat(names)))
        return orig(self, amount, source)
    Player.gain_vp = gv
    perm = PERMS[g % 24]
    agents = [make_agent(specs[perm[s]], seed=g * 4 + s) for s in range(4)]
    if with_leaders:                           # random real Leaders, seats rotate with the agents
        import random as _r
        from src.game.leader.leader_definitions import LEADER_NAMES
        lineup = _r.Random(700000 + g).sample(list(LEADER_NAMES), 4)
        gs = setup_game(num_players=4, seed=700000 + g, leaders=lineup, use_bloodlines=True)
    else:
        gs = setup_game(num_players=4, seed=700000 + g, neutral_leaders=True, use_bloodlines=True)
    real = {id(p) for p in gs.players}
    afford = collections.Counter(); took = collections.Counter()
    n = 0
    vp_r6 = None                               # VP when round 6 is over (scoring pace)
    while not gs.game_over and n < 3000:
        if vp_r6 is None and gs.round > 6:
            vp_r6 = [p.victory_points for p in gs.players]
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        a = agents[pid].select_action(gs, pid, valid)
        if any(v.action_type == ActionType.ACQUIRE_RESERVE and v.reserve_type == "spice_must_flow" for v in valid):
            afford[pid] += 1
            if a.action_type == ActionType.ACQUIRE_RESERVE and a.reserve_type == "spice_must_flow":
                took[pid] += 1
        gs.step(a); n += 1
    Player.gain_vp = orig
    vp = [p.victory_points for p in gs.players]
    if vp_r6 is None:
        vp_r6 = vp                             # finished in round 6 or earlier
    win = gs.winner if gs.winner is not None else max(range(4), key=lambda s: vp[s])
    out = {}
    for s in range(4):
        src = collections.Counter()
        for pid, amt, c in log:
            if pid == s: src[c] += amt
        out[perm[s]] = {"win": int(win == s), "vp": vp[s], "vp_r6": vp_r6[s], "src": dict(src),
                        "afford": afford[s], "took": took[s], "stats": _seat_stats(gs, s)}
    return g, gs.round, out


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("games", type=int)
    ap.add_argument("specs", nargs=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=24)
    A = ap.parse_args()
    N, specs = A.games, A.specs
    results = {}
    if A.out and os.path.exists(A.out):
        for line in open(A.out):
            d = json.loads(line)
            results[d["game"]] = (d["rounds"], {int(k): v for k, v in d["out"].items()})
    todo = [(g, specs) for g in range(N) if g not in results]
    f = open(A.out, "a") if A.out else None
    with Pool(A.workers) as pool:
        for g, r, out in pool.imap_unordered(game, todo):
            results[g] = (r, out)
            if f:
                f.write(json.dumps({"game": g, "rounds": r, "out": out}) + "\n"); f.flush()
                print(f"[{len(results)}/{N}] wins " + " ".join(
                    str(sum(o[i]["win"] for _, o in results.values())) for i in range(4)), flush=True)
    agg = [collections.Counter() for _ in range(4)]
    rounds = []
    for r, out in results.values():
            rounds.append(r)
            for i, o in out.items():
                a = agg[i]; a["win"] += o["win"]; a["vp"] += o["vp"]
                a["afford"] += o["afford"]; a["took"] += o["took"]
                for k, v in o["src"].items(): a[k] += v
    N = len(results)
    print(f"games {N}  avg rounds {sum(rounds)/N:.2f}  dist {sorted(collections.Counter(rounds).items())}")
    for i, s in enumerate(specs):
        a = agg[i]
        print(f"{s.split('/')[-1]:28s} win {a['win']/N:.3f}  VP {a['vp']/N:.2f}  "
              + " ".join(f"{k} {a[k]/N:.2f}" for k in ("friendship", "alliance", "tsmf", "icons", "conflict", "other"))
              + f"  TSMF bought {a['took']}/{a['afford']} offers")
