"""Four-way arena with VP-source accounting per agent.
usage: python -m src.selfplay.gap_arena GAMES SPEC1 SPEC2 SPEC3 SPEC4"""
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


def game(args):
    g, specs = args
    from src.game.player.player import Player
    from src.game.gameState import ActionType
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    log = []
    orig = Player.gain_vp

    def gv(self, amount):
        f = sys._getframe(1); names = []
        while f and len(names) < 3:
            names.append(f.f_code.co_name); f = f.f_back
        log.append((self.id, amount, cat(names)))
        return orig(self, amount)
    Player.gain_vp = gv
    perm = PERMS[g % 24]
    agents = [make_agent(specs[perm[s]], seed=g * 4 + s) for s in range(4)]
    gs = setup_game(num_players=4, seed=700000 + g, neutral_leaders=True, use_bloodlines=True)
    afford = collections.Counter(); took = collections.Counter()
    n = 0
    while not gs.game_over and n < 3000:
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
    win = gs.winner if gs.winner is not None else max(range(4), key=lambda s: vp[s])
    out = {}
    for s in range(4):
        src = collections.Counter()
        for pid, amt, c in log:
            if pid == s: src[c] += amt
        out[perm[s]] = {"win": int(win == s), "vp": vp[s], "src": dict(src),
                        "afford": afford[s], "took": took[s]}
    return gs.round, out


if __name__ == "__main__":
    N = int(sys.argv[1]); specs = sys.argv[2:6]
    agg = [collections.Counter() for _ in range(4)]
    rounds = []
    with Pool(24) as pool:
        for r, out in pool.imap_unordered(game, [(g, specs) for g in range(N)]):
            rounds.append(r)
            for i, o in out.items():
                a = agg[i]; a["win"] += o["win"]; a["vp"] += o["vp"]
                a["afford"] += o["afford"]; a["took"] += o["took"]
                for k, v in o["src"].items(): a[k] += v
    print(f"games {N}  avg rounds {sum(rounds)/N:.2f}  dist {sorted(collections.Counter(rounds).items())}")
    for i, s in enumerate(specs):
        a = agg[i]
        print(f"{s.split('/')[-1]:28s} win {a['win']/N:.3f}  VP {a['vp']/N:.2f}  "
              + " ".join(f"{k} {a[k]/N:.2f}" for k in ("friendship", "alliance", "tsmf", "icons", "conflict", "other"))
              + f"  TSMF bought {a['took']}/{a['afford']} offers")
