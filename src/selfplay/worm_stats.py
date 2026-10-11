"""How the AI uses sandworms, the Shield Wall and combat.
usage: python -m src.selfplay.worm_stats GAMES SPEC"""
import sys, collections
from multiprocessing import Pool


def game(args):
    g, spec = args
    from src.game.gameState import GameState
    from src.game.board.board import SHIELD_WALL_PROTECTED
    from src.ai.agents import make_agent
    from src.data.card_definitions import setup_game
    gs = setup_game(num_players=4, seed=800000 + g, neutral_leaders=True, use_bloodlines=True)
    agents = [make_agent(spec, seed=g * 4 + s) for s in range(4)]
    combats = []
    orig = GameState.resolve_combat

    def rc(self):
        cc = self.current_conflict
        before = [p.victory_points for p in self.players]
        worms = dict(self.sandworms_in_conflict)
        troops = dict(self.troops_in_conflict)
        strength = dict(getattr(self, "combat_strength", {}) or {})
        res = orig(self)
        after = [p.victory_points for p in self.players]
        if cc is not None:
            combats.append({"round": self.round, "name": cc.name, "level": cc.conflict_level,
                            "protected": cc.location in SHIELD_WALL_PROTECTED,
                            "wall": self.shield_wall_intact,
                            "worms": worms, "troops": troops, "strength": strength,
                            "winner": getattr(res, "winner", None),
                            "gain": [a - b for a, b in zip(after, before)]})
        return res
    GameState.resolve_combat = rc
    wall_break = None
    n = 0
    while not gs.game_over and n < 3000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        intact = gs.shield_wall_intact
        gs.step(agents[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        if intact and not gs.shield_wall_intact and wall_break is None:
            wall_break = (gs.round, pid)
        n += 1
    GameState.resolve_combat = orig
    return {"rounds": gs.round, "combats": combats, "wall_break": wall_break,
            "vp": [p.victory_points for p in gs.players]}


if __name__ == "__main__":
    N, spec = int(sys.argv[1]), sys.argv[2]
    with Pool(24) as pool:
        R = pool.map(game, [(g, spec) for g in range(N)])
    c = collections.Counter()
    gains = collections.Counter()
    for r in R:
        if r["wall_break"]:
            c["wall_broken"] += 1
            c["wall_round_sum"] += r["wall_break"][0]
        for cb in r["combats"]:
            c["combats"] += 1
            wormers = [p for p, w in cb["worms"].items() if w > 0]
            prot_open = cb["protected"] and not cb["wall"]
            if cb["protected"]:
                c["protected"] += 1
                c["protected_wall_open"] += prot_open
            if cb["level"] == 3:
                c["lvl3"] += 1
            if wormers:
                c["combats_with_worm"] += 1
                c["worm_players"] += len(wormers)
                if cb["winner"] in wormers:
                    c["worm_won"] += 1
                    c["worm_won_vp"] += cb["gain"][cb["winner"]]
                    if cb["level"] == 3:
                        c["worm_won_lvl3"] += 1
            w = cb["winner"]
            if w is not None:
                gains[max(cb["gain"])] += 1
                if cb["level"] == 3:
                    c["lvl3_winner_vp"] += cb["gain"][w]
                    c["lvl3_won"] += 1
    print(f"games {N}  rounds {sum(r['rounds'] for r in R)/N:.2f}")
    print(f"wall broken in {c['wall_broken']/N:.0%} of games, avg round {c['wall_round_sum']/max(1,c['wall_broken']):.1f}")
    print(f"combats/game {c['combats']/N:.1f}; protected-location {c['protected']/N:.2f}/game, of those with wall open {c['protected_wall_open']/max(1,c['protected']):.0%}")
    print(f"combats with any worm {c['combats_with_worm']/c['combats']:.0%} ({c['combats_with_worm']/N:.2f}/game), worm players per such combat {c['worm_players']/max(1,c['combats_with_worm']):.2f}")
    print(f"worm player won {c['worm_won']/max(1,c['combats_with_worm']):.0%}; avg VP when worm wins {c['worm_won_vp']/max(1,c['worm_won']):.2f}; worm wins of level-3 {c['worm_won_lvl3']/N:.2f}/game")
    print(f"level-3 combats {c['lvl3']/N:.2f}/game, winner VP from it avg {c['lvl3_winner_vp']/max(1,c['lvl3_won']):.2f}")
    print("max VP gained in one combat:", sorted(gains.items()))
