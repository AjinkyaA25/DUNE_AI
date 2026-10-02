"""Where AI players get their VP, by round: every gain_vp call is tagged by
the function that awarded it (friendship / alliance / conflict / battle-icon
pair / contract / card or intrigue effect / tech ...)."""
import collections
import sys
import traceback
from multiprocessing import Pool

SPEC = sys.argv[1] if len(sys.argv) > 1 else "heuristic:tuned=config/heuristic_me.json"

LABELS = [
    ("gain_influence", "faction friendship"), ("_check_and_update_alliance", "alliance"),
    ("_apply_combat_reward", "conflict reward"), ("_award_battle_icon", "battle-icon pair"),
    ("_resolve_endgame_battle_icons", "battle icons (endgame)"),
    ("complete_contract", "contract"), ("_trigger_acquire_effects", "card acquired (e.g. TSMF)"),
    ("bloodlines", "Bloodlines tech/endgame"), ("_resolve_endgame_intrigues", "endgame intrigue"),
    ("resolve_single_effect", "card / intrigue / space effect"),
]


def source():
    stack = traceback.extract_stack()[:-3]
    names = [f.name for f in stack] + [f.filename for f in stack]
    for key, lab in LABELS:
        if any(key in n for n in names):
            return lab
    return stack[-1].name


def run(g):
    from src.game.player import player as P
    orig = P.Player.gain_vp
    log = []

    def gain_vp(self, amount):
        if amount:
            log.append((self.id, amount, source()))
        return orig(self, amount)
    P.Player.gain_vp = gain_vp
    from src.data.card_definitions import setup_game
    from src.ai.agents import make_agent
    gs = setup_game(num_players=4, seed=95_000 + g, neutral_leaders=True, use_bloodlines=True)
    ag = [make_agent(SPEC, seed=g * 4 + i) for i in range(4)]
    out = collections.Counter()
    n = 0
    while not gs.game_over and n < 4000:
        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        r = gs.round
        k = len(log)
        gs.step(ag[pid].select_action(gs, pid, gs.get_valid_actions(pid)))
        for q, amt, src in log[k:]:
            out[(min(r, 9) if r <= 3 else ("4-6" if r <= 6 else "7+"), src)] += amt
        n += 1
    return out, gs.round


if __name__ == "__main__":
    with Pool(8) as pool:
        res = pool.map(run, range(int(sys.argv[2]) if len(sys.argv) > 2 else 120))
    tot = sum((r[0] for r in res), collections.Counter())
    G = len(res)
    srcs = sorted({s for _, s in tot}, key=lambda s: -sum(v for (b, x), v in tot.items() if x == s))
    buckets = [1, 2, 3, "4-6", "7+"]
    print(f"VP gained per PLAYER per game, by source and round ({SPEC}, {G} games)")
    print(f"{'source':34s}" + "".join(f"{str(b):>7s}" for b in buckets) + "  total")
    for s in srcs:
        row = [tot.get((b, s), 0) / (4 * G) for b in buckets]
        print(f"{s[:34]:34s}" + "".join(f"{v:7.2f}" for v in row) + f"  {sum(row):5.2f}")
