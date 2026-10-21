"""Turn finished UI games (game_logs/ui_*.json) into AlphaZero examples of the
human's decisions (data/human_az/), like ui/server.py records live.

The game is rebuilt from its seed by replaying each recorded move; a move
that no longer matches the current rules (repr differs) stops the import of
that game there, so only positions the engine reproduces exactly are kept.

    python -m src.az.import_ui_games [--only-won] [files...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def import_game(path: str, out_dir: str) -> tuple:
    from src.az.encode import encode_actions, encode_obs
    from src.az.mcts import legal, terminal_values
    from src.data.card_definitions import setup_game
    d = json.load(open(path, encoding="utf-8"))
    seat = d["human_id"]
    gs = setup_game(num_players=4, seed=d["seed"], neutral_leaders=True,
                    use_bloodlines=bool(d.get("bloodlines", True)))
    gs.manual_end_turn = {seat}
    rec, matched = [], 0
    from src.game.gameState import ActionType as T
    for m in d["moves"]:
        pid = m["active_player"]
        want = m.get("repr", "")
        act = None
        for _ in range(6):                          # bridge today's rule changes
            valid = gs.get_valid_actions(pid)
            act = next((a for a in valid if repr(a) == want), None)
            if act is not None:
                break
            cur = gs.player_in_reveal_buy if gs.player_in_reveal_buy is not None else gs.get_current_player_id()
            end = [a for a in gs.get_valid_actions(cur) if a.action_type == T.END_TURN]
            if end:                                 # a window the old rules didn't have
                gs.step(end[0]); continue
            break
        if act is None:
            if "RESOLVE_DEPLOY" in want.upper() and "n=0" in want.replace(" ", ""):
                matched += 1                        # "deploy 0" no longer exists
                continue
            break                                   # can't follow: stop here
        if pid == seat and gs.bl:
            acts = legal(gs, seat)
            k = next((i for i, a in enumerate(acts) if repr(a) == repr(act)), None)
            if k is not None and len(acts) > 1:
                dense, idx = encode_actions(gs, seat, acts)
                rec.append((encode_obs(gs, seat).astype(np.float16), dense.astype(np.float16),
                            idx.astype(np.int16), k))
        gs.step(act)
        matched += 1
    if not rec:
        return 0, matched, len(d["moves"])
    full = matched == len(d["moves"]) and gs.game_over
    n = gs.num_players
    if full:
        zt = terminal_values(gs)
        z = np.array([zt[(seat + k) % n] for k in range(n)], np.float32)
    else:
        z = np.full(n, np.nan, np.float32)        # partial replay: moves only
    S, D, I, P, off = [], [], [], [], [0]
    for obs, dense, idx, k in rec:
        pi = np.zeros(len(dense), np.float32)
        pi[k] = 1.0
        S.append(obs); D.append(dense); I.append(idx); P.append(pi); off.append(off[-1] + len(dense))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "import_" + os.path.basename(path).replace(".json", ".npz"))
    np.savez(out, s=np.stack(S), z=np.tile(z, (len(S), 1)), off=np.asarray(off, np.int64),
             dense=np.concatenate(D), idx=np.concatenate(I), pi=np.concatenate(P),
             vs=np.zeros((0, 1), np.float16), vz=np.zeros((0, 4), np.float32))
    return len(S), matched, len(d["moves"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--only-won", action="store_true")
    ap.add_argument("--out", default=os.path.join("data", "human_az"))
    A = ap.parse_args()
    files = A.files or sorted(glob.glob(os.path.join("game_logs", "ui_*.json")))
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        if not d.get("finished") or d.get("human_id") is None or (A.only_won and not d.get("you_won")):
            continue
        n, matched, total = import_game(f, A.out)
        print(f"{os.path.basename(f)}: won={d.get('you_won')} replayed {matched}/{total} moves, "
              f"{n} of your decisions saved")


if __name__ == "__main__":
    main()
