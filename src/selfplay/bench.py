"""CPU seconds per training-mix game (src/selfplay/parity.py games), best of
--reps runs, so other programs' load on the machine matters less.

    python -m src.selfplay.bench [--games 8] [--reps 3] [--nocache]
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--nocache", action="store_true", help="per-decision cache off (A/B)")
    A = ap.parse_args()
    if A.nocache:
        import src.ai.agents as ag

        class _Off:
            def __enter__(self): pass
            def __exit__(self, *e): pass
        ag.decision_cache = _Off
    from src.selfplay.parity import play
    best = None
    for _ in range(A.reps):
        t = time.process_time()
        for g in range(A.games):
            play(777000 + g, trace=False)
        dt = (time.process_time() - t) / A.games
        best = dt if best is None else min(best, dt)
    print(f"{best:.3f} CPU s/game ({'no cache' if A.nocache else 'cache'})")


if __name__ == "__main__":
    main()
