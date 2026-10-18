"""Heuristic evolution progress page: is the champion getting stronger, and
does it end the game sooner?

    python -m src.selfplay.evolve_progress [--out config/evolve]

Reads <out>/log.jsonl (one line per generation) and <out>/bench.jsonl
(champion vs 3 copies of the starting weights) and writes <out>/progress.html.
src.selfplay.evolve rebuilds it after every generation.
"""
from __future__ import annotations

import argparse
import html
import json
import os

from src.az.progress import _chart, _page, _read

OUT = os.path.join("config", "evolve")


def _avg(pts, k=5):
    """Trailing moving average of [(gen, value)]."""
    out = []
    for i, (g, _) in enumerate(pts):
        w = [v for _, v in pts[max(0, i - k + 1):i + 1]]
        out.append((g, sum(w) / len(w)))
    return out


def build(out_dir=OUT):
    log = _read(os.path.join(out_dir, "log.jsonl"))
    bench = _read(os.path.join(out_dir, "bench.jsonl"))
    if not log:
        return None
    gmax = max([r["gen"] for r in log] + [b["gen"] for b in bench] + [1])
    swaps = [{"gen": r["gen"], "change": ("new champion: " if "leader" not in r else f"{r['leader']} weights: ") + ", ".join(
        f"{k} {v:g}" for k, v in r.get("changes", {}).items())} for r in log if r.get("new_champion")]
    log = [r for r in log if "rounds" in r]          # Leader A/B generations have their own records
    if not log:
        return None
    rounds = [(r["gen"], r["rounds"]) for r in log]
    r6 = [(r["gen"], r["vp_r6"][0]) for r in log if "vp_r6" in r]
    charts = [
        _chart("Rounds per game",
               [("evolution games (5-gen avg)", "#2f9e6e", _avg(rounds)),
                ("evolution games", "#a6d5bf", rounds),
                ("champion vs start weights", "#d9822b", [(b["gen"], b["rounds"]) for b in bench])],
               [], gmax, "{:.2f}",
               note="Lower = someone reaches 10 VP sooner. Strong human games end around round 7-8."),
        _chart("VP at the end of round 6 (scoring pace)",
               [("champion vs start weights", "#d9822b", [(b["gen"], b["vp_r6"]) for b in bench]),
                ("start weights in those games", "#8a8f98", [(b["gen"], b["opp_vp_r6"]) for b in bench]),
                ("champion in evolution games (5-gen avg)", "#c9a227", _avg(r6))],
               [], gmax, "{:.2f}"),
        _chart("Champion win share vs 3 copies of the starting weights",
               [("champion", "#d9822b", [(b["gen"], b["share"]) for b in bench])],
               [], gmax, "{:.0%}", ref=0.25,
               note="Dashed grey = 25%, no better than the start. 400 games per point (about ±4%)."),
        _chart("Games ended by round 8",
               [("evolution games", "#9b6cd6", _avg([(r["gen"], r["ended_by_r8"]) for r in log if "ended_by_r8" in r])),
                ("champion vs start weights", "#d9822b", [(b["gen"], b["ended_by_r8"]) for b in bench])],
               [], gmax, "{:.0%}"),
        _chart("Fitness per game (wins weighted by speed)",
               [("champion vs start weights", "#d9822b", [(b["gen"], b["fit"]) for b in bench if "fit" in b]),
                ("champion in evolution games (5-gen avg)", "#3b82c4",
                 _avg([(r["gen"], r["fit"][0]) for r in log if "fit" in r]))],
               [], gmax, "{:.3f}",
               note="A win is worth 1 + speed x (rounds before round 10); this is what selection maximises.")
        if any("fit" in r for r in log) else "",
        _chart("Champion's share in its own generation",
               [("champion (5-gen avg)", "#3b82c4", _avg([(r["gen"], r["share"][0]) for r in log]))],
               swaps, gmax, "{:.0%}", ref=0.25,
               note="Gold dashed lines = a mutant took over (hover for the weights it changed)."),
    ]
    start_p, champ_p = os.path.join(out_dir, "start.json"), os.path.join(out_dir, "champion.json")
    table = ""
    if os.path.exists(start_p) and os.path.exists(champ_p):
        from src.selfplay.evolve import effective
        s, c = effective(json.load(open(start_p))), effective(json.load(open(champ_p)))
        rows = [(k, s.get(k), c[k]) for k in sorted(c) if isinstance(c[k], (int, float))
                and s.get(k) != c[k]]
        rows.sort(key=lambda t: -abs((t[2] or 0) / (t[1] or 1e-9) - 1) if t[1] else 0)
        table = ("<section><h2>Champion weights that differ from the start</h2><table class='t'>"
                 "<tr><th>weight</th><th>start</th><th>champion</th></tr>" +
                 "".join(f"<tr><td>{html.escape(k)}</td><td>{'-' if a is None else f'{a:g}'}</td>"
                         f"<td>{b:g}</td></tr>" for k, a, b in rows) + "</table></section>")
    last = log[-1]
    nb = bench[-1] if bench else None
    intro = (f"<p class='sub'>Generation {last['gen']} · {sum(1 for r in log if r.get('new_champion'))} "
             f"champion swaps · last generation averaged {last['rounds']:.2f} rounds"
             + (f" · latest benchmark (gen {nb['gen']}): {nb['share']:.0%} win share vs start, "
                f"{nb['rounds']:.2f} rounds, {nb['vp_r6']:.2f} VP after round 6" if nb else "")
             + ". Refresh the page to update.</p>")
    page = _page("Heuristic evolution progress", intro, charts + [table]).replace(
        "<title>AlphaZero Progress</title>", "<title>Heuristic Evolution</title>")
    path = os.path.join(out_dir, "progress.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    print("wrote", build(ap.parse_args().out))


if __name__ == "__main__":
    main()
