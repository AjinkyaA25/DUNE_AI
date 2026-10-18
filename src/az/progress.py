"""AlphaZero run progress: a table in the terminal and charts in an HTML page.

    python -m src.az.progress [--out models_az]
    python -m src.az.progress --out models_az_scratch --compare models_az

Reads <out>/log.jsonl (one line per generation, written by src.az.train) and
<out>/changes.jsonl (rule / card-pool changes, drawn as dashed lines), and
writes <out>/progress.html. Safe to run while training is going.
"""
from __future__ import annotations

import argparse
import html
import json
import os


def _read(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:      # a line being written right now
                    pass
    return out


def _chart(title, series, changes, gmax, y_fmt="{:.2f}", ref=None, note="", x_label="generation"):
    """series: [(label, colour, [(x, value), ...])] -> an inline SVG card
    (a label ending in " (compare)" is drawn dashed)."""
    W, H, L, R, T, B = 640, 230, 48, 12, 14, 26
    pts = [v for _, _, s in series for _, v in s] + ([ref] if ref is not None else [])
    if not pts:
        return ""
    lo, hi = min(pts), max(pts)
    pad = (hi - lo) * 0.1 or abs(hi) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    gx = lambda g: L + (W - L - R) * g / max(gmax, 1)
    gy = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = gy(v)
        svg.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{L - 6}" y="{y + 4:.1f}" class="ax" text-anchor="end">'
                   f'{y_fmt.format(v)}</text>')
    step = max(1, round(gmax / 8))
    for g in range(0, int(gmax) + 1, step):
        svg.append(f'<text x="{gx(g):.1f}" y="{H - 8}" class="ax" text-anchor="middle">{g}</text>')
    if ref is not None:
        svg.append(f'<line x1="{L}" x2="{W - R}" y1="{gy(ref):.1f}" y2="{gy(ref):.1f}" class="ref"/>')
    for c in changes:
        x = gx(c["gen"])
        svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{H - B}" class="chg">'
                   f'<title>gen {c["gen"]}: {html.escape(c["change"])}</title></line>')
    for label, col, s in series:
        if not s:
            continue
        d = " ".join(f'{"M" if i == 0 else "L"}{gx(g):.1f},{gy(v):.1f}' for i, (g, v) in enumerate(s))
        dash = ' stroke-dasharray="5 4" opacity="0.75"' if label.endswith("(compare)") else ""
        svg.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
        for g, v in s:
            svg.append(f'<circle cx="{gx(g):.1f}" cy="{gy(v):.1f}" r="2.6" fill="{col}">'
                       f'<title>{html.escape(label)} gen {g}: {y_fmt.format(v)}</title></circle>')
    svg.append("</svg>")
    legend = "".join(f'<span><i style="background:{c}"></i>{html.escape(l)}</span>'
                     for l, c, s in series if s)
    return (f'<section><h2>{html.escape(title)}</h2><div class="lg">{legend}</div>'
            f'{"".join(svg)}<p class="note">x: {html.escape(x_label)}. {html.escape(note)}</p></section>')


def _series(log):
    """(per-generation records, policy evals, search evals), x = thousand games."""
    per = {r["gen"]: r.get("games", 0) for r in log}
    k = lambda g: per.get(g, 0) / 1000
    gens = [r for r in log if r.get("gen", 0) > 0]
    return ([(k(r["gen"]), r) for r in gens],
            [(k(r["gen"]), r["eval_policy"]) for r in log if r.get("eval_policy")],
            [(k(r["gen"]), r["eval_search"]) for r in log if r.get("eval_search")])


def build_compare(out_dir, cmp_dir):
    """Two runs on one set of charts, by games played."""
    A, B = _series(_read(os.path.join(out_dir, "log.jsonl"))), _series(_read(os.path.join(cmp_dir, "log.jsonl")))
    na, nb = os.path.basename(out_dir.rstrip("/\\")), os.path.basename(cmp_dir.rstrip("/\\")) + " (compare)"
    xmax = max([x for s in (A, B) for part in s for x, _ in part] + [1])
    def cfg(d):
        try:
            with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
                return [json.load(f)]
        except (OSError, ValueError):
            return [{}]
    charts = [
        _chart("Win share vs 3 tuned heuristic players",
               [(f"{na} net", "#d9822b", [(x, e["win_share"]) for x, e in A[1]]),
                (f"{na} net + search", "#3b82c4", [(x, e["win_share"]) for x, e in A[2]]),
                (f"{nb[:-10]} net (compare)", "#d9822b", [(x, e["win_share"]) for x, e in B[1]]),
                (f"{nb[:-10]} net + search (compare)", "#3b82c4", [(x, e["win_share"]) for x, e in B[2]])],
               [], xmax, "{:.0%}", ref=0.25, x_label="thousand self-play games",
               note="Solid = this run, dashed = comparison. Grey = 25%."),
        _chart("Rounds per game vs heuristic (lower = finishes faster)",
               [(na, "#2f9e6e", [(x, e["rounds"]) for x, e in A[1]]),
                (nb, "#2f9e6e", [(x, e["rounds"]) for x, e in B[1]])],
               [], xmax, "{:.1f}", x_label="thousand self-play games"),
        _chart("Self-play rounds per game",
               [(na, "#9b6cd6", [(x, r["rounds"]) for x, r in A[0]]),
                (nb, "#9b6cd6", [(x, r["rounds"]) for x, r in B[0]])],
               [], xmax, "{:.1f}", x_label="thousand self-play games"),
        _chart("Final VP vs heuristic",
               [(na, "#c4553b", [(x, e["vp"]) for x, e in A[1]]),
                (nb, "#c4553b", [(x, e["vp"]) for x, e in B[1]])],
               [], xmax, "{:.1f}", x_label="thousand self-play games"),
    ]
    def row(name, s):
        last = s[1][-1][1] if s[1] else {}
        g = s[0][-1][1] if s[0] else {}
        return (f"<tr><td>{html.escape(name)}</td><td>{g.get('gen', 0)}</td><td>{g.get('games', 0):,}</td>"
                f"<td>{last.get('win_share', 0):.1%}</td><td>{last.get('rounds', 0):.2f}</td>"
                f"<td>{g.get('eps', 0) if g else 0}</td></tr>")
    print(f"{'run':24s} {'gen':>4} {'games':>9} {'win vs heur':>11} {'rounds':>7}")
    for name, s in ((na, A), (nb, B)):
        last = s[1][-1][1] if s[1] else {}
        g = s[0][-1][1] if s[0] else {}
        print(f"{name:24s} {g.get('gen', 0):>4} {g.get('games', 0):>9,} {last.get('win_share', 0):>11.1%} {last.get('rounds', 0):>7.2f}")
    settings = "".join(f"<li><b>{html.escape(os.path.basename(d.rstrip('/')))}</b>: "
                       f"{html.escape(json.dumps({k: v for k, v in cfg(d)[0].items() if k in ('eps', 'eps_min', 'eps_decay', 'margin', 'speed', 'per_gen', 'workers', 'sims', 'p_full')}))}</li>"
                       for d in (out_dir, cmp_dir))
    page = _page(f"{na} vs {nb[:-10]}",
                 f"<table class='t'><tr><th>run</th><th>gen</th><th>games</th><th>latest win vs heuristic</th><th>rounds</th><th>random-move share</th></tr>"
                 f"{row(na, A)}{row(nb[:-10], B)}</table><ul>{settings}</ul>", charts)
    path = os.path.join(out_dir, "compare.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"\nwrote {path}")


def build(out_dir):
    log = _read(os.path.join(out_dir, "log.jsonl"))
    changes = _read(os.path.join(out_dir, "changes.jsonl"))
    gens = [r for r in log if r.get("gen", 0) > 0]
    evp = [(r["gen"], r["eval_policy"]) for r in log if r.get("eval_policy")]
    evs = [(r["gen"], r["eval_search"]) for r in log if r.get("eval_search")]
    gmax = max([r["gen"] for r in log] + [c["gen"] for c in changes] + [1])

    # terminal table
    print(f"{'gen':>4} {'games':>8} {'selfVP':>6} {'winVP':>6} {'rounds':>6} "
          f"{'pol.loss':>8} {'val.loss':>8}   vs heuristic (policy / search)")
    evp_d, evs_d = dict(evp), dict(evs)
    for r in log:
        g = r.get("gen", 0)
        e = []
        if g in evp_d:
            x = evp_d[g]
            e.append(f"policy {x['win_share']:.1%} VP {x['vp']:.2f} vs {x['opp_vp']:.2f} r{x['rounds']:.1f}")
        if g in evs_d:
            x = evs_d[g]
            e.append(f"search {x['win_share']:.1%} VP {x['vp']:.2f}")
        if g == 0:
            print(f"{0:>4} {'-':>8} {'':>6} {'':>6} {'':>6} {'':>8} {'':>8}   " + " | ".join(e))
            continue
        print(f"{g:>4} {r['games']:>8} {r['vp_mean']:>6.2f} {r['vp_max']:>6.2f} {r['rounds']:>6.2f} "
              f"{r.get('pl', 0):>8.3f} {r.get('vl', 0):>8.4f}   " + " | ".join(e))
    for c in changes:
        print(f"  change at gen {c['gen']}: {c['change']}")

    charts = [
        _chart("Win share vs 3 tuned heuristic players",
               [("net alone (no search)", "#d9822b", [(g, e["win_share"]) for g, e in evp]),
                ("net + search", "#3b82c4", [(g, e["win_share"]) for g, e in evs])],
               changes, gmax, "{:.0%}", ref=0.25,
               note="Dashed grey = 25%, an even share among 4 players. Measured every 5 / 20 generations."),
        _chart("Final VP in those evaluation games",
               [("AI", "#d9822b", [(g, e["vp"]) for g, e in evp]),
                ("heuristic opponents (avg)", "#8a8f98", [(g, e["opp_vp"]) for g, e in evp])],
               changes, gmax, "{:.1f}"),
        _chart("Self-play VP",
               [("winner's VP", "#2f9e6e", [(r["gen"], r["vp_max"]) for r in gens]),
                ("average VP", "#9b6cd6", [(r["gen"], r["vp_mean"]) for r in gens])],
               changes, gmax, "{:.1f}",
               note="Self-play samples moves to keep exploring, so it scores below the evaluation games."),
        _chart("VP at the end of round 6 (scoring pace)",
               [("self-play: leader", "#c9a227", [(r["gen"], r["vp_r6_max"]) for r in gens if "vp_r6_max" in r]),
                ("self-play: average", "#9b6cd6", [(r["gen"], r["vp_r6_mean"]) for r in gens if "vp_r6_mean" in r]),
                ("vs heuristic: AI", "#d9822b", [(g, e["vp_r6"]) for g, e in evp if "vp_r6" in e])],
               changes, gmax, "{:.1f}",
               note="Human games usually end in round 7-8; the round-6 replay's winner had 11 VP after round 6.")
        if any("vp_r6_max" in r for r in gens) else "",
        _chart("Rounds per game",
               [("self-play", "#2f9e6e", [(r["gen"], r["rounds"]) for r in gens]),
                ("vs heuristic", "#d9822b", [(g, e["rounds"]) for g, e in evp])],
               changes, gmax, "{:.1f}",
               note="Lower = the AI reaches 10 VP sooner. Strong human games end around round 7-8."),
        _chart("Self-play win share by persona (personas runs)",
               [(s, c, [(r["gen"], r["styles"][s]["win"]) for r in gens if s in r.get("styles", {})])
                for s, c in (("sword", "#c4553b"), ("deck", "#3b82c4"),
                             ("resource", "#2f9e6e"), ("faction", "#9b6cd6"), ("smf", "#c9a227"), ("commanders", "#8a8f98"))],
               changes, gmax, "{:.0%}", ref=0.25,
               note="Which playstyle wins its self-play games; 25% = even.") if any(r.get("styles") for r in gens) else "",
        _chart("Training loss",
               [("policy loss", "#c4553b", [(r["gen"], r["pl"]) for r in gens if "pl" in r]),
                ("value loss x10", "#3b82c4", [(r["gen"], 10 * r["vl"]) for r in gens if "vl" in r])],
               changes, gmax, "{:.2f}"),
    ]
    chg = "".join(f"<li><b>gen {c['gen']}</b> {html.escape(c['change'])}</li>" for c in changes)
    last = gens[-1] if gens else {}
    intro = (f"<p class='sub'>Generation {last.get('gen', 0)} · {last.get('games', 0):,} self-play games. "
             f"Rerun <code>python -m src.az.progress</code> to refresh.</p>"
             + (f'<p class="sub">Rule / card-pool changes (gold dashed lines):</p><ul>{chg}</ul>' if chg else ''))
    page = _page("AlphaZero training progress", intro, charts)
    path = os.path.join(out_dir, "progress.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"\nwrote {path}")


def _page(title, intro, charts):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaZero Progress</title>
<style>
:root{{--bg:#f6f5f2;--card:#fff;--fg:#1d1f23;--mut:#6b7079;--grid:#e3e1dc}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--card:#1f2125;--fg:#e8e6e1;--mut:#9aa0a8;--grid:#2e3036}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1340px;margin:0 auto;padding:20px 16px}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 16px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:14px}}
section{{background:var(--card);border-radius:10px;padding:12px 14px}}
h2{{font-size:15px;margin:0 0 4px}} svg{{width:100%;height:auto;display:block}}
.grid{{stroke:var(--grid)}} .ax{{fill:var(--mut);font-size:11px}}
.ref{{stroke:var(--mut);stroke-dasharray:3 4}} .chg{{stroke:#c9a227;stroke-dasharray:6 4;stroke-width:1.5}}
.lg{{display:flex;flex-wrap:wrap;gap:12px;font-size:13px;color:var(--mut)}}
.lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}
.note{{color:var(--mut);font-size:12.5px;margin:4px 0 0}} ul{{margin:6px 0 14px;padding-left:18px}}
.t{{border-collapse:collapse;margin:6px 0 10px;font-size:14px}} .t td,.t th{{padding:4px 10px;border-bottom:1px solid var(--grid);text-align:left}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
{intro}
<div class="grid2">{''.join(charts)}</div>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models_az")
    ap.add_argument("--compare", default=None, help="another run's folder to overlay")
    A = ap.parse_args()
    if A.compare:
        build_compare(A.out, A.compare)
    else:
        build(A.out)


if __name__ == "__main__":
    main()
