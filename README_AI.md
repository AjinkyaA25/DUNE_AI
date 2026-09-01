# Dune Imperium: Uprising — self-play AI

## Setup

```bash
python -m venv .venv           # already present
.venv/Scripts/pip install -r requirements.txt   # NumPy + pytest only
python -m pytest tests/ -q
```

## Play a game

```bash
# You (P0) vs 3 heuristic bots
python play_game.py --players 4 --human 0

# Watch bots play each other; mix agents per seat
python play_game.py --players 4 --human -1 --quiet-ai \
    --agents heuristic,value:models/value_best.npz,heuristic,random
```

Agent specs: `random`, `heuristic`, `heuristic:T0.7` (softmax temp),
`value` (1-ply lookahead, heuristic leaf), `value:models/value_best.npz`,
`value:models/value_best.npz:T0.5`.

## Train by self-play

```bash
python train.py --iterations 15 --games-per-iter 400 --workers 8 --players 4
```

Each iteration: generate self-play games with the current best agent →
refit the value network on a replay buffer → `arena` gate → promote if it
beats the previous best. Artifacts:

- `models/value_vNN.npz` — checkpoint per iteration
- `models/value_best.npz` — current best
- `models/metrics.csv` — iter, val log-loss, arena result, promotion
- `data/selfplay/*.npz` — training shards (+ `.json` manifests)

Flags: `--no-book` (disable opening priors), `--temperature`, `--hidden`,
`--replay-k`, `--arena-games`.

## Evaluate

```bash
python -m src.selfplay.arena --a value:models/value_best.npz --b heuristic \
    --games 200 --players 4
```

`a_vs_fair` > 1.0 means agent A beats an even split of wins.

## Teach it openings

Edit `config/openings.json` (see `config/openings.README.md`). Rules are
**soft** score nudges gated by round / player-count / leader / influence.
They bias both live play and self-play exploration. `"rules": []` disables it.

## What's modelled / simplified

- Full turn flow: agent turns alternate, combat deployment on Combat spaces,
  special spaces (Swordmaster / High Council / Gather Support / Sietch Tabr /
  Spice Refinery / Maker choices), Plot + Combat Intrigue play, CHOAM contracts,
  9 Leaders (trigger-hook framework).
- **Approximations**: leader ability wording (verify vs cards — see each
  `Leader.notes`), a curated ~34-card Imperium set (not all 66), no
  WIN/ENDGAME intrigue timing, tech tiles stubbed, no 6-player mode.
