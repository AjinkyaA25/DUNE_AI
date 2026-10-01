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
`value:models/value_best.npz:T0.5`,
`heuristic:tuned=config/<weights>.json` (fitted weights, see below),
`search:K5:M24` (round search: top-5 heuristic moves, 24 full-game playouts
each; `M48` = 48 playouts, `:tuned=<json>` = heuristic used inside the search,
`:B` = also search card buys).

## Play in the browser

Double-click **`play_in_browser.bat`** (project folder), or run

```bash
.venv/Scripts/python.exe ui/server.py      # then open http://localhost:8765
```

"New game" → pick your seat and each opponent (Search 48 / 24 / 8 playouts,
Winners' style, Your style, human-tuned or default heuristic). Click a card
in your hand, then a highlighted board space; at reveal click a highlighted
Imperium Row card to buy it. Every AI move in the log expands to what it
considered: search opponents show their candidate moves with the playout win
chance, heuristic opponents their top-scored moves. "Show AI hands" reveals
the opponents' cards for review. Finished games are saved to `game_logs/`,
your decisions to `data/human_games/` (training format). Code: `ui/server.py`
(engine + JSON API) and `ui/static/` (page).

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

## Learning from video games (Dinosaur11 / Consules streams)

| Step | Tool |
|---|---|
| Extract every downloaded stream (board scan + chat OCR → game files), then delete the video | `video_scrape/process_videos.py --workers 8` |
| Merge the chat log into a stream's games | `video_scrape/merge_chat.py <video id>` |
| Replay games in the engine → training samples | `video_scrape/replay.py` → `data/video_games/` |
| Both of the above, automatically, for each finished video | `video_scrape/auto_ingest.sh` |
| Where humans and the AI disagree (your seat + winner tagged) | `video_scrape/compare_ai.py` → `reports/human_vs_ai.json` |
| Fit heuristic weights to human moves | `video_scrape/tune_heuristic.py --subset all\|winners\|me --out config/<name>.json` |
| Compare fits / cross-validate them | `video_scrape/compare_models.py`, `video_scrape/cv_models.py` |

Notes: chat OCR is capped at 3 threads per process (`OCR_THREADS`), or each
process takes ~12 cores; a video whose scan finds no game of 5+ rounds is
kept and logged as SUSPECT instead of being deleted.

Weight files in `config/`: `heuristic_tuned.json` (all players, 39 games),
`heuristic_tuned_all88.json` (all players, 88 games), `heuristic_winners.json`
(game winners' moves), `heuristic_me.json` (Dinosaur11's moves).
Situational knobs (new): `early.` / `mid.` / `late.` prefixed keys apply in
rounds 1-3 / 4-7 / 8-10 only, and `sat_<resource>` makes a gain worth less the
more of that resource you already hold.

## Results log

All arenas: 4 players, Bloodlines, neutral leaders. "Fair" share is 25%.
Arenas are not bit-for-bit repeatable (the same config re-run moves by
about ±0.5 percentage points).

### 1. Every AI version vs 3× the original heuristic (300 games each)

| Agent | Win share |
|---|---|
| Random | 0% |
| Heuristic, Bloodlines-blind | 18.0% |
| Heuristic (original) | 27.0% |
| Value net v1 | 28.3% |
| Value + policy v2 / v3 | 30.0% / 30.0% |
| Nets trained on search self-play (s1) | 25.7% |
| Heuristic tuned to human games (16 games) | **32.7%** |

### 2. Human-tuned heuristic, refit as video data grew (vs 3× default heuristic, 800 games)

| Fit | Matches human moves (held-out) | Win share |
|---|---|---|
| Default heuristic | 31.5% | 25% (it is the field) |
| Fitted on 16 games | 38.1%* | 32.6% |
| Fitted on 39 games | 38.1%* | **33.8%** |

*same 8 unseen games; the two fits tie, so the 16-game fit was not overfit.

### 3. Four agents at one table (96 games)

| Agent | Win share |
|---|---|
| Search, 24 playouts | **66.7%** |
| Search, 8 playouts | 24.0% |
| Heuristic, human-tuned | 9.4% |
| Random | 0% |

More playouts = much stronger (8 → 24 playouts: 24% → 67%).

### 4. Human-tuned heuristic inside search (96 games, one table)

| Agent | Win share |
|---|---|
| Search 24, default heuristic | **50.0%** |
| Search 24, human-tuned heuristic | 43.8% |
| Heuristic, human-tuned | 4.2% |
| Heuristic, default | 2.1% |

No gain: against AI opponents, the default heuristic models them at least as
well. Search keeps the default heuristic.

### 5. Search with 48 vs 24 playouts (one table; 96 games, in progress)

At 81/96 games: 48 playouts **43 wins (53%)**, 24 playouts 32 (40%),
human-tuned heuristic 3, default heuristic 3.

### 6. Winners' style vs your style (88 video games)

Data: 90 replayed games, 8,815 decisions; your seat identified in 51 games
(you won 17 = **33%**), 1,211 of your decisions, 2,207 winners' decisions.

Matching human moves, 5-fold cross-validated (rows = whose moves the
weights were fitted to, columns = whose moves are predicted):

| Fitted to | everyone | winners | you |
|---|---|---|---|
| (default) | 30.8% | 29.1% | 30.1% |
| everyone | **36.0%** | **34.5%** | **35.3%** |
| winners | 35.7% | 34.2% | 34.9% |
| you | 34.0% | 33.3% | 34.4% |

Weights that came out the same in at least 4 of 5 folds (×1 = default):

| Value placed on | Winners | You |
|---|---|---|
| Spies | ×3 | ×1 |
| Water | ×1.5 | ×0.5 |
| Faction influence | ×1 | ×2 |
| Card choice | situational (tier list 30%) | tier list 70% |
| Revealing early | yes (+2 to +3) | no |
| Solari | ×0.5 | mixed |
| Combat / troops | ×0.25 / ×0.5 | ×0.25 / ×0.5 |

Playing strength, all four at one table (960 games):

| Weights | Win share |
|---|---|
| Your style | **28.2%** |
| Winners' style | 25.8% |
| Default | 24.1% |
| Everyone | 21.9% |

### 7. Human vs AI habits (first 39 games, per player per game)

You use Hagga Basin 1.17× vs opponents 0.70× and Imperial Basin 0.56× vs 0.96×
(same total spice visits, but paying water); winners use Research Station 0.54×
vs 0.28× for non-winners and Imperial Privilege 0.41× vs 0.28×, and High Council
0.44× vs 0.63×. Full tables: `reports/human_vs_ai_40.txt`.

### In progress

Expert iteration: 48-playout search self-play with buys searched
(`python -m src.selfplay.search_positions`) records every searched decision,
the candidate moves and their playout values; the heuristic's situational
weights are then fitted to rank moves the way the search does, and tested
first as a heuristic and then inside search.

## What's modelled / simplified

- Full turn flow: agent turns alternate, combat deployment on Combat spaces,
  special spaces (Swordmaster / High Council / Gather Support / Sietch Tabr /
  Spice Refinery / Maker choices), Plot + Combat Intrigue play, CHOAM contracts,
  9 Leaders (trigger-hook framework).
- **Approximations**: leader ability wording (verify vs cards — see each
  `Leader.notes`), a curated ~34-card Imperium set (not all 66), no
  WIN/ENDGAME intrigue timing, tech tiles stubbed, no 6-player mode.
