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
the opponents' cards for review. The Bloodlines panels show the tech market
(top tile of each stack), the 4-tile Sardaukar skill row, which commander
spaces still hold a commander, and the face-up contracts with how to complete
them. Code: `ui/server.py` (engine + JSON API) and `ui/static/` (page).

**Your games are tracked** (autosaved after every one of your moves, so an
unfinished game is kept too):

| File | What |
|---|---|
| `game_logs/ui_<time>_<id>.json` | every move by every player, the AI's reasoning for each of its moves, what the AI would have played on each of yours, opponents, seed, result |
| `data/ui_games/ui_<time>_<id>.pkl` | each of your decisions as a position (board, options, your choice, won/lost); finished games are picked up automatically by `tune_heuristic.py`, `compare_models.py` and `cv_models.py` as "you" decisions |
| `data/human_games/human_*.npz` | your decisions in the neural-net training format (finished games) |

`python ui/games_report.py` summarises them: your record against each
opponent line-up, how often your moves matched the AI's, and your most common
differences. Only one server can run at a time; if the page doesn't change
after an update, stop the running server (Ctrl+C) and start it again.

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

### ★ Old rules vs corrected engine (2026-10-02)

The engine had several rules bugs (list under *Rules fixes* below). Every AI
version was re-benchmarked on the corrected engine against the **same fixed
opponent** (3× the original heuristic with its original scoring,
`config/fight_old.json`) on the **same deals**, so the two columns compare
directly. Fair share 25%; ±5 points is noise at 300 games (±7 at 120).

| AI version | Old rules | Corrected engine | Avg round game ended |
|---|---|---|---|
| Random | 0% | 0% | 9.6 |
| Heuristic, Bloodlines-blind | 18.0% | 25.0% | 9.7 |
| Heuristic (original) — the opponent itself | 27.0% | 24.0% | 9.7 |
| Value net v1 (self-play RL) | 28.3% | **36.7%** | 9.7 |
| Value + AWR policy v2 (self-play RL) | 30.0% | **42.7%** | 9.7 |
| Value + AWR policy v3 (self-play RL) | 30.0% | **40.0%** | 9.7 |
| Nets trained on search self-play (s1) | 25.7% | 30.3% | 9.7 |
| Heuristic tuned to human games (all players) | 32.7% | **38.3%** | 9.7 |
| **New:** heuristic + fight model | — | 31.0% | 9.7 |
| **New:** your style (Dinosaur11 fit) + fight model | — | 34.7% | 9.6 |
| **New:** winners' style + fight model | — | 28.3% | 9.6 |
| **New:** RL + AWR retrained on the corrected engine | — | 34.3% | 9.7 |
| Search, 8 playouts (120 games) | 70% * | **77.5%** | 9.6 |
| Search, 24 playouts (120 games) | 84% * | **90.0%** | 9.3 |

\* old-rules search numbers are from the earlier `search_arena` runs (200 /
120 games). Raw results: `reports/benchmark.json` (old) and
`reports/benchmark_current.log`, `reports/benchmark_rl_fixed.log` (new).

**Reading it.** Every trained model gained 8-13 points on the corrected
engine: their decisions lean on the heuristic, which now has the fight model
and real intrigue choices. Retraining the RL value net + AWR policy on the
corrected rules did **not** help: it improved once (iteration 2: 28.5% vs the
current heuristic) and then plateaued for six iterations; on the same engine
it is weaker than the old v2 (34.3% vs 42.7%). Search remains far ahead.

**Head-to-head tests of the new decision models** (2 seats each, 1920 games,
fair 50% / 50%):

| Change | New | Old |
|---|---|---|
| Fight model (deploy + combat intrigues by expected reward vs holding value) | **59.0%** | 41.0% |
| Deck-quality buying | 50.8% | 49.2% |
| Deck-quality buying + Prepare the Way / Weirding Woman rated D (test only) | **54.6%** | 45.4% |
| Search 48 vs 24 playouts (old rules, 92 games, one table) | **49 wins** | 37 wins |

**Game length: still the main gap.** Self-play, all four seats the same AI
(200 games each):

| AI playing itself | Avg round game ends | Games ended by 10 VP |
|---|---|---|
| Humans (89 video games) | **7.7** | — |
| Winners' style + fight model | 9.3 | 84% |
| Your style + fight model | 9.4 | 82% |
| Original heuristic | 9.7 | 64% |
| Heuristic + fight model | 9.8 | 60% |
| Value + policy v2 / RL retrained | 9.8 | 50% / 61% |

**Final head-to-head** (one table, every game: your style, search 24,
search 8, RL retrained; 96 games): _see below, filled in when it finishes._

**Open issues found along the way**

- Game tempo: AIs finish ~2 rounds later than humans; nothing so far moves it
  except the human-fitted weights (slightly).
- RL self-play + AWR policy plateaus quickly; search and heuristic fixes are
  where the gains come from.
- Tier list: Prepare the Way and Weirding Woman are B in
  `config/consules_tierlist_DRAFT.md`; the user rates them as near-useless and
  the test above agrees. Awaiting the user's call before changing the list.
- Spies: the Landsraad post (High Council / Swordmaster) is almost never used
  by the AI.


### 1. Every AI version vs 3× the original heuristic (300 games each, OLD rules)

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

### 5. Search with 48 vs 24 playouts (one table, old rules)

Stopped at 92/96 games: 48 playouts **49 wins (53%)**, 24 playouts 37 (40%),
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

### 8. Game phases: build early, convert late (hand-set test, 800 games each vs 3× default)

Phases: early = rounds 1-3 (build resources), mid = 4-5, late = 6+ (convert
resources and techs into VP through faction influence and combat).

| Variant | Win share |
|---|---|
| Late: influence & combat ×1.5, solari/spice/water ×0.6, draw ×0.7 | 26.1% |
| Late: influence & combat ×2, resources ×0.4, draw ×0.5 | 24.4% |
| Early resources ×1.3 + the ×1.5 late shift | 25.1% |

Guessed multipliers do not move the win rate, so the phase weights are being
learned from game results instead (`python -m src.selfplay.tune_by_wins`:
keeps a change only if it wins more against a mixed field of default,
winners' style and your style, and still does on fresh deals).

### Rules fixes

- Sardaukar skills (2026-10-01): 14 tiles (2 of each of the 7 skills); 4 are
  face-up in the skill row. Recruiting a commander from the board or Plasteel
  Blades' bonus takes one of those 4 (one you don't hold), and the slot is
  refilled from the 10 face-down tiles. Plasteel Blades is now a choice (which
  row skill, or keep the tile). Previously any skill could be taken.

- First player (2026-10-02): the First Player token only advanced every
  OTHER round (0,1,1,2,2,3,…); now it moves every round (0,1,2,3,0,…).
  Every engine game before this fix was affected.
- Intrigue choices (2026-10-02, all 59 intrigue cards checked against their
  art): choices inside cards were made by fixed rules for both players and
  AI. They are now real decisions (`src/game/choices.py`): Poison Snooper
  draw/trash, Inspire Awe / Impress card, faction picks (Bribery, Buy Access,
  Imperium Politics, Sietch Ritual, Change Allegiances), Manipulate, Market
  Opportunity, Opportunism (optional), troops to retreat (Go to Ground, Reach
  Agreement, Tactical Option), Special Mission, Emperor's Invitation,
  Coercive Negotiation, "deploy up to N", which card to discard. OR cards
  (Detonation, Counterattack, Backed by CHOAM, Insider Information, Sleeper
  Unit, Spice is Power) are one-or-the-other; Questionable Methods' influence
  loss is optional; Tenuous Bond (combat) trashes a 1+ card from your discard;
  Call to Arms counts acquisitions made after playing it.
- Intrigue requirements (2026-10-02): an intrigue can only be played when at
  least one of its effects can happen (cost payable, condition met, option
  available); "deploy up to N" includes 0, so Detonation is always playable.
- Prepare the Way (2026-10-02): agent effect is "2 Bene Gesserit influence:
  draw a card" (engine had 1 solari).
- Infiltrate (2026-10-02): an occupied space could be entered with a plain
  placement that skipped spending the Spy and overwrote the first Agent (a
  free Infiltrate, ~3 per game in AI play). Now only Infiltrating (which spends
  the Spy) gets you in, and both Agents stay on the space.

### 9. Deploying troops and playing combat intrigues (2026-10-02)

The AI deployed the maximum 84% of the time (garrison emptied in 55-75% of
deploys) and played every combat intrigue it could (only 33% changed its
placing). Now both decisions use an expected-reward model of the Conflict
against the value of keeping the troops / card.

| | Old | New |
|---|---|---|
| Max deploy, rounds 1-3 | 83% | 46% |
| Garrison emptied, rounds 1-3 | 55% | 20% |
| Win share, 2 seats each, 1920 games | 41.0% | **59.0%** |

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
