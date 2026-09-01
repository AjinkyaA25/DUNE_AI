# Opening priors (`openings.json`)

These are **soft** nudges. The agent adds `weight` to an action's score when the
action matches a `prefer` entry of an *active* rule. It can still ignore the
nudge if its value network / heuristic strongly prefers something else. They
also bias self-play exploration so training starts from strong lines.

## Rule shape

```json
{
  "name": "human-readable id",
  "match": {
    "max_round": 4,                     // rule active while gs.round <= 4
    "min_round": 2,                     // and >= 2
    "num_players": [2, 3, 4],           // only these player counts
    "my_leader": "Muad'Dib",            // only for this leader (omit = any)
    "my_influence_below": { "fremen": 2 } // only while my fremen influence < 2
  },
  "prefer": [
    { "space": "High Council", "weight": 3.0 },
    { "card": "Dagger", "space": "Sardaukar", "weight": 2.0 },  // specific card+space
    { "buy_card": "Mentat", "weight": 2.0 },
    { "buy_reserve": "spice_must_flow", "weight": 3.0 },
    { "action": "reveal_turn", "weight": -1.0 }                 // discourage revealing early
  ]
}
```

## The seeded lines (Claude's proposed 4-player openings)

| Rule | Idea |
|---|---|
| `economy-spine-high-council` | Get the **Councilor** (+2 persuasion every Reveal) early; buy the **Swordmaster** (3rd agent) as soon as affordable. Tempo compounds. |
| `snowball-persuasion-turn1` | Grab a cheap persuasion card turn 1 (Mentat / Arrakis Liaison / Landsraad Council) so your buys snowball. |
| `push-one-faction-to-friendship` | Drive **one** faction to Influence 2 fast — a free VP — then toward 4 for the Alliance. Weighted across all faction spaces so the agent takes whichever suits its hand/leader. |
| `spice-water-economy` | Keep the resource engine fed (Imperial Basin, Hagga Basin, Accept Contract). |
| `troops-before-level-2-conflicts` | Pick up a troop-producing card before the Level-II conflicts arrive so you can actually contest combat. |
| `contest-victory-conflicts` | From round 3, fight for the controlled locations / VP conflicts (Arrakeen, Imperial Basin). |

Edit weights, add `my_leader`-specific lines, or delete rules you disagree with —
then re-run training. An empty `"rules": []` disables the book entirely.
