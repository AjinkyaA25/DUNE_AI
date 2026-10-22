#!/bin/bash
# Website AI refresh (2026-10-22): PACE self-play iter 6240 and local critic iter 300 vs the current UI net
# (iter 3170) and vs Best AI; base 82000000 = the seeds where iter 3170 scored 71.9% vs Best AI.
cd "$(dirname "$0")/.."
T=tuned=config/ppo/base.json:greedy
U="ppo:net=models_ui/ppo.npz:$T"
H="heuristic:tuned=config/ppo/base.json"
arena() {
  python -m src.selfplay.ppo_arena --spec "$4" --spec "$6" --spec "$6" --spec "$6" \
    --name "$3" --name "$5" --name "$5" --name "$5" --games 1000 --base "$2" --workers 6 --out "$1"
}
for c in self6240 critic300; do
  C="ppo:net=models_ppo/eval/$c.npz:$T"
  arena reports/pick2_${c}_vs_ui3170.json 93000000 $c "$C" ui3170 "$U"
  arena reports/pick2_${c}_vs_heur.json 82000000 $c "$C" H "$H"
done
echo ALLDONE
