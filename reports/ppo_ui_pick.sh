#!/bin/bash
# Pick the website AI: self-play iter 3170 / 3948 vs the current UI net (iter 1340) and vs Best AI, paired seeds
cd "$(dirname "$0")/.."
T=tuned=config/ppo/base.json:greedy
P1340="ppo:net=models_ppo/eval/iter1340_best.npz:$T"
H="heuristic:tuned=config/ppo/base.json"
arena() {  # out base name1 spec1 name2 spec2 (seats 2-4 = spec2)
  python -m src.selfplay.ppo_arena --spec "$4" --spec "$6" --spec "$6" --spec "$6" \
    --name "$3" --name "$5" --name "$5" --name "$5" --games 1000 --base "$2" --workers 20 --out "$1"
}
for c in self3170 self3948; do
  C="ppo:net=models_ppo/eval/$c.npz:$T"
  arena reports/pick_${c}_vs_1340.json 81000000 $c "$C" i1340 "$P1340"
  arena reports/pick_${c}_vs_heur.json 82000000 $c "$C" H "$H"
done
arena reports/pick_i1340_vs_heur.json 82000000 i1340 "$P1340" H "$H"
echo ALLDONE
