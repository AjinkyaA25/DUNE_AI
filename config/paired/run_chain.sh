#!/bin/bash
# endgame confirm (fresh seeds, timed) -> league check -> restart the loop
export PYTHONHASHSEED=0
L="look:D2:K3:M1:ev=models_eval_leaders/evaluator.npz:tuned=config/best_ai.json"
s=$(date +%s); python -u -m src.selfplay.paired_test "$L" --games 1200 --base 85000000 --out reports/endgame_confirm_plain.json > reports/endgame_confirm_plain.txt 2>&1; e=$(date +%s); echo "plain seconds $((e-s))" >> reports/endgame_confirm_time.txt
s=$(date +%s); python -u -m src.selfplay.paired_test "$L:E6:EK3:EM4" --games 1200 --base 85000000 --out reports/endgame_confirm_e6.json > reports/endgame_confirm_e6.txt 2>&1; e=$(date +%s); echo "e6 seconds $((e-s))" >> reports/endgame_confirm_time.txt
python -u -m src.selfplay.paired_test config/best_ai.json config/best_ai_history/*.json config/evolve_plan/champion.json config/evolve_fixed/champion.json config/gap/best_ref2.json config/evolve_look/champion.json --games 2000 --base 87000000 --out reports/league_check.json > reports/league_check.txt 2>&1
