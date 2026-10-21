#!/bin/bash
cd /c/Users/argon/Ajinkya/Dune_AI
export PYTHONHASHSEED=0
P="ppo:tuned=config/ppo/base.json:greedy:net=models_ppo/eval"
H="heuristic:tuned=config/best_ai.json"
E6="look:D2:K3:M1:ev=models_eval_leaders/evaluator.npz:tuned=config/best_ai.json:E6:EK3:EM4"
A="python -u -m src.selfplay.ppo_arena --workers 28"
$A --games 4000 --base 93000000 --spec "$P/iter1340_best.npz" --spec "$P/iter470_best.npz" --spec "$P/iter470_best.npz" --spec "$P/iter470_best.npz" \
   --name "PPO new (932k)" --name "PPO old a" --name "PPO old b" --name "PPO old c" --out reports/ppo_new_vs_3old.json
$A --games 4000 --base 94000000 --spec "$P/iter1340_best.npz" --spec "$P/iter1340_best.npz" --spec "$P/iter470_best.npz" --spec "$P/iter470_best.npz" \
   --name "PPO new a" --name "PPO new b" --name "PPO old a" --name "PPO old b" --out reports/ppo_new2_vs_old2.json
$A --games 4000 --base 95000000 --spec "$P/iter1340_best.npz" --spec "$P/iter470_best.npz" --spec "$P/iter150.npz" --spec "$H" \
   --name "PPO new (iter 1340)" --name "PPO old (iter 470)" --name "PPO 63k (iter 150)" --name "Best AI heuristic" --out reports/ppo_ladder2.json
$A --games 800 --base 92000000 --spec "$P/iter1340_best.npz" --spec "$E6" --spec "$E6" --spec "$E6" \
   --name "PPO new" --name E6a --name E6b --name E6c --out reports/ppo_new_vs_e6.json
echo ALL DONE
