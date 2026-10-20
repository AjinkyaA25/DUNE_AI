#!/bin/bash
# PPO strength tests (2026-10-20): ladder of PPO checkpoints, mirror games, PPO vs the E6 endgame solver.
cd /c/Users/argon/Ajinkya/Dune_AI
export PYTHONHASHSEED=0
P="ppo:tuned=config/ppo/base.json:greedy:net=models_ppo/eval"
H="heuristic:tuned=config/best_ai.json"
E6="look:D2:K3:M1:ev=models_eval_leaders/evaluator.npz:tuned=config/best_ai.json:E6:EK3:EM4"
A="python -u -m src.selfplay.ppo_arena --workers 28"
$A --games 4000 --spec "$P/iter470_best.npz" --spec "$P/iter284.npz" --spec "$P/iter150.npz" --spec "$H" \
   --name "PPO 285k games (best)" --name "PPO 157k games" --name "PPO 63k games" --name "Best AI heuristic" --out reports/ppo_ladder.json
$A --games 2000 --spec "$P/iter470_best.npz" --spec "$P/iter470_best.npz" --spec "$P/iter470_best.npz" --spec "$P/iter470_best.npz" \
   --name PPO1 --name PPO2 --name PPO3 --name PPO4 --out reports/ppo_mirror.json
$A --games 2000 --spec "$H" --spec "$H" --spec "$H" --spec "$H" --name H1 --name H2 --name H3 --name H4 --out reports/heur_mirror.json
$A --games 800 --base 92000000 --spec "$P/iter470_best.npz" --spec "$E6" --spec "$E6" --spec "$E6" \
   --name "PPO best" --name E6a --name E6b --name E6c --out reports/ppo_vs_e6.json
$A --games 800 --base 92000000 --spec "$H" --spec "$E6" --spec "$E6" --spec "$E6" \
   --name "Best AI heuristic" --name E6a --name E6b --name E6c --out reports/heur_vs_e6.json
echo ALL DONE
