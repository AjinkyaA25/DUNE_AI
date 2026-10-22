#!/bin/bash
# Privileged-critic run (user, 2026-10-21): identical to models_ppo_local/run.sh (from the UI's Best AI,
# self-play league, dense reward, every legal move, eval vs 3 x frozen iter 3170) PLUS --critic, so the
# two eval curves compare directly. Resumes from its checkpoint when restarted.
cd "$(dirname "$0")/.."
P=tuned=config/ppo/base.json:greedy
python -u -m src.selfplay.ppo_train --out models_ppo_critic --init models_ppo_critic/pool/self3170.npz \
    --games 300 --workers 10 --k 0 --dense 0.1 --critic --critic-warmup 8 \
    --cur-frac 0.5 --pool "models_ppo_critic/pool/*.npz" --self-frac 0.6 --league-frac 0.1 \
    --eval-every 10 --eval-games 600 \
    --eval-opp "ppo:net=models_ppo_critic/pool/self3170.npz:$P" --eval-base "ppo:net=models_ppo_critic/pool/self3170.npz:$P"
