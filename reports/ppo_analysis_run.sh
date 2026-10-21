#!/bin/bash
# PPO move analysis (2026-10-20): overrules + playstyle, and agreement with the video games' human moves.
cd /c/Users/argon/Ajinkya/Dune_AI
export PYTHONHASHSEED=0
python -u -m src.selfplay.ppo_overrule --net models_ui/ppo.npz --games 600 --style-games 2000 --workers 28 --out reports/ppo_overrule
python -u video_scrape/judge_ai.py config/ppo/base.json "ppo:tuned=config/ppo/base.json:greedy:net=models_ui/ppo.npz" \
    --workers 16 --out reports/video_judgement_ppo.json > reports/video_judgement_ppo.txt 2>&1
echo ALL DONE
