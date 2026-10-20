#!/bin/bash
# Commits and pushes models_ppo (best.npz/best.json + resumable checkpoint) every
# $1 seconds (default 3600; pass "0 once" for a single push) while PPO trains. Started by pace/ppo_autopush.sbatch.
INTERVAL=${1:-3600}
cd "$(dirname "$0")/.."
while true; do
    sleep "$INTERVAL"
    # don't grab checkpoint.pt mid-write
    while [ $(( $(date +%s) - $(stat -c %Y models_ppo/checkpoint.pt) )) -lt 5 ]; do sleep 5; done
    last=$(tail -1 models_ppo/log.jsonl)
    it=$(echo "$last" | python -c "import json,sys; print(json.load(sys.stdin)['iter'])")
    games=$(echo "$last" | python -c "import json,sys; print(json.load(sys.stdin)['games'])")
    best=$(python -c "import json; b=json.load(open('models_ppo/best.json')); print(f\"best iter {b['iter']}, eval {b['share']:.1%} vs Best AI {b['baseline']:.1%}\")")
    git add models_ppo/best.json models_ppo/best.npz models_ppo/checkpoint.pt models_ppo/net.npz models_ppo/log.jsonl
    git commit -q -m "PPO checkpoint: iteration $it ($games games); $best" \
        -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
        && echo "[autopush $(date -Is)] committed iter $it" \
        && (git push -q origin HEAD 2>&1 || echo "[autopush] push failed, will retry next hour")
    [ "$2" = once ] && break
done
