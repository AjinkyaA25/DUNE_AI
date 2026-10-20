#!/bin/bash
# Commits and pushes models_ppo (best.npz/best.json + resumable checkpoint) every
# $1 seconds (default 3600; pass "0 once" for a single push) while PPO trains. Started by pace/ppo_autopush.sbatch.
# MODEL_DIR=<dir> pushes another run's folder instead (pace/ppo_wide.sbatch).
INTERVAL=${1:-3600}
D=${MODEL_DIR:-models_ppo}
cd "$(dirname "$0")/.."
while true; do
    sleep "$INTERVAL"
    # don't grab checkpoint.pt mid-write
    while [ $(( $(date +%s) - $(stat -c %Y $D/checkpoint.pt) )) -lt 5 ]; do sleep 5; done
    last=$(tail -1 $D/log.jsonl)
    it=$(echo "$last" | python -c "import json,sys; print(json.load(sys.stdin)['iter'])")
    games=$(echo "$last" | python -c "import json,sys; print(json.load(sys.stdin)['games'])")
    best=$(python -c "import json; b=json.load(open('$D/best.json')); print(f\"best iter {b['iter']}, eval {b['share']:.1%} vs Best AI {b['baseline']:.1%}\")")
    git add $D/best.json $D/best.npz $D/checkpoint.pt $D/net.npz $D/log.jsonl
    git commit -q -m "PPO checkpoint ($D): iteration $it ($games games); $best" \
        -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
        && echo "[autopush $(date -Is)] committed iter $it" \
        && (git push -q origin HEAD 2>&1 \
            || (git pull -q --rebase --autostash origin "$(git branch --show-current)" && git push -q origin HEAD) 2>&1 \
            || echo "[autopush] push failed, will retry next hour")
    [ "$2" = once ] && break
done
