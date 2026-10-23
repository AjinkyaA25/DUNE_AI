"""Cross-feed two PPO runs' opponent pools so they train off each other.

The PACE self-play run (models_ppo_self, pushed to git by pace/autopush.sh) and the local
privileged-critic run (models_ppo_critic) each re-read their --pool glob every iteration.
Every --every seconds this:
  * copies the self-play run's newest best.npz from origin into the critic run's pool
    (models_ppo_critic/pool/selfbest<iter>.npz), and
  * pushes the critic run's current net into the self-play run's pool on origin
    (models_ppo_self/pool/critic<iter>.npz), at most once per --push-every seconds,
    through a separate git worktree so the training checkout is never rebased.
Files are only ever added (never replaced or deleted): a running trainer may still be
loading an older one, and Elo keys pool nets by file name.

    python -m src.selfplay.ppo_cross_pool --worktree ../Dune_AI_sync
"""
import argparse
import json
import os
import shutil
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("--branch", default="video-scrape")
ap.add_argument("--self-dir", default="models_ppo_self")
ap.add_argument("--critic-dir", default="models_ppo_critic")
ap.add_argument("--worktree", default=os.path.join("..", "Dune_AI_sync"))
ap.add_argument("--every", type=int, default=3600)
ap.add_argument("--push-every", type=int, default=6 * 3600)
ap.add_argument("--once", action="store_true")
A = ap.parse_args()
REF = f"origin/{A.branch}"


def git(*args, cwd=".", binary=False):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=not binary).stdout


def log(msg):
    print(f"[cross_pool {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def pull_self():
    best = json.loads(git("show", f"{REF}:{A.self_dir}/best.json"))
    dst = os.path.join(A.critic_dir, "pool", f"selfbest{best['iter']}.npz")
    if os.path.exists(dst):
        return
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(git("show", f"{REF}:{A.self_dir}/best.npz", binary=True))
    os.replace(tmp, dst)          # the trainer globs *.npz, so it never sees a half-written file
    log(f"self-play best iter {best['iter']} ({best['share']:.1%}) -> {dst}")


def push_critic():
    it = json.loads(open(os.path.join(A.critic_dir, "log.jsonl")).readlines()[-1])["iter"]
    name = f"{A.self_dir}/pool/critic{it}.npz"
    wt = A.worktree
    if not os.path.isdir(wt):
        git("worktree", "add", "--detach", wt, REF)
    for attempt in range(3):
        git("fetch", "-q", "origin", A.branch, cwd=wt)
        git("reset", "-q", "--hard", REF, cwd=wt)
        shutil.copy(os.path.join(A.critic_dir, "net.npz"), os.path.join(wt, name))
        git("add", name, cwd=wt)
        git("commit", "-q", "-m", f"PPO cross-pool: privileged-critic run iteration {it} joins the self-play pool",
            "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", cwd=wt)
        try:
            git("push", "-q", "origin", f"HEAD:{A.branch}", cwd=wt)
            log(f"critic iter {it} -> {name} (pushed)")
            return
        except subprocess.CalledProcessError as e:
            log(f"push rejected ({e.stderr.strip()[:200]}), retrying")
            time.sleep(30)
    log("push failed 3 times, will retry next cycle")
    raise RuntimeError("push failed")


last_push = 0.0
while True:
    try:
        git("fetch", "-q", "origin", A.branch)
        pull_self()
        if time.time() - last_push >= A.push_every:
            push_critic()
            last_push = time.time()
    except Exception as e:      # network hiccups etc.: try again next cycle
        log(f"error: {e}")
    if A.once:
        break
    time.sleep(A.every)
