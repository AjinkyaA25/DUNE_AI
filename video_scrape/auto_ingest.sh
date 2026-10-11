#!/bin/bash
# Watch raw_streams/ for videos process_videos.py has finished (<id>.done) and
# feed each one in: merge its chat log, replay its games into data/video_games.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
while true; do
  for d in video_scrape/raw_streams/*.done; do
    v=$(basename "$d" .done)
    [ -e "video_scrape/raw_streams/$v.ingested" ] && continue
    $PY video_scrape/merge_chat.py "$v" 2>&1 | cut -c1-160
    $PY video_scrape/replay.py video_scrape/games/"$v"_g*.json --out data/video_games 2>&1 \
      | $PY -c "import sys,json
for l in sys.stdin:
    try: d=json.loads(l); print('REPLAY', d.get('game'), 'rounds', d.get('rounds'), 'samples', d.get('samples'), 'quality', d.get('quality'))
    except Exception: print(l.rstrip()[:200])"
    touch "video_scrape/raw_streams/$v.ingested"
    echo "INGESTED $v total_games=$(ls data/video_games/*.npz | wc -l)"
  done
  sleep 60
done
