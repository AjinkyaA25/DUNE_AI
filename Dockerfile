# The public Dune AI table (ui/server.py in website mode): you vs the Best AI.
# Only what the server needs is copied - no training data, logs or checkpoints.
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir "numpy>=1.24"

COPY src/ src/
COPY ui/ ui/
COPY play_game.py ./
COPY ["Tech tiles/", "Tech tiles/"]
COPY sard_skills/ sard_skills/
COPY hub_images/ hub_images/
COPY video_scrape/cards/ video_scrape/cards/
COPY config/best_ai.json config/best_ai.json
COPY models_eval_leaders/evaluator.npz models_eval_leaders/evaluator.npz

ENV DUNE_PUBLIC=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PORT=8080
EXPOSE 8080
CMD ["python", "ui/server.py"]
