FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE_DIR=/app

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY ml_service ./ml_service
COPY baseline_lgbm ./baseline_lgbm
COPY .env.example ./.env.example
COPY train_team_track.parquet ./train_team_track.parquet
COPY test_team_track.parquet ./test_team_track.parquet

EXPOSE 8090

CMD ["python", "ml_service/server.py"]
