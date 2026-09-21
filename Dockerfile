# В образе только HTTP-сервис. Модель на удалённом сервере, эмбеддинги — в Ollama на хосте.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY knowledge ./knowledge

# Данные в томе: база переживает пересборку образа, индекс правил строится при первом старте.
ENV DB_PATH=/data/bank.sqlite3 \
    CHECKPOINTS_PATH=/data/checkpoints.sqlite3 \
    INDEX_PATH=/data/policies.sqlite3
VOLUME /data
EXPOSE 8000

# Без базы агенту не с чем работать, поэтому при первом запуске наполняем её.
CMD ["sh", "-c", "[ -f \"$DB_PATH\" ] || python -m app.seed; exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
