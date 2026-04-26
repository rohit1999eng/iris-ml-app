# ── Stage 1: Train the model ─────────────────────────────────────────────────
FROM python:3.11-slim AS trainer

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
RUN python train.py

# ── Stage 2: Production API server ───────────────────────────────────────────
FROM python:3.11-slim AS production

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=trainer /build/model/model.pkl ./model/model.pkl
COPY --from=trainer /build/model/metadata.json ./model/metadata.json

COPY app.py .
COPY templates ./templates 

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]