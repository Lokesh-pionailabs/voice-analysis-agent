# syntax=docker/dockerfile:1.7

# --- Builder stage: install dependencies into a venv ------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Runtime stage: minimal image with only what's needed to run -----------
FROM python:3.12-slim AS runtime

# Run as a non-root user (Cloud Run and general security best practice).
RUN groupadd --system app && useradd --system --gid app --home-dir /app app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

COPY app ./app

RUN chown -R app:app /app
USER app

EXPOSE 8080

# Single worker per container: Cloud Run scales via container concurrency
# and horizontal instance count, not multi-process workers. Adjust
# --workers if deploying on a platform without that model.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
