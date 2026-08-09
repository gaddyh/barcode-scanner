# --- Stage 1: Build the React frontend --------------------------------------
FROM node:20-slim AS frontend

WORKDIR /web

COPY web/package.json web/package-lock.json* ./
RUN npm ci --silent || npm install --silent

COPY web/ ./
RUN npm run build

# --- Stage 2: Python backend ------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

# Copy the built frontend so FastAPI can serve it at /
COPY --from=frontend /web/dist ./web/dist

EXPOSE 8000

# Render assigns the port via $PORT; fall back to 8000 for local/docker-compose.
CMD sh -c "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"
