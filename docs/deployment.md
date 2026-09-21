# Deployment

## Direct upload experiment (web/)

Tiny Vite + React + TS mobile page that uploads the original phone photo
(no canvas, no compression, no base64) to the existing `/barcode/scan`
endpoint and shows dimensions, file size, barcodes, server scan latency,
and total request latency. No WhatsApp, no Gemini, no chat UI, no auth.

### Run locally

Backend (exposes `/health` and `/barcode/scan`):

```bash
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```bash
cd web
npm install
npm run dev -- --host 0.0.0.0
```

Open `http://localhost:5173`. Default API URL is `http://localhost:8000`
(see `web/.env.example`).

### With ngrok (phone needs HTTPS)

Expose the frontend and API separately:

```bash
ngrok http 5173   # frontend
ngrok http 8000   # API
```

Set the frontend API URL to the API tunnel:

```bash
# web/.env
VITE_API_BASE_URL=https://<api-tunnel>.ngrok-free.app
```

Restart Vite after changing `.env`. Then open the frontend ngrok URL on
the phone. ngrok free tier shows an interstitial page on first visit —
tap through once.

### Direct-vs-WhatsApp comparison

Take one photo and preserve both versions. For each run, record:

- source (direct / WhatsApp)
- filename
- dimensions
- bytes
- server scan latency (`elapsed_ms` from the response)
- total request latency (`performance.now()` in the browser)
- decoded count
- decoded values

Direct version: upload from the page above (Take photo or Choose
existing photo).

WhatsApp version: send the same image through WhatsApp, download the
exact media received by the backend, then scan that file locally.

Expected shape of the comparison:

```
Source       Dimensions    Bytes       Decoded
Direct       4032×3024     4.6 MB      6
WhatsApp     1005×1280     216 KB      1
```

## Deploy to Render (Docker)

The Dockerfile is a multi-stage build: Node stage builds the React
frontend, Python stage runs the backend and serves the built frontend at
`/` via `StaticFiles`. One image, one URL, same-origin — no CORS or
`VITE_API_BASE_URL` config needed in prod.

```bash
# Local Docker test (same as Render)
docker build -t barcode-scanner .
docker run --rm -p 8000:8000 -e D360_API_KEY=dummy barcode-scanner
# Open http://localhost:8000 — both frontend and API are served from here
```

On Render:

1. Create a **Web Service** from this repo (Render detects `render.yaml`
   automatically, or point it to the Dockerfile).
2. Set env vars (Render dashboard or `render.yaml`):
   - `D360_API_KEY=dummy` — required to boot even for scanner-only use
     (config.py raises without it). Set the real key when WhatsApp
     webhooks are needed.
   - `APP_ENV=production`
   - `MAX_UPLOAD_BYTES=15728640`
   - `ALLOWED_IMAGE_TYPES=image/jpeg,image/png,image/webp`
3. Render assigns `$PORT` automatically — the CMD handles it.
4. Health check: `/health`.

The deployed URL serves the upload page at `/` and the API at
`/barcode/scan`. Open the Render URL on your phone — it's HTTPS, no
ngrok needed.

## LangSmith monitoring dashboard

The repository includes an idempotent provisioning script for the first
scanner health dashboard. It uses the LangSmith REST API directly because the
installed Python SDK does not expose custom dashboard helpers.

Required environment variables:

```bash
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT_ID=<tracing-project-uuid>
```

Optional variables:

```bash
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TENANT_ID=<langsmith-tenant-uuid>
```

Run from the repository root:

```bash
source .venv/bin/activate
python scripts/provision_langsmith_dashboard.py --dry-run
python scripts/provision_langsmith_dashboard.py
python scripts/provision_langsmith_dashboard.py --check
```

The dashboard is named `Barcode Scanner Production Health` and currently
contains upload volume by source, outcome distribution, recovery attempts,
user-confirmed correctness, completed analyses, P50 analysis latency, and
recovery labels resolved. The script creates or updates resources by
stable dashboard/chart metadata and does not run as part of application
startup.
