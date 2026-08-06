# AGENTS.md

Build, test, and lint commands for the barcode-scanner project.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m app.cli scan ./samples/multi_clear_6_boxes.jpeg
python -m app.cli audit ./samples/multi_clear_6_boxes.jpeg --time
python -m app.cli pipeline ./samples/multi_clear_6_boxes.jpeg --time --pretty
```

Requires `GEMINI_API_KEY` in `.env` or the environment for `audit` and `pipeline`.

## Architecture

The pipeline is a **clean happy path** — two independent branches run in
parallel on one image, then a single containment match joins them:

1. **Deterministic scanner** (`barcode_scanner.py`) — zxing-cpp + OpenCV
   label fallback. Decodes barcode values with full-resolution pixel bboxes.
2. **Gemini Flash spatial audit** (`gemini_box_audit.py`) — locates every
   visible product label and its barcode region, returns pixel bboxes.

`pipeline.py` runs both in a `ThreadPoolExecutor(max_workers=2)`, then
`spatial_reconciliation.match_scanner_to_labels()` assigns each scanner
detection to the Gemini label whose barcode region contains it.

When labels remain unmatched after reconciliation, a **Gemini-guided
recovery** step crops each missing label's `barcode_bbox` from the
full-resolution image with 20% padding and scans it aggressively
(`scan_crop_with_recovery` — CLAHE, Otsu, adaptive, aggressive sharpen,
invert, plus an explicit 90° rotation attempt). Any newly decoded barcodes
are merged back and reconciliation is re-run. This only runs on the failure
path — the happy path is unaffected.

`analyze.py` reshapes the pipeline summary into the product response
(`complete` / `needs_better_photo` / `retryable_error`).

### Dependency direction

```
gemini_box_audit.py ──→ spatial_geometry.py ←── spatial_reconciliation.py
                              ↑
                         pipeline.py
                              ↑
                         analyze.py
```

- `spatial_geometry.py` — generic coordinate math only (no Gemini/scanner imports).
- `spatial_reconciliation.py` — imports only `spatial_geometry`. Receives
  scanner detections and Gemini labels as plain dicts.
- All Gemini audit functions consume EXIF-normalized RGB JPEG bytes via
  `load_normalized_image()`. If the original exceeds 1600px on either side,
  the Gemini copy is resized (LANCZOS, JPEG quality 85); smaller images are
  left untouched. Gemini's normalized 0..1000 coordinates are
  resolution-independent and convert directly to the original full-resolution
  pixel frame. The scanner keeps full resolution independently.
- Reconciliation uses padded center-in-box containment with global
  nearest-first assignment. Target selection is strict: when `barcode_bbox`
  is present, only it is used (no fallback to the larger `label_bbox`).

## Product API

`app/services/analyze.py` exposes `analyze_image()` — the product hot path.

```python
from app.services.analyze import analyze_image

result = analyze_image(image_bytes_or_path)

if result["outcome"] == "complete":
    for item in result["found"]:
        print(item["barcode_value"], item["label_index"])
elif result["outcome"] == "needs_better_photo":
    for m in result["missing"]:
        print(m["label_index"], m["label_bbox"], m["barcode_bbox"])
else:  # "retryable_error"
    print(result.get("error"))
```

### Response schema

| Field | Type | Description |
|---|---|---|
| `ok` | bool | Function executed (false = invalid input / unhandled error). |
| `outcome` | str | `complete` / `needs_better_photo` / `retryable_error`. |
| `audit_available` | bool | Whether the Gemini audit succeeded. |
| `image_width` / `image_height` | int | Original image dimensions (pixels). |
| `found` | list | Barcodes matched to a Gemini label. Each entry: `label_index`, `barcode_value`, `barcode_format`, `barcode_bbox`, `label_bbox`, `match_basis`. |
| `missing` | list | Gemini labels with no decoded barcode. Each entry: `label_index`, `status`, `label_bbox`, `barcode_bbox`. |
| `unassigned` | list | Scanner detections not matched to any Gemini label. Each entry: `barcode_value`, `barcode_format`, `barcode_bbox`. |
| `summary` | object | `visible_label_count`, `found_count`, `missing_count`, `unassigned_count`, `all_found`. |
| `error` | object | Present on `retryable_error`: `{code, message}`. |
| `annotated_image_b64` | str | Present on `needs_better_photo`: base64 PNG with red circles. |
| `annotated_image_width` / `annotated_image_height` | int | Present on `needs_better_photo`. |
| `message` | str | Present on `needs_better_photo`: human-readable prompt. |

### Outcome decision

- `complete` — valid audit, `visible_label_count > 0`, no missing labels.
- `needs_better_photo` — valid audit, but labels remain missing (or zero
  labels found). Do NOT ask for a better photo when Gemini itself failed.
- `retryable_error` — scan error or Gemini audit failure. The client retries.

## Test

```bash
pytest                              # all tests
pytest tests/test_barcode_scanner.py
pytest tests/test_spatial_geometry.py
pytest tests/test_spatial_reconciliation.py
pytest tests/test_analyze.py
pytest tests/test_cli.py
pytest tests/test_api.py
pytest tests/eval/                  # eval harness scoring logic (offline)
```

Tests mock `zxingcpp.read_barcodes` and `pipeline._traced_audit` — no real
barcode images or Gemini API calls are required.

## Lint

```bash
ruff check .
```

Pre-existing warnings in `app/api/routes.py` (B008),
`app/services/barcode_scanner.py` (B905),
`app/services/gemini_box_audit.py` (UP042),
`app/services/modal_transcriber.py` (E501), and
`app/services/transcribe.py` (UP022) are intentionally left to preserve
existing style consistency.

## Offline evaluation (LangSmith)

```bash
python -m tests.eval.runner                 # live (charged, needs GEMINI_API_KEY)
python -m tests.eval.runner --scanner-only  # scanner-only, no Gemini
```

Runs `analyze_image()` on the ground-truth dataset
(`tests/eval/dataset.json`) and scores each result with LangSmith
`evaluate()`:

- **value_recall** — fraction of expected decoded barcodes found.
- **value_precision** — fraction of found barcodes matching an expected value.
- **outcome_correct** — did the pipeline report the right outcome?
- **count_exact** — found_count == expected decoded count.

A summary evaluator applies soft aggregate thresholds (mean recall >= 0.90,
mean precision >= 0.95, outcome accuracy >= 0.80). Results upload to
LangSmith as an experiment under `LANGSMITH_PROJECT`. Set
`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to enable upload.

`tests/eval/test_eval.py` is deterministic and runs in every CI — it asserts
the dataset loads and the evaluators score correctly with stub predictions,
without calling LangSmith or Gemini.

## Direct upload experiment (web/)

Tiny Vite + React + TS mobile page that uploads the original phone photo
(no canvas, no compression, no base64) to the existing `/barcode/scan`
endpoint and shows dimensions, file size, barcodes, server scan latency,
and total request latency. No WhatsApp, no Gemini, no chat UI, no auth.

### Run locally

Backend (exposes `/health` and `/barcode/scan`):

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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

### Deploy to Render (Docker)

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


