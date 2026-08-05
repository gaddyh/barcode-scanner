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
detection to the Gemini label whose barcode region contains it. No dual-audit,
no recovery, no crop retries.

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
