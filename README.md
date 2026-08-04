# Barcode Scanner Service

Deterministic barcode-scanning service for product photos. It is intentionally independent of WhatsApp and Priority ERP.

## Flow

```text
product photo -> FastAPI upload -> Pillow EXIF orientation normalization
  -> tiled multi-scale scan -> ZXing-C++ -> position-aware deduplication -> JSON
```

The scanner runs many decode passes per image to maximize recall on difficult
warehouse photos:

1. **Full-image pass** — scan the original image once.
2. **Tiling** — split the image into a 3×2 grid of overlapping tiles (20% overlap)
   and scan each tile independently so small barcodes fill more of the frame.
3. **Preprocessing variants** — for every region (full image + each tile), four
   variants are decoded:
   - Original (or upscaled 2× / 3× via LANCZOS)
   - Grayscale
   - Grayscale + 1.8× contrast
   - Grayscale + contrast + UnsharpMask sharpening
4. **Deduplication** — the same physical barcode is detected many times across
   tiles, scales, and variants. Detections are grouped by `(value, format)` and
   merged when their positions overlap (center-distance or IoU). The detection
   with the largest bounding box is kept. Results are sorted top-to-bottom,
   left-to-right.

ZXing-C++ handles rotation and inverted-image attempts internally via
`try_rotate` / `try_invert`.

## Requirements

- Python 3.11+
- No database
- No cloud vision account

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

## Scan a photo

```bash
curl -X POST http://localhost:8000/barcode/scan \
  -H "accept: application/json" \
  -F "file=@./product.jpg"
```

Example response:

```json
{
  "status": "found",
  "count": 2,
  "image_width": 2000,
  "image_height": 1500,
  "barcodes": [
    {
      "value": "7297501098442",
      "format": "Code 128",
      "content_type": "ContentType.Text",
      "orientation": -90,
      "position": [
        {"x": 619, "y": 878},
        {"x": 619, "y": 424},
        {"x": 559, "y": 415},
        {"x": 559, "y": 876}
      ],
      "bounding_box": {
        "x1": 559,
        "y1": 415,
        "x2": 619,
        "y2": 878
      }
    },
    {
      "value": "900439-42",
      "format": "Code 128",
      "content_type": "ContentType.Text",
      "orientation": 0,
      "position": [
        {"x": 1171, "y": 679},
        {"x": 1465, "y": 679},
        {"x": 1459, "y": 779},
        {"x": 1170, "y": 779}
      ],
      "bounding_box": {
        "x1": 1170,
        "y1": 679,
        "x2": 1465,
        "y2": 779
      }
    }
  ]
}
```

No barcode is still a successful API request:

```json
{
  "status": "not_found",
  "count": 0,
  "image_width": 2000,
  "image_height": 1500,
  "barcodes": []
}
```

## Scan from the command line

The `barcode-scan` console script (installed by `pip install -e ".[dev]"`) has
three subcommands: `scan`, `audit`, and `pipeline`.

### scan — deterministic barcode scan

```bash
barcode-scan scan ./product.jpg
barcode-scan scan --pretty ./a.png ./b.png
barcode-scan scan --time ./product.jpg
```

Each image produces one JSON object in a top-level array. `--time` prints
per-image wall-clock timing to stderr. Unreadable files and invalid images are
reported per-entry with `"status": "error"` and the process exits non-zero if
any image failed.

```json
[
  {
    "path": "./product.jpg",
    "status": "found",
    "count": 2,
    "barcodes": [
      {
        "value": "7297501098442",
        "format": "Code 128",
        "content_type": "ContentType.Text",
        "orientation": -90,
        "position": [
          {"x": 619, "y": 878},
          {"x": 619, "y": 424},
          {"x": 559, "y": 415},
          {"x": 559, "y": 876}
        ],
        "bounding_box": {
          "x1": 559,
          "y1": 415,
          "x2": 619,
          "y2": 878
        }
      }
    ]
  }
]
```

### audit — Gemini visual audit

Runs a Gemini visual audit on shoebox images. By default it returns a fast
counts-only result (~1.5s) using `gemini-3.5-flash-lite`. Use `--full` for the
detailed audit (bounding boxes, OCR text, per-box observations; ~25s).

```bash
barcode-scan audit ./product.jpg --time --pretty
barcode-scan audit ./product.jpg --full --pretty
barcode-scan audit ./product.jpg --model gemini-3.5-flash
```

Requires `GEMINI_API_KEY` in `.env` or the environment. The fast audit returns
four count fields:

```json
[
  {
    "path": "./product.jpg",
    "status": "ok",
    "audit": {
      "visible_product_barcode_label_count": 1,
      "clear_product_barcode_label_count": 1,
      "boxes_without_visible_product_barcode": 3,
      "partially_obscured_product_barcode_count": 0
    }
  }
]
```

### pipeline — scan + audit in parallel

Runs the deterministic scanner and the Gemini audit concurrently, then returns
a combined summary with reconciliation metrics. Wall-clock time is the slower
of the two (~1.5–2s), not their sum.

```bash
barcode-scan pipeline ./product.jpg --time --pretty
barcode-scan pipeline ./a.jpg ./b.jpg ./c.jpg --time
```

Output includes a summary table on stderr and full JSON on stdout:

```text
Image                             Decoded/Visible  Match    Time
----------------------------------------------------------------
marny_brown_42.jpeg                           2/1   DIFF   1.85s
multi_clear_6_boxes.jpeg                      6/6     OK   2.04s
multi_12_clean.jpeg                         11/12   DIFF   1.50s
----------------------------------------------------------------
```

The `decoded_vs_visible` field reconciles scanner output with Gemini's visual
count — `match: true` means the scanner decoded exactly as many barcodes as
Gemini saw labels; a negative `difference` means the scanner missed some.

```json
[
  {
    "path": "samples/multi_clear_6_boxes.jpeg",
    "scan_status": "found",
    "audit_status": "ok",
    "decoded_count": 6,
    "unique_values": ["7297500243416", "7297500243423", "7297500243430", "7297500243447"],
    "unique_value_count": 4,
    "visible_labels": 6,
    "clear_labels": 6,
    "boxes_without_label": 0,
    "partially_obscured": 0,
    "decoded_vs_visible": {"decoded": 6, "visible": 6, "match": true, "difference": 0},
    "ok": true
  }
]
```

You can also run any subcommand without installing the entry point:

```bash
python -m app.cli scan ./product.jpg
python -m app.cli audit ./product.jpg --time
python -m app.cli pipeline ./product.jpg --time --pretty
```

## LangSmith tracing

The `pipeline` subcommand is instrumented with LangSmith tracing. When
`LANGSMITH_TRACING=true` is set in `.env` (along with `LANGSMITH_API_KEY`,
`LANGSMITH_PROJECT`, and `LANGSMITH_ENDPOINT`), each pipeline run is traced as
a nested span tree:

```text
pipeline (chain)
├── barcode_scan (tool)     — deterministic scanner
└── gemini_audit (tool)     — Gemini counts audit
```

Traces are visible at [smith.langchain.com](https://smith.langchain.com) under
the configured project. Tracing is automatically disabled when
`LANGSMITH_TRACING` is not set, so tests and non-tracing runs are unaffected.

## Run tests

```bash
pytest
```

The scanner tests monkeypatch `zxingcpp.read_barcodes` to inject deterministic
results, so they do not depend on a real barcode image. The CLI regression tests
run the actual `barcode-scan` command against the sample images in `samples/`.

## Benchmark

A ground-truth benchmark dataset and runner live in `tests/benchmark/`. They
freeze the deterministic scanner as a stable baseline so that any change which
improves one image while silently breaking another is caught.

### Dataset

`tests/benchmark/dataset.json` holds a rich per-box ground truth for each
sample image. The top-level count field is `expected_barcode_symbol_count`,
because the benchmark measures **barcode-symbol detection**, not physical
boxes. (For `marny_brown_42.jpeg` the value is 2 — two barcode symbols on one
physical box; for the multi-box images it equals the number of primary barcode
symbols.)

Each box is one of:

- `decoded` — the barcode value is known and the scanner is expected to find it.
- `unreadable` — the barcode is visible in the image but cannot be reliably
  decoded from the available pixels. The box records `bounding_box`,
  `location`, `visible_metadata`, and `reason` for future analysis.

### Run the benchmark

```bash
make bench
# or
python -m tests.benchmark.runner
```

The runner scans each image 10 times in a single process, matches scanner
detections to expected boxes, and prints a report table. It exits 0 if the
baseline passes and 1 on regression.

Matching uses center-in-box with a minimum absolute padding of 25px (so
line-like ZXing boxes get real tolerance) and global distance-based assignment:
all valid `(expected, detection)` candidate pairs are sorted by center distance
and assigned nearest-first, making matching deterministic and order-independent.

### Metrics

| Metric | Definition |
|---|---|
| **Symbols** | `expected_barcode_symbol_count` for the image |
| **Decoded** | Number of expected `decoded` boxes |
| **Found** | Total scanner detections returned (`len(detections)`), distinct from matched count |
| **Exact** | Expected-decoded boxes matched with the correct value (`exact/expected_decoded`) |
| **UniqueCases** | Unique expected-decoded values found in scanner output (`found/expected_unique_values`). The aggregate is the **sum of unique expected values per image**, not a global set union. |
| **FP** | False positives — scanner detections not matched to any expected box, plus mismatches |
| **Bonus** | Expected-`unreadable` boxes where the scanner found a value. Printed but not counted as a false positive and not a success criterion. |
| **Mismatch** | Matched pairs where the expected value differs from the scanner value. Each mismatch counts as both a miss and a false positive. |
| **Median / P95** | Wall-clock latency per scan across warm runs (runs 2..N). P95 uses nearest-rank. |
| **First** | First-run latency — scanner construction + first scan in the current process. Not a true process cold start (imports and native-library loading happen before the runner executes). |

### Pass criteria

The runner exits 0 only when:

- `exact_matches == expected_decoded`
- `false_positives == 0`
- `mismatches == 0`
- `unique_values_found == expected_unique_values`

Unreadable boxes do **not** block success. Bonus detections are reported but
cause neither failure nor a false positive.

### Frozen baseline

| Metric | Value |
|---|---|
| Expected barcode symbols | 20 |
| Expected decoded symbols | 19 |
| Exact decoded occurrences | 19/19 |
| UniqueCases | 14/14 |
| False positives | 0 |
| Mismatches | 0 |
| Bonuses | 0 |

`multi_16_fuzzy.jpeg` is excluded until its per-label ground truth is manually
annotated. The dataset format supports adding it later.

### Regression test

```bash
pytest tests/benchmark/test_baseline.py
```

Asserts the frozen baseline (19/19 exact, 14/14 unique cases, 0 false
positives, 0 mismatches). Latency is not asserted — it is reported for human
monitoring only.

## Docker

```bash
docker compose up --build
```

## Project layout

```text
app/
  api/routes.py                 HTTP validation and endpoint
  cli.py                        CLI with scan / audit / pipeline subcommands
  core/config.py                Environment configuration
  models/barcode.py             API response contract (ScanResponse)
  services/
    barcode_scanner.py          BarcodeScanner — tiling, preprocessing, dedup
    gemini_box_audit.py         Gemini visual audit (fast counts + full audit)
  main.py                       FastAPI application
samples/
  marny_brown_42.jpeg           Sample photo with 2 barcodes
  multi_clear_6_boxes.jpeg      Sample photo with 6 barcodes
  multi_12_clean.jpeg           Sample photo with 12 barcodes
  multi_16_fuzzy.jpeg           Difficult photo (16 boxes, fuzzy barcodes)
tests/
  benchmark/                    Ground-truth dataset + benchmark runner
  test_api.py                   API endpoint tests
  test_barcode_scanner.py       Scanner logic tests (monkeypatched zxing)
  test_cli.py                   CLI unit tests + regression tests on samples
```

## Current boundary

This version does not:

- call WhatsApp
- call Priority ERP
- run object detection to find box regions (tiling is a fixed grid, not adaptive)
- validate barcode check digits
- expose per-detection confidence scores

The Gemini visual audit is advisory only — it does not decode barcodes or
replace the deterministic scanner. The pipeline reconciles the two but does not
yet auto-retry failed decodes with targeted crops guided by Gemini bounding boxes.

The next step is to test it against real warehouse photos and record exact-match
success and failure reasons.
