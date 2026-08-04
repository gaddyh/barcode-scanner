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

Runs a Gemini visual audit on shoebox images. Three modes are available:

- **Counts (default, fast ~1.5s):** `gemini-3.5-flash-lite` returns four count
  fields only. Good for quick diagnostics.
- **Labels (`--labels`, spatial):** `gemini-3.5-flash` locates every product
  label and its barcode region in pixel coordinates. The middle mode — faster
  than `--full`, more useful than counts because it returns spatial regions.
- **Full (`--full`, detailed ~25s):** bounding boxes, OCR text, per-box
  observations, and quality analysis.

```bash
barcode-scan audit ./product.jpg --time --pretty
barcode-scan audit ./product.jpg --labels --pretty
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

The `--labels` audit returns pixel-space label and barcode bounding boxes:

```json
[
  {
    "path": "./product.jpg",
    "status": "ok",
    "audit": {
      "image_width": 4032,
      "image_height": 3024,
      "labels": [
        {
          "label_index": 1,
          "label_bbox": {"x1": 412, "y1": 301, "x2": 1044, "y2": 698},
          "barcode_bbox": {"x1": 785, "y1": 391, "x2": 1010, "y2": 652},
          "status": "clear",
          "confidence": "high"
        }
      ]
    }
  }
]
```

All Gemini modes consume EXIF-normalized RGB JPEG bytes so their pixel
coordinate system matches the deterministic scanner.

### pipeline — scan + spatial audit in parallel

Runs the deterministic scanner and the Gemini spatial label audit concurrently,
then matches scanner detections to Gemini product labels geometrically.
Wall-clock time is the slower of the two.

```
                         ┌─ ZXing scanner
input image ─ normalize ─┤     → decoded values + pixel bounding boxes
                         │
                         └─ Gemini spatial audit
                               → product-label boxes + barcode boxes

scanner detection centers  +  Gemini barcode/label regions
        ↓
spatial reconciliation
        ↓
matched labels · unmatched labels · unassigned scanner detections
```

```bash
barcode-scan pipeline ./product.jpg --time --pretty
barcode-scan pipeline ./a.jpg ./b.jpg ./c.jpg --time
```

Output includes a summary table on stderr and full JSON on stdout:

```text
Image                             Matched/Visible  Match    Time
------------------------------------------------------------------
marny_brown_42.jpeg                           1/2   DIFF   3.85s
multi_clear_6_boxes.jpeg                      6/6     OK   4.04s
multi_12_clean.jpeg                         11/12   DIFF   3.50s
------------------------------------------------------------------
```

The pipeline preserves full scanner detections (with bounding boxes), Gemini
labels, and a `reconciliation` object with matches, unmatched labels, and
unassigned scanner detections:

```json
[
  {
    "path": "samples/multi_12_clean.jpeg",
    "scan_status": "found",
    "audit_status": "ok",
    "decoded_count": 11,
    "unique_values": ["..."],
    "unique_value_count": 8,
    "scanner_detections": [
      {
        "value": "7297501154117",
        "format": "Code128",
        "bounding_box": {"x1": 100, "y1": 200, "x2": 300, "y2": 260}
      }
    ],
    "visible_labels": 12,
    "clear_labels": 12,
    "gemini_labels": [
      {
        "label_index": 1,
        "label_bbox": {"x1": 412, "y1": 301, "x2": 1044, "y2": 698},
        "barcode_bbox": {"x1": 785, "y1": 391, "x2": 1010, "y2": 652},
        "status": "clear",
        "confidence": "high"
      }
    ],
    "reconciliation": {
      "matches": [
        {
          "label_index": 1,
          "scanner_detection_index": 4,
          "barcode_value": "7297501154117",
          "match_basis": "barcode_bbox",
          "center_distance": 0.08
        }
      ],
      "unmatched_labels": [
        {
          "label_index": 7,
          "label_bbox": {"x1": 2140, "y1": 1210, "x2": 2750, "y2": 1660},
          "barcode_bbox": {"x1": 2420, "y1": 1320, "x2": 2690, "y2": 1530},
          "status": "clear"
        }
      ],
      "unassigned_scanner_detections": [],
      "matched_label_count": 11,
      "visible_label_count": 12,
      "all_labels_matched": false
    },
    "decoded_vs_visible": {
      "decoded": 11,
      "visible": 12,
      "match": false,
      "difference": -1,
      "matched_labels": 11,
      "all_labels_matched": false
    },
    "ok": true
  }
]
```

`all_labels_matched` means only "every Gemini product label has at least one
scanner match." It does **not** mean every barcode was decoded, every value is
correct, or the order is ready for Priority. `unassigned_scanner_detections`
are detections not assigned to any Gemini label — not necessarily false
positives, since one physical label can legitimately carry multiple barcodes.

When the Gemini audit fails, the scanner result is still returned with
`audit_status: "error"` and no `reconciliation` block — there is no silent
counts fallback, because counts alone cannot identify which label was missed.

### Gemini-guided recovery

When reconciliation leaves unmatched Gemini labels (a label Gemini sees but
the scanner didn't decode), the pipeline crops each unmatched label's
`barcode_bbox` from the full-resolution image and runs the aggressive
label-crop preprocessing pipeline on it. If the tight barcode crop fails, it
falls back to the wider `label_bbox`. If that also fails, it tries the exact
(unpadded) barcode region at high scales (6x–12x) — this recovers very small
barcodes that only decode at high magnification. Recovered detections are
merged with existing scanner detections and reconciliation re-runs.

```
scan everything once (parallel with Gemini audit)
        ↓
reconcile scanner detections ↔ Gemini labels
        ↓
if unmatched_labels:
    crop barcode_bbox (25% pad) → aggressive decode (2x–3x, 8 variants)
        ↓ (if nothing found)
    crop label_bbox (10% pad) → aggressive decode
        ↓ (if nothing found)
    crop exact barcode_bbox (no pad) → high-scale decode (6x–12x)
        ↓
    merge recovered + existing → re-reconcile
        ↓
output: initial_reconciliation + recovery + final reconciliation
```

Use `--recovery-debug DIR` to save each crop and preprocessing variant as PNG
for visual debugging:

```bash
barcode-scan pipeline ./multi_12_clean.jpeg --recovery-debug /tmp/recovery_debug
```

A label is only counted as recovered when the final reconciliation assigns a
recovered detection to that attempted label — a crop that accidentally finds
a nearby barcode belonging to a different label does not count. The output
preserves both `initial_reconciliation` and final `reconciliation` plus a
`recovery` section with provenance:

```json
{
  "initial_reconciliation": {
    "matched_label_count": 11,
    "visible_label_count": 12,
    "all_labels_matched": false
  },
  "recovery": {
    "attempted_label_count": 1,
    "attempted_label_indexes": [10],
    "recovered_labels": [
      {
        "label_index": 10,
        "barcode_value": "7297501153998",
        "crop_basis": "barcode_bbox",
        "crop_box": {"x1": 358, "y1": 1450, "x2": 410, "y2": 1632}
      }
    ],
    "recovered_label_count": 1,
    "recovered_detection_count": 1,
    "still_unmatched_labels": []
  },
  "reconciliation": {
    "matched_label_count": 12,
    "visible_label_count": 12,
    "all_labels_matched": true
  }
}
```

Recovery only fires on mismatch — when all labels match initially, the output
is unchanged (no `initial_reconciliation` or `recovery` keys).

### Image resizing for Gemini

The scanner needs full resolution to decode thin barcode lines, but Gemini only
needs enough resolution to locate product labels. If the original image exceeds
1600px on either side, the Gemini copy is resized (aspect ratio preserved,
LANCZOS resampling, JPEG quality 85) before upload. Images already smaller than
1600px are left untouched. Gemini's normalized 0..1000 coordinates are
resolution-independent, so they convert directly to the original
full-resolution pixel frame — no intermediate resized-pixel step:

```
original image (e.g. 4032×3024)
    ├─ scanner              — full resolution, decodes barcodes
    └─ Gemini copy          — resized to 1600×1200 (if needed), locates labels
            ↓
        normalized 0..1000 boxes
            ↓
        round(normalized × original_dimension / 1000) → original-image pixels
```

This reduces encoding time, request size, network upload time, and Gemini
image-processing work on large phone photos without affecting scanner recall
or the downstream coordinate system.

You can also run any subcommand without installing the entry point:

```bash
python -m app.cli scan ./product.jpg
python -m app.cli audit ./product.jpg --time
python -m app.cli audit ./product.jpg --labels --time
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
└── gemini_audit (tool)     — Gemini spatial label audit
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

`fuzzy_16_labels.jpeg` is excluded until its per-label ground truth is manually
annotated. The dataset format supports adding it later.

### Regression test

```bash
pytest tests/benchmark/test_baseline.py
```

Asserts the frozen baseline (19/19 exact, 14/14 unique cases, 0 false
positives, 0 mismatches). Latency is not asserted — it is reported for human
monitoring only.

## Spatial benchmark (Gemini)

A second benchmark (`tests/benchmark_spatial/`) evaluates the Gemini spatial
pipeline (label detection + reconciliation) separately from the deterministic
scanner benchmark. It uses 9 images from `samples/` and tracks image-level
metrics (visible label count, unmatched label count, extra labels) plus
per-label spatial metrics (center distance, IoU, center-inside-ground-truth)
once ground-truth boxes are frozen.

### Live runner

```bash
make bench-spatial
# or
python -m tests.benchmark_spatial.runner --runs 5
```

Requires `GEMINI_API_KEY`. Runs each image N times and reports median latency,
label-count accuracy, unmatched-label accuracy, and extra labels. Exits 0 if
all active image-level expectations pass.

### Snapshot capture and regression

```bash
python -m tests.benchmark_spatial.runner --capture-snapshots   # run once
pytest tests/benchmark_spatial/test_regression.py
```

`--capture-snapshots` saves the first run's Gemini response to
`tests/benchmark_spatial/snapshots/gemini_responses.json`. Commit that file so
the snapshot regression test can replay it offline (no API calls, no
`GEMINI_API_KEY` needed). The test asserts dataset-derived totals and the
frozen baseline (9/9 label-count correct, 0 extra labels).

### Live test (charged, gated)

```bash
RUN_LIVE_GEMINI=1 pytest -m live_gemini
```

Marked `live_gemini` and gated by `RUN_LIVE_GEMINI=1` so plain `pytest` never
makes charged API calls.

### Annotation workflow (per-label ground truth)

Per-label ground-truth boxes are not frozen yet (`labels: []` in
`dataset.json`). To freeze them:

```bash
# 1. Generate a draft annotation from Gemini
python -m tests.benchmark_spatial.annotate draft multi_12_clean.jpeg

# 2. Edit the JSON by hand: move/add/delete boxes
#    tests/benchmark_spatial/annotations/multi_12_clean.jpeg.json

# 3. Re-render the preview PNG to verify your edits
python -m tests.benchmark_spatial.annotate review multi_12_clean.jpeg

# 4. Approve when satisfied
python -m tests.benchmark_spatial.annotate review multi_12_clean.jpeg --approve

# 5. Freeze approved labels into dataset.json
python -m tests.benchmark_spatial.annotate freeze multi_12_clean.jpeg
```

`freeze` refuses unreviewed annotations and validates that both image
dimensions and the `coordinate_space` string match the source image and
dataset. Hard spatial assertions (spatial recall, barcode localization, exact
rectangles) become active only after annotations are frozen.

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
    barcode_scanner.py          BarcodeScanner — tiling, preprocessing, dedup,
                                Gemini-guided crop recovery (scan_label_crops)
    gemini_box_audit.py         Gemini visual audit (counts + spatial labels + full)
    spatial_geometry.py         Pixel bounding boxes, normalized→pixel conversion
    spatial_reconciliation.py   Match scanner detections to Gemini product labels;
                                RecoveryResult / RecoveredLabel models
  main.py                       FastAPI application
samples/
  marny_brown_42.jpeg           Sample photo with 2 barcodes
  multi_clear_6_boxes.jpeg      Sample photo with 6 barcodes
  multi_12_clean.jpeg           Sample photo with 12 barcodes
  stacked_6_labels.jpeg         Stacked photo with 6 labels
  topdown_12_labels_a.jpeg      Top-down photo with 12 labels
  topdown_12_labels_b.jpeg      Top-down photo with 12 labels
  vegan_12_labels_a.jpeg        Vegan product photo with 12 labels
  vegan_12_labels_b.jpeg        Vegan product photo with 12 labels
  fuzzy_16_labels.jpeg          Difficult photo (16 boxes, fuzzy barcodes)
tests/
  benchmark/                    Ground-truth dataset + benchmark runner
  benchmark_spatial/            Gemini spatial pipeline benchmark
    dataset.json                9 image-level expectations + per-label ground truth
    models.py                   Pydantic dataset + metrics models
    metrics.py                  Pure metric functions + aggregation
    runner.py                   Live benchmark + snapshot replay CLI
    annotate.py                 Draft/review/freeze annotation workflow
    snapshots/                  Captured Gemini responses for offline regression
    annotations/                Per-image ground-truth boxes (manual review)
    preview/                    Rendered PNG previews (gitignored)
    test_metrics.py             Deterministic metric tests
    test_regression.py          Snapshot baseline + gated live test
  test_api.py                   API endpoint tests
  test_barcode_scanner.py       Scanner logic tests (monkeypatched zxing)
  test_cli.py                   CLI unit tests + regression tests on samples
  test_gemini_box_audit.py      Gemini schema, EXIF normalization, pixel conversion
  test_spatial_geometry.py      Pure coordinate mathematics
  test_spatial_reconciliation.py  Scanner↔label matching rules
```

## Current boundary

This version does not:

- call WhatsApp
- call Priority ERP
- run object detection to find box regions (tiling is a fixed grid, not adaptive)
- validate barcode check digits
- expose per-detection confidence scores

The Gemini visual audit is advisory only — it does not decode barcodes or
replace the deterministic scanner. The pipeline reconciles the two spatially
(matched labels, unmatched labels, unassigned scanner detections) but does not
yet auto-retry failed decodes with targeted crops guided by Gemini bounding
boxes.

The next step is to test it against real warehouse photos and record exact-match
success and failure reasons.
