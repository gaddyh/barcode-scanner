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
python -m app.cli audit ./samples/multi_clear_6_boxes.jpeg --labels --time
python -m app.cli pipeline ./samples/multi_clear_6_boxes.jpeg --time --pretty
```

Requires `GEMINI_API_KEY` in `.env` or the environment for `audit` and `pipeline`.

## Test

```bash
pytest                          # all tests
pytest tests/test_spatial_geometry.py
pytest tests/test_spatial_reconciliation.py
pytest tests/test_gemini_box_audit.py
pytest tests/test_cli.py
pytest tests/test_analyze.py
```

Tests mock `zxingcpp.read_barcodes` and `genai.Client` — no real barcode
images or Gemini API calls are required. CLI regression tests run against
`samples/` and are skipped if that directory is absent.

## Lint

```bash
ruff check .
```

Pre-existing warnings in `app/api/routes.py`, `app/services/barcode_scanner.py`,
and `app/services/gemini_box_audit.py` (B008, B905, UP042, UP037, I001) are
intentionally left to preserve existing style consistency.

## Benchmark

```bash
make bench
# or
python -m tests.benchmark.runner
```

Exits 0 if the frozen scanner baseline passes (19/19 exact, 14/14 unique
cases, 0 false positives, 0 mismatches). The benchmark exercises the
deterministic scanner only — it does not call Gemini.

## Spatial benchmark

A second benchmark (`tests/benchmark_spatial/`) evaluates the Gemini spatial
pipeline (label detection + reconciliation) separately from the scanner
benchmark. Images are referenced from `samples/` via a `source` field in
`dataset.json` — no images are duplicated.

### Live runner (charged, needs `GEMINI_API_KEY`)

```bash
make bench-spatial
# or
python -m tests.benchmark_spatial.runner --runs 5
python -m tests.benchmark_spatial.runner --capture-snapshots   # run once
```

`--capture-snapshots` saves the first run's Gemini response to
`tests/benchmark_spatial/snapshots/gemini_responses.json`. Commit that file so
the snapshot regression test can replay it offline.

### Tests

```bash
pytest tests/benchmark_spatial/                 # metric tests + snapshot regression
pytest tests/benchmark_spatial/test_metrics.py  # deterministic metric tests only
RUN_LIVE_GEMINI=1 pytest -m live_gemini         # explicit live Gemini run
```

- `test_metrics.py` is deterministic and runs on every commit.
- `test_regression.py::test_spatial_snapshot_baseline` skips cleanly until
  `snapshots/gemini_responses.json` exists; then it replays the snapshots
  through the real scanner + metrics and asserts dataset-derived totals.
- `test_regression.py::test_live_spatial_benchmark` is marked `live_gemini` and
  gated by `RUN_LIVE_GEMINI=1`, so plain `pytest` never makes charged API calls.

### Annotation workflow (per-label ground truth)

Per-label ground-truth boxes are NOT frozen yet (`labels: []` in `dataset.json`).
To freeze them:

```bash
python -m tests.benchmark_spatial.annotate draft <image>          # Gemini proposes boxes
# edit tests/benchmark_spatial/annotations/<image>.json by hand
python -m tests.benchmark_spatial.annotate review <image>         # re-render preview
python -m tests.benchmark_spatial.annotate review <image> --approve
python -m tests.benchmark_spatial.annotate freeze <image>         # copy into dataset.json
```

`freeze` refuses unreviewed annotations and validates that both image
dimensions and the `coordinate_space` string match the source image and
dataset. Hard spatial assertions (spatial recall, barcode localization, exact
rectangles) are deferred to a follow-up PR after annotations are frozen.

## Architecture notes

- `spatial_geometry.py` contains generic coordinate math only — no Gemini or
  scanner imports. Dependency direction:
  `gemini_box_audit → spatial_geometry ← spatial_reconciliation`.
- All Gemini audit functions consume EXIF-normalized RGB JPEG bytes via
  `load_normalized_image()`. If the original exceeds 1600px on either side,
  the Gemini copy is resized (LANCZOS, JPEG quality 85); smaller images are
  left untouched. Gemini's normalized 0..1000 coordinates are
  resolution-independent and convert directly to the original full-resolution
  pixel frame — no intermediate resized-pixel step. The scanner keeps full
  resolution independently.
- The pipeline calls `audit_shoebox_labels()` (spatial) — not the fast counts
  audit. Counts are derived from the labels array. On Gemini failure, the
  scanner result is still returned with no silent counts fallback.
- Reconciliation uses padded center-in-box containment with global
  nearest-first assignment. Target selection is strict: when `barcode_bbox`
  is present, only it is used (no fallback to the larger `label_bbox`).
- **Gemini-guided recovery**: When reconciliation leaves unmatched Gemini
  labels, the pipeline crops each unmatched label's `barcode_bbox` (25% pad)
  from the full-resolution image and runs the aggressive label-crop
  preprocessing (`_decode_crop_variants`) on it. If the tight barcode crop
  fails, it falls back to the wider `label_bbox` (10% pad). If that also fails,
  it tries the exact (unpadded) barcode region at high scales (6x, 8x, 10x,
  12x) via `_decode_crop_high_scale` — this recovers very small barcodes that
  only decode at high magnification and where any surrounding padding kills
  detection. Recovered detections are merged with existing scanner detections
  and reconciliation re-runs. A label is only counted as recovered when the
  final reconciliation assigns a recovered detection to that attempted label.
  The output preserves both `initial_reconciliation` and final
  `reconciliation` plus a `recovery` section with provenance (`crop_basis`,
  `label_index`). Recovery only fires on mismatch — when all labels match, the
  output is unchanged (backward compatible). The shared decoding logic lives
  in `BarcodeScanner._decode_crop_variants()`, used by both the OpenCV
  label-candidate fallback and the Gemini-guided recovery path. The
  `--recovery-debug DIR` CLI flag saves each crop and preprocessing variant as
  PNG for visual debugging.
- **Pipeline service layer**: `app/services/pipeline.py` contains the shared
  orchestration (`pipeline_path`, `select_best_spatial_audit`,
  `reconcile_with_recovery`, `scan_path`) used by `app/cli.py` (CLI
  presentation), `app/services/analyze.py` (product API), and the warehouse
  benchmark runner. It contains no argument parsing or CLI presentation code.

## Product API

`app/services/analyze.py` exposes `analyze_image()` — the product hot path.
A user uploads an image and the function returns a clean JSON dict with an
explicit `outcome` and three arrays.

```python
from app.services.analyze import analyze_image

result = analyze_image(image_bytes_or_path)

if result["outcome"] == "complete":
    # every label has a decoded barcode — create the draft order
    for item in result["found"]:
        print(item["barcode_value"], item["label_index"])
elif result["outcome"] == "needs_better_photo":
    # show the missing regions to the user
    for m in result["missing"]:
        print(m["label_index"], m["label_bbox"], m["barcode_bbox"])
else:  # "retryable_error"
    # retry the service (Gemini failure, not the user's fault)
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
| `missing` | list | Gemini labels with no decoded barcode. Each entry: `label_index`, `status`, `label_bbox`, `barcode_bbox` (pixel coords for re-photographing). |
| `unassigned` | list | Scanner detections not matched to any Gemini label. Each entry: `barcode_value`, `barcode_format`, `barcode_bbox`. |
| `summary` | object | `visible_label_count`, `found_count`, `missing_count`, `unassigned_count`, `all_found`. |
| `error` | object | Present on `retryable_error`: `{code, message}`. |

### Outcome decision

- `complete` — valid audit, `visible_label_count > 0`, no missing labels.
- `needs_better_photo` — valid audit, but labels remain missing (or zero
  labels found). Do NOT ask for a better photo when Gemini itself failed.
- `retryable_error` — scan error or Gemini audit failure (timeout, 429,
  etc.). The client should retry the service.

### Test

```bash
pytest tests/test_analyze.py
```

Tests mock `BarcodeScanner` and `pipeline._traced_audit` — no real barcode
images or Gemini API calls are required.

## Warehouse benchmark

A third benchmark (`tests/benchmark_warehouse/`) evaluates the **full
end-to-end production pipeline** (scanner + dual Gemini audit + audit selection
+ reconciliation + Gemini-guided recovery) against all 9 sample images. It
measures baseline recall, recovery uplift, false positives, across-run
consistency, latency distribution, and exact failure reasons.

### Live runner (charged, needs `GEMINI_API_KEY`)

```bash
make bench-warehouse
# or
python -m tests.benchmark_warehouse.runner --runs 5
python -m tests.benchmark_warehouse.runner --capture-snapshots --runs 1
```

`--capture-snapshots` saves both Gemini audit candidates per image to
`tests/benchmark_warehouse/snapshots/pipeline_responses.json`. Commit that file
so the snapshot regression test can replay it offline.

### Snapshot replay (deterministic, offline)

```bash
python -m tests.benchmark_warehouse.runner --snapshot
```

Replays the frozen audit candidates through the **current** scanner +
`select_best_spatial_audit()` + `reconcile_with_recovery()`. This exercises
scanner, dual-audit selection, reconciliation, and recovery code changes
against frozen Gemini inputs — the most useful regression test.

The snapshot stores **both** audit candidates per image so the real
dual-audit selection algorithm is exercised during replay. Stored scanner
detections are diagnostic only and are not used during replay.

### Tests

```bash
pytest tests/benchmark_warehouse/test_snapshot.py   # offline, soft thresholds
RUN_LIVE_GEMINI=1 pytest -m live_gemini tests/benchmark_warehouse/test_live.py
```

- `test_snapshot.py` runs in normal CI with soft aggregate thresholds (label
  count accuracy ≥ 85%, baseline recall ≥ 95%, final recall ≥ 90%, recovery
  uplift within ±2, FP violations ≤ 1, reason mismatches ≤ 2). Skips cleanly
  when snapshots are absent.
- `test_live.py` is marked `live_gemini` and gated by `RUN_LIVE_GEMINI=1`.
  It asserts exact first-run expectations + p95 latency. Across-run
  consistency is recorded and printed but **not** asserted in the first PR —
  once variance is measured and proves stable, promote the commented-out
  `assert result.all_runs_consistent` to a hard gate.

### False-positive accounting

Raw counts are kept separate: `unassigned_scanner_detection_count` (scanner
detections not matched to any Gemini label) and `extra_gemini_label_count`
(Gemini returned more labels than expected). A derived
`false_positive_violation_count` respects per-image
`allow_extra_scanner_detections` — e.g. Marny has 1 unassigned detection
(legitimate: one label with two barcodes) but 0 violations.

### Snapshot staleness

When Gemini model behavior shifts, re-capture with
`--capture-snapshots --runs 1`. The dataset's `expected_*` fields may need
updating in lockstep — they are frozen from live discovery runs, not from
snapshot replay.

