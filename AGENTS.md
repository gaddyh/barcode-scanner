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
