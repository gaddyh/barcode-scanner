# Evaluation

## Deterministic vs live evaluation

Two evaluation modes exist with different purposes:

- **`make eval`** — deterministic scanner-only regression. No Gemini, no
  LangSmith, no Postgres, no network. Gates merges. Reproducible.
- **`make eval-live`** — scanner + Gemini full-pipeline evaluation.
  Requires `GEMINI_API_KEY`. Observational, NOT a merge gate (Gemini is
  nondeterministic).

## Product API

`src/ingest/analyze.py` exposes `analyze_image()` — the product hot path.

```python
from src.ingest.analyze import analyze_image

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

## LangSmith tracing

The `pipeline` subcommand is instrumented with LangSmith tracing. When
`LANGSMITH_TRACING=true` is set in `.env` (along with `LANGSMITH_API_KEY`,
`LANGSMITH_PROJECT`, and `LANGSMITH_ENDPOINT`), each pipeline run is traced as
a nested span tree:

```text
pipeline (chain)
├── barcode_scan (tool)     — deterministic scanner
├── gemini_audit (tool)     — Gemini spatial label audit (1st)
└── gemini_audit (tool)     — Gemini spatial label audit (2nd, dual)
```

Traces are visible at [smith.langchain.com](https://smith.langchain.com) under
the configured project. Tracing is automatically disabled when
`LANGSMITH_TRACING` is not set, so tests and non-tracing runs are unaffected.

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
