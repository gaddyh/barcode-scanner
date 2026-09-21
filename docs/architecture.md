# Architecture

## Pipeline overview

The pipeline is a **clean happy path** — two independent branches run in
parallel on one image, then a single containment match joins them:

1. **Deterministic scanner** (`src/ingest/scanner.py`) — zxing-cpp + OpenCV
   label fallback. Decodes barcode values with full-resolution pixel bboxes.
2. **Gemini Flash spatial audit** (`src/ingest/vision.py`) — locates every
   visible product label and its barcode region, returns pixel bboxes.

`src/ingest/graph.py` orchestrates the pipeline as a LangGraph `StateGraph`
(M15A). `scan` and `audit` run in parallel (LangGraph superstep fan-out),
then `src.ingest.reconciliation.match_scanner_to_labels()` assigns each
scanner detection to the Gemini label whose barcode region contains it.
`src/ingest/pipeline.py` is a thin traced facade that delegates to
`run_scan_graph()` — the summary dict contract is unchanged.

When labels remain unmatched after reconciliation, a **Gemini-guided
recovery** step crops each missing label's `barcode_bbox` from the
full-resolution image with 20% padding and scans it aggressively
(`scan_crop_with_recovery` — CLAHE, Otsu, adaptive, aggressive sharpen,
invert, plus an explicit 90° rotation attempt). Any newly decoded barcodes
are merged back and reconciliation is re-run via a conditional edge cycle
(`reconcile → recover → reconcile`). This only runs on the failure
path — the happy path is unaffected.

`src/ingest/analyze.py` reshapes the pipeline summary into the product response
(`complete` / `needs_better_photo` / `retryable_error`).

## Dependency direction

```
vision.py ──→ geometry.py ←── reconciliation.py
                    ↑
                graph.py
                    ↑
               pipeline.py (facade)
                    ↑
               analyze.py
```

- `src/ingest/geometry.py` — generic coordinate math only (no Gemini/scanner imports).
- `src/ingest/reconciliation.py` — imports only `src.ingest.geometry`. Receives
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

## Image resizing for Gemini

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

## Gemini-guided recovery

When reconciliation leaves unmatched Gemini labels (a label Gemini sees but
the scanner didn't decode), the pipeline crops each unmatched label's
`barcode_bbox` from the full-resolution image and runs the aggressive
label-crop preprocessing pipeline on it. If the tight barcode crop fails, it
falls back to the wider `label_bbox`. If that also fails, it tries the exact
(unpadded) barcode region at high scales (6x–12x) — this recovers very small
barcodes that only decode at high magnification. Recovered detections are
merged with existing scanner detections and reconciliation re-runs.

The Gemini prompt explicitly describes barcode bars as the striped
black-and-white bar pattern (not the human-readable digits, product name, or
brand text) and notes the expected aspect ratio. Combined with the dual-audit
strategy, this prevents the most common failure mode where Gemini's
`barcode_bbox` points at product text instead of the actual barcode.

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
a nearby barcode belonging to a different label does not count.

Recovery only fires on mismatch — when all labels match initially, the output
is unchanged (no `initial_reconciliation` or `recovery` keys).

## LangGraph orchestration

`src/ingest/graph.py` orchestrates the pipeline as a LangGraph `StateGraph`:
- `scan` and `audit` run in parallel (superstep fan-out).
- `reconciliation` joins them.
- A conditional edge cycles `reconcile → recover → reconcile` on the failure path.
- The happy path is unaffected (no recovery node runs).

`src/ingest/checkpoint.py` provides LangGraph checkpointer persistence
(M15C) using a separate psycopg pool for the checkpoint tables in the same
Postgres instance. Non-fatal: if it fails, the app boots without checkpointing.

## Messaging (optional adapter)

`src/messaging/` contains the WhatsApp/360dialog adapter. It is NOT required
to run the product — the web upload page (`web/`) is the primary demo path.
The messaging code remains as an optional adapter for WhatsApp-based photo
intake but is not part of the canonical product flow.

## Runtime reliability + idempotency

External irreversible writes go through `src/runtime/executor.py` with a
declared `ExecutionPolicy` and optional idempotency store. The executor
normalizes exceptions into a structured taxonomy:

- `RetryableError` — transient failure (timeout, connection refused).
  Retried per policy.
- `PermanentError` — non-retryable failure (invalid input, unique
  constraint violation). Not retried.
- `IndeterminateError` — unknown outcome of an irreversible write
  (timeout/disconnect after the request may have crossed the network
  boundary). Not retried; the idempotency store replays the cached
  indeterminate outcome on retry.
- `InvalidInputError` — caller violated a precondition.

Product-specific policies (timeouts derived from current P95
measurements, NOT copied from echo-v2):

| Policy | Use | `max_attempts` | `timeout_seconds` | `irreversible_write` |
|---|---|---|---|---|
| `SCAN_COMPUTE` | Local deterministic scanner | 1 | 7.0 | false |
| `EXTERNAL_READ` | Gemini audit (retryable) | 3 | 15.0 | false |
| `EXTERNAL_WRITE` | Priority draft order | 1 | 10.0 | true |

### INDETERMINATE classification lives at the integration boundary

The executor does NOT upgrade a generic unexpected exception to
`IndeterminateError` on its own. The adapter (e.g.
`LocalPriorityGateway` / future `RealPriorityGateway`) classifies:
failure before submission → `RetryableError`/`PermanentError`;
timeout/disconnect after submission may have occurred →
`IndeterminateError`. The executor preserves and persists what the
adapter raises.

### Runtime layering

```
application/service
    ↓ runtime.execute(policy=EXTERNAL_WRITE, idempotency_key=...)
PriorityGateway (protocol)
    ↓
LocalPriorityGateway / RealPriorityGateway (adapter performs the op + classifies)
```

The gateway just performs the external operation and classifies errors.
Runtime wrapping stays in the application/service layer, NOT inside the
gateway. Mixing these layers would make the gateway untestable in
isolation and obscure the idempotency boundary.

### Idempotency stores

- `InMemoryIdempotencyStore` — for tests and the vertical slice.
- `PostgresIdempotencyStore` — production. Owner-token fencing prevents
  lost updates from slow writers; lease expiry allows reclaim.

## Priority boundary

ERP access goes ONLY through `PriorityGateway`
(`src/integrations/priority/port.py`). No direct Priority API calls from
routes/UI/workflows. The local fake Priority (`LocalPriorityGateway`)
remains usable for tests/demo.

Domain models (`src/integrations/priority/models.py`): `Customer`,
`Branch`, `OrderLineItem`, `CreateDraftOrderRequest`,
`CreateDraftOrderResult` (frozen dataclasses).

## Receiving flow (end-to-end vertical slice)

`src/domain/receiving.py` is the product-level model of one customer +
branch + set of photos + draft order. Physical boxes are occurrences
(multiset) — duplicate barcode values count separately.

`src/api/receiving.py` wires the flow:

```
POST /receiving/sessions          — create session (customer + branch + action)
POST /receiving/sessions/{id}/images — upload photo, run analyze_image(), append boxes
POST /receiving/sessions/{id}/submit   — freeze, create draft order via runtime
GET  /receiving/sessions/{id}     — inspect session state
```

### Session submission state machine

```
ACTIVE
   ↓ freeze payload
SUBMITTING
   ↓ success
SUBMITTED

SUBMITTING
   ↓ unknown external outcome
SUBMISSION_UNKNOWN
```

- On entering `SUBMITTING`, session contents are FROZEN (immutable).
  Retrying uses the same `priority:draft:{session_id}` key and exactly
  the same frozen payload.
- On success: persist `external_order_id`, transition to `SUBMITTED`.
- On indeterminate: transition to `SUBMISSION_UNKNOWN`. The idempotency
  store replays the indeterminate outcome on retry; the user must NOT
  create another session/order blindly.
- If the call fails BEFORE submission: stay `ACTIVE`, return error, user
  can retry/edit.
- `priority_orders.session_id` has a UNIQUE partial index as
  defense-in-depth — even if the runtime idempotency layer has a bug, the
  fake ERP cannot create two draft orders for one submitted session.

## Evaluation

- `make eval` — deterministic scanner-only regression gate (no Gemini,
  no LangSmith, no Postgres, no network). Exits non-zero on quality
  regression against `tests/eval/baseline_frozen.json`. Gates merges.
- `make eval-live` — scanner + Gemini (observational, NOT a gate).
- `make eval-freeze` — the only way to rewrite `baseline_frozen.json`.
  Normal `make eval` runs must never silently rewrite the baseline.
- Barcode accuracy is occurrence/multiset based (duplicate values =
  separate physical boxes). Use `collections.Counter`, not `set()`.
- Gate on per-image and aggregate occurrence recall + false positives.
  Do NOT gate on latency (workstation/CI latency fluctuates; record
  P50/P95 as informational).
