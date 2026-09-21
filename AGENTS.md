# AGENTS.md

Rules for safely changing the barcode-scanner repository without breaking its
important properties. Operational details live in `README.md` and `docs/`.

## Repository status

`barcode-scanner` is the **canonical product repo** for Visual Receiving /
Order Intake. `naot-poc` is archived as reference after baseline. `echo-v2`
donates runtime reliability patterns but remains independent. No fourth repo.

**Baseline status:** PRs #0–#6 complete. The repo supports the full
vertical slice — real shoebox photos → multi-image scan → count duplicate
physical boxes correctly → review → choose customer + branch → create one
local Priority-compatible draft → safely retry without duplicate creation.
Coverage gate at 90% (source-only, `--cov=src`), mypy gating, ruff clean, `make eval` deterministic
gate in CI. Tagged `v0.2.0-baseline`.

## Git workflow

- Never work directly on `main`.
- Feature branch names must be unique and meaningful (e.g. `feature/eval-freeze`,
  `fix/scanner-fp`).
- Start from up-to-date main:
  `git switch main && git pull --ff-only origin main && git switch -c feature/<short-name>`.
- Push the feature branch, open a PR targeting `main`.
- Enable GitHub auto-merge with squash: `gh pr merge --auto --squash <PR_NUMBER>`.
- `main` requires CI checks; merge happens automatically after they pass.
- Delete the feature branch after merge: `git switch main && git pull --ff-only origin main && git branch -d <branch-name>`.

## Verification before merge

- `pytest --cov=src --cov-fail-under=90` — all tests, no live Gemini
  (tests mock `zxingcpp.read_barcodes` and `graph._traced_audit`).
  Postgres service in CI runs DB tests and runtime idempotency tests.
- `mypy` — gating (0 errors). Was non-gating in PR #0 via
  `continue-on-error`; promoted to gating once all errors were fixed.
- `ruff check .` — genuinely green (per-file ignores encoded in `pyproject.toml`).
- `make eval` (deterministic scanner-only, gates merges).
  `make eval-live` (scanner + Gemini) is observational, NOT a gate.
- `make eval-freeze` is the only way to rewrite `baseline_frozen.json`.
  Normal `make eval` runs must never silently rewrite the baseline.
- **Hard rule:** Scanner/recovery changes must not merge unless the frozen
  deterministic eval baseline passes.

## Architecture invariants

- `barcode-scanner` is the canonical product repo.
- Deterministic scanning remains independent of Gemini (two parallel branches,
  joined by reconciliation). Scanner-only operation must remain usable without
  Gemini or Priority.
- Physical boxes are occurrences (multiset), not unique barcode values. Duplicate
  barcode values represent separate physical boxes and count separately.
- The ingest pipeline must remain usable independently of Priority.
- ERP access goes ONLY through `PriorityGateway` (after PR #4). No direct
  Priority API calls from routes/UI/workflows.
- The local fake Priority remains usable for tests/demo.
- External irreversible writes must go through the runtime executor + idempotency
  (after PR #3).
- Do not maintain two production scanner implementations. Candidate algorithms
  can exist only temporarily for A/B evaluation (PR #2).

## Runtime safety rules

Product-specific policies with timeouts derived from current P95 measurements
(NOT copied from echo-v2):

- `SCAN_COMPUTE` — local deterministic scanner. No retry (deterministic; retrying
  the same image is pointless). Timeout = scanner P95 × 3.
- `EXTERNAL_READ` — Gemini audit (retryable). `max_attempts=3`, bounded timeout.
- `EXTERNAL_WRITE` — Priority draft order (irreversible). `max_attempts=1`,
  `irreversible_write=True`, idempotency required.

Unknown write outcome → `INDETERMINATE` (decided at the integration boundary by
the adapter, not blindly by the executor), never blind retry.

### INDETERMINATE classification

- The executor does NOT upgrade a generic unexpected exception to
  `IndeterminateError` on its own.
- The adapter (e.g. `LocalPriorityGateway` / future `RealPriorityGateway`)
  classifies: failure before submission → `RetryableError`/`PermanentError`;
  timeout/disconnect after submission may have occurred → `IndeterminateError`.
- The executor preserves and persists what the adapter raises.
- **One exception:** an executor-enforced `asyncio.wait_for()` timeout around an
  irreversible write is conservatively `IndeterminateError` — the executor cannot
  prove the request did not cross the network boundary.
- Better: let the Priority adapter own its HTTP timeout and classify pre-submit
  vs post-submit failures; the executor timeout is a larger emergency bound.

### Runtime layering

```
application/service
    ↓ runtime.execute(policy=EXTERNAL_WRITE, idempotency_key=...)
PriorityGateway (protocol)
    ↓
LocalPriorityGateway / RealPriorityGateway (adapter performs the op + classifies)
```

The gateway just performs the external operation and classifies errors. Runtime
wrapping stays in the application/service layer, NOT inside the gateway.

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

- On entering `SUBMITTING`, session contents are FROZEN (immutable). Retrying
  uses the same `priority:draft:{session_id}` key and exactly the same frozen
  payload.
- On success: persist `external_order_id`, transition to `SUBMITTED`.
- On indeterminate: transition to `SUBMISSION_UNKNOWN`. Do NOT let the user
  create another session/order blindly — the idempotency store replays the
  indeterminate outcome on retry.
- If the call fails BEFORE submission: stay `ACTIVE`, return error, user can
  retry/edit.
- `priority_orders.session_id` has a UNIQUE constraint as defense-in-depth.

### Known gaps (deferred to MVP)

- **SUBMISSION_UNKNOWN reconciliation via ERP external-reference lookup** is
  not yet implemented. Until then, a `SUBMISSION_UNKNOWN` session blocks new
  session creation for the same participant (one unresolved receiving session
  per participant). The user must retry the existing session; the idempotency
  store replays the indeterminate outcome. The full flow — lookup ERP by
  external reference, transition to `SUBMITTED` if found, retry same request
  if definitely absent — is MVP work.
- **Barcode → SKU/catalog mapping** is not yet implemented. The receiving
  flow aggregates by barcode value; model/color/size resolution is MVP work.

## Scanner/evaluation rules

- Barcode accuracy is occurrence/multiset based (duplicate values = separate
  physical boxes). Use `collections.Counter`, not `set()`.
- Zero false positives is a hard priority.
- New scanner algorithms must be benchmarked against the canonical dataset
  (`tests/eval/barcode_baseline.json` after PR #1).
- `make eval` is deterministic scanner-only (no Gemini, no LangSmith, no
  Postgres, no network). Gates merges. Exits non-zero on regression.
- `make eval-live` is scanner + Gemini (observational, NOT a gate).
- `make eval-freeze` / `--write-baseline` is the only way to update
  `baseline_frozen.json`.
- Gate on per-image and aggregate occurrence recall + false positives. Do NOT
  gate on latency (workstation/CI latency fluctuates; record P50/P95 as
  informational).

## Pipeline overview

Two independent branches run in parallel on one image, then a single
containment match joins them:

1. **Deterministic scanner** (`src/ingest/scanner.py`) — zxing-cpp + OpenCV
   label fallback. Decodes barcode values with full-resolution pixel bboxes.
2. **Gemini Flash spatial audit** (`src/ingest/vision.py`) — locates every
   visible product label and its barcode region, returns pixel bboxes.

`src/ingest/graph.py` orchestrates as a LangGraph `StateGraph`. On the failure
path, Gemini-guided recovery crops missing barcode regions and retries
reconciliation. See `docs/architecture.md` for full detail.

### Dependency direction

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
- `src/ingest/reconciliation.py` — imports only `src.ingest.geometry`.
- Reconciliation uses padded center-in-box containment with global
  nearest-first assignment. When `barcode_bbox` is present, only it is used.

## Product API

`src/ingest/analyze.py` exposes `analyze_image()` — the product hot path.
Returns `complete` / `needs_better_photo` / `retryable_error`. Full response
schema in `docs/evaluation.md`.

## Commands

```bash
# Environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run
python -m src.cli_app scan ./samples/multi_clear_6_boxes.jpeg
python -m src.cli_app audit ./samples/multi_clear_6_boxes.jpeg --time   # needs GEMINI_API_KEY
python -m src.cli_app pipeline ./samples/multi_clear_6_boxes.jpeg --time --pretty

# Verify
pytest --cov=src --cov-fail-under=90
ruff check .
mypy
make eval          # deterministic scanner-only (gates merges)
make eval-live     # scanner + Gemini, observational (NOT a gate)
make eval-freeze   # rewrite baseline_frozen.json (the ONLY way to update it)
```

## Lint exceptions

Encoded as per-file ignores in `pyproject.toml` under
`[tool.ruff.lint.per-file-ignores]` (machine-enforced, not a manual note).
Run `ruff check .` — it is genuinely green.

## NOT imported from echo-v2

- Coverage gate: started at 74% in PR #0, ramped to 95% across PRs #0–#6
  (counting test files). PR A switched to source-only coverage
  (`--cov=src`) with a floor of 90% — the honest metric.
- Strict mypy (gating once all errors fixed; was non-gating in PR #0).
- WaitingListQueryService, SQLAlchemy/UoW notes.
- Python 3.10/3.11 matrix (barcode-scanner requires Python >=3.12).
- Echo-specific timeout values (use product-specific P95-derived timeouts).
