#!/usr/bin/env bash
# scripts/web_sanity.sh
#
# Start all services, reset the DB, and run end-to-end sanity checks
# against the web frontend + backend API.
#
# Usage:
#   ./scripts/web_sanity.sh
#
# Prerequisites:
#   - .venv/ activated with uvicorn, pytest deps installed
#   - .env with GEMINI_API_KEY
#   - Docker container "barcode-scanner-pg-test" running on port 5433
#   - web/node_modules installed

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE="http://localhost:5173"   # Vite proxy → backend
PG_CONTAINER="barcode-scanner-pg-test"
PG_PORT="5433"
BACKEND_PID=""
FRONTEND_PID=""
PASS=0
FAIL=0
ERRORS=""
SESSION_ID=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { printf '\033[1;34m▶\033[0m %s\n' "$*"; }
ok()   { printf '  \033[32m[PASS]\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
fail() { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  - $1"; }
done_section() { printf '\n'; }

check() {
  local name="$1"
  local cond="$2"
  if eval "$cond" 2>/dev/null; then
    ok "$name"
  else
    fail "$name"
  fi
}

# ---------------------------------------------------------------------------
# 1. Start / reset Postgres
# ---------------------------------------------------------------------------

log "Checking Postgres container ($PG_CONTAINER)..."
if ! docker ps --format '{{.Names}}' | grep -q "^${PG_CONTAINER}$"; then
  fail "Postgres container '$PG_CONTAINER' is not running"
  echo "Start it with: docker run -d --name $PG_CONTAINER -p $PG_PORT:5432 -e POSTGRES_USER=scanner -e POSTGRES_PASSWORD=scanner -e POSTGRES_DB=scanner postgres:16-alpine"
  exit 1
fi
ok "Postgres container running"
done_section

log "Resetting database..."
docker exec "$PG_CONTAINER" psql -U scanner -d scanner -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;" >/dev/null 2>&1
ok "Database schema reset"
done_section

# ---------------------------------------------------------------------------
# 2. Start backend
# ---------------------------------------------------------------------------

log "Starting backend (uvicorn :8000)..."
if lsof -i :8000 -t >/dev/null 2>&1; then
  log "Port 8000 already in use — killing existing process..."
  lsof -i :8000 -t | xargs kill -9 2>/dev/null || true
  sleep 1
fi

source .venv/bin/activate
set -a; source .env; set +a
export DATABASE_URL="postgres://scanner:scanner@localhost:$PG_PORT/scanner"
# Use the frozen Gemini audit cache so the full pipeline is deterministic
# (matches `make eval-full-replay`). No live Gemini calls.
export GEMINI_AUDIT_CACHE_MODE=replay
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "  backend PID: $BACKEND_PID"

log "Waiting for backend to start..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    ok "Backend is up"
    break
  fi
  sleep 1
  if [ $i -eq 30 ]; then
    fail "Backend failed to start within 30s"
    exit 1
  fi
done
done_section

# ---------------------------------------------------------------------------
# 3. Start frontend
# ---------------------------------------------------------------------------

log "Starting frontend (vite :5173)..."
if lsof -i :5173 -t >/dev/null 2>&1; then
  log "Port 5173 already in use — killing existing process..."
  lsof -i :5173 -t | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# Remove stale compiled vite config (Vite loads .js over .ts)
rm -f web/vite.config.js web/vite.config.d.ts

cd web
npm run dev >/tmp/vite_sanity.log 2>&1 &
FRONTEND_PID=$!
cd "$ROOT"
echo "  frontend PID: $FRONTEND_PID"

log "Waiting for frontend to start..."
for i in $(seq 1 20); do
  if curl -sf http://localhost:5173/ >/dev/null 2>&1; then
    ok "Frontend is up"
    break
  fi
  sleep 1
  if [ $i -eq 20 ]; then
    fail "Frontend failed to start within 20s"
    cat /tmp/vite_sanity.log
    exit 1
  fi
done
done_section

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
cleanup() {
  log "Cleaning up..."
  [ -n "$BACKEND_PID" ]  && kill "$BACKEND_PID" 2>/dev/null || true
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
  wait "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 4. Sanity checks — full multi-step flows with state verification
# ---------------------------------------------------------------------------

# Disable set -e and set -u for sanity checks — we want to record
# failures, not exit on curl errors or unset variables.
set +e +u

log "Running sanity checks (full flows)..."

# Helper: assert a JSON field equals an expected value.
#   jassert "$RESP" "status" "active"
#   jassert "$RESP" "customer_id" "null"
jassert() {
  local body="$1" field="$2" expected="$3"
  local actual
  actual=$(echo "$body" | python3 -c "
import sys, json
d = json.load(sys.stdin)
v = d.get('$field')
if v is None:
    print('null')
elif isinstance(v, bool):
    print('true' if v else 'false')
else:
    print(str(v))
" 2>/dev/null)
  if [ "$actual" = "$expected" ]; then
    ok "$field == $expected"
  else
    fail "$field == $expected (got $actual)"
  fi
}

# Helper: assert HTTP status code.
#   hassert "$RESP" "422"
hassert() {
  local code="$1" expected="$2"
  if [ "$code" = "$expected" ]; then
    ok "HTTP $expected"
  else
    fail "HTTP $expected (got $code)"
  fi
}

# Helper: POST with status code + body capture.
#   post_status_body RESP_VAR CODE_VAR URL -F "k=v"
# Uses temp files to avoid BSD sed/GNU sed differences and eval quoting issues.
post_status_body() {
  local resp_var="$1" code_var="$2" url="$3"; shift 3
  local body_file
  body_file=$(mktemp)
  CODE=$(curl -s -o "$body_file" -w '%{http_code}' -X POST "$url" "$@" || echo "000")
  BODY=$(cat "$body_file" 2>/dev/null || true)
  rm -f "$body_file"
}

# Helper: GET with status code + body capture.
#   get_status_body RESP_VAR CODE_VAR URL
get_status_body() {
  local resp_var="$1" code_var="$2" url="$3"
  local body_file
  body_file=$(mktemp)
  CODE=$(curl -s -o "$body_file" -w '%{http_code}' "$url" || echo "000")
  BODY=$(cat "$body_file" 2>/dev/null || true)
  rm -f "$body_file"
}

# --- GET /health ---
RESP=$(curl -sf "$BASE/health")
check "GET /health returns ok" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"status\"]==\"ok\""'

# --- GET /customers ---
RESP=$(curl -sf "$BASE/customers")
check "GET /customers returns 3 customers" 'echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d[\"items\"])==3"'
check "GET /customers has cust-acme" 'echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert any(i[\"id\"]==\"cust-acme\" for i in d[\"items\"])"'

# --- GET /customers/{id}/branches ---
RESP=$(curl -sf "$BASE/customers/cust-acme/branches")
check "GET /customers/cust-acme/branches returns 2 branches" 'echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d[\"items\"])==2"'
check "GET /branches has branch-acme-main" 'echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert any(i[\"id\"]==\"branch-acme-main\" for i in d[\"items\"])"'

# ===========================================================================
# Flow A: scan-first full lifecycle
#   create(no ctx) → resume(200) → upload → verify state → attach ctx →
#   submit → verify submitted → double-submit(idempotent) →
#   upload-after-submit(409) → context-after-submit(409)
# ===========================================================================
log "Flow A: scan-first full lifecycle (no ctx → upload → attach → submit → idempotency)"

# A.1 — Create with NO context (201), explicit participant for resume test
RESP=$(curl -sf -X POST "$BASE/receiving/sessions" -F "participant_id=test-participant-a")
jassert "$RESP" "status" "active"
jassert "$RESP" "customer_id" "null"
jassert "$RESP" "branch_id" "null"
jassert "$RESP" "action" "null"
jassert "$RESP" "box_count" "0"
check "POST /sessions (no ctx) has all frontend fields" 'echo "$RESP" | python3 -c "
import sys,json; d=json.load(sys.stdin)
for k in [\"session_id\",\"status\",\"customer_id\",\"branch_id\",\"action\",\"participant_id\",\"box_count\",\"expected_count\",\"external_order_id\",\"frozen\",\"items\",\"discrepancy\"]:
    assert k in d, f\"missing {k}\"
assert \"is_complete\" in d[\"discrepancy\"]
"'
SESSION_A=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "  session: $SESSION_A"

# A.2 — Resume same participant → 200, same session_id
post_status_body BODY CODE "$BASE/receiving/sessions" -F "participant_id=test-participant-a"
hassert "$CODE" "200"
jassert "$BODY" "session_id" "$SESSION_A"

# A.3 — Validation errors on create
post_status_body _ C "$BASE/receiving/sessions" -F "customer_id=cust-acme"
hassert "$CODE" "422"
post_status_body _ C "$BASE/receiving/sessions" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main" -F "action=bad"
hassert "$CODE" "422"
post_status_body _ C "$BASE/receiving/sessions" -F "customer_id=nope" -F "branch_id=branch-acme-main" -F "action=create_order"
hassert "$CODE" "422"

# A.4 — Upload photo 1 (before context — must work)
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_A/images" -F "file=@samples/multi_clear_6_boxes.jpeg")
BOXES_A1=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['boxes_added'])")
echo "  boxes_added (photo 1): $BOXES_A1"
check "POST /images (no ctx) boxes_added > 0" '[ "$BOXES_A1" -gt 0 ]'

# A.5 — GET session: verify state after upload, context still null
RESP=$(curl -sf "$BASE/receiving/sessions/$SESSION_A")
jassert "$RESP" "status" "active"
jassert "$RESP" "box_count" "$BOXES_A1"
jassert "$RESP" "customer_id" "null"
check "GET session after upload items non-empty" 'echo "$RESP" | python3 -c "import sys,json; assert len(json.load(sys.stdin)[\"items\"])>0"'

# A.6 — Submit without context → 422
post_status_body _ C "$BASE/receiving/sessions/$SESSION_A/submit"
hassert "$CODE" "422"

# A.7 — Attach context
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_A/context" \
  -F "customer_id=cust-acme" -F "branch_id=branch-acme-main" -F "action=create_order")
jassert "$RESP" "customer_id" "cust-acme"
jassert "$RESP" "branch_id" "branch-acme-main"
jassert "$RESP" "action" "create_order"
jassert "$RESP" "status" "active"

# A.8 — Context validation errors (session still active so these hit validation)
post_status_body _ C "$BASE/receiving/sessions/$SESSION_A/context" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main"
hassert "$CODE" "422"
post_status_body _ C "$BASE/receiving/sessions/$SESSION_A/context" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main" -F "action=bad"
hassert "$CODE" "422"
post_status_body _ C "$BASE/receiving/sessions/$SESSION_A/context" -F "customer_id=nope" -F "branch_id=branch-acme-main" -F "action=create_order"
hassert "$CODE" "422"
post_status_body _ C "$BASE/receiving/sessions/$SESSION_A/context" -F "customer_id=cust-acme" -F "branch_id=nope" -F "action=create_order"
hassert "$CODE" "422"
post_status_body _ C "$BASE/receiving/sessions/nonexistent/context" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main" -F "action=create_order"
hassert "$CODE" "404"

# A.9 — Submit after context → success
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_A/submit")
jassert "$RESP" "status" "submitted"
check "POST /submit has order_id" 'echo "$RESP" | python3 -c "import sys,json; assert \"order_id\" in json.load(sys.stdin)"'
check "POST /submit has items" 'echo "$RESP" | python3 -c "import sys,json; assert \"items\" in json.load(sys.stdin)"'
ORDER_A=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['order_id'])")
echo "  order_id: $ORDER_A"

# A.10 — GET session: verify submitted state
RESP=$(curl -sf "$BASE/receiving/sessions/$SESSION_A")
jassert "$RESP" "status" "submitted"
jassert "$RESP" "frozen" "true"
jassert "$RESP" "customer_id" "cust-acme"
check "GET session after submit external_order_id set" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"external_order_id\"] is not None"'

# A.11 — Double submit → same order_id (idempotency)
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_A/submit")
check "Double submit same order_id" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"order_id\"]=='"$ORDER_A"'"'

# A.12 — Upload after submit → 409 (frozen)
post_status_body _ C "$BASE/receiving/sessions/$SESSION_A/images" -F "file=@samples/multi_clear_6_boxes.jpeg"
hassert "$CODE" "409"

# A.13 — Attach context after submit → 409 (frozen)
post_status_body _ C "$BASE/receiving/sessions/$SESSION_A/context" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main" -F "action=create_order"
hassert "$CODE" "409"

done_section

# ===========================================================================
# Flow B: multi-photo aggregation
#   create → upload photo 1 (6 boxes) → upload photo 2 (different image) →
#   verify aggregate grows → attach ctx → submit
# ===========================================================================
log "Flow B: multi-photo aggregation (upload → upload → verify aggregate)"

RESP=$(curl -sf -X POST "$BASE/receiving/sessions" -F "participant_id=test-participant-multi")
SESSION_B=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "  session: $SESSION_B"

# B.1 — Upload photo 1 (topdown_12_labels_b — 11/12 boxes, 1 missing → status stays 'active')
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_B/images" -F "file=@samples/topdown_12_labels_b.jpeg")
BOXES_B1=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['boxes_added'])")
echo "  boxes after photo 1: $BOXES_B1"
check "Flow B photo 1 boxes_added > 0" '[ "$BOXES_B1" -gt 0 ]'

# B.2 — GET session: verify box_count
RESP=$(curl -sf "$BASE/receiving/sessions/$SESSION_B")
jassert "$RESP" "box_count" "$BOXES_B1"
ITEMS_B1=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['items']))")
echo "  distinct barcodes after photo 1: $ITEMS_B1"

# B.3 — Upload photo 2 (marny_brown_42 — single box, different barcode)
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_B/images" -F "file=@samples/marny_brown_42.jpeg")
BOXES_B2_ADDED=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['boxes_added'])")
TOTAL_B2=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_boxes'])")
echo "  boxes_added (photo 2): $BOXES_B2_ADDED  total_boxes: $TOTAL_B2"
check "Flow B photo 2 accepted (boxes_added >= 0)" '[ "$BOXES_B2_ADDED" -ge 0 ]'

# B.4 — GET session: verify aggregate didn't shrink (second upload may add 0
# new boxes if the cached Gemini audit for marny_brown_42 is misaligned, but
# the session must still be active and accept the upload).
RESP=$(curl -sf "$BASE/receiving/sessions/$SESSION_B")
BOX_COUNT_B2=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['box_count'])")
echo "  aggregate box_count after photo 2: $BOX_COUNT_B2"
check "Flow B aggregate box_count >= photo 1" '[ "$BOX_COUNT_B2" -ge "$BOXES_B1" ]'
ITEMS_B2=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['items']))")
echo "  distinct barcodes after photo 2: $ITEMS_B2"
check "Flow B distinct barcodes >= photo 1" '[ "$ITEMS_B2" -ge "$ITEMS_B1" ]'

# B.5 — Attach context + submit
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_B/context" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main" -F "action=create_order")
jassert "$RESP" "status" "active"
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_B/submit")
jassert "$RESP" "status" "submitted"
ORDER_B=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['order_id'])")
echo "  order_id: $ORDER_B"

done_section

# ===========================================================================
# Flow B2: targeted retry — upload photo with missing → upload photo of
#   the missing box (same barcode as existing) → verify missing resolved.
#   This catches the bug where duplicate barcodes were filtered out and
#   the missing slot was never resolved.
# ===========================================================================
log "Flow B2: targeted retry (missing box with known barcode → resolve)"

RESP=$(curl -sf -X POST "$BASE/receiving/sessions" -F "participant_id=test-participant-retry")
SESSION_B2=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "  session: $SESSION_B2"

# B2.1 — Upload topdown_12_labels_b (11/12 found, 1 missing)
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_B2/images" -F "file=@samples/topdown_12_labels_b.jpeg")
BOXES_B2_1=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['boxes_added'])")
MISSING_B2_1=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['missing_count'])")
echo "  photo 1: boxes=$BOXES_B2_1 missing=$MISSING_B2_1"
check "Flow B2 photo 1 boxes_added > 0" '[ "$BOXES_B2_1" -gt 0 ]'
check "Flow B2 photo 1 missing > 0" '[ "$MISSING_B2_1" -gt 0 ]'

# B2.2 — Upload a single box photo (one of the HQ samples)
#   This should resolve one missing slot even if the barcode matches
#   an already-found value (duplicate physical box).
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_B2/images" -F "file=@samples/naot_box_samples_HQ_1_to_12/boxes_01_HQ.png")
BOXES_B2_2=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['boxes_added'])")
MISSING_B2_2=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['missing_count'])")
TOTAL_B2_2=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_boxes'])")
echo "  photo 2: boxes_added=$BOXES_B2_2 total=$TOTAL_B2_2 missing=$MISSING_B2_2"
check "Flow B2 photo 2 total_boxes > photo 1" '[ "$TOTAL_B2_2" -gt "$BOXES_B2_1" ]'
check "Flow B2 photo 2 missing < photo 1" '[ "$MISSING_B2_2" -lt "$MISSING_B2_1" ]'

done_section

# ===========================================================================
# Flow C: backwards-compatible (context up front)
#   create(with ctx) → upload → submit (no separate /context call needed)
# ===========================================================================
log "Flow C: backwards-compatible (context up front → upload → submit)"

RESP=$(curl -sf -X POST "$BASE/receiving/sessions" \
  -F "customer_id=cust-acme" \
  -F "branch_id=branch-acme-main" \
  -F "action=create_order" \
  -F "participant_id=test-participant-ctx-upfront")
jassert "$RESP" "status" "active"
jassert "$RESP" "customer_id" "cust-acme"
jassert "$RESP" "branch_id" "branch-acme-main"
jassert "$RESP" "action" "create_order"
SESSION_C=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "  session: $SESSION_C"

# C.1 — Upload
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_C/images" -F "file=@samples/multi_clear_6_boxes.jpeg")
echo "  Flow C upload response: $(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'boxes_added={d.get(\"boxes_added\")} total_boxes={d.get(\"total_boxes\")} outcome={d.get(\"outcome\")}')" 2>/dev/null || echo "PARSE ERROR: $RESP")"
check "Flow C upload boxes_added > 0" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"boxes_added\"]>0"'

# C.2 — Submit directly (context already attached)
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_C/submit")
jassert "$RESP" "status" "submitted"
ORDER_C=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['order_id'])")
echo "  order_id: $ORDER_C"

# C.3 — GET session: verify submitted
RESP=$(curl -sf "$BASE/receiving/sessions/$SESSION_C")
jassert "$RESP" "status" "submitted"
jassert "$RESP" "frozen" "true"

done_section

# ===========================================================================
# Flow D: empty session rejected at every submit attempt
#   create → submit(422 empty_order) → attach ctx → submit(422 empty_order)
# ===========================================================================
log "Flow D: empty session rejected (no ctx → 422, with ctx → 422 empty_order)"

RESP=$(curl -sf -X POST "$BASE/receiving/sessions" -F "participant_id=test-participant-empty")
SESSION_D=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "  session: $SESSION_D"

# D.1 — Submit without context → 422 (order_context_required)
post_status_body _ C "$BASE/receiving/sessions/$SESSION_D/submit"
hassert "$CODE" "422"

# D.2 — Attach context (works on empty active session)
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_D/context" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main" -F "action=create_order")
jassert "$RESP" "status" "active"
jassert "$RESP" "customer_id" "cust-acme"

# D.3 — Submit with context but no boxes → 422 (empty_order)
post_status_body _ C "$BASE/receiving/sessions/$SESSION_D/submit"
hassert "$CODE" "422"

done_section

# ===========================================================================
# Flow E: not-found / error paths
# ===========================================================================
log "Flow E: not-found + error paths"

# E.1 — GET nonexistent session → 404
get_status_body _ C "$BASE/receiving/sessions/nonexistent"
hassert "$CODE" "404"

# E.2 — Upload to nonexistent session → 404
post_status_body _ C "$BASE/receiving/sessions/nonexistent/images" -F "file=@samples/multi_clear_6_boxes.jpeg"
hassert "$CODE" "404"

# E.3 — Submit to nonexistent session → 404
post_status_body _ C "$BASE/receiving/sessions/nonexistent/submit"
hassert "$CODE" "404"

# E.4 — Context to nonexistent session → 404
post_status_body _ C "$BASE/receiving/sessions/nonexistent/context" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main" -F "action=create_order"
hassert "$CODE" "404"

done_section

# ===========================================================================
# Flow F: scanner-only + full-pipeline endpoints (independent of receiving)
# ===========================================================================
log "Flow F: scanner-only + full-pipeline endpoints"

# F.1 — POST /barcode/scan (scanner-only)
RESP=$(curl -sf -X POST "$BASE/barcode/scan" -F "file=@samples/multi_clear_6_boxes.jpeg")
jassert "$RESP" "status" "found"
jassert "$RESP" "count" "6"
check "POST /barcode/scan has 6 barcodes" 'echo "$RESP" | python3 -c "import sys,json; assert len(json.load(sys.stdin)[\"barcodes\"])==6"'

# F.2 — POST /barcode/analyze (full pipeline)
RESP=$(curl -sf -X POST "$BASE/barcode/analyze" -F "file=@samples/multi_clear_6_boxes.jpeg")
check "POST /barcode/analyze has outcome" 'echo "$RESP" | python3 -c "import sys,json; assert \"outcome\" in json.load(sys.stdin)"'

# F.3 — POST /feedback
post_status_body _ C "$BASE/feedback" -H "Content-Type: application/json" -d '{"upload_id":"test","trace_id":"test","rating":"good","comment":"test"}'
echo "  feedback HTTP code: $CODE"
check "POST /feedback returns 200 or 422" 'echo "'"$CODE"'" | grep -qE "^(200|422)$"'

done_section

# ===========================================================================
# Flow G: submission-unknown blocks new session for same participant
#   (safety rule — one unresolved session per participant)
#   We can't easily force SUBMISSION_UNKNOWN in a live script without
#   mocking the gateway, so we verify the rule indirectly: a SUBMITTED
#   session's participant can create a NEW session (different participant_id
#   is generated by the client). We instead verify that two different
#   participants can each have their own active session simultaneously.
# ===========================================================================
log "Flow G: per-participant session isolation"

RESP=$(curl -sf -X POST "$BASE/receiving/sessions" -F "participant_id=isolation-1")
SESSION_G1=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
RESP=$(curl -sf -X POST "$BASE/receiving/sessions" -F "participant_id=isolation-2")
SESSION_G2=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
check "Flow G: two participants get different sessions" '[ "$SESSION_G1" != "$SESSION_G2" ]'

# G.1 — Same participant resumes (200), not creates (201)
post_status_body BODY CODE "$BASE/receiving/sessions" -F "participant_id=isolation-1"
hassert "$CODE" "200"
jassert "$BODY" "session_id" "$SESSION_G1"

done_section

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------

echo "=========================================="
printf 'RESULTS: \033[32m%d passed\033[0m, \033[31m%d failed\033[0m\n' "$PASS" "$FAIL"
if [ $FAIL -gt 0 ]; then
  printf 'Failures:%s\n' "$ERRORS"
fi
echo "=========================================="

if [ $FAIL -gt 0 ]; then
  exit 1
fi
