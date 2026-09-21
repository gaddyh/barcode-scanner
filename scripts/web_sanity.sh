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
#   - .env with GEMINI_API_KEY, D360_API_KEY
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
# 4. Sanity checks
# ---------------------------------------------------------------------------

log "Running sanity checks..."

# --- /health ---
RESP=$(curl -sf "$BASE/health")
check "GET /health returns ok" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"status\"]==\"ok\""'

# --- /customers ---
RESP=$(curl -sf "$BASE/customers")
check "GET /customers returns 3 customers" 'echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d[\"items\"])==3"'
check "GET /customers has cust-acme" 'echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert any(i[\"id\"]==\"cust-acme\" for i in d[\"items\"])"'

# --- /customers/{id}/branches ---
RESP=$(curl -sf "$BASE/customers/cust-acme/branches")
check "GET /customers/cust-acme/branches returns 2 branches" 'echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d[\"items\"])==2"'
check "GET /branches has branch-acme-main" 'echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert any(i[\"id\"]==\"branch-acme-main\" for i in d[\"items\"])"'

# --- POST /receiving/sessions (create) ---
RESP=$(curl -sf -X POST "$BASE/receiving/sessions" \
  -F "customer_id=cust-acme" \
  -F "branch_id=branch-acme-main" \
  -F "action=create_order")
check "POST /receiving/sessions returns session_id" 'echo "$RESP" | python3 -c "import sys,json; assert \"session_id\" in json.load(sys.stdin)"'
check "POST /receiving/sessions status is active" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"status\"]==\"active\""'
check "POST /receiving/sessions has participant_id" 'echo "$RESP" | python3 -c "import sys,json; assert \"participant_id\" in json.load(sys.stdin)"'
check "POST /receiving/sessions has expected_count" 'echo "$RESP" | python3 -c "import sys,json; assert \"expected_count\" in json.load(sys.stdin)"'
check "POST /receiving/sessions has external_order_id" 'echo "$RESP" | python3 -c "import sys,json; assert \"external_order_id\" in json.load(sys.stdin)"'
check "POST /receiving/sessions has frozen" 'echo "$RESP" | python3 -c "import sys,json; assert \"frozen\" in json.load(sys.stdin)"'
check "POST /receiving/sessions has items" 'echo "$RESP" | python3 -c "import sys,json; assert \"items\" in json.load(sys.stdin)"'
check "POST /receiving/sessions has discrepancy" 'echo "$RESP" | python3 -c "import sys,json; assert \"discrepancy\" in json.load(sys.stdin)"'
check "POST /receiving/sessions discrepancy has is_complete" 'echo "$RESP" | python3 -c "import sys,json; assert \"is_complete\" in json.load(sys.stdin)[\"discrepancy\"]"'
SESSION_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "  session: $SESSION_ID"

# --- POST /receiving/sessions (validation errors) ---
RESP=$(curl -s -w '%{http_code}' -o /dev/null -X POST "$BASE/receiving/sessions" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main")
check "POST /receiving/sessions missing action → 422" 'echo "$RESP" | grep -q "422"'
RESP=$(curl -s -w '%{http_code}' -o /dev/null -X POST "$BASE/receiving/sessions" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main" -F "action=bad")
check "POST /receiving/sessions invalid action → 422" 'echo "$RESP" | grep -q "422"'
RESP=$(curl -s -w '%{http_code}' -o /dev/null -X POST "$BASE/receiving/sessions" -F "customer_id=nope" -F "branch_id=branch-acme-main" -F "action=create_order")
check "POST /receiving/sessions unknown customer → 422" 'echo "$RESP" | grep -q "422"'

# --- GET /receiving/sessions/{id} ---
RESP=$(curl -sf "$BASE/receiving/sessions/$SESSION_ID")
check "GET /receiving/sessions/{id} returns correct id" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"session_id\"]==\"'"$SESSION_ID"'\""'
check "GET /receiving/sessions/{id} has all fields" 'echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); missing=[k for k in [\"session_id\",\"status\",\"customer_id\",\"branch_id\",\"action\",\"participant_id\",\"box_count\",\"expected_count\",\"external_order_id\",\"frozen\",\"items\",\"discrepancy\"] if k not in d]; assert not missing, missing"'
RESP=$(curl -s -w '%{http_code}' -o /dev/null "$BASE/receiving/sessions/nonexistent")
check "GET /receiving/sessions/nonexistent → 404" 'echo "$RESP" | grep -q "404"'

# --- POST /receiving/sessions/{id}/images (upload) ---
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_ID/images" -F "file=@samples/multi_clear_6_boxes.jpeg")
check "POST /images returns session_id" 'echo "$RESP" | python3 -c "import sys,json; assert \"session_id\" in json.load(sys.stdin)"'
check "POST /images has outcome" 'echo "$RESP" | python3 -c "import sys,json; assert \"outcome\" in json.load(sys.stdin)"'
check "POST /images has boxes_added" 'echo "$RESP" | python3 -c "import sys,json; assert \"boxes_added\" in json.load(sys.stdin)"'
check "POST /images has total_boxes" 'echo "$RESP" | python3 -c "import sys,json; assert \"total_boxes\" in json.load(sys.stdin)"'
check "POST /images has expected_count" 'echo "$RESP" | python3 -c "import sys,json; assert \"expected_count\" in json.load(sys.stdin)"'
check "POST /images has missing_count" 'echo "$RESP" | python3 -c "import sys,json; assert \"missing_count\" in json.load(sys.stdin)"'
BOXES=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('boxes_added',0))")
echo "  boxes_added: $BOXES"

# --- GET /receiving/sessions/{id} (after upload) ---
RESP=$(curl -sf "$BASE/receiving/sessions/$SESSION_ID")
check "GET session after upload box_count > 0" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"box_count\"]>0"'
check "GET session after upload expected_count > 0" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"expected_count\"]>0"'
check "GET session after upload items non-empty" 'echo "$RESP" | python3 -c "import sys,json; assert len(json.load(sys.stdin)[\"items\"])>0"'
check "GET session after upload discrepancy.is_complete" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"discrepancy\"][\"is_complete\"]==True"'

# --- POST /receiving/sessions/{id}/submit ---
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_ID/submit")
check "POST /submit status is submitted" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"status\"]==\"submitted\""'
check "POST /submit has order_id" 'echo "$RESP" | python3 -c "import sys,json; assert \"order_id\" in json.load(sys.stdin)"'
check "POST /submit has items" 'echo "$RESP" | python3 -c "import sys,json; assert \"items\" in json.load(sys.stdin)"'
ORDER_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['order_id'])")
echo "  order_id: $ORDER_ID"

# --- GET /receiving/sessions/{id} (after submit) ---
RESP=$(curl -sf "$BASE/receiving/sessions/$SESSION_ID")
check "GET session after submit status=submitted" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"status\"]==\"submitted\""'
check "GET session after submit external_order_id set" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"external_order_id\"] is not None"'
check "GET session after submit frozen=true" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"frozen\"]==True"'

# --- Double submit returns same order_id ---
RESP=$(curl -sf -X POST "$BASE/receiving/sessions/$SESSION_ID/submit")
check "Double submit returns same order_id" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"order_id\"]=='"$ORDER_ID"'"'

# --- Upload after submit → 409 ---
RESP=$(curl -s -w '%{http_code}' -o /dev/null -X POST "$BASE/receiving/sessions/$SESSION_ID/images" -F "file=@samples/multi_clear_6_boxes.jpeg")
check "Upload after submit → 409" 'echo "$RESP" | grep -q "409"'

# --- Empty session submit → 422 ---
EMPTY_SESSION=$(curl -sf -X POST "$BASE/receiving/sessions" -F "customer_id=cust-acme" -F "branch_id=branch-acme-main" -F "action=create_order" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
RESP=$(curl -s -w '%{http_code}' -o /dev/null -X POST "$BASE/receiving/sessions/$EMPTY_SESSION/submit")
check "Empty session submit → 422" 'echo "$RESP" | grep -q "422"'

# --- POST /barcode/scan (scanner-only) ---
RESP=$(curl -sf -X POST "$BASE/barcode/scan" -F "file=@samples/multi_clear_6_boxes.jpeg")
check "POST /barcode/scan status=found" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"status\"]==\"found\""'
check "POST /barcode/scan count=6" 'echo "$RESP" | python3 -c "import sys,json; assert json.load(sys.stdin)[\"count\"]==6"'
check "POST /barcode/scan has 6 barcodes" 'echo "$RESP" | python3 -c "import sys,json; assert len(json.load(sys.stdin)[\"barcodes\"])==6"'

# --- POST /barcode/analyze (full pipeline) ---
RESP=$(curl -sf -X POST "$BASE/barcode/analyze" -F "file=@samples/multi_clear_6_boxes.jpeg")
check "POST /barcode/analyze has outcome" 'echo "$RESP" | python3 -c "import sys,json; assert \"outcome\" in json.load(sys.stdin)"'

# --- POST /feedback ---
RESP=$(curl -s -w '%{http_code}' -o /dev/null -X POST "$BASE/feedback" -H "Content-Type: application/json" -d '{"upload_id":"test","trace_id":"test","rating":"good","comment":"test"}')
check "POST /feedback returns 200 or 422" 'echo "$RESP" | grep -qE "200|422"'

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
