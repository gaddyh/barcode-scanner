#!/usr/bin/env python3
"""Test retry scenarios against a running backend using synthetic images.

Scenarios:
1. Exactly the missing count — upload 12 (3 missing) → upload 3 boxes → complete
2. Less than missing — upload 12 (3 missing) → upload 1 box → 2 still missing
3. More than missing — upload 12 (3 missing) → upload 5 boxes → needs better photo
4. Aggregate complete then add more — upload 6 (complete) → upload 5 more → 11 total

Usage:
  DATABASE_URL=... GEMINI_API_KEY=... python scripts/test_retry_scenarios.py
"""
import json
import sys
import urllib.request

BASE = "http://localhost:8001"


def post(url, data=None, files=None):
    import urllib.parse
    if files:
        boundary = "----boundary"
        body = b""
        for k, v in (data or {}).items():
            body += (
                f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{k}"\r\n\r\n{v}\r\n'
            ).encode()
        for k, (fn, fb) in files.items():
            body += (
                f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{k}"; filename="{fn}"\r\n'
                f"Content-Type: image/png\r\n\r\n"
            ).encode()
            body += fb + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    else:
        req = urllib.request.Request(
            url, data=json.dumps(data).encode(), method="POST"
        )
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def upload_image(session_id, filepath):
    with open(filepath, "rb") as f:
        fb = f.read()
    fn = filepath.split("/")[-1]
    return post(
        f"{BASE}/receiving/sessions/{session_id}/images",
        files={"file": (fn, fb)},
    )


def create_session(participant):
    return post(f"{BASE}/receiving/sessions", data={"participant_id": participant})


def get_session(session_id):
    req = urllib.request.Request(f"{BASE}/receiving/sessions/{session_id}")
    return json.loads(urllib.request.urlopen(req).read())


def main():
    all_ok = True
    SYNTH = "samples/synthetic"

    # --- Scenario 1: exactly the missing count ---
    print("\n--- Scenario 1: exactly the missing count ---")
    code, sess = create_session("retry-exact")
    s1 = sess["session_id"]
    print(f"  session: {s1}")
    code, r1 = upload_image(s1, f"{SYNTH}/photo1_12_9found.png")
    print(
        f"  photo 1: found={r1['boxes_added']} "
        f"expected={r1['expected_count']} missing={r1['missing_count']}"
    )
    missing_before = r1["missing_count"]
    if missing_before == 0:
        print("  [SKIP] photo 1 already complete")
    else:
        code, r2 = upload_image(s1, f"{SYNTH}/retry_exact_3.png")
        print(
            f"  photo 2: found={r2['boxes_added']} "
            f"total={r2['total_boxes']} missing={r2['missing_count']}"
        )
        sess_final = get_session(s1)
        missing_after = sess_final["discrepancy"]["missing"]
        print(
            f"  final: box_count={sess_final['box_count']} "
            f"missing={missing_after}"
        )
        if missing_after == 0 and sess_final["box_count"] == 12:
            print("  [PASS] all missing resolved, 12 boxes")
        else:
            print(f"  [FAIL] missing={missing_after} boxes={sess_final['box_count']}")
            all_ok = False

    # --- Scenario 2: less than missing ---
    print("\n--- Scenario 2: less than missing ---")
    code, sess = create_session("retry-less")
    s2 = sess["session_id"]
    print(f"  session: {s2}")
    code, r1 = upload_image(s2, f"{SYNTH}/photo1_12_9found.png")
    print(
        f"  photo 1: found={r1['boxes_added']} "
        f"expected={r1['expected_count']} missing={r1['missing_count']}"
    )
    missing_before = r1["missing_count"]
    if missing_before == 0:
        print("  [SKIP] photo 1 already complete")
    else:
        code, r2 = upload_image(s2, f"{SYNTH}/retry_less_1.png")
        print(
            f"  photo 2: found={r2['boxes_added']} "
            f"total={r2['total_boxes']} missing={r2['missing_count']}"
        )
        sess_final = get_session(s2)
        missing_after = sess_final["discrepancy"]["missing"]
        print(
            f"  final: box_count={sess_final['box_count']} "
            f"missing={missing_after}"
        )
        if missing_after == missing_before - 1 and sess_final["box_count"] == 10:
            print(
                f"  [PASS] missing decreased {missing_before} → {missing_after}, "
                f"10 boxes"
            )
        else:
            print(
                f"  [FAIL] missing={missing_after} "
                f"boxes={sess_final['box_count']} (expected 10 boxes, "
                f"{missing_before - 1} missing)"
            )
            all_ok = False

    # --- Scenario 3: more than missing ---
    print("\n--- Scenario 3: more than missing ---")
    code, sess = create_session("retry-more")
    s3 = sess["session_id"]
    print(f"  session: {s3}")
    code, r1 = upload_image(s3, f"{SYNTH}/photo1_12_9found.png")
    print(
        f"  photo 1: found={r1['boxes_added']} "
        f"expected={r1['expected_count']} missing={r1['missing_count']}"
    )
    missing_before = r1["missing_count"]
    if missing_before == 0:
        print("  [SKIP] photo 1 already complete")
    else:
        code, r2 = upload_image(s3, f"{SYNTH}/retry_more_5.png")
        print(
            f"  photo 2: found={r2['boxes_added']} "
            f"total={r2['total_boxes']} missing={r2['missing_count']}"
        )
        sess_final = get_session(s3)
        missing_after = sess_final["discrepancy"]["missing"]
        print(
            f"  final: box_count={sess_final['box_count']} "
            f"missing={missing_after} status={sess_final['status']}"
        )
        # 5 candidates > 3 missing → filter known neighbors (2) → 3 new → resolve
        # OR if all 5 are new → AMBIGUOUS
        # In our case: 3 new + 2 known → filter → 3 new = 3 missing → resolve all
        if missing_after == 0 and sess_final["box_count"] == 12:
            print("  [PASS] filtered neighbors, resolved all 3 missing")
        else:
            print(
                f"  [FAIL] missing={missing_after} "
                f"boxes={sess_final['box_count']} status={sess_final['status']}"
            )
            all_ok = False

    # --- Scenario 4: aggregate complete then add more ---
    print("\n--- Scenario 4: aggregate complete then add more ---")
    code, sess = create_session("retry-aggregate")
    s4 = sess["session_id"]
    print(f"  session: {s4}")
    code, r1 = upload_image(s4, f"{SYNTH}/agg_photo1_6.png")
    print(
        f"  photo 1: found={r1['boxes_added']} "
        f"expected={r1['expected_count']} missing={r1['missing_count']} "
        f"status={r1['status']}"
    )
    boxes_after_1 = r1["total_boxes"]
    code, r2 = upload_image(s4, f"{SYNTH}/agg_photo2_5.png")
    print(
        f"  photo 2: found={r2['boxes_added']} "
        f"total={r2['total_boxes']} missing={r2['missing_count']} "
        f"status={r2['status']}"
    )
    sess_final = get_session(s4)
    print(
        f"  final: box_count={sess_final['box_count']} "
        f"expected={sess_final['expected_count']}"
    )
    if sess_final["box_count"] == 11:
        print("  [PASS] aggregated 6 + 5 = 11 boxes")
    else:
        print(
            f"  [FAIL] did not aggregate "
            f"({boxes_after_1} → {sess_final['box_count']}, expected 11)"
        )
        all_ok = False

    print(f"\n{'=' * 60}")
    if all_ok:
        print("ALL SCENARIOS PASSED")
    else:
        print("SOME SCENARIOS FAILED")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
