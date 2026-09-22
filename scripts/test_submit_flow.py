#!/usr/bin/env python3
"""Test that submitting an order refreshes the session — next photo
creates a NEW session (not aggregated into the submitted one)."""
import json
import sys
import urllib.request

BASE = "http://localhost:8001"


def post(url, data=None, files=None):
    import urllib.parse
    if files or data:
        boundary = "----boundary"
        body = b""
        for k, v in (data or {}).items():
            body += (
                f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{k}"\r\n\r\n{v}\r\n'
            ).encode()
        if files:
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
        req = urllib.request.Request(url, method="POST")
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(url):
    req = urllib.request.Request(url)
    return json.loads(urllib.request.urlopen(req).read())


def upload_image(session_id, filepath):
    with open(filepath, "rb") as f:
        fb = f.read()
    fn = filepath.split("/")[-1]
    return post(
        f"{BASE}/receiving/sessions/{session_id}/images",
        files={"file": (fn, fb)},
    )


def main():
    SYNTH = "samples/synthetic"
    all_ok = True

    # 1. Create session
    code, sess = post(
        f"{BASE}/receiving/sessions",
        data={"participant_id": "test-submit-flow"},
    )
    s1 = sess["session_id"]
    print(f"1. session: {s1}")

    # 2. Upload photo (complete with 6 boxes)
    code, r1 = upload_image(s1, f"{SYNTH}/agg_photo1_6.png")
    print(
        f"2. upload: found={r1['boxes_added']} total={r1['total_boxes']} "
        f"status={r1['status']}"
    )

    # 3. Get valid customer/branch from the API
    customers = get(f"{BASE}/customers")
    customers = customers.get("items", customers) if isinstance(customers, dict) else customers
    if not customers:
        print("  [SKIP] no customers available — cannot test submit")
        return 0
    customer_id = customers[0]["id"]
    branches = get(f"{BASE}/customers/{customer_id}/branches")
    branches = branches.get("items", branches) if isinstance(branches, dict) else branches
    if not branches:
        print("  [SKIP] no branches available — cannot test submit")
        return 0
    branch_id = branches[0]["id"]
    print(f"3. customer={customer_id} branch={branch_id}")

    # 4. Attach context
    code, ctx = post(
        f"{BASE}/receiving/sessions/{s1}/context",
        data={
            "customer_id": customer_id,
            "branch_id": branch_id,
            "action": "create_order",
        },
    )
    print(f"4. context: status={ctx.get('status', '?')}")

    # 5. Submit order
    code, sub = post(f"{BASE}/receiving/sessions/{s1}/submit")
    print(
        f"5. submit: status={sub.get('status', '?')} "
        f"order_id={sub.get('order_id', '?')}"
    )

    # 6. Upload another photo — should create a NEW session
    code, r2 = upload_image(s1, f"{SYNTH}/agg_photo2_5.png")
    print(
        f"6. upload after submit: code={code} "
        f"status={r2.get('status', '?')} "
        f"session_id={r2.get('session_id', '?')}"
    )

    if code == 409:
        # Session is submitted — upload rejected. Good.
        print("  [PASS] submitted session rejects new images (409)")
    else:
        print(f"  [FAIL] expected 409, got {code}")
        all_ok = False

    # 7. Create a new session for the same participant
    code, sess2 = post(
        f"{BASE}/receiving/sessions",
        data={"participant_id": "test-submit-flow"},
    )
    s2 = sess2["session_id"]
    print(f"7. new session: {s2}")

    if s2 != s1:
        print("  [PASS] new session created (different ID)")
    else:
        print("  [FAIL] same session reused after submit")
        all_ok = False

    # 8. Upload to the new session
    code, r3 = upload_image(s2, f"{SYNTH}/agg_photo2_5.png")
    print(
        f"8. upload to new session: found={r3['boxes_added']} "
        f"total={r3['total_boxes']} status={r3['status']}"
    )

    if r3["total_boxes"] == 5:
        print("  [PASS] new session starts fresh (5 boxes, not 11)")
    else:
        print(f"  [FAIL] expected 5 boxes, got {r3['total_boxes']}")
        all_ok = False

    print(f"\n{'=' * 60}")
    if all_ok:
        print("SUBMIT FLOW PASSED")
    else:
        print("SUBMIT FLOW FAILED")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
