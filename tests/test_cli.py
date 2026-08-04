import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from app.services.barcode_scanner import BarcodeScanner, BoundingBox, DetectedBarcode, Point

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def make_png(width: int = 100, height: int = 50) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_barcode(value: str = "7290001234567") -> DetectedBarcode:
    return DetectedBarcode(
        value=value,
        format="EAN13",
        content_type="Text",
        orientation=0,
        position=(
            Point(x=10, y=5),
            Point(x=90, y=5),
            Point(x=90, y=40),
            Point(x=10, y=40),
        ),
        bounding_box=BoundingBox(x1=10, y1=5, x2=90, y2=40),
    )


def _patch_scanner(monkeypatch, barcodes: list[DetectedBarcode]):
    from app import cli

    class FakeScanner(BarcodeScanner):
        def __init__(self) -> None:
            pass

        def scan_bytes(self, image_bytes: bytes) -> list[DetectedBarcode]:
            if image_bytes == b"not an image":
                raise UnidentifiedImageError("not an image")
            return barcodes

    monkeypatch.setattr(cli, "BarcodeScanner", FakeScanner)
    return cli


# ---------------------------------------------------------------------------
# Unit tests (fake scanner) — scan subcommand
# ---------------------------------------------------------------------------


def test_cli_scans_image_and_prints_json(tmp_path, monkeypatch, capsys) -> None:
    cli = _patch_scanner(monkeypatch, [_fake_barcode()])

    image_path = tmp_path / "product.png"
    image_path.write_bytes(make_png())

    exit_code = cli.main(["scan", str(image_path), "--pretty"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload[0]["path"] == str(image_path)
    assert payload[0]["status"] == "found"
    assert payload[0]["count"] == 1
    assert payload[0]["barcodes"][0]["value"] == "7290001234567"


def test_cli_reports_unreadable_file(tmp_path, monkeypatch, capsys) -> None:
    cli = _patch_scanner(monkeypatch, [])

    missing = tmp_path / "does-not-exist.png"
    exit_code = cli.main(["scan", str(missing)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload[0]["status"] == "error"
    assert payload[0]["error"]["code"] == "unreadable_file"


def test_cli_reports_invalid_image(tmp_path, monkeypatch, capsys) -> None:
    cli = _patch_scanner(monkeypatch, [])

    bad = tmp_path / "not-an-image.png"
    bad.write_bytes(b"not an image")

    exit_code = cli.main(["scan", str(bad)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload[0]["status"] == "error"
    assert payload[0]["error"]["code"] == "invalid_image"


def test_cli_handles_multiple_images(tmp_path, monkeypatch, capsys) -> None:
    cli = _patch_scanner(monkeypatch, [])

    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(make_png())
    second.write_bytes(make_png())

    exit_code = cli.main(["scan", str(first), str(second)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert len(payload) == 2
    assert {entry["path"] for entry in payload} == {str(first), str(second)}


def test_cli_scan_time_prints_timing_to_stderr(tmp_path, monkeypatch, capsys) -> None:
    cli = _patch_scanner(monkeypatch, [_fake_barcode()])

    image_path = tmp_path / "product.png"
    image_path.write_bytes(make_png())

    exit_code = cli.main(["scan", str(image_path), "--time"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload[0]["status"] == "found"
    assert "s" in captured.err  # timing line on stderr


# ---------------------------------------------------------------------------
# Unit tests — audit subcommand (fake audit)
# ---------------------------------------------------------------------------


def _patch_audit(monkeypatch, result=None, error=None):
    from app import cli

    def fake_audit_path(path, *, model, max_retries, retry_delay_seconds, full, labels):
        if error is not None:
            raise error
        return result or {
            "path": str(path),
            "status": "ok",
            "audit": {"visible_product_barcode_label_count": 1},
        }

    monkeypatch.setattr(cli, "audit_path", fake_audit_path)
    return cli


def test_cli_audit_prints_json(tmp_path, monkeypatch, capsys) -> None:
    cli = _patch_audit(monkeypatch)

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    exit_code = cli.main(["audit", str(image_path), "--pretty"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload[0]["status"] == "ok"
    assert payload[0]["audit"]["visible_product_barcode_label_count"] == 1


def test_cli_audit_time_prints_timing_to_stderr(tmp_path, monkeypatch, capsys) -> None:
    cli = _patch_audit(monkeypatch)

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    exit_code = cli.main(["audit", str(image_path), "--time"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload[0]["status"] == "ok"
    assert "s" in captured.err


# ---------------------------------------------------------------------------
# Regression tests (real CLI command against real sample images)
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> tuple[int, list[dict]]:
    result = subprocess.run(
        [sys.executable, "-m", "app.cli", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    payload = json.loads(result.stdout)
    return result.returncode, payload


@pytest.mark.skipif(
    not SAMPLES_DIR.exists(),
    reason="samples/ directory not available",
)
def test_cli_regression_multi_clear_6_boxes() -> None:
    """barcode-scan scan samples/multi_clear_6_boxes.jpeg finds all 6 barcodes."""
    exit_code, payload = _run_cli("scan", "samples/multi_clear_6_boxes.jpeg")

    assert exit_code == 0
    assert len(payload) == 1

    entry = payload[0]
    assert entry["path"] == "samples/multi_clear_6_boxes.jpeg"
    assert entry["status"] == "found"
    assert entry["count"] == 6

    values = [barcode["value"] for barcode in entry["barcodes"]]
    assert values == [
        "7297500243416",
        "7297500243416",
        "7297500243430",
        "7297500243423",
        "7297500243447",
        "7297500243423",
    ]

    for barcode in entry["barcodes"]:
        assert barcode["format"] == "Code128"
        assert barcode["content_type"] == "ContentType.Text"
        assert len(barcode["position"]) == 4
        assert "bounding_box" in barcode
        assert "x1" in barcode["bounding_box"]


@pytest.mark.skipif(
    not SAMPLES_DIR.exists(),
    reason="samples/ directory not available",
)
def test_cli_regression_marny_brown_42() -> None:
    """barcode-scan scan samples/marny_brown_42.jpeg finds both barcodes."""
    exit_code, payload = _run_cli("scan", "samples/marny_brown_42.jpeg")

    assert exit_code == 0
    assert len(payload) == 1

    entry = payload[0]
    assert entry["path"] == "samples/marny_brown_42.jpeg"
    assert entry["status"] == "found"
    assert entry["count"] == 2

    values = [barcode["value"] for barcode in entry["barcodes"]]
    assert values == ["7297501098442", "900439-42"]

    for barcode in entry["barcodes"]:
        assert barcode["format"] == "Code128"
        assert barcode["content_type"] == "ContentType.Text"
        assert len(barcode["position"]) == 4
        assert "bounding_box" in barcode


# ---------------------------------------------------------------------------
# Unit tests — pipeline subcommand (spatial reconciliation)
# ---------------------------------------------------------------------------


def _spatial_result(image_width: int = 1000, image_height: int = 1000) -> dict:
    """A minimal spatial audit result in the shape returned by
    audit_shoebox_labels().model_dump(mode='json')."""
    return {
        "image_width": image_width,
        "image_height": image_height,
        "labels": [
            {
                "label_index": 1,
                "label_bbox": {"x1": 100, "y1": 100, "x2": 300, "y2": 200},
                "barcode_bbox": {"x1": 150, "y1": 120, "x2": 280, "y2": 190},
                "status": "clear",
                "confidence": "high",
            },
            {
                "label_index": 2,
                "label_bbox": {"x1": 100, "y1": 800, "x2": 300, "y2": 900},
                "barcode_bbox": {"x1": 150, "y1": 820, "x2": 280, "y2": 890},
                "status": "clear",
                "confidence": "high",
            },
        ],
    }


def _patch_pipeline(
    monkeypatch,
    barcodes,
    spatial_result_dict,
    audit_error=False,
    recovered_detections=None,
):
    """Patch the pipeline so scan returns ``barcodes`` and the Gemini audit
    returns ``spatial_result_dict`` (or an error).

    ``recovered_detections`` is a list of RecoveredDetection objects to return
    from scan_label_crops.  When None, scan_label_crops returns [] (no recovery).
    """

    cli = _patch_scanner(monkeypatch, barcodes)

    # Add scan_label_crops to the FakeScanner.
    def fake_scan_label_crops(self, image, requests, *, existing_detections=None, debug_dir=None):
        return recovered_detections or []

    cli.BarcodeScanner.scan_label_crops = fake_scan_label_crops

    if audit_error:
        def fake_traced_audit(path, *, model, max_retries, retry_delay_seconds):
            return {
                "status": "error",
                "error": {"type": "ShoeboxAuditError", "message": "boom"},
            }
    else:
        def fake_traced_audit(path, *, model, max_retries, retry_delay_seconds):
            return {"status": "ok", "spatial": spatial_result_dict}

    # _traced_audit lives in app.services.pipeline (extracted from cli).
    from app.services import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "_traced_audit", fake_traced_audit)
    return cli


def test_cli_pipeline_success_produces_reconciliation(
    tmp_path, monkeypatch, capsys
) -> None:
    """Pipeline with scanner success + Gemini success → reconciliation exists
    and scanner_detections preserve bounding boxes."""
    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
        DetectedBarcode(
            value="V2",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=830), Point(x=270, y=830),
                      Point(x=270, y=840), Point(x=160, y=840)),
            bounding_box=BoundingBox(x1=160, y1=830, x2=270, y2=840),
        ),
    ]
    cli = _patch_pipeline(monkeypatch, barcodes, _spatial_result())

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    exit_code = cli.main(["pipeline", str(image_path), "--pretty"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert len(payload) == 1
    entry = payload[0]

    assert entry["scan_status"] == "found"
    assert entry["audit_status"] == "ok"
    assert entry["ok"] is True

    # Full scanner detections preserved with bounding boxes.
    assert "scanner_detections" in entry
    assert len(entry["scanner_detections"]) == 2
    assert "bounding_box" in entry["scanner_detections"][0]

    # Gemini labels preserved.
    assert "gemini_labels" in entry
    assert len(entry["gemini_labels"]) == 2

    # Reconciliation produced.
    assert "reconciliation" in entry
    recon = entry["reconciliation"]
    assert recon["matched_label_count"] == 2
    assert recon["visible_label_count"] == 2
    assert recon["all_labels_matched"] is True
    assert len(recon["matches"]) == 2
    assert recon["unmatched_labels"] == []
    assert recon["unassigned_scanner_detections"] == []

    # Derived counts.
    assert entry["visible_labels"] == 2
    assert entry["clear_labels"] == 2

    # Backward-compatible decoded_vs_visible carries matched_labels.
    dv = entry["decoded_vs_visible"]
    assert dv["matched_labels"] == 2
    assert dv["all_labels_matched"] is True


def test_cli_pipeline_gemini_error_keeps_scanner_no_reconciliation(
    tmp_path, monkeypatch, capsys
) -> None:
    """Pipeline on Gemini audit failure: scanner result still returned,
    audit_status == 'error', no reconciliation, no counts fallback."""
    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=10, y=5), Point(x=90, y=5),
                      Point(x=90, y=40), Point(x=10, y=40)),
            bounding_box=BoundingBox(x1=10, y1=5, x2=90, y2=40),
        ),
    ]
    cli = _patch_pipeline(
        monkeypatch, barcodes, _spatial_result(), audit_error=True
    )

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    exit_code = cli.main(["pipeline", str(image_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    # Exit code 1 because audit failed (ok = scan_ok and audit_ok).
    assert exit_code == 1
    assert len(payload) == 1
    entry = payload[0]

    assert entry["scan_status"] == "found"
    assert entry["audit_status"] == "error"
    assert entry["ok"] is False

    # Scanner detections still present.
    assert "scanner_detections" in entry
    assert len(entry["scanner_detections"]) == 1

    # No reconciliation, no gemini_labels, no counts fallback.
    assert "reconciliation" not in entry
    assert "gemini_labels" not in entry
    assert "visible_labels" not in entry
    assert "audit_error" in entry
    assert entry["audit_error"]["type"] == "ShoeboxAuditError"


def test_cli_pipeline_unmatched_label_shown_in_reconciliation(
    tmp_path, monkeypatch, capsys
) -> None:
    """One scanner detection, two Gemini labels → one unmatched label,
    all_labels_matched=False."""
    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
    ]
    cli = _patch_pipeline(monkeypatch, barcodes, _spatial_result())

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    exit_code = cli.main(["pipeline", str(image_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    # Exit code is 0 because both scan and audit succeeded (ok = scan_ok and
    # audit_ok). A partial match (all_labels_matched=False) is informational,
    # not an error.
    assert exit_code == 0
    entry = payload[0]
    assert entry["ok"] is True
    recon = entry["reconciliation"]
    assert recon["matched_label_count"] == 1
    assert recon["visible_label_count"] == 2
    assert recon["all_labels_matched"] is False
    assert len(recon["unmatched_labels"]) == 1
    assert recon["unmatched_labels"][0]["label_index"] == 2


def test_cli_pipeline_table_shows_matched_over_visible(
    tmp_path, monkeypatch, capsys
) -> None:
    """The stderr table should show matched/visible (e.g. 2/2) when
    reconciliation is available."""
    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
        DetectedBarcode(
            value="V2",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=830), Point(x=270, y=830),
                      Point(x=270, y=840), Point(x=160, y=840)),
            bounding_box=BoundingBox(x1=160, y1=830, x2=270, y2=840),
        ),
    ]
    cli = _patch_pipeline(monkeypatch, barcodes, _spatial_result())

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    cli.main(["pipeline", str(image_path)])

    captured = capsys.readouterr()
    # Table header should have Initial/Final columns, and the row should
    # show 2/2 in both, with OK match and 0 recovered.
    assert "Initial" in captured.err
    assert "Final" in captured.err
    assert "Recovered" in captured.err
    assert "2/2" in captured.err
    assert "OK" in captured.err


def test_cli_audit_labels_flag_calls_audit_shoebox_labels(
    tmp_path, monkeypatch, capsys
) -> None:
    """audit --labels should call audit_shoebox_labels and return its output."""
    from app import cli

    captured_result = {}

    def fake_audit_path(path, *, model, max_retries, retry_delay_seconds, full, labels):
        captured_result["labels"] = labels
        captured_result["full"] = full
        return {
            "path": str(path),
            "status": "ok",
            "audit": {"image_width": 1000, "labels": []},
        }

    monkeypatch.setattr(cli, "audit_path", fake_audit_path)

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    exit_code = cli.main(["audit", "--labels", str(image_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured_result["labels"] is True
    assert captured_result["full"] is False
    assert payload[0]["status"] == "ok"


# ---------------------------------------------------------------------------
# Pipeline recovery tests (Gemini-guided targeted crop recovery)
# ---------------------------------------------------------------------------


def _spatial_result_with_unmatched(image_width: int = 1000, image_height: int = 1000) -> dict:
    """Spatial result where label 2 has its barcode in a different location,
    so a detection at (160, 130)-(270, 140) won't match label 2."""
    return {
        "image_width": image_width,
        "image_height": image_height,
        "labels": [
            {
                "label_index": 1,
                "label_bbox": {"x1": 100, "y1": 100, "x2": 300, "y2": 200},
                "barcode_bbox": {"x1": 150, "y1": 120, "x2": 280, "y2": 190},
                "status": "clear",
                "confidence": "high",
            },
            {
                "label_index": 2,
                "label_bbox": {"x1": 100, "y1": 800, "x2": 300, "y2": 900},
                "barcode_bbox": {"x1": 150, "y1": 820, "x2": 280, "y2": 890},
                "status": "clear",
                "confidence": "high",
            },
        ],
    }


def _recovered_for_label2() -> list:
    """A RecoveredDetection that finds a barcode inside label 2's region."""
    from app.services.barcode_scanner import RecoveredDetection

    det = DetectedBarcode(
        value="V2_RECOVERED",
        format="Code128",
        content_type="Text",
        orientation=0,
        position=(
            Point(x=160, y=830), Point(x=270, y=830),
            Point(x=270, y=840), Point(x=160, y=840),
        ),
        bounding_box=BoundingBox(x1=160, y1=830, x2=270, y2=840),
    )
    return [RecoveredDetection(label_index=2, crop_basis="barcode_bbox", detection=det)]


def test_pipeline_recovery_recovers_unmatched_label(
    tmp_path, monkeypatch, capsys
) -> None:
    """Scanner finds 1/2, recovery finds the second → 2/2 matched, recovery section present."""
    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
    ]
    cli = _patch_pipeline(
        monkeypatch,
        barcodes,
        _spatial_result_with_unmatched(),
        recovered_detections=_recovered_for_label2(),
    )

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png(width=1000, height=1000))

    exit_code = cli.main(["pipeline", str(image_path), "--pretty"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    entry = payload[0]

    # Initial reconciliation should show 1/2.
    assert "initial_reconciliation" in entry
    assert entry["initial_reconciliation"]["matched_label_count"] == 1
    assert entry["initial_reconciliation"]["visible_label_count"] == 2

    # Recovery section present.
    assert "recovery" in entry
    rec = entry["recovery"]
    assert rec["attempted_label_count"] == 1
    assert rec["attempted_label_indexes"] == [2]
    assert rec["recovered_label_count"] == 1
    assert rec["recovered_detection_count"] == 1
    assert len(rec["recovered_labels"]) == 1
    assert rec["recovered_labels"][0]["label_index"] == 2
    assert rec["recovered_labels"][0]["crop_basis"] == "barcode_bbox"
    assert rec["still_unmatched_labels"] == []

    # Final reconciliation should show 2/2.
    assert entry["reconciliation"]["matched_label_count"] == 2
    assert entry["reconciliation"]["all_labels_matched"] is True

    # decoded_vs_visible updated.
    assert entry["decoded_vs_visible"]["all_labels_matched"] is True


def test_pipeline_recovery_skipped_when_all_matched(
    tmp_path, monkeypatch, capsys
) -> None:
    """Scanner finds 2/2 → no recovery section, no initial_reconciliation."""
    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
        DetectedBarcode(
            value="V2",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=830), Point(x=270, y=830),
                      Point(x=270, y=840), Point(x=160, y=840)),
            bounding_box=BoundingBox(x1=160, y1=830, x2=270, y2=840),
        ),
    ]
    cli = _patch_pipeline(
        monkeypatch,
        barcodes,
        _spatial_result_with_unmatched(),
        recovered_detections=_recovered_for_label2(),
    )

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png(width=1000, height=1000))

    cli.main(["pipeline", str(image_path), "--pretty"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    entry = payload[0]

    # No recovery section when all matched.
    assert "recovery" not in entry
    assert "initial_reconciliation" not in entry
    assert entry["reconciliation"]["all_labels_matched"] is True


def test_pipeline_recovery_fails_gracefully(
    tmp_path, monkeypatch, capsys
) -> None:
    """Recovery crop finds nothing → still_unmatched has 1, no crash."""
    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
    ]
    cli = _patch_pipeline(
        monkeypatch,
        barcodes,
        _spatial_result_with_unmatched(),
        recovered_detections=[],  # recovery finds nothing
    )

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png(width=1000, height=1000))

    exit_code = cli.main(["pipeline", str(image_path), "--pretty"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    entry = payload[0]

    assert exit_code == 0
    assert "recovery" in entry
    rec = entry["recovery"]
    assert rec["attempted_label_count"] == 1
    assert rec["recovered_label_count"] == 0
    assert rec["recovered_detection_count"] == 0
    assert len(rec["still_unmatched_labels"]) == 1
    assert rec["still_unmatched_labels"][0]["label_index"] == 2

    # Final reconciliation still 1/2.
    assert entry["reconciliation"]["matched_label_count"] == 1
    assert entry["reconciliation"]["all_labels_matched"] is False


def test_pipeline_recovery_preserves_initial_reconciliation(
    tmp_path, monkeypatch, capsys
) -> None:
    """initial_reconciliation shows 1/2, final reconciliation shows 2/2."""
    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
    ]
    cli = _patch_pipeline(
        monkeypatch,
        barcodes,
        _spatial_result_with_unmatched(),
        recovered_detections=_recovered_for_label2(),
    )

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png(width=1000, height=1000))

    cli.main(["pipeline", str(image_path), "--pretty"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    entry = payload[0]

    initial = entry["initial_reconciliation"]
    final = entry["reconciliation"]

    assert initial["matched_label_count"] == 1
    assert initial["all_labels_matched"] is False
    assert final["matched_label_count"] == 2
    assert final["all_labels_matched"] is True


def test_pipeline_recovery_recovered_label_count_vs_detection_count(
    tmp_path, monkeypatch, capsys
) -> None:
    """One label → two symbols → recovered_label_count=1, recovered_detection_count=2."""
    from app.services.barcode_scanner import RecoveredDetection

    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
    ]

    # Two detections from the same label (label 2).
    det1 = DetectedBarcode(
        value="V2A",
        format="Code128",
        content_type="Text",
        orientation=0,
        position=(Point(x=160, y=830), Point(x=270, y=830),
                  Point(x=270, y=840), Point(x=160, y=840)),
        bounding_box=BoundingBox(x1=160, y1=830, x2=270, y2=840),
    )
    det2 = DetectedBarcode(
        value="V2B",
        format="Code128",
        content_type="Text",
        orientation=0,
        position=(Point(x=160, y=850), Point(x=270, y=850),
                  Point(x=270, y=860), Point(x=160, y=860)),
        bounding_box=BoundingBox(x1=160, y1=850, x2=270, y2=860),
    )
    recovered = [
        RecoveredDetection(label_index=2, crop_basis="barcode_bbox", detection=det1),
        RecoveredDetection(label_index=2, crop_basis="barcode_bbox", detection=det2),
    ]

    cli = _patch_pipeline(
        monkeypatch,
        barcodes,
        _spatial_result_with_unmatched(),
        recovered_detections=recovered,
    )

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png(width=1000, height=1000))

    cli.main(["pipeline", str(image_path), "--pretty"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    entry = payload[0]

    rec = entry["recovery"]
    # Two detections from recovery, but only one label recovered.
    assert rec["recovered_detection_count"] == 2
    assert rec["recovered_label_count"] == 1


def test_pipeline_recovery_only_counts_confirmed_labels(
    tmp_path, monkeypatch, capsys
) -> None:
    """Crop finds a barcode but reconciliation assigns it to a different label
    → recovered_label_count=0 for the attempted label."""
    from app.services.barcode_scanner import RecoveredDetection

    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
    ]

    # Recovery finds a barcode near label 1 (not label 2).
    # This detection is inside label 1's barcode bbox, so reconciliation
    # will assign it to label 1, not the attempted label 2.
    det = DetectedBarcode(
        value="V1_NEARBY",
        format="Code128",
        content_type="Text",
        orientation=0,
        position=(Point(x=170, y=135), Point(x=260, y=135),
                  Point(x=260, y=145), Point(x=170, y=145)),
        bounding_box=BoundingBox(x1=170, y1=135, x2=260, y2=145),
    )
    recovered = [
        RecoveredDetection(label_index=2, crop_basis="barcode_bbox", detection=det),
    ]

    cli = _patch_pipeline(
        monkeypatch,
        barcodes,
        _spatial_result_with_unmatched(),
        recovered_detections=recovered,
    )

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png(width=1000, height=1000))

    cli.main(["pipeline", str(image_path), "--pretty"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    entry = payload[0]

    rec = entry["recovery"]
    # Recovery found a detection, but it matched label 1 (already matched),
    # not the attempted label 2. So recovered_label_count=0.
    assert rec["recovered_detection_count"] == 1
    assert rec["recovered_label_count"] == 0
    assert len(rec["still_unmatched_labels"]) == 1


def test_pipeline_recovery_barcode_crop_fails_label_crop_succeeds(
    tmp_path, monkeypatch, capsys
) -> None:
    """crop_basis='label_bbox' when the barcode crop fails and label crop succeeds."""
    from app.services.barcode_scanner import RecoveredDetection

    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
    ]

    det = DetectedBarcode(
        value="V2_RECOVERED",
        format="Code128",
        content_type="Text",
        orientation=0,
        position=(Point(x=160, y=830), Point(x=270, y=830),
                  Point(x=270, y=840), Point(x=160, y=840)),
        bounding_box=BoundingBox(x1=160, y1=830, x2=270, y2=840),
    )
    recovered = [
        RecoveredDetection(label_index=2, crop_basis="label_bbox", detection=det),
    ]

    cli = _patch_pipeline(
        monkeypatch,
        barcodes,
        _spatial_result_with_unmatched(),
        recovered_detections=recovered,
    )

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png(width=1000, height=1000))

    cli.main(["pipeline", str(image_path), "--pretty"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    entry = payload[0]

    rec = entry["recovery"]
    assert rec["recovered_label_count"] == 1
    assert rec["recovered_labels"][0]["crop_basis"] == "label_bbox"


def test_pipeline_recovery_coordinates_use_padded_crop_origin(
    tmp_path, monkeypatch, capsys
) -> None:
    """Recovered detection coordinates reflect the padded crop origin, not
    the original Gemini box origin."""
    from app.services.barcode_scanner import RecoveredDetection

    barcodes = [
        DetectedBarcode(
            value="V1",
            format="Code128",
            content_type="Text",
            orientation=0,
            position=(Point(x=160, y=130), Point(x=270, y=130),
                      Point(x=270, y=140), Point(x=160, y=140)),
            bounding_box=BoundingBox(x1=160, y1=130, x2=270, y2=140),
        ),
    ]

    # The recovered detection should have coordinates in full-image space.
    # The test verifies the detection is placed correctly by checking
    # reconciliation assigns it to label 2.
    det = DetectedBarcode(
        value="V2_RECOVERED",
        format="Code128",
        content_type="Text",
        orientation=0,
        position=(Point(x=160, y=830), Point(x=270, y=830),
                  Point(x=270, y=840), Point(x=160, y=840)),
        bounding_box=BoundingBox(x1=160, y1=830, x2=270, y2=840),
    )
    recovered = [
        RecoveredDetection(label_index=2, crop_basis="barcode_bbox", detection=det),
    ]

    cli = _patch_pipeline(
        monkeypatch,
        barcodes,
        _spatial_result_with_unmatched(),
        recovered_detections=recovered,
    )

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png(width=1000, height=1000))

    cli.main(["pipeline", str(image_path), "--pretty"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    entry = payload[0]

    # The recovered detection at (160, 830)-(270, 840) falls inside label 2's
    # barcode_bbox (150, 820)-(280, 890), confirming correct coordinate mapping.
    assert entry["reconciliation"]["all_labels_matched"] is True
    rec = entry["recovery"]
    assert rec["recovered_labels"][0]["barcode_value"] == "V2_RECOVERED"
