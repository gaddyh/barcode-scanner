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

    def fake_audit_path(path, *, model, max_retries, retry_delay_seconds, full):
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
