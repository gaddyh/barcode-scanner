"""Tests for the PrimaryShoeboxBarcodePolicy — product barcode contract.

The policy filters raw scanner detections to primary shoebox EAN-13
barcodes (13 digits, valid mod-10 checksum). Non-primary detections
(Code128, UPC-A, partial reads, noise) are rejected.
"""

from __future__ import annotations

import pytest

from src.ingest.barcode_policy import (
    PrimaryShoeboxBarcodePolicy,
    _ean13_checksum_valid,
)
from src.ingest.scanner import BoundingBox, DetectedBarcode, Point


def _det(value: str) -> DetectedBarcode:
    """Build a minimal DetectedBarcode for testing."""
    return DetectedBarcode(
        value=value,
        format="EAN-13",
        content_type="text",
        orientation=0,
        position=(Point(x=0, y=0),),
        bounding_box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
    )


# ---------------------------------------------------------------------------
# EAN-13 checksum validation
# ---------------------------------------------------------------------------


class TestEan13Checksum:
    def test_valid_ean13(self) -> None:
        # 7297501098442 is a real Israeli EAN-13 from the eval dataset.
        assert _ean13_checksum_valid("7297501098442")

    def test_valid_ean13_another(self) -> None:
        assert _ean13_checksum_valid("7297500243423")

    def test_invalid_checksum(self) -> None:
        # Change the last digit — checksum should fail.
        assert not _ean13_checksum_valid("7297501098443")

    def test_wrong_length(self) -> None:
        assert not _ean13_checksum_valid("729750109844")  # 12 digits
        assert not _ean13_checksum_valid("72975010984422")  # 14 digits

    def test_non_digit(self) -> None:
        assert not _ean13_checksum_valid("729750109844A")

    def test_empty(self) -> None:
        assert not _ean13_checksum_valid("")

    def test_all_zeros(self) -> None:
        # 0000000000000 — checksum of 000000000000 is 0, so this is valid.
        assert _ean13_checksum_valid("0000000000000")


# ---------------------------------------------------------------------------
# is_primary
# ---------------------------------------------------------------------------


class TestIsPrimary:
    def setup_method(self) -> None:
        self.policy = PrimaryShoeboxBarcodePolicy()

    def test_valid_ean13(self) -> None:
        assert self.policy.is_primary("7297501098442")

    def test_invalid_checksum_rejected(self) -> None:
        assert not self.policy.is_primary("7297501098443")

    def test_12_digit_upc_rejected(self) -> None:
        assert not self.policy.is_primary("123456789012")

    def test_14_digit_rejected(self) -> None:
        assert not self.policy.is_primary("12345678901234")

    def test_code128_rejected(self) -> None:
        assert not self.policy.is_primary("ABC123DEF456")

    def test_partial_digits_rejected(self) -> None:
        assert not self.policy.is_primary("12345")

    def test_empty_rejected(self) -> None:
        assert not self.policy.is_primary("")

    def test_non_digit_rejected(self) -> None:
        assert not self.policy.is_primary("729750109844A")

    def test_disable_checksum(self) -> None:
        policy = PrimaryShoeboxBarcodePolicy(require_checksum=False)
        # Wrong checksum but right length + all digits → accepted.
        assert policy.is_primary("7297501098443")
        # Still rejects wrong length.
        assert not policy.is_primary("12345")


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


class TestFilter:
    def test_filters_non_primary(self) -> None:
        policy = PrimaryShoeboxBarcodePolicy()
        detections = [
            _det("7297501098442"),  # valid EAN-13
            _det("ABC123DEF456"),   # Code128 — rejected
            _det("12345"),          # partial — rejected
            _det("7297500243423"),  # valid EAN-13
        ]
        result = policy.filter(detections)
        assert len(result) == 2
        assert result[0].value == "7297501098442"
        assert result[1].value == "7297500243423"

    def test_empty_input(self) -> None:
        policy = PrimaryShoeboxBarcodePolicy()
        result = policy.filter([])
        assert result == []

    def test_all_valid(self) -> None:
        policy = PrimaryShoeboxBarcodePolicy()
        detections = [
            _det("7297501098442"),
            _det("7297500243423"),
        ]
        result = policy.filter(detections)
        assert len(result) == 2

    def test_all_rejected(self) -> None:
        policy = PrimaryShoeboxBarcodePolicy()
        detections = [
            _det("ABC123"),
            _det("12345"),
        ]
        result = policy.filter(detections)
        assert result == []

    def test_duplicate_eans_preserved(self) -> None:
        """Duplicate EAN-13 values are separate physical occurrences —
        the policy does NOT deduplicate by value."""
        policy = PrimaryShoeboxBarcodePolicy()
        detections = [
            _det("7297500243423"),
            _det("7297500243423"),  # same EAN, different physical box
        ]
        result = policy.filter(detections)
        assert len(result) == 2  # both kept — multiset semantics


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


class TestInstrumentation:
    def test_counts_recorded(self) -> None:
        policy = PrimaryShoeboxBarcodePolicy()
        detections = [
            _det("7297501098442"),  # valid
            _det("ABC123"),         # rejected
            _det("12345"),          # rejected
            _det("7297500243423"),  # valid
        ]
        policy.filter(detections)
        assert policy.last_raw_count == 4
        assert policy.last_matched_count == 2
        assert policy.last_rejected_count == 2

    def test_empty_counts(self) -> None:
        policy = PrimaryShoeboxBarcodePolicy()
        policy.filter([])
        assert policy.last_raw_count == 0
        assert policy.last_matched_count == 0
        assert policy.last_rejected_count == 0

    def test_all_valid_counts(self) -> None:
        policy = PrimaryShoeboxBarcodePolicy()
        policy.filter([_det("7297501098442"), _det("7297500243423")])
        assert policy.last_raw_count == 2
        assert policy.last_matched_count == 2
        assert policy.last_rejected_count == 0


# ---------------------------------------------------------------------------
# Graph integration — policy filters detections in _reconcile_node
# ---------------------------------------------------------------------------


class TestGraphIntegration:
    """Verify the policy is applied in the graph's reconcile node."""

    @pytest.mark.asyncio
    async def test_reconcile_filters_non_primary(self) -> None:
        """When a barcode_policy is in the config, non-primary detections
        are filtered before reconciliation."""
        from src.ingest.graph import _reconcile_node

        # One valid EAN-13 inside a label, one Code128 outside all labels.
        spatial = {
            "image_width": 800,
            "image_height": 600,
            "labels": [
                {
                    "label_index": 1,
                    "label_bbox": {"x1": 50, "y1": 50, "x2": 250, "y2": 350},
                    "barcode_bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 300},
                },
            ],
        }
        det_valid = {
            "value": "7297501098442",
            "format": "EAN-13",
            "content_type": "text",
            "orientation": 0,
            "position": [{"x": 110, "y": 110}, {"x": 190, "y": 110}],
            "bounding_box": {"x1": 110, "y1": 110, "x2": 190, "y2": 290},
        }
        det_code128 = {
            "value": "SHIP123456",
            "format": "Code128",
            "content_type": "text",
            "orientation": 0,
            "position": [{"x": 700, "y": 500}, {"x": 780, "y": 500}],
            "bounding_box": {"x1": 700, "y1": 500, "x2": 780, "y2": 580},
        }
        state = {
            "scan_ok": True,
            "audit_ok": True,
            "barcodes": [det_valid, det_code128],
            "audit_result": {"status": "ok", "spatial": spatial},
        }
        policy = PrimaryShoeboxBarcodePolicy()
        config = {"configurable": {"barcode_policy": policy}}

        out = await _reconcile_node(state, config)
        # The Code128 detection was filtered; only the EAN-13 was reconciled.
        barcodes = out.get("barcodes", [])
        assert len(barcodes) == 1
        assert barcodes[0]["value"] == "7297501098442"

    @pytest.mark.asyncio
    async def test_reconcile_no_policy_passes_all(self) -> None:
        """Without a barcode_policy in config, all detections pass through."""
        from src.ingest.graph import _reconcile_node

        spatial = {
            "image_width": 800,
            "image_height": 600,
            "labels": [
                {
                    "label_index": 1,
                    "label_bbox": {"x1": 50, "y1": 50, "x2": 250, "y2": 350},
                    "barcode_bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 300},
                },
            ],
        }
        det_valid = {
            "value": "7297501098442",
            "format": "EAN-13",
            "content_type": "text",
            "orientation": 0,
            "position": [{"x": 110, "y": 110}, {"x": 190, "y": 110}],
            "bounding_box": {"x1": 110, "y1": 110, "x2": 190, "y2": 290},
        }
        det_code128 = {
            "value": "SHIP123456",
            "format": "Code128",
            "content_type": "text",
            "orientation": 0,
            "position": [{"x": 700, "y": 500}, {"x": 780, "y": 500}],
            "bounding_box": {"x1": 700, "y1": 500, "x2": 780, "y2": 580},
        }
        state = {
            "scan_ok": True,
            "audit_ok": True,
            "barcodes": [det_valid, det_code128],
            "audit_result": {"status": "ok", "spatial": spatial},
        }
        config = {"configurable": {}}

        out = await _reconcile_node(state, config)
        # No policy → both detections pass through.
        barcodes = out.get("barcodes", [])
        assert len(barcodes) == 2
