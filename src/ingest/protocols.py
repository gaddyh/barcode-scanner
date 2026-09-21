"""Protocols (ports) for the ingest pipeline.

These protocols define the boundaries between the pipeline orchestration
and the concrete scanner/recovery implementations. The existing concrete
``BarcodeScanner`` in ``src/ingest/scanner.py`` conforms structurally to
``BarcodeScannerPort`` — no rename or subclassing needed.

The protocols are intentionally named with a ``Port`` suffix to avoid
clashing with the concrete ``BarcodeScanner`` class used throughout the
codebase (routes, tests, pipeline).
"""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from src.ingest.scanner import DetectedBarcode


class BarcodeScannerPort(Protocol):
    """Scanner port — the minimal interface the pipeline depends on.

    The concrete ``BarcodeScanner`` in ``src/ingest/scanner.py`` conforms
    to this protocol structurally. Adapters for candidate scanners (e.g.
    the naot-poc enhanced scanner) implement this protocol so they can be
    A/B tested without touching the pipeline.
    """

    def scan_bytes(self, image_bytes: bytes) -> list[DetectedBarcode]:
        """Scan raw image bytes and return all detected barcodes."""
        ...

    def scan_image(self, image: Image.Image) -> list[DetectedBarcode]:
        """Scan a PIL Image and return all detected barcodes."""
        ...


class TargetedBarcodeRecovery(Protocol):
    """Targeted recovery port — crop-level scanning for Gemini-guided recovery.

    The concrete ``BarcodeScanner`` implements this via
    ``scan_crop_with_recovery``. Candidate scanners may provide a
    diagnostics variant (``scan_crop_with_recovery_diagnostics``) that
    returns per-attempt timing and values for A/B analysis.
    """

    def scan_crop_with_recovery(
        self,
        crop: Image.Image,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[DetectedBarcode]:
        """Aggressively scan a crop for barcodes missed by the fast path."""
        ...
