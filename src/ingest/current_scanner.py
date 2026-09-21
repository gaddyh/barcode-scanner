"""Current scanner adapter — thin wrapper around the existing BarcodeScanner.

This adapter exists so the A/B comparison script can treat both scanners
uniformly via the ``BarcodeScannerPort`` protocol. The wrapper adds no
behavior change — it delegates directly to ``BarcodeScanner``.
"""

from __future__ import annotations

from PIL import Image

from src.ingest.scanner import BarcodeScanner, DetectedBarcode


class CurrentScanner:
    """Adapter wrapping the existing ``BarcodeScanner`` (scanner-0.8).

    Conforms to ``BarcodeScannerPort``. No behavior change — delegates
    directly to the concrete scanner.
    """

    VERSION = "scanner-0.8"

    def __init__(self) -> None:
        self._scanner = BarcodeScanner()

    def scan_bytes(self, image_bytes: bytes) -> list[DetectedBarcode]:
        return self._scanner.scan_bytes(image_bytes)

    def scan_image(self, image: Image.Image) -> list[DetectedBarcode]:
        return self._scanner.scan_image(image)
