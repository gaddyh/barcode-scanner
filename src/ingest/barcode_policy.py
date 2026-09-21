"""Product-level barcode contract — filter raw scanner detections to
primary shoebox barcodes.

The deterministic scanner (`BarcodeScanner`) is intentionally generic: it
decodes whatever barcodes it can find (Code128, EAN-13, UPC-A, etc.) and
returns all of them. The product, however, only cares about **primary
shoebox barcodes** — the EAN-13 product identifier printed on the
shoebox label.

This module defines `PrimaryShoeboxBarcodePolicy`, which:

- Accepts only 13-digit EAN-13 values (rejects UPC-A-12, Code128, etc.).
- Validates the EAN-13 checksum (mod-10).
- Filters raw scanner detections to those that pass the policy.
- Records instrumentation: how many raw detections were seen, how many
  matched the policy, and how many were rejected.

The policy is a separate class so the scanner stays generic and reusable
for non-shoebox use cases. The filter is applied in the graph/pipeline
layer, NOT inside the scanner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.ingest.scanner import DetectedBarcode


class BarcodePolicy(Protocol):
    """Protocol for barcode filtering policies."""

    def is_primary(self, value: str) -> bool: ...

    def filter(self, detections: list[DetectedBarcode]) -> list[DetectedBarcode]: ...


@dataclass(frozen=True)
class PrimaryShoeboxBarcodePolicy:
    """Filter raw scanner detections to primary shoebox EAN-13 barcodes.

    A primary shoebox barcode is a 13-digit EAN-13 with a valid mod-10
    checksum. This is the product-level identifier printed on shoebox
    labels. Other detections (Code128 shipping codes, UPC-A, partial
    reads, noise) are rejected.

    The policy also records instrumentation counters for matched/expected
    and matched/found analysis.
    """

    require_checksum: bool = True
    require_length: int = 13

    # Instrumentation: counts from the most recent `filter()` call.
    # Not part of equality/hashing — use `field(compare=False)`.
    last_raw_count: int = field(default=0, compare=False)
    last_matched_count: int = field(default=0, compare=False)
    last_rejected_count: int = field(default=0, compare=False)

    def is_primary(self, value: str) -> bool:
        """Return True if `value` is a valid primary shoebox barcode.

        Checks:
        1. Exactly `require_length` digits (default 13 for EAN-13).
        2. All characters are digits.
        3. (Optional, default on) EAN-13 mod-10 checksum is valid.
        """
        if len(value) != self.require_length:
            return False
        if not value.isdigit():
            return False
        if self.require_checksum and not _ean13_checksum_valid(value):
            return False
        return True

    def filter(self, detections: list[DetectedBarcode]) -> list[DetectedBarcode]:
        """Filter raw scanner detections to primary shoebox barcodes.

        Updates instrumentation counters (`last_raw_count`,
        `last_matched_count`, `last_rejected_count`).
        """
        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "last_raw_count", len(detections))
        matched = [d for d in detections if self.is_primary(d.value)]
        object.__setattr__(self, "last_matched_count", len(matched))
        object.__setattr__(self, "last_rejected_count", len(detections) - len(matched))
        return matched


def _ean13_checksum_valid(value: str) -> bool:
    """Validate an EAN-13 mod-10 checksum.

    EAN-13 checksum algorithm:
    1. Sum digits in odd positions (1-indexed: positions 1, 3, 5, ... 11).
    2. Sum digits in even positions (positions 2, 4, 6, ... 12) and multiply by 3.
    3. Total = odd_sum + 3 * even_sum.
    4. Checksum digit = (10 - (total % 10)) % 10.
    5. Valid if checksum digit == the 13th digit.
    """
    if len(value) != 13 or not value.isdigit():
        return False

    digits = [int(c) for c in value]
    # First 12 digits: positions 1..12 (0-indexed: 0..11).
    # Odd positions (1-indexed): indices 0, 2, 4, 6, 8, 10.
    # Even positions (1-indexed): indices 1, 3, 5, 7, 9, 11.
    odd_sum = sum(digits[i] for i in range(0, 12, 2))
    even_sum = sum(digits[i] for i in range(1, 12, 2))
    total = odd_sum + 3 * even_sum
    checksum = (10 - (total % 10)) % 10
    return checksum == digits[12]
