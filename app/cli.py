from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path

from PIL import UnidentifiedImageError

from app.services.barcode_scanner import BarcodeScanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="barcode-scan",
        description="Scan barcodes directly from one or more image files (no HTTP server).",
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Path(s) to image file(s) to scan (JPEG, PNG, or WebP).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output with indentation.",
    )
    return parser


def _to_jsonable(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def scan_path(path: Path, scanner: BarcodeScanner) -> dict[str, object]:
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": {"code": "unreadable_file", "message": str(exc)},
        }

    try:
        barcodes = scanner.scan_bytes(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": {"code": "invalid_image", "message": str(exc)},
        }

    return {
        "path": str(path),
        "status": "found" if barcodes else "not_found",
        "count": len(barcodes),
        "barcodes": [_to_jsonable(b) for b in barcodes],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scanner = BarcodeScanner()

    results = [scan_path(path, scanner) for path in args.images]
    print(json.dumps(results, indent=2 if args.pretty else None))

    return 1 if any(result.get("status") == "error" for result in results) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
