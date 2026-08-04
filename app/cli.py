from __future__ import annotations

import argparse
import contextvars
import json
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path

from dotenv import load_dotenv
from PIL import UnidentifiedImageError

from app.services.barcode_scanner import BarcodeScanner
from app.services.gemini_box_audit import (
    DEFAULT_COUNTS_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_RETRY_DELAY_SECONDS,
    ShoeboxAuditError,
    audit_shoebox_counts,
    audit_shoebox_image,
)

# Load .env early so LANGSMITH_* vars are set before langsmith is imported.
load_dotenv()

# langsmith is optional — tracing is enabled when LANGSMITH_TRACING=true.
_TRACING = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
if _TRACING:
    from langsmith import traceable
else:
    # no-op decorator fallback when tracing is disabled.
    def traceable(*args, **kwargs):  # type: ignore[misc]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _to_jsonable(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _print_timing(label: str, seconds: float) -> None:
    print(f"{label:<40} {seconds:.2f}s", file=sys.stderr)


# ---------------------------------------------------------------------------
# scan subcommand
# ---------------------------------------------------------------------------


@traceable(run_type="tool", name="barcode_scan")
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


def _run_scan(args: argparse.Namespace) -> int:
    scanner = BarcodeScanner()

    results: list[dict[str, object]] = []
    for path in args.images:
        if args.time:
            t0 = time.perf_counter()
            result = scan_path(path, scanner)
            elapsed = time.perf_counter() - t0
            _print_timing(path.name, elapsed)
        else:
            result = scan_path(path, scanner)
        results.append(result)

    print(json.dumps(results, indent=2 if args.pretty else None))

    return 1 if any(result.get("status") == "error" for result in results) else 0


# ---------------------------------------------------------------------------
# audit subcommand
# ---------------------------------------------------------------------------


def audit_path(
    path: Path,
    *,
    model: str | None,
    max_retries: int,
    retry_delay_seconds: float,
    full: bool,
) -> dict[str, object]:
    try:
        if full:
            result = audit_shoebox_image(
                path,
                model=model,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
        else:
            result = audit_shoebox_counts(
                path,
                model=model,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
    except FileNotFoundError as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": {"code": "unreadable_file", "message": str(exc)},
        }
    except (ValueError, ShoeboxAuditError) as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": {"code": "audit_failed", "message": str(exc)},
        }

    return {
        "path": str(path),
        "status": "ok",
        "audit": result.model_dump(mode="json"),
    }


def _run_audit(args: argparse.Namespace) -> int:
    results: list[dict[str, object]] = []
    for path in args.images:
        if args.time:
            t0 = time.perf_counter()
            result = audit_path(
                path,
                model=args.model,
                max_retries=args.max_retries,
                retry_delay_seconds=DEFAULT_RETRY_DELAY_SECONDS,
                full=args.full,
            )
            elapsed = time.perf_counter() - t0
            _print_timing(path.name, elapsed)
        else:
            result = audit_path(
                path,
                model=args.model,
                max_retries=args.max_retries,
                retry_delay_seconds=DEFAULT_RETRY_DELAY_SECONDS,
                full=args.full,
            )
        results.append(result)

    print(json.dumps(results, indent=2 if args.pretty else None))

    return 1 if any(result.get("status") == "error" for result in results) else 0


# ---------------------------------------------------------------------------
# pipeline subcommand — run scan + audit in parallel, return combined summary
# ---------------------------------------------------------------------------


@traceable(run_type="tool", name="gemini_audit")
def _traced_audit(
    path: Path,
    *,
    model: str | None,
    max_retries: int,
    retry_delay_seconds: float,
) -> dict[str, object]:
    """Traced wrapper for the Gemini counts audit."""
    try:
        counts = audit_shoebox_counts(
            path,
            model=model,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "error": {"code": "unreadable_file", "message": str(exc)},
        }
    except (ValueError, ShoeboxAuditError) as exc:
        return {
            "status": "error",
            "error": {"code": "audit_failed", "message": str(exc)},
        }
    return {"status": "ok", "counts": counts.model_dump(mode="json")}


@traceable(run_type="chain", name="pipeline")
def pipeline_path(
    path: Path,
    scanner: BarcodeScanner,
    *,
    model: str | None,
    max_retries: int,
    retry_delay_seconds: float,
) -> dict[str, object]:
    """Run deterministic scan and Gemini audit in parallel, return combined summary."""

    def _do_scan() -> dict[str, object]:
        return scan_path(path, scanner)

    def _do_audit() -> dict[str, object]:
        return _traced_audit(
            path,
            model=model,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )

    # Copy the current context so langsmith trace context propagates to threads.
    # Each thread needs its own copy — a single Context cannot be entered twice.
    scan_ctx = contextvars.copy_context()
    audit_ctx = contextvars.copy_context()

    with ThreadPoolExecutor(max_workers=2) as pool:
        scan_future = pool.submit(scan_ctx.run, _do_scan)
        audit_future = pool.submit(audit_ctx.run, _do_audit)
        scan_result = scan_future.result()
        audit_result = audit_future.result()

    # Build summary reconciliation
    scan_ok = scan_result.get("status") in ("found", "not_found")
    audit_ok = audit_result.get("status") == "ok"

    summary: dict[str, object] = {
        "path": str(path),
        "scan_status": scan_result.get("status"),
        "audit_status": audit_result.get("status"),
    }

    if scan_ok:
        barcodes = scan_result.get("barcodes", [])
        values = [b["value"] for b in barcodes]  # type: ignore[index]
        summary["decoded_count"] = scan_result.get("count", 0)
        summary["unique_values"] = sorted(set(values))
        summary["unique_value_count"] = len(set(values))

    if audit_ok:
        counts = audit_result.get("counts", {})
        summary["visible_labels"] = counts.get("visible_product_barcode_label_count")
        summary["clear_labels"] = counts.get("clear_product_barcode_label_count")
        summary["boxes_without_label"] = counts.get("boxes_without_visible_product_barcode")
        summary["partially_obscured"] = counts.get("partially_obscured_product_barcode_count")

    if scan_ok and audit_ok:
        decoded = summary.get("decoded_count", 0)
        visible = summary.get("visible_labels", 0)
        summary["decoded_vs_visible"] = {
            "decoded": decoded,
            "visible": visible,
            "match": decoded == visible,
            "difference": decoded - visible,
        }

    if not scan_ok:
        summary["scan_error"] = scan_result.get("error", {})
    if not audit_ok:
        summary["audit_error"] = audit_result.get("error", {})

    summary["ok"] = scan_ok and audit_ok
    return summary


def _print_pipeline_table(
    rows: list[tuple[str, str, str, float | None]],
) -> None:
    """Print a rich summary table to stderr.

    Each row: (image, decoded/visible, match, time).
    """
    header = (
        f"{'Image':<32} {'Decoded/Visible':>16} {'Match':>6} {'Time':>7}"
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for image, dv, match, elapsed in rows:
        time_str = f"{elapsed:.2f}s" if elapsed is not None else "-"
        print(
            f"{image:<32} {dv:>16} {match:>6} {time_str:>7}",
            file=sys.stderr,
        )

    print("-" * len(header), file=sys.stderr)


def _run_pipeline(args: argparse.Namespace) -> int:
    scanner = BarcodeScanner()

    results: list[dict[str, object]] = []
    table_rows: list[tuple[str, str, str, float | None]] = []

    for path in args.images:
        t0 = time.perf_counter()
        result = pipeline_path(
            path,
            scanner,
            model=args.model,
            max_retries=args.max_retries,
            retry_delay_seconds=DEFAULT_RETRY_DELAY_SECONDS,
        )
        elapsed = time.perf_counter() - t0
        results.append(result)

        if args.time:
            _print_timing(path.name, elapsed)

        # Build table row
        dv = result.get("decoded_vs_visible", {})
        decoded = dv.get("decoded", "-")
        visible = dv.get("visible", "-")
        match = dv.get("match")
        if match is True:
            match_str = "OK"
        elif match is False:
            match_str = "DIFF"
        else:
            match_str = "ERR"
        dv_str = f"{decoded}/{visible}" if decoded != "-" else "-/-"
        table_rows.append((path.name, dv_str, match_str, elapsed))

    _print_pipeline_table(table_rows)

    print(json.dumps(results, indent=2 if args.pretty else None))

    return 1 if any(not result.get("ok") for result in results) else 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="barcode-scan",
        description="Scan barcodes and audit shoebox images directly from image files (no HTTP server).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan ---------------------------------------------------------------
    scan_parser = subparsers.add_parser(
        "scan",
        help="Deterministic barcode scan of one or more images.",
    )
    scan_parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Path(s) to image file(s) to scan (JPEG, PNG, or WebP).",
    )
    scan_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output with indentation.",
    )
    scan_parser.add_argument(
        "--time",
        action="store_true",
        help="Print per-image wall-clock timing to stderr.",
    )
    scan_parser.set_defaults(func=_run_scan)

    # audit --------------------------------------------------------------
    audit_parser = subparsers.add_parser(
        "audit",
        help="Gemini visual audit of shoebox images (structured JSON).",
    )
    audit_parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Path(s) to image file(s) to audit (JPEG, PNG, WebP, HEIC, or HEIF).",
    )
    audit_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output with indentation.",
    )
    audit_parser.add_argument(
        "--time",
        action="store_true",
        help="Print per-image wall-clock timing to stderr.",
    )
    audit_parser.add_argument(
        "--model",
        default=None,
        help=(
            "Gemini model name. Defaults to GEMINI_MODEL, then "
            f"{DEFAULT_COUNTS_MODEL!r} (fast) or {DEFAULT_MODEL!r} (--full)."
        ),
    )
    audit_parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Retries after the first request. Default: {DEFAULT_MAX_RETRIES}.",
    )
    audit_parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run the full detailed audit (bounding boxes, OCR text, per-box "
            "observations). Much slower; default is the fast counts-only audit."
        ),
    )
    audit_parser.set_defaults(func=_run_audit)

    # pipeline -----------------------------------------------------------
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run scan + audit in parallel and return a combined summary.",
    )
    pipeline_parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Path(s) to image file(s) to process.",
    )
    pipeline_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output with indentation.",
    )
    pipeline_parser.add_argument(
        "--time",
        action="store_true",
        help="Print per-image wall-clock timing to stderr.",
    )
    pipeline_parser.add_argument(
        "--model",
        default=None,
        help=(
            "Gemini model for the audit step. Defaults to GEMINI_MODEL or "
            f"{DEFAULT_COUNTS_MODEL!r}."
        ),
    )
    pipeline_parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Retries after the first request. Default: {DEFAULT_MAX_RETRIES}.",
    )
    pipeline_parser.set_defaults(func=_run_pipeline)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
