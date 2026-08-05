from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from app.services.barcode_scanner import BarcodeScanner
from app.services.gemini_box_audit import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_RETRY_DELAY_SECONDS,
    ShoeboxAuditError,
    audit_shoebox_image,
    audit_shoebox_labels,
)
from app.services.pipeline import (
    pipeline_path,
    scan_path,
)

# ---------------------------------------------------------------------------
# CLI presentation helpers
# ---------------------------------------------------------------------------


def _print_timing(label: str, seconds: float) -> None:
    print(f"{label:<40} {seconds:.2f}s", file=sys.stderr)


def _print_scan_table(
    rows: list[tuple[str, str, int, float | None]],
) -> None:
    """Print a rich summary table to stderr.

    Each row: (image, status, count, time).
    """
    header = (
        f"{'Image':<32} {'Status':>10} {'Barcodes':>9} {'Time':>7}"
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for image, status, count, elapsed in rows:
        time_str = f"{elapsed:.2f}s" if elapsed is not None else "-"
        print(
            f"{image:<32} {status:>10} {count:>9} {time_str:>7}",
            file=sys.stderr,
        )

    print("-" * len(header), file=sys.stderr)


def _print_audit_table(
    rows: list[tuple[str, str, str, str, float | None]],
) -> None:
    """Print a rich summary table to stderr.

    Each row: (image, status, visible, clear, time).
    """
    header = (
        f"{'Image':<32} {'Status':>10} {'Visible':>8} {'Clear':>7} {'Time':>7}"
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for image, status, visible, clear, elapsed in rows:
        time_str = f"{elapsed:.2f}s" if elapsed is not None else "-"
        print(
            f"{image:<32} {status:>10} {visible:>8} {clear:>7} {time_str:>7}",
            file=sys.stderr,
        )

    print("-" * len(header), file=sys.stderr)


# ---------------------------------------------------------------------------
# scan subcommand
# ---------------------------------------------------------------------------


def _run_scan(args: argparse.Namespace) -> int:
    scanner = BarcodeScanner()

    results: list[dict[str, object]] = []
    table_rows: list[tuple[str, str, int, float | None]] = []

    for path in args.images:
        t0 = time.perf_counter()
        result = scan_path(path, scanner)
        elapsed = time.perf_counter() - t0
        results.append(result)

        if args.time:
            _print_timing(path.name, elapsed)

        status = str(result.get("status", "error"))
        count = int(result.get("count", 0))
        table_rows.append((path.name, status, count, elapsed if args.time else None))

    _print_scan_table(table_rows)

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
            result = audit_shoebox_labels(
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
    table_rows: list[tuple[str, str, str, str, float | None]] = []

    for path in args.images:
        t0 = time.perf_counter()
        result = audit_path(
            path,
            model=args.model,
            max_retries=args.max_retries,
            retry_delay_seconds=DEFAULT_RETRY_DELAY_SECONDS,
            full=args.full,
        )
        elapsed = time.perf_counter() - t0
        results.append(result)

        if args.time:
            _print_timing(path.name, elapsed)

        status = str(result.get("status", "error"))
        audit_data = result.get("audit", {})  # type: ignore[union-attr]
        labels = audit_data.get("labels", [])
        if labels:
            visible = str(len(labels))
            clear = str(sum(1 for lbl in labels if lbl.get("status") == "clear"))
        else:
            visible = str(audit_data.get("visible_product_barcode_label_count", "-"))
            clear = str(audit_data.get("clear_product_barcode_label_count", "-"))
        table_rows.append((path.name, status, visible, clear, elapsed if args.time else None))

    _print_audit_table(table_rows)

    print(json.dumps(results, indent=2 if args.pretty else None))

    return 1 if any(result.get("status") == "error" for result in results) else 0


# ---------------------------------------------------------------------------
# pipeline subcommand — run scan + audit in parallel, return combined summary
# ---------------------------------------------------------------------------


def _print_pipeline_table(
    rows: list[tuple[str, str, str, int, float | None]],
) -> None:
    """Print a rich summary table to stderr.

    Each row: (image, matched, match, decoded, time).
    """
    header = (
        f"{'Image':<32} {'Matched':>9} {'Match':>6} {'Decoded':>8} {'Time':>7}"
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for image, matched, match, decoded, elapsed in rows:
        time_str = f"{elapsed:.2f}s" if elapsed is not None else "-"
        print(
            f"{image:<32} {matched:>9} {match:>6} {decoded:>8} {time_str:>7}",
            file=sys.stderr,
        )

    print("-" * len(header), file=sys.stderr)


def _run_pipeline(args: argparse.Namespace) -> int:
    scanner = BarcodeScanner()

    results: list[dict[str, object]] = []
    table_rows: list[tuple[str, str, str, int, float | None]] = []

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

        # Build table row with matched/visible, match status, decoded count.
        dv = result.get("decoded_vs_visible", {})
        visible = dv.get("visible", "-")
        matched = dv.get("matched_labels", "-")
        all_matched = dv.get("all_labels_matched")
        if all_matched is True:
            match_str = "OK"
        elif all_matched is False:
            match_str = "DIFF"
        else:
            match_str = "ERR"

        decoded = result.get("decoded_count", 0)
        matched_str = f"{matched}/{visible}" if visible != "-" else "-/-"
        table_rows.append(
            (path.name, matched_str, match_str, decoded,
             elapsed if args.time else None)
        )

    _print_pipeline_table(table_rows)

    print(json.dumps(results, indent=2 if args.pretty else None))

    return 1 if any(not result.get("ok") for result in results) else 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="barcode-scan",
        description=(
            "Scan barcodes and audit shoebox images directly from image files "
            "(no HTTP server)."
        ),
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
            f"{DEFAULT_MODEL!r}."
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
            "observations). Much slower; default is the spatial label audit "
            "used by the pipeline."
        ),
    )
    audit_parser.set_defaults(func=_run_audit)

    # pipeline -----------------------------------------------------------
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run scan + Gemini audit in parallel and return a combined summary.",
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
            f"{DEFAULT_MODEL!r}."
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
