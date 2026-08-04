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
from PIL import Image, ImageOps, UnidentifiedImageError

from app.services.barcode_scanner import (
    BarcodeScanner,
    BoundingBox,
    DetectedBarcode,
    LabelCropRequest,
)
from app.services.gemini_box_audit import (
    DEFAULT_COUNTS_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_RETRY_DELAY_SECONDS,
    ShoeboxAuditError,
    audit_shoebox_counts,
    audit_shoebox_image,
    audit_shoebox_labels,
)
from app.services.spatial_reconciliation import (
    RecoveredLabel,
    RecoveryResult,
    UnmatchedLabel,
    match_scanner_to_labels,
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


def _dict_to_detected_barcode(d: dict) -> DetectedBarcode:
    """Reconstruct a DetectedBarcode from its dict form (asdict output)."""
    from app.services.barcode_scanner import Point as ScanPoint

    box = d["bounding_box"]
    position = tuple(
        ScanPoint(x=p["x"], y=p["y"]) for p in d.get("position", [])
    )
    return DetectedBarcode(
        value=d["value"],
        format=d["format"],
        content_type=d["content_type"],
        orientation=d["orientation"],
        position=position,
        bounding_box=BoundingBox(
            x1=box["x1"], y1=box["y1"], x2=box["x2"], y2=box["y2"]
        ),
    )


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
    labels: bool,
) -> dict[str, object]:
    try:
        if full:
            result = audit_shoebox_image(
                path,
                model=model,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
        elif labels:
            result = audit_shoebox_labels(
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
                labels=args.labels,
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
                labels=args.labels,
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
    """Traced wrapper for the Gemini spatial label audit."""
    try:
        spatial = audit_shoebox_labels(
            path,
            model=model,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "error": {"type": "FileNotFoundError", "message": str(exc)},
        }
    except (ValueError, ShoeboxAuditError) as exc:
        return {
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {"status": "ok", "spatial": spatial.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# Gemini-guided recovery helpers
# ---------------------------------------------------------------------------


def _pad_bbox(
    box: dict,
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> BoundingBox:
    """Pad a bbox dict by a ratio and clamp to image bounds."""
    width = box["x2"] - box["x1"]
    height = box["y2"] - box["y1"]
    pad_x = round(width * padding_ratio)
    pad_y = round(height * padding_ratio)
    return BoundingBox(
        x1=max(0, box["x1"] - pad_x),
        y1=max(0, box["y1"] - pad_y),
        x2=min(image_width, box["x2"] + pad_x),
        y2=min(image_height, box["y2"] + pad_y),
    )


def build_recovery_requests(
    unmatched_labels: list[UnmatchedLabel],
    *,
    image_width: int,
    image_height: int,
) -> list[LabelCropRequest]:
    """Build crop requests from unmatched Gemini labels.

    For each unmatched label, the barcode_bbox is padded by 25% (tight, first
    attempt) and the label_bbox by 10% (wider, fallback).  When barcode_bbox is
    None, only the label crop is used.
    """
    requests: list[LabelCropRequest] = []
    for unmatched in unmatched_labels:
        barcode_dict = unmatched.barcode_bbox
        label_dict = unmatched.label_bbox

        if barcode_dict is not None:
            barcode_crop = _pad_bbox(
                barcode_dict,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=0.25,
            )
            # Exact (unpadded) barcode bbox for high-scale attempts.
            exact_barcode_crop = BoundingBox(
                x1=max(0, barcode_dict["x1"]),
                y1=max(0, barcode_dict["y1"]),
                x2=min(image_width, barcode_dict["x2"]),
                y2=min(image_height, barcode_dict["y2"]),
            )
        else:
            # No barcode bbox — use the label bbox as the primary crop.
            barcode_crop = _pad_bbox(
                label_dict,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=0.25,
            )
            exact_barcode_crop = BoundingBox(
                x1=max(0, label_dict["x1"]),
                y1=max(0, label_dict["y1"]),
                x2=min(image_width, label_dict["x2"]),
                y2=min(image_height, label_dict["y2"]),
            )

        label_crop = _pad_bbox(
            label_dict,
            image_width=image_width,
            image_height=image_height,
            padding_ratio=0.10,
        )

        requests.append(
            LabelCropRequest(
                label_index=unmatched.label_index,
                barcode_crop=barcode_crop,
                exact_barcode_crop=exact_barcode_crop,
                label_crop=label_crop,
            )
        )
    return requests


def _build_recovery_result(
    *,
    attempted_indexes: list[int],
    recovered_detections: list,  # list[RecoveredDetection]
    merged_detections: list[dict],  # list of detection dicts (asdict output)
    final_reconciliation,  # SpatialReconciliation
    image_width: int,
    image_height: int,
) -> RecoveryResult:
    """Build RecoveryResult from the final reconciliation.

    A label is only counted as recovered when the final reconciliation assigns
    a recovered detection to that attempted label.
    """
    attempted_set = set(attempted_indexes)

    # Map recovered detection identity → (label_index, crop_basis).
    # We match by value + bbox since the merged list may have reordered.
    recovered_by_key: dict[tuple[str, int, int, int, int], tuple[int, str]] = {}
    for rec in recovered_detections:
        box = rec.detection.bounding_box
        key = (rec.detection.value, box.x1, box.y1, box.x2, box.y2)
        recovered_by_key[key] = (rec.label_index, rec.crop_basis)

    # Find which final matches correspond to recovered detections on attempted labels.
    recovered_labels: list[RecoveredLabel] = []
    for match in final_reconciliation.matches:
        if match.label_index not in attempted_set:
            continue
        det = merged_detections[match.scanner_detection_index]
        det_box = det["bounding_box"]
        key = (det["value"], det_box["x1"], det_box["y1"], det_box["x2"], det_box["y2"])
        if key not in recovered_by_key:
            continue
        orig_label_index, crop_basis = recovered_by_key[key]
        recovered_labels.append(
            RecoveredLabel(
                label_index=match.label_index,
                barcode_value=match.barcode_value,
                scanner_detection_index=match.scanner_detection_index,
                crop_basis=crop_basis,
                crop_box=det_box,
            )
        )

    recovered_label_indexes = {rl.label_index for rl in recovered_labels}
    still_unmatched = [
        ul
        for ul in final_reconciliation.unmatched_labels
        if ul.label_index in attempted_set
    ]

    return RecoveryResult(
        attempted_label_count=len(attempted_indexes),
        attempted_label_indexes=attempted_indexes,
        recovered_labels=recovered_labels,
        recovered_label_count=len(recovered_label_indexes),
        recovered_detection_count=len(recovered_detections),
        still_unmatched_labels=still_unmatched,
    )


@traceable(run_type="chain", name="pipeline")
def pipeline_path(
    path: Path,
    scanner: BarcodeScanner,
    *,
    model: str | None,
    max_retries: int,
    retry_delay_seconds: float,
    recovery_debug_dir: Path | None = None,
) -> dict[str, object]:
    """Run deterministic scan and Gemini audit in parallel, return combined summary.

    When ``recovery_debug_dir`` is set and recovery is triggered, crop images
    and preprocessing variants are saved as PNGs for debugging.
    """

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

    barcodes: list[dict] = []
    if scan_ok:
        barcodes = scan_result.get("barcodes", [])  # type: ignore[assignment]
        values = [b["value"] for b in barcodes]  # type: ignore[index]
        summary["decoded_count"] = scan_result.get("count", 0)
        summary["unique_values"] = sorted(set(values))
        summary["unique_value_count"] = len(set(values))
        # Preserve full scanner detections (with bounding boxes) for spatial
        # reconciliation, debugging, visualization, and targeted crop retries.
        summary["scanner_detections"] = barcodes

    if audit_ok:
        spatial = audit_result.get("spatial", {})
        labels = spatial.get("labels", [])
        summary["visible_labels"] = len(labels)
        summary["clear_labels"] = sum(
            1 for label in labels if label.get("status") == "clear"
        )
        summary["gemini_labels"] = labels

    if scan_ok and audit_ok:
        spatial = audit_result.get("spatial", {})
        labels = spatial.get("labels", [])
        image_width = spatial["image_width"]
        image_height = spatial["image_height"]

        # Initial reconciliation.
        initial_reconciliation = match_scanner_to_labels(
            barcodes,
            labels,
            image_width=image_width,
            image_height=image_height,
        )

        # Gemini-guided recovery for unmatched labels.
        if initial_reconciliation.unmatched_labels:
            summary["initial_reconciliation"] = initial_reconciliation.model_dump(
                mode="json"
            )

            # Load the full-resolution image for cropping.
            try:
                pil_image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except (OSError, ValueError) as exc:
                summary["recovery_error"] = {"code": "image_load_failed", "message": str(exc)}
                pil_image = None

            if pil_image is not None:
                requests = build_recovery_requests(
                    initial_reconciliation.unmatched_labels,
                    image_width=image_width,
                    image_height=image_height,
                )
                attempted_indexes = [r.label_index for r in requests]

                recovered = scanner.scan_label_crops(
                    pil_image,
                    requests,
                    existing_detections=[
                        # Reconstruct DetectedBarcode-like objects for the
                        # existing-detection check.  We only need bounding boxes
                        # and values, so we use the scanner's own type.
                        _dict_to_detected_barcode(b) for b in barcodes
                    ],
                    debug_dir=recovery_debug_dir,
                )

                if recovered:
                    recovered_detections = [r.detection for r in recovered]
                    merged = scanner.merge_detections(
                        [_dict_to_detected_barcode(b) for b in barcodes],
                        recovered_detections,
                    )
                    merged_dicts = [asdict(d) for d in merged]

                    final_reconciliation = match_scanner_to_labels(
                        merged_dicts,
                        labels,
                        image_width=image_width,
                        image_height=image_height,
                    )

                    recovery_result = _build_recovery_result(
                        attempted_indexes=attempted_indexes,
                        recovered_detections=recovered,
                        merged_detections=merged_dicts,
                        final_reconciliation=final_reconciliation,
                        image_width=image_width,
                        image_height=image_height,
                    )

                    summary["recovery"] = recovery_result.model_dump(mode="json")
                    summary["reconciliation"] = final_reconciliation.model_dump(
                        mode="json"
                    )
                    summary["scanner_detections"] = merged_dicts
                    barcodes = merged_dicts
                    summary["decoded_count"] = len(merged_dicts)
                    values = [d["value"] for d in merged_dicts]
                    summary["unique_values"] = sorted(set(values))
                    summary["unique_value_count"] = len(set(values))

                    reconciliation = final_reconciliation
                else:
                    # Recovery attempted but found nothing.
                    recovery_result = _build_recovery_result(
                        attempted_indexes=attempted_indexes,
                        recovered_detections=[],
                        merged_detections=barcodes,
                        final_reconciliation=initial_reconciliation,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    summary["recovery"] = recovery_result.model_dump(mode="json")
                    summary["reconciliation"] = initial_reconciliation.model_dump(
                        mode="json"
                    )
                    reconciliation = initial_reconciliation
            else:
                summary["reconciliation"] = initial_reconciliation.model_dump(
                    mode="json"
                )
                reconciliation = initial_reconciliation
        else:
            # No unmatched labels — no recovery needed (backward compatible).
            summary["reconciliation"] = initial_reconciliation.model_dump(
                mode="json"
            )
            reconciliation = initial_reconciliation

        # Backward-compatible decoded-vs-visible summary for table rendering.
        decoded = summary.get("decoded_count", 0)
        visible = summary.get("visible_labels", 0)
        summary["decoded_vs_visible"] = {
            "decoded": decoded,
            "visible": visible,
            "match": decoded == visible,
            "difference": decoded - visible,
            "matched_labels": reconciliation.matched_label_count,
            "all_labels_matched": reconciliation.all_labels_matched,
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

    Each row: (image, matched/visible, match, time).
    """
    header = (
        f"{'Image':<32} {'Matched/Visible':>16} {'Match':>6} {'Time':>7}"
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for image, mv, match, elapsed in rows:
        time_str = f"{elapsed:.2f}s" if elapsed is not None else "-"
        print(
            f"{image:<32} {mv:>16} {match:>6} {time_str:>7}",
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
            recovery_debug_dir=args.recovery_debug,
        )
        elapsed = time.perf_counter() - t0
        results.append(result)

        if args.time:
            _print_timing(path.name, elapsed)

        # Build table row — prefer reconciliation matched/visible when available.
        dv = result.get("decoded_vs_visible", {})
        if "matched_labels" in dv:
            matched = dv.get("matched_labels", "-")
            visible = dv.get("visible", "-")
            all_matched = dv.get("all_labels_matched")
            if all_matched is True:
                match_str = "OK"
            elif all_matched is False:
                match_str = "DIFF"
            else:
                match_str = "ERR"
            mv_str = f"{matched}/{visible}" if matched != "-" else "-/-"
        else:
            match_str = "ERR"
            mv_str = "-/-"
        table_rows.append((path.name, mv_str, match_str, elapsed))

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
    audit_parser.add_argument(
        "--labels",
        action="store_true",
        help=(
            "Run the spatial label audit: locate every product label and its "
            "barcode region in pixel coordinates. Middle ground between the "
            "fast counts audit and the full detailed audit."
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
    pipeline_parser.add_argument(
        "--recovery-debug",
        type=Path,
        default=None,
        help=(
            "Save recovery crops and preprocessing variants as PNGs to this "
            "directory for debugging. Only fires when recovery is triggered "
            "(unmatched labels exist)."
        ),
    )
    pipeline_parser.set_defaults(func=_run_pipeline)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
