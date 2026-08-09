"""CLI for the new runtime-based ingest pipeline.

Run ingest_one() on an image with LangSmith tracing enabled and print
the typed IngestResult. The trace tree (ingest_one → pipeline → scan /
audit / recovery) appears in LangSmith under the configured project.

Usage::

    python -m src.cli ./samples/multi_clear_6_boxes.jpeg
    python -m src.cli ./samples/multi_clear_6_boxes.jpeg --pretty
    python -m src.cli ./samples/multi_clear_6_boxes.jpeg --time

Requires GEMINI_API_KEY for the full pipeline (Gemini audit).
Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to see traces.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from app.models.upload import generate_upload_id
from src.evals.annotation_sink import register_annotation_sink
from src.ingest import IngestStatus, ingest_one
from src.observability import is_tracing
from src.runtime import RunContext, execute

load_dotenv()


def _print_result(result, *, pretty: bool, elapsed: float | None) -> None:
    """Print the IngestResult as JSON to stdout."""
    data = result.model_dump(mode="json")
    if pretty:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(json.dumps(data, default=str))

    # Summary to stderr.
    print("", file=sys.stderr)
    print(f"  status:     {result.status.value}", file=sys.stderr)
    print(f"  items:      {len(result.items)}", file=sys.stderr)
    print(f"  missing:    {len(result.missing)}", file=sys.stderr)
    print(f"  unassigned: {len(result.unassigned)}", file=sys.stderr)
    print(f"  issues:     {len(result.issues)}", file=sys.stderr)
    for issue in result.issues:
        print(
            f"    [{issue.severity}] {issue.code}: {issue.message}",
            file=sys.stderr,
        )
    print(f"  metrics:    scanner={result.metrics.scanner_count} "
          f"vision={result.metrics.vision_count} "
          f"recovery={'yes' if result.metrics.recovery_attempted else 'no'}",
          file=sys.stderr)
    if elapsed is not None:
        print(f"  wall time:  {elapsed:.2f}s", file=sys.stderr)
    if result.annotated_image_b64:
        print(f"  annotated:  {result.annotated_image_width}x{result.annotated_image_height} PNG",
              file=sys.stderr)
    print("", file=sys.stderr)


async def _run(args: argparse.Namespace) -> int:
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        return 1

    ctx = RunContext(
        run_id=str(uuid4()),
        session_id=generate_upload_id(),
        user_id=None,
        source="cli",
        metadata={"filename": image_path.name},
    )

    print(f"run_id:      {ctx.run_id}", file=sys.stderr)
    print(f"session_id:  {ctx.session_id}", file=sys.stderr)
    print(f"source:      {ctx.source}", file=sys.stderr)
    print(f"tracing:     {'enabled' if is_tracing() else 'disabled'}", file=sys.stderr)
    print(f"image:       {image_path}", file=sys.stderr)
    print("", file=sys.stderr)

    # Register the annotation sink so interesting failures are captured.
    register_annotation_sink()

    t0 = time.perf_counter()
    try:
        result = await execute(
            ingest_one,
            image_path,
            ctx,
            name="ingest_one",
            tags=["barcode-scanner", "cli"],
            image_ref=str(image_path),
        )

    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        _flush_tracing()
        return 1

    elapsed = time.perf_counter() - t0
    _print_result(result, pretty=args.pretty, elapsed=elapsed)

    # Flush LangSmith's background upload thread before the process exits.
    # Without this, traces + metadata are lost when the interpreter shuts down.
    _flush_tracing()

    return 0 if result.status == IngestStatus.COMPLETE else 1


def _flush_tracing(timeout: float = 10.0) -> None:
    """Flush the LangSmith client so traces are uploaded before exit."""
    if not is_tracing():
        return
    try:
        from langsmith import Client
        client = Client()
        client.flush(timeout=timeout)
    except Exception as exc:
        print(f"  (trace flush failed: {exc})", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Run ingest_one() with LangSmith tracing and print the typed result.",
    )
    parser.add_argument(
        "image",
        help="Path to image file (JPEG, PNG, or WebP).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="Print wall-clock timing to stderr.",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
