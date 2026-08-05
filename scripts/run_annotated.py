"""Run analyze_image() on a sample image and save the annotated preview.

Usage:
    python scripts/run_annotated.py samples/multi_clear_6_boxes.jpeg
    python scripts/run_annotated.py samples/multi_clear_6_boxes.jpeg --outdir output

Saves the annotated PNG (red circles around missing regions) to
<outdir>/<image-stem>_annotated.png when outcome == needs_better_photo.
Also dumps the full JSON result to <outdir>/<image-stem>_result.json.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from app.services.analyze import analyze_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the image to analyze.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output"),
        help="Where to write the annotated PNG + JSON result (default: output/).",
    )
    args = parser.parse_args()

    image_path: Path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 1

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    print(f"Analyzing {image_path.name} ...", file=sys.stderr)
    result = analyze_image(image_path)

    # Always dump the full JSON result.
    json_path = outdir / f"{stem}_result.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Result JSON:  {json_path}", file=sys.stderr)
    print(f"  outcome:           {result['outcome']}", file=sys.stderr)
    print(f"  found_count:       {result['summary']['found_count']}", file=sys.stderr)
    print(f"  missing_count:     {result['summary']['missing_count']}", file=sys.stderr)
    print(f"  unassigned_count:  {result['summary']['unassigned_count']}", file=sys.stderr)

    b64 = result.get("annotated_image_b64")
    if b64:
        png_path = outdir / f"{stem}_annotated.png"
        png_path.write_bytes(base64.b64decode(b64))
        print(f"Annotated PNG: {png_path}", file=sys.stderr)
        print(
            f"  dimensions: {result['annotated_image_width']}x"
            f"{result['annotated_image_height']}",
            file=sys.stderr,
        )
        print(f"  message: {result.get('message')}", file=sys.stderr)
    else:
        print(
            "No annotated image (outcome is not needs_better_photo).",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
