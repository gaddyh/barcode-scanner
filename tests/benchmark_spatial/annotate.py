"""Annotation helper for freezing per-label ground-truth boxes.

Workflow::

    python -m tests.benchmark_spatial.annotate draft <image>
        # Runs Gemini, writes a draft annotations/<image>.json with
        # reviewed=false, renders preview/<image>.png.

    # Edit annotations/<image>.json by hand: move/add/delete boxes.

    python -m tests.benchmark_spatial.annotate review <image>
        # Re-renders preview/<image>.png from the edited JSON so you can
        # verify your corrections.

    python -m tests.benchmark_spatial.annotate review <image> --approve
        # Sets reviewed=true in the annotation file.

    python -m tests.benchmark_spatial.annotate freeze <image>
        # Copies the approved labels into dataset.json. Refuses to freeze
        # unreviewed annotations or annotations whose dimensions / coordinate
        # space do not match the source image and dataset.

Boxes are edited by hand in the JSON file; the preview PNG makes the edits
verifiable. True drag-and-drop interactivity would require matplotlib/Qt and is
intentionally out of scope for this first version.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.services.gemini_box_audit import audit_shoebox_labels, load_normalized_image
from tests.benchmark_spatial.models import COORDINATE_SPACE, GroundTruthImage, SpatialDataset
from tests.benchmark_spatial.runner import DATASET_PATH, SAMPLES_DIR, load_dataset

BENCH_DIR = Path(__file__).resolve().parent
ANNOTATIONS_DIR = BENCH_DIR / "annotations"
PREVIEW_DIR = BENCH_DIR / "preview"


# ---------------------------------------------------------------------------
# Annotation file model (plain dict + helpers; kept lightweight)
# ---------------------------------------------------------------------------


def _annotation_path(image: str) -> Path:
    return ANNOTATIONS_DIR / f"{image}.json"


def _preview_path(image: str) -> Path:
    return PREVIEW_DIR / f"{image}.png"


def _load_annotation(image: str) -> dict[str, Any]:
    path = _annotation_path(image)
    if not path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {path}. Run `annotate draft {image}` first."
        )
    return json.loads(path.read_text())


def _save_annotation(annotation: dict[str, Any], image: str) -> None:
    path = _annotation_path(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotation, indent=2))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _try_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_boxes(
    image: Image.Image,
    labels: list[dict[str, Any]],
) -> Image.Image:
    """Draw label boxes (green) and barcode boxes (blue) with index numbers."""
    draw = ImageDraw.Draw(image)
    font = _try_font(max(16, min(32, image.width // 80)))

    for label in labels:
        label_bbox = label.get("label_bbox")
        if label_bbox is not None:
            box = (label_bbox["x1"], label_bbox["y1"], label_bbox["x2"], label_bbox["y2"])
            draw.rectangle(box, outline="green", width=max(2, image.width // 600))
            idx_text = str(label.get("label_index", "?"))
            draw.text(
                (box[0] + 4, box[1] + 2), idx_text, fill="green", font=font
            )

        barcode_bbox = label.get("barcode_bbox")
        if barcode_bbox is not None:
            box = (barcode_bbox["x1"], barcode_bbox["y1"], barcode_bbox["x2"], barcode_bbox["y2"])
            draw.rectangle(box, outline="blue", width=max(2, image.width // 600))

    return image


def _render_preview(image_name: str, annotation: dict[str, Any]) -> Path:
    image_path = SAMPLES_DIR / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        annotated = _draw_boxes(rgb, annotation.get("labels", []))
        preview_path = _preview_path(image_name)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(preview_path, format="PNG")
    return preview_path


# ---------------------------------------------------------------------------
# draft
# ---------------------------------------------------------------------------


def _normalize_image_arg(image: str) -> str:
    """Strip a leading samples/ prefix if the user included it."""
    prefix = "samples/"
    return image[len(prefix):] if image.startswith(prefix) else image


def _find_gt_image(dataset: SpatialDataset, image: str) -> GroundTruthImage | None:
    image = _normalize_image_arg(image)
    return next((i for i in dataset.images if i.image == image), None)


def _print_available_images(dataset: SpatialDataset) -> None:
    print("Available images in dataset.json:", file=sys.stderr)
    for img in dataset.images:
        print(f"  {img.image}", file=sys.stderr)


def cmd_draft(args: argparse.Namespace) -> int:
    image = _normalize_image_arg(args.image)
    dataset = load_dataset()
    gt_image = _find_gt_image(dataset, image)
    if gt_image is None:
        print(f"ERROR: image '{image}' not found in dataset.json", file=sys.stderr)
        _print_available_images(dataset)
        return 1

    source_path = SAMPLES_DIR / gt_image.image
    if not source_path.exists():
        print(f"ERROR: source image not found: {source_path}", file=sys.stderr)
        return 1

    print(f"Running Gemini spatial audit on {source_path.name} ...", file=sys.stderr)
    spatial = audit_shoebox_labels(source_path, model=args.model)

    normalized = load_normalized_image(source_path)
    labels_out: list[dict[str, Any]] = []
    for lab in spatial.labels:
        labels_out.append({
            "label_index": lab.label_index,
            "label_bbox": lab.label_bbox.model_dump(),
            "barcode_bbox": (
                lab.barcode_bbox.model_dump() if lab.barcode_bbox is not None else None
            ),
            "status": lab.status.value,
            "confidence": lab.confidence.value,
        })

    annotation = {
        "image": image,
        "coordinate_space": COORDINATE_SPACE,
        "image_width": normalized.original_width,
        "image_height": normalized.original_height,
        "reviewed": False,
        "labels": labels_out,
    }
    _save_annotation(annotation, image)

    preview_path = _render_preview(image, annotation)
    print(f"Draft annotation: {_annotation_path(image)}", file=sys.stderr)
    print(f"Preview:          {preview_path}", file=sys.stderr)
    print(
        "Edit the JSON by hand, then run `annotate review "
        f"{image} --approve` and `annotate freeze {image}`.",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


def cmd_review(args: argparse.Namespace) -> int:
    image = _normalize_image_arg(args.image)
    annotation = _load_annotation(image)

    if args.approve:
        annotation["reviewed"] = True
        _save_annotation(annotation, image)
        print(f"Marked {image} as reviewed=true.", file=sys.stderr)

    preview_path = _render_preview(image, annotation)
    print(f"Preview: {preview_path}", file=sys.stderr)
    if not annotation.get("reviewed"):
        print(
            "Annotation is still reviewed=false. "
            f"Run `annotate review {image} --approve` when satisfied.",
            file=sys.stderr,
        )
    return 0


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    image = _normalize_image_arg(args.image)
    annotation = _load_annotation(image)

    # Safety check 1: must be reviewed.
    if not annotation.get("reviewed"):
        raise ValueError(
            f"Annotation for {image} must be approved before freezing. "
            f"Run `annotate review {image} --approve`."
        )

    # Safety check 2: both dimensions must match the image.
    image_path = SAMPLES_DIR / image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    normalized = load_normalized_image(image_path)
    if (
        annotation["image_width"] != normalized.original_width
        or annotation["image_height"] != normalized.original_height
    ):
        raise ValueError(
            f"Annotation dimensions do not match the image: "
            f"annotation={annotation['image_width']}x{annotation['image_height']} "
            f"source={normalized.original_width}x{normalized.original_height}"
        )

    # Safety check 3: coordinate space must match the dataset.
    dataset = load_dataset()
    if annotation.get("coordinate_space") != dataset.coordinate_space:
        raise ValueError(
            f"Annotation coordinate space does not match the dataset: "
            f"annotation={annotation.get('coordinate_space')!r} "
            f"dataset={dataset.coordinate_space!r}"
        )

    # Convert annotation labels into GroundTruthLabel records.
    gt_image = next((i for i in dataset.images if i.image == image), None)
    if gt_image is None:
        raise ValueError(f"Image '{image}' not found in dataset.json")

    frozen_labels: list[dict[str, Any]] = []
    for lab in annotation.get("labels", []):
        label_bbox = lab.get("label_bbox")
        barcode_bbox = lab.get("barcode_bbox")
        frozen_labels.append({
            "label_id": lab.get("label_id") or f"l{lab['label_index']}",
            "row": lab.get("row", 1),
            "column": lab.get("column", lab["label_index"]),
            "label_bbox": label_bbox,
            "barcode_bbox": barcode_bbox,
            "expected_scanner_status": lab.get("expected_scanner_status", "decoded"),
            "expected_unmatched": lab.get("expected_unmatched", False),
            "visible_metadata": lab.get("visible_metadata"),
        })

    # Update dataset.json in place.
    raw = json.loads(DATASET_PATH.read_text())
    for entry in raw["images"]:
        if entry["image"] == image:
            entry["labels"] = frozen_labels
            break
    DATASET_PATH.write_text(json.dumps(raw, indent=2))
    print(
        f"Froze {len(frozen_labels)} label(s) for {image} into {DATASET_PATH}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotate",
        description="Annotation helper for spatial benchmark ground-truth boxes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    draft = sub.add_parser("draft", help="Generate a draft annotation from Gemini.")
    draft.add_argument("image", help="Benchmark image name (e.g. clean_12_labels.jpeg).")
    draft.add_argument("--model", default=None, help="Gemini model name.")
    draft.set_defaults(func=cmd_draft)

    review = sub.add_parser("review", help="Re-render the preview from the edited JSON.")
    review.add_argument("image", help="Benchmark image name.")
    review.add_argument(
        "--approve",
        action="store_true",
        help="Set reviewed=true in the annotation file.",
    )
    review.set_defaults(func=cmd_review)

    freeze = sub.add_parser("freeze", help="Copy approved labels into dataset.json.")
    freeze.add_argument("image", help="Benchmark image name.")
    freeze.set_defaults(func=cmd_freeze)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
