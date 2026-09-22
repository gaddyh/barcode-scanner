#!/usr/bin/env python3
"""Generate Gemini audit cache entries for synthetic test images.

Since the synthetic images have known label positions, we can generate
the audit cache entries programmatically without calling Gemini.
"""
import json
from pathlib import Path

from PIL import Image

from src.evals.gemini_cache import GeminiAuditCache, _cache_key
from src.ingest.vision import DEFAULT_MODEL

SYNTH_DIR = Path("samples/synthetic")
CACHE_PATH = Path("tests/eval/gemini_audit_cache.json")
DEFAULT_MODEL_STR = DEFAULT_MODEL


def build_spatial_entry(image_path: Path, label_count: int, blur_indices: set[int] | None = None):
    """Build a spatial audit entry for a synthetic grid image.

    The grid layout matches generate_test_images.py:
    - 4 columns, variable rows
    - padding=60, label_w=bc_img.width+80, label_h=bc_img.height+80
    """
    blur_indices = blur_indices or set()
    img = Image.open(image_path)
    img_w, img_h = img.size

    # Match the layout from generate_test_images.py
    # bc_img is rendered with module_width=0.5, module_height=50
    # Each label is bc_img.width+80 x bc_img.height+80
    # Grid: 4 cols, padding=60
    # We need to compute label positions based on the actual image size

    cols = 4
    rows = (label_count + cols - 1) // cols
    padding = 60
    label_w = (img_w - (cols + 1) * padding) // cols
    label_h = (img_h - (rows + 1) * padding) // rows

    labels = []
    for i in range(label_count):
        r = i // cols
        c = i % cols
        x1 = padding + c * (label_w + padding)
        y1 = padding + r * (label_h + padding)
        x2 = x1 + label_w
        y2 = y1 + label_h

        # Barcode region is inside the label (40px margin from label edges)
        bc_x1 = x1 + 40
        bc_y1 = y1 + 30
        bc_x2 = x2 - 40
        bc_y2 = y1 + 30 + int((label_h - 80) * 0.6)  # barcode takes ~60% of label height

        is_blurred = i in blur_indices
        labels.append({
            "label_index": i + 1,
            "label_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "barcode_bbox": {"x1": bc_x1, "y1": bc_y1, "x2": bc_x2, "y2": bc_y2},
            "confidence": "high",
            "status": "blurred" if is_blurred else "clear",
        })

    return {
        "image_height": img_h,
        "image_width": img_w,
        "labels": labels,
    }


def main():
    cache = GeminiAuditCache(CACHE_PATH)
    cache._load()

    entries = [
        ("photo1_12_9found.png", 12, {9, 10, 11}),
        ("retry_exact_3.png", 3, set()),
        ("retry_less_1.png", 1, set()),
        ("retry_more_5.png", 5, set()),
        ("agg_photo1_6.png", 6, set()),
        ("agg_photo2_5.png", 5, set()),
    ]

    for name, count, blur in entries:
        path = SYNTH_DIR / name
        if not path.exists():
            print(f"  SKIP {name} (not found)")
            continue
        spatial = build_spatial_entry(path, count, blur)
        key = _cache_key(path, DEFAULT_MODEL_STR)
        cache._cache[key] = spatial
        print(f"  {name}: {count} labels ({len(blur)} blurred) → key={key}")

    # Save the cache
    with CACHE_PATH.open("w") as f:
        json.dump(cache._cache, f, indent=2)
    print(f"Cache saved: {len(cache._cache)} entries → {CACHE_PATH}")


if __name__ == "__main__":
    main()
