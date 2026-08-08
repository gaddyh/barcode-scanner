"""Ground-truth dataset loading for offline evaluation.

Moved from ``tests/eval/runner.py``. Same dataset format, same logic.
The dataset lives at ``tests/eval/dataset.json`` and references images
in ``samples/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Paths are resolved relative to the repo root, not this module.
# This keeps the dataset location stable regardless of where the
# eval runner is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = _REPO_ROOT / "tests" / "eval" / "dataset.json"
SAMPLES_DIR = _REPO_ROOT / "samples"


def load_dataset() -> list[dict[str, Any]]:
    """Load the ground-truth dataset and resolve each image to an absolute path.

    Each example in the returned list has:
        image_name, image_path, expected_barcode_symbol_count,
        expected_decoded_count, expected_values, expected_unique_values,
        expected_unique_count, expected_outcome
    """
    with DATASET_PATH.open() as f:
        data = json.load(f)

    examples: list[dict[str, Any]] = []
    for img_entry in data["images"]:
        image_name = img_entry["image"]
        image_path = SAMPLES_DIR / image_name
        if not image_path.exists():
            logger.warning("Sample image missing, skipping: %s", image_path)
            continue

        decoded_boxes = [b for b in img_entry["boxes"] if b.get("status") == "decoded"]
        expected_values = [b["value"] for b in decoded_boxes if b.get("value")]
        expected_unique = sorted(set(expected_values))

        examples.append({
            "image_name": image_name,
            "image_path": str(image_path),
            "expected_barcode_symbol_count": img_entry["expected_barcode_symbol_count"],
            "expected_decoded_count": len(decoded_boxes),
            "expected_values": expected_values,
            "expected_unique_values": expected_unique,
            "expected_unique_count": len(expected_unique),
            "expected_outcome": "complete",
        })
    return examples
