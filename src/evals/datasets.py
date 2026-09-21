"""Ground-truth dataset loading for offline evaluation.

The canonical dataset lives at ``tests/eval/barcode_baseline.json`` and
references images in ``samples/``. It is the deduplicated union of the
original barcode-scanner dataset and the naot-poc ground truth.

Schema (one case per image)::

    {
      "name": "barcode_baseline",
      "cases": [
        {
          "id": "multi_12_clean",
          "inputs": {"image": "samples/multi_12_clean.jpeg"},
          "reference_outputs": {"barcodes": ["7297501154056", ...]},
          "metadata": {
            "ground_truth_status": "verified",
            "expected_box_count": 12,
            "exclude_from_eval": false,
            ...
          }
        }
      ]
    }

``reference_outputs.barcodes`` is a **multiset** (list). Duplicate values
represent separate physical boxes and count separately in occurrence-level
metrics.

The legacy ``tests/eval/dataset.json`` is still loaded by
``load_legacy_dataset()`` for backward compatibility with the live
LangSmith eval harness.
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
CANONICAL_DATASET_PATH = _REPO_ROOT / "tests" / "eval" / "barcode_baseline.json"
LEGACY_DATASET_PATH = _REPO_ROOT / "tests" / "eval" / "dataset.json"
# Backward-compat alias — points to the legacy dataset (still used by the
# live LangSmith eval harness via load_legacy_dataset()).
DATASET_PATH = LEGACY_DATASET_PATH
SAMPLES_DIR = _REPO_ROOT / "samples"


# ---------------------------------------------------------------------------
# Canonical dataset (barcode_baseline.json — multiset semantics)
# ---------------------------------------------------------------------------


def load_dataset() -> list[dict[str, Any]]:
    """Load the canonical ground-truth dataset.

    Returns one dict per scorable case (cases with
    ``metadata.exclude_from_eval != True``). Each dict has::

        id, image_name, image_path, expected_barcodes (list, multiset)

    The ``expected_barcodes`` list preserves duplicates — two identical
    values mean two physical boxes and count as two occurrences.
    """
    with CANONICAL_DATASET_PATH.open() as f:
        data = json.load(f)

    examples: list[dict[str, Any]] = []
    for case in data["cases"]:
        meta = case.get("metadata", {})
        if meta.get("exclude_from_eval", False):
            logger.info("Skipping excluded case: %s", case["id"])
            continue

        image_rel = case["inputs"]["image"]
        # Strip "samples/" prefix if present — we resolve against SAMPLES_DIR.
        image_name = image_rel.split("/", 1)[-1] if image_rel.startswith("samples/") else image_rel
        image_path = SAMPLES_DIR / image_name
        if not image_path.exists():
            logger.warning("Sample image missing, skipping: %s", image_path)
            continue

        barcodes = case["reference_outputs"]["barcodes"]
        examples.append({
            "id": case["id"],
            "image_name": image_name,
            "image_path": str(image_path),
            "expected_barcodes": list(barcodes),
            "expected_count": len(barcodes),
            "expected_unique_count": len(set(barcodes)),
            "expected_outcome": "complete" if barcodes else "needs_user_input",
        })
    return examples


# ---------------------------------------------------------------------------
# Legacy dataset (dataset.json — set semantics, for live LangSmith eval)
# ---------------------------------------------------------------------------


def load_legacy_dataset() -> list[dict[str, Any]]:
    """Load the legacy dataset for the live LangSmith eval harness.

    Each example has the original schema with ``expected_values`` (list)
    and ``expected_unique_values`` (sorted set) so the existing
    LangSmith evaluators continue to work unchanged.
    """
    with LEGACY_DATASET_PATH.open() as f:
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
