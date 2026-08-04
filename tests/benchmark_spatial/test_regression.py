"""Regression guards for the spatial Gemini benchmark.

Two tests:

- ``test_spatial_snapshot_baseline`` — replays a captured
  ``snapshots/gemini_responses.json`` through the real scanner + metrics.
  Skips cleanly when snapshots are absent (run
  ``python -m tests.benchmark_spatial.runner --capture-snapshots`` once with
  ``GEMINI_API_KEY`` set to populate them). Assertions are derived from the
  dataset, not hardcoded.

- ``test_live_spatial_benchmark`` — runs the real Gemini API. Marked
  ``live_gemini`` and gated by ``RUN_LIVE_GEMINI=1`` so plain ``pytest`` never
  makes charged API calls. Run explicitly via::

      RUN_LIVE_GEMINI=1 pytest -m live_gemini

Per-label spatial assertions (spatial recall, barcode localization, exact
rectangles, ``false_label_regions``) are NOT included yet — they become active
only after per-label ground-truth boxes are manually reviewed and frozen.
"""

from __future__ import annotations

import os

import pytest

from tests.benchmark_spatial.runner import (
    SNAPSHOTS_PATH,
    load_dataset,
    run_live_benchmark,
    run_snapshot_benchmark,
)


def test_spatial_snapshot_baseline() -> None:
    """Replay captured Gemini snapshots through the scanner + metrics.

    Skips when snapshots are absent. Assertions are derived from the dataset
    so adding an image does not require editing this test.
    """
    if not SNAPSHOTS_PATH.exists():
        pytest.skip(
            f"Snapshot file absent: {SNAPSHOTS_PATH}. Run "
            f"`python -m tests.benchmark_spatial.runner --capture-snapshots` "
            f"once with GEMINI_API_KEY set to populate it."
        )

    results = run_snapshot_benchmark()
    agg = results.aggregate
    dataset = load_dataset()

    # Dataset-derived totals (no hardcoded numbers).
    assert agg.image_count == len(dataset.images)
    assert agg.expected_visible_labels == sum(
        image.expected_visible_label_count for image in dataset.images
    )
    assert agg.expected_unmatched_labels == sum(
        image.expected_unmatched_label_count
        for image in dataset.images
        if image.expected_unmatched_label_count is not None
    )

    # Frozen baseline behavior.
    # Gemini currently returns 16/16 labels for fuzzy_16_labels after the
    # count was corrected from 17 to 16. All 9 images now match their
    # expected visible label counts.
    assert agg.label_count_correct_images == 9
    assert agg.label_count_accuracy == 1.0
    assert agg.unmatched_label_accuracy == 1.0
    assert agg.extra_labels == 0


@pytest.mark.live_gemini
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI") != "1",
    reason="Set RUN_LIVE_GEMINI=1 to run charged Gemini tests.",
)
def test_live_spatial_benchmark() -> None:
    """Live Gemini run with soft acceptance thresholds.

    Excluded from ordinary ``pytest`` by the ``skipif`` guard. Run explicitly::

        RUN_LIVE_GEMINI=1 pytest -m live_gemini
    """
    results = run_live_benchmark(runs_per_image=3)
    agg = results.aggregate

    assert agg.label_count_accuracy >= 0.95
    assert agg.unmatched_label_accuracy == 1.0
    assert agg.extra_labels <= 1
