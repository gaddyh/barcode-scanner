"""Backward-compat shim — re-exports from src.evals.

The eval harness has moved to ``src/evals/``. This module keeps the old
import paths working for existing scripts and tests that patch
``SAMPLES_DIR`` on ``tests.eval.runner``.

New code should import from ``src.evals`` directly.
"""

from __future__ import annotations

import sys

from src.evals.datasets import DATASET_PATH, SAMPLES_DIR, load_dataset
from src.evals.evaluators import (
    aggregate_thresholds,
    count_exact,
    latency,
    outcome_correct,
    recovery_gain,
    value_precision,
    value_recall,
)
from src.evals.runner import run_eval

# Expose SAMPLES_DIR as a module-level attribute so tests can monkeypatch it.
# The import above binds it, but we make it explicit for clarity.
# Note: tests that patch tests.eval.runner.SAMPLES_DIR will affect this
# module's reference, but src.evals.datasets.SAMPLES_DIR is the canonical
# one. The dataset loader reads from src.evals.datasets.SAMPLES_DIR, so
# tests need to patch that module instead. See test_eval.py updates.

__all__ = [
    "load_dataset",
    "DATASET_PATH",
    "SAMPLES_DIR",
    "value_recall",
    "value_precision",
    "outcome_correct",
    "count_exact",
    "recovery_gain",
    "latency",
    "aggregate_thresholds",
    "run_eval",
]


def main(argv: list[str] | None = None) -> int:
    """Delegate to src.evals.runner.main."""
    from src.evals.runner import main as _main
    return _main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
