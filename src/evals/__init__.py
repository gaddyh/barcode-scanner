"""Evals package — datasets, evaluators, and offline eval runner."""

from src.evals.datasets import DATASET_PATH, SAMPLES_DIR, load_dataset
from src.evals.evaluators import (
    LATENCY_THRESHOLD_MS,
    aggregate_thresholds,
    count_exact,
    latency,
    outcome_correct,
    recovery_gain,
    value_precision,
    value_recall,
)
from src.evals.runner import run_eval

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
    "LATENCY_THRESHOLD_MS",
    "run_eval",
]
