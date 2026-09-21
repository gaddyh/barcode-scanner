"""Evals package — datasets, evaluators, offline runner, online feedback, and annotation queue."""

from src.evals.annotation_sink import AnnotationCandidateSink, register_annotation_sink
from src.evals.annotation_store import (
    AnnotationCandidate,
    create_candidate,
    export_to_dataset_json,
    get_candidate,
    get_stats,
    init_db,
    list_pending,
    list_reviewed,
    submit_review,
)
from src.evals.datasets import (
    CANONICAL_DATASET_PATH,
    DATASET_PATH,
    LEGACY_DATASET_PATH,
    SAMPLES_DIR,
    load_dataset,
    load_legacy_dataset,
)
from src.evals.evaluators import (
    LATENCY_THRESHOLD_MS,
    aggregate_thresholds,
    barcode_accuracy,
    count_exact,
    latency,
    occurrence_precision,
    occurrence_recall,
    outcome_correct,
    recovery_gain,
    value_precision,
    value_recall,
)
from src.evals.online import LATENCY_THRESHOLD_MS as ONLINE_LATENCY_THRESHOLD_MS
from src.evals.online import evaluate_production_run
from src.evals.runner import run_eval

__all__ = [
    "load_dataset",
    "load_legacy_dataset",
    "CANONICAL_DATASET_PATH",
    "DATASET_PATH",
    "LEGACY_DATASET_PATH",
    "SAMPLES_DIR",
    "value_recall",
    "value_precision",
    "outcome_correct",
    "count_exact",
    "recovery_gain",
    "latency",
    "aggregate_thresholds",
    "occurrence_recall",
    "occurrence_precision",
    "barcode_accuracy",
    "LATENCY_THRESHOLD_MS",
    "run_eval",
    "evaluate_production_run",
    "ONLINE_LATENCY_THRESHOLD_MS",
    "AnnotationCandidate",
    "AnnotationCandidateSink",
    "register_annotation_sink",
    "init_db",
    "create_candidate",
    "list_pending",
    "list_reviewed",
    "get_candidate",
    "submit_review",
    "get_stats",
    "export_to_dataset_json",
]
