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
from src.evals.online import LATENCY_THRESHOLD_MS as ONLINE_LATENCY_THRESHOLD_MS
from src.evals.online import evaluate_production_run
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
