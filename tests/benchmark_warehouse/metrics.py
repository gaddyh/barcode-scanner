"""Pure metric functions for the warehouse benchmark.

This module knows nothing about Gemini, ZXing, or the CLI. It operates on plain
``RunMetrics`` and ``GroundTruthImage`` values. It is safe to unit-test
deterministically.

Functions:
- ``derive_failure_reason`` — categorize a run's outcome.
- ``false_positive_violations`` — count FP violations for one run vs ground truth.
- ``compute_run_metrics`` — extract ``RunMetrics`` from a pipeline summary dict.
- ``run_matches_expected`` — check one run against all expected fields.
- ``aggregate_runs`` — aggregate first-run metrics + consistency across runs.
- ``consistency_report`` — per-image across-run consistency.
"""

from __future__ import annotations

from typing import Any

from tests.benchmark_warehouse.models import (
    AggregateMetrics,
    GroundTruthImage,
    ImageConsistency,
    RunMetrics,
)

# ---------------------------------------------------------------------------
# Failure reason derivation
# ---------------------------------------------------------------------------

FAILURE_REASONS = (
    "scan_error",
    "audit_error",
    "gemini_label_count_mismatch",
    "scanner_miss",
    "recovery_partial",
    "recovery_failed",
    "false_positive",
    "all_matched",
)


def derive_failure_reason(
    run: RunMetrics,
    gt: GroundTruthImage,
) -> str:
    """Categorize a run's outcome in priority order.

    See the plan for the full enumeration. The first matching category wins.
    """
    if run.scan_status not in ("found", "not_found"):
        return "scan_error"
    if run.audit_status != "ok":
        return "audit_error"
    if (
        run.visible_label_count is not None
        and run.visible_label_count != gt.expected_visible_label_count
    ):
        return "gemini_label_count_mismatch"
    if (
        run.baseline_matched_labels is not None
        and run.baseline_matched_labels < gt.expected_baseline_matched_labels
    ):
        return "scanner_miss"

    recovery_fired = run.recovery_fired
    if recovery_fired:
        if run.recovered_label_count > 0:
            # Recovery found something. If not all labels matched, it's partial.
            if run.all_labels_matched is False:
                return "recovery_partial"
            # Recovery found something and all labels matched — success.
        else:
            # Recovery fired but found nothing. If labels remain unmatched,
            # it's a failure.
            if run.all_labels_matched is False:
                return "recovery_failed"
            # Recovery fired, found nothing, but all labels were already matched
            # (recovery was triggered by a transient mismatch that resolved) — success.

    # No recovery fired but labels unmatched beyond expectation.
    if (
        run.final_matched_labels is not None
        and run.final_matched_labels < gt.expected_final_matched_labels
        and run.baseline_matched_labels is not None
        and run.baseline_matched_labels < gt.expected_final_matched_labels
    ):
        if run.recovered_label_count == 0 and recovery_fired:
            return "recovery_failed"

    if false_positive_violations(run, gt) > 0:
        return "false_positive"

    return "all_matched"


# ---------------------------------------------------------------------------
# False-positive accounting
# ---------------------------------------------------------------------------


def false_positive_violations(run: RunMetrics, gt: GroundTruthImage) -> int:
    """Count false-positive violations for one run vs ground truth.

    Separate raw counts (``unassigned_scanner_detections``,
    ``extra_gemini_labels``) are kept on ``RunMetrics``; this function derives
    the violation count based on per-image expectations and
    ``allow_extra_scanner_detections``.
    """
    violations = 0
    if run.extra_gemini_labels > gt.expected_extra_labels:
        violations += run.extra_gemini_labels - gt.expected_extra_labels
    if not gt.allow_extra_scanner_detections:
        if run.unassigned_scanner_detections > gt.expected_unassigned_scanner_detections:
            violations += (
                run.unassigned_scanner_detections
                - gt.expected_unassigned_scanner_detections
            )
    return violations


# ---------------------------------------------------------------------------
# Run metrics extraction
# ---------------------------------------------------------------------------


def compute_run_metrics(
    image_name: str,
    summary: dict[str, Any],
    gt: GroundTruthImage,
    latency: float | None = None,
) -> RunMetrics:
    """Extract ``RunMetrics`` from a ``pipeline_path()`` summary dict."""
    dv = summary.get("decoded_vs_visible", {})
    initial_recon = summary.get("initial_reconciliation", {})
    recovery = summary.get("recovery", {})
    final_recon = summary.get("reconciliation", {})

    visible = summary.get("visible_labels")
    clear = summary.get("clear_labels")
    decoded_count = summary.get("decoded_count", 0)
    unique_count = summary.get("unique_value_count", 0)

    # baseline_matched: from initial_reconciliation if recovery fired, else from
    # the final reconciliation (which is the same as initial when no recovery).
    if initial_recon:
        baseline_matched = initial_recon.get("matched_label_count")
    else:
        baseline_matched = dv.get("matched_labels")

    final_matched = dv.get("matched_labels")
    all_matched = dv.get("all_labels_matched")
    recovered = recovery.get("recovered_label_count", 0)
    unassigned = len(final_recon.get("unassigned_scanner_detections", []))
    extra_labels = max(0, (visible or 0) - gt.expected_visible_label_count)

    run = RunMetrics(
        image=image_name,
        scan_status=summary.get("scan_status"),
        audit_status=summary.get("audit_status"),
        ok=summary.get("ok", False),
        scanner_detection_count=decoded_count if decoded_count is not None else 0,
        unique_value_count=unique_count if unique_count is not None else 0,
        baseline_matched_labels=baseline_matched,
        visible_label_count=visible,
        clear_label_count=clear,
        recovered_label_count=recovered,
        final_matched_labels=final_matched,
        all_labels_matched=all_matched,
        unassigned_scanner_detections=unassigned,
        extra_gemini_labels=extra_labels,
        latency_seconds=latency,
        failure_reason="",  # set below
        recovery_fired="recovery" in summary,
        scan_error=summary.get("scan_error"),
        audit_error=summary.get("audit_error"),
        recovery_error=summary.get("recovery_error"),
    )
    run.failure_reason = derive_failure_reason(run, gt)
    return run


# ---------------------------------------------------------------------------
# Expected-match check
# ---------------------------------------------------------------------------


def run_matches_expected(run: RunMetrics, gt: GroundTruthImage) -> bool:
    """Check one run against all expected fields exactly."""
    if run.scan_status not in ("found", "not_found"):
        return False
    if run.audit_status != "ok":
        return False
    if run.visible_label_count != gt.expected_visible_label_count:
        return False
    if run.clear_label_count != gt.expected_clear_label_count:
        return False
    if run.scanner_detection_count != gt.expected_scanner_detection_count:
        return False
    if run.unique_value_count != gt.expected_unique_value_count:
        return False
    if run.baseline_matched_labels != gt.expected_baseline_matched_labels:
        return False
    if run.final_matched_labels != gt.expected_final_matched_labels:
        return False
    if run.recovered_label_count != gt.expected_recovered_label_count:
        return False
    if run.unassigned_scanner_detections != gt.expected_unassigned_scanner_detections:
        return False
    if run.extra_gemini_labels != gt.expected_extra_labels:
        return False
    if run.all_labels_matched != gt.expected_all_labels_matched:
        return False
    if run.failure_reason != gt.expected_failure_reason:
        return False
    if false_positive_violations(run, gt) > 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def consistency_report(
    image: str,
    runs: list[RunMetrics],
) -> ImageConsistency:
    """Check whether all runs agree on key counts."""
    visible = [r.visible_label_count for r in runs]
    final = [r.final_matched_labels for r in runs]
    recovered = [r.recovered_label_count for r in runs]
    unassigned = [r.unassigned_scanner_detections for r in runs]

    consistent = (
        len(set(visible)) <= 1
        and len(set(final)) <= 1
        and len(set(recovered)) <= 1
        and len(set(unassigned)) <= 1
    )

    return ImageConsistency(
        image=image,
        visible_label_counts=visible,
        final_matched_counts=final,
        recovered_counts=recovered,
        unassigned_counts=unassigned,
        consistent=consistent,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_runs(
    run_metrics_by_image: dict[str, list[RunMetrics]],
    dataset_images: list[GroundTruthImage],
) -> AggregateMetrics:
    """Aggregate first-run metrics across all images + consistency across runs.

    ``run_metrics_by_image`` maps image name to a list of ``RunMetrics`` (one
    per run). The first run is used for aggregate counts; all runs are used for
    the consistency report.
    """
    agg = AggregateMetrics()
    agg.image_count = len(dataset_images)

    gt_by_image = {gt.image: gt for gt in dataset_images}
    agg._expected_visible_by_image = {
        gt.image: gt.expected_visible_label_count for gt in dataset_images
    }

    for gt in dataset_images:
        runs = run_metrics_by_image.get(gt.image, [])
        if not runs:
            continue
        first = runs[0]
        agg._first_run_metrics.append(first)

        agg.expected_scanner_detections += gt.expected_scanner_detection_count
        agg.actual_scanner_detections += first.scanner_detection_count

        agg.expected_baseline_matched += gt.expected_baseline_matched_labels
        if first.baseline_matched_labels is not None:
            agg.actual_baseline_matched += first.baseline_matched_labels

        agg.expected_visible_labels += gt.expected_visible_label_count
        if first.visible_label_count is not None:
            agg.actual_visible_labels += first.visible_label_count

        agg.expected_clear_labels += gt.expected_clear_label_count
        if first.clear_label_count is not None:
            agg.actual_clear_labels += first.clear_label_count

        agg.expected_recovered += gt.expected_recovered_label_count
        agg.actual_recovered += first.recovered_label_count

        agg.expected_final_matched += gt.expected_final_matched_labels
        if first.final_matched_labels is not None:
            agg.actual_final_matched += first.final_matched_labels

        agg.unassigned_scanner_detection_count += first.unassigned_scanner_detections
        agg.extra_gemini_label_count += first.extra_gemini_labels
        agg.false_positive_violation_count += false_positive_violations(first, gt)

        if first.failure_reason != gt.expected_failure_reason:
            agg.failure_reason_mismatch_count += 1

        if first.latency_seconds is not None:
            agg.latencies.append(first.latency_seconds)

        # Consistency across all runs (live only — snapshot has 1 run).
        if len(runs) > 1:
            report = consistency_report(gt.image, runs)
            agg.consistency_reports.append(report)

    # First-run exact match.
    agg.all_first_runs_match_expected = all(
        run_matches_expected(runs[0], gt_by_image[image])
        for image, runs in run_metrics_by_image.items()
        if runs and image in gt_by_image
    )

    # Consistency (only when we have multi-run data).
    if agg.consistency_reports:
        agg.all_runs_consistent = all(r.consistent for r in agg.consistency_reports)

    return agg
