"""Tests for src.evals.regression — the offline regression gate.

Covers _run_scanner, _run_full_pipeline, run_regression, _aggregate,
_format_results, compare_against_baseline, write_baseline, and main.
Uses monkeypatching to avoid real Gemini calls and real dataset images.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evals import regression

# ---------------------------------------------------------------------------
# _run_scanner
# ---------------------------------------------------------------------------


class TestRunScanner:
    def test_returns_items_and_metrics(self, tmp_path: Path) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")

        class _Det:
            def __init__(self, value: str) -> None:
                self.value = value

        class _FakeScanner:
            def scan_bytes(self, b: bytes):
                return [_Det("7297501098442"), _Det("SHIP123")]

        with patch.object(regression, "BarcodeScanner", lambda: _FakeScanner()):
            result = regression._run_scanner(str(img))

        assert result["status"] == "complete"
        assert len(result["items"]) == 1  # policy filtered out SHIP123
        assert result["items"][0]["barcode_value"] == "7297501098442"
        assert result["metrics"]["raw_scanner_count"] == 2
        assert result["metrics"]["policy_rejected_count"] == 1
        assert "elapsed_ms" in result["metrics"]

    def test_no_detections(self, tmp_path: Path) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")

        class _FakeScanner:
            def scan_bytes(self, b: bytes):
                return []

        with patch.object(regression, "BarcodeScanner", lambda: _FakeScanner()):
            result = regression._run_scanner(str(img))

        assert result["status"] == "needs_user_input"
        assert result["items"] == []
        assert result["metrics"]["raw_scanner_count"] == 0

    def test_all_rejected_by_policy(self, tmp_path: Path) -> None:
        """All detections non-EAN-13 → policy rejects all → needs_user_input."""
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")

        class _Det:
            def __init__(self, value: str) -> None:
                self.value = value

        class _FakeScanner:
            def scan_bytes(self, b: bytes):
                return [_Det("SHIP123"), _Det("ABC")]

        with patch.object(regression, "BarcodeScanner", lambda: _FakeScanner()):
            result = regression._run_scanner(str(img))

        assert result["status"] == "needs_user_input"
        assert result["items"] == []
        assert result["metrics"]["raw_scanner_count"] == 2
        assert result["metrics"]["policy_rejected_count"] == 2

    def test_duplicate_eans_preserved(self, tmp_path: Path) -> None:
        """Duplicate EAN-13 values are separate occurrences (multiset)."""
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")

        class _Det:
            def __init__(self, value: str) -> None:
                self.value = value

        class _FakeScanner:
            def scan_bytes(self, b: bytes):
                return [_Det("7297501098442"), _Det("7297501098442")]

        with patch.object(regression, "BarcodeScanner", lambda: _FakeScanner()):
            result = regression._run_scanner(str(img))

        assert len(result["items"]) == 2
        assert result["metrics"]["policy_rejected_count"] == 0

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Missing image file → OSError from open()."""
        missing = tmp_path / "missing.png"

        class _EmptyScanner:
            def scan_bytes(self, b: bytes):
                return []

        with patch.object(regression, "BarcodeScanner", lambda: _EmptyScanner()):
            with pytest.raises(OSError):
                regression._run_scanner(str(missing))


# ---------------------------------------------------------------------------
# _run_full_pipeline
# ---------------------------------------------------------------------------


class TestRunFullPipeline:
    def test_returns_found_items(self, tmp_path: Path) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")

        def _fake_analyze(image_path, **kwargs):
            return {
                "outcome": "complete",
                "found": [{"barcode_value": "7297501098442"}],
                "summary": {"audit_latency_ms": 100},
            }

        with patch("src.ingest.analyze.analyze_image", _fake_analyze):
            result = regression._run_full_pipeline(str(img))

        assert result["status"] == "complete"
        assert len(result["items"]) == 1
        assert result["metrics"]["audit_latency_ms"] == 100


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_empty(self) -> None:
        agg = regression._aggregate([])
        assert agg == {"n": 0}

    def test_basic_aggregation(self) -> None:
        results = [
            {
                "id": "a", "expected_count": 2, "found_count": 2,
                "matched_count": 2, "occurrence_recall": 1.0,
                "occurrence_precision": 1.0, "barcode_accuracy": 1.0,
                "elapsed_ms": 100, "audit_latency_ms": 0,
                "raw_scanner_count": 2, "policy_rejected_count": 0,
            },
            {
                "id": "b", "expected_count": 3, "found_count": 2,
                "matched_count": 2, "occurrence_recall": 0.667,
                "occurrence_precision": 1.0, "barcode_accuracy": 0.0,
                "elapsed_ms": 200, "audit_latency_ms": 50,
                "raw_scanner_count": 2, "policy_rejected_count": 1,
            },
        ]
        agg = regression._aggregate(results)
        assert agg["n"] == 2
        assert agg["total_matched_occurrences"] == 4
        assert agg["total_expected_occurrences"] == 5
        assert agg["total_found_occurrences"] == 4
        assert agg["total_raw_scanner_detections"] == 4
        assert agg["total_policy_rejected"] == 1
        assert 0 < agg["mean_occurrence_recall"] < 1

    def test_single_result_p50_p95(self) -> None:
        """n=1: p50 and p95 both index 0."""
        results = [{
            "id": "x", "expected_count": 1, "found_count": 1,
            "matched_count": 1, "occurrence_recall": 1.0,
            "occurrence_precision": 1.0, "barcode_accuracy": 1.0,
            "elapsed_ms": 500, "audit_latency_ms": 10,
            "raw_scanner_count": 1, "policy_rejected_count": 0,
        }]
        agg = regression._aggregate(results)
        assert agg["p50_latency_ms"] == 500
        assert agg["p95_latency_ms"] == 500
        assert agg["p50_audit_latency_ms"] == 10

    def test_all_zero_counts(self) -> None:
        """No expected, no found — recall/precision vacuously 1.0."""
        results = [{
            "id": "x", "expected_count": 0, "found_count": 0,
            "matched_count": 0, "occurrence_recall": 1.0,
            "occurrence_precision": 1.0, "barcode_accuracy": 1.0,
            "elapsed_ms": 100, "audit_latency_ms": 0,
            "raw_scanner_count": 0, "policy_rejected_count": 0,
        }]
        agg = regression._aggregate(results)
        assert agg["total_expected_occurrences"] == 0
        assert agg["total_found_occurrences"] == 0
        assert agg["total_matched_occurrences"] == 0


# ---------------------------------------------------------------------------
# _format_results
# ---------------------------------------------------------------------------


class TestFormatResults:
    def test_formats_report(self) -> None:
        results = [
            {
                "id": "case1", "expected_count": 2, "found_count": 2,
                "matched_count": 2, "occurrence_recall": 1.0,
                "occurrence_precision": 1.0, "barcode_accuracy": 1.0,
                "elapsed_ms": 100, "audit_latency_ms": 0,
                "raw_scanner_count": 2, "policy_rejected_count": 0,
            },
        ]
        agg = regression._aggregate(results)
        report = regression._format_results(results, agg)
        assert "Barcode Scanner Regression Report" in report
        assert "Matched/Expected:" in report
        assert "Matched/Found:" in report
        assert "Raw scanner detections:" in report
        assert "Policy rejected:" in report
        assert "case1" in report
        assert "[PASS]" in report


# ---------------------------------------------------------------------------
# compare_against_baseline
# ---------------------------------------------------------------------------


class TestCompareAgainstBaseline:
    def _result(
        self, recall: float = 1.0, precision: float = 1.0, accuracy: float = 1.0,
        case_id: str = "case1", expected: int = 2, found: int = 2,
    ) -> dict:
        return {
            "id": case_id, "expected_count": expected, "found_count": found,
            "matched_count": min(expected, found), "occurrence_recall": recall,
            "occurrence_precision": precision, "barcode_accuracy": accuracy,
            "elapsed_ms": 100, "audit_latency_ms": 0,
            "raw_scanner_count": found, "policy_rejected_count": 0,
        }

    def test_no_baseline_file(self, tmp_path: Path) -> None:
        # No baseline → warning, returns 0.
        results = [self._result()]
        agg = regression._aggregate(results)
        rc = regression.compare_against_baseline(
            results, agg, baseline_path=tmp_path / "missing.json"
        )
        assert rc == 0

    def test_no_regression(self, tmp_path: Path) -> None:
        results = [self._result()]
        agg = regression._aggregate(results)
        baseline = {"aggregate": agg, "cases": results}
        bp = tmp_path / "baseline.json"
        bp.write_text(json.dumps(baseline))
        rc = regression.compare_against_baseline(results, agg, baseline_path=bp)
        assert rc == 0

    def test_recall_regression(self, tmp_path: Path) -> None:
        results = [self._result(recall=0.5)]
        agg = regression._aggregate(results)
        baseline_agg = regression._aggregate([self._result(recall=1.0)])
        baseline = {"aggregate": baseline_agg, "cases": []}
        bp = tmp_path / "baseline.json"
        bp.write_text(json.dumps(baseline))
        rc = regression.compare_against_baseline(results, agg, baseline_path=bp)
        assert rc == 1

    def test_precision_regression(self, tmp_path: Path) -> None:
        results = [self._result(precision=0.5)]
        agg = regression._aggregate(results)
        baseline_agg = regression._aggregate([self._result(precision=1.0)])
        baseline = {"aggregate": baseline_agg, "cases": []}
        bp = tmp_path / "baseline.json"
        bp.write_text(json.dumps(baseline))
        rc = regression.compare_against_baseline(results, agg, baseline_path=bp)
        assert rc == 1

    def test_acceptance_fail_recall(self, tmp_path: Path) -> None:
        # Recall below 0.65 acceptance target.
        results = [self._result(recall=0.5, expected=10, found=5)]
        agg = regression._aggregate(results)
        # Baseline also low so no regression, but acceptance fails.
        baseline = {"aggregate": agg, "cases": []}
        bp = tmp_path / "baseline.json"
        bp.write_text(json.dumps(baseline))
        rc = regression.compare_against_baseline(results, agg, baseline_path=bp)
        assert rc == 1

    def test_acceptance_fail_precision(self, tmp_path: Path) -> None:
        # Precision below 0.90 acceptance target.
        results = [self._result(precision=0.5, expected=2, found=4)]
        agg = regression._aggregate(results)
        baseline = {"aggregate": agg, "cases": []}
        bp = tmp_path / "baseline.json"
        bp.write_text(json.dumps(baseline))
        rc = regression.compare_against_baseline(results, agg, baseline_path=bp)
        assert rc == 1

    def test_per_case_regression(self, tmp_path: Path) -> None:
        results = [self._result(accuracy=0.0, case_id="case1")]
        agg = regression._aggregate(results)
        baseline = {
            "aggregate": agg,
            "cases": [{"id": "case1", "barcode_accuracy": 1.0}],
        }
        bp = tmp_path / "baseline.json"
        bp.write_text(json.dumps(baseline))
        rc = regression.compare_against_baseline(results, agg, baseline_path=bp)
        assert rc == 1

    def test_per_case_regression_unknown_case_ignored(self, tmp_path: Path) -> None:
        """Case not in baseline → skipped (no regression)."""
        results = [self._result(accuracy=0.0, case_id="new_case")]
        agg = regression._aggregate(results)
        baseline = {"aggregate": agg, "cases": []}
        bp = tmp_path / "baseline.json"
        bp.write_text(json.dumps(baseline))
        rc = regression.compare_against_baseline(results, agg, baseline_path=bp)
        assert rc == 0

    def test_multiple_errors_at_once(self, tmp_path: Path) -> None:
        """Recall regression + precision regression + acceptance fail."""
        results = [self._result(recall=0.3, precision=0.3, expected=10, found=10)]
        agg = regression._aggregate(results)
        baseline_agg = regression._aggregate(
            [self._result(recall=1.0, precision=1.0, expected=10, found=10)]
        )
        baseline = {"aggregate": baseline_agg, "cases": []}
        bp = tmp_path / "baseline.json"
        bp.write_text(json.dumps(baseline))
        rc = regression.compare_against_baseline(results, agg, baseline_path=bp)
        assert rc == 1

    def test_latency_info_only(self, tmp_path: Path, capsys) -> None:
        results = [self._result()]
        results[0]["elapsed_ms"] = 10000
        agg = regression._aggregate(results)
        baseline_agg = regression._aggregate([self._result()])
        baseline_agg["p95_latency_ms"] = 100
        baseline = {"aggregate": baseline_agg, "cases": []}
        bp = tmp_path / "baseline.json"
        bp.write_text(json.dumps(baseline))
        rc = regression.compare_against_baseline(results, agg, baseline_path=bp)
        # Latency is informational — should still return 0 if no quality regression.
        assert rc == 0


# ---------------------------------------------------------------------------
# write_baseline
# ---------------------------------------------------------------------------


class TestWriteBaseline:
    def test_writes_file(self, tmp_path: Path) -> None:
        results = [
            {
                "id": "case1", "expected_count": 2, "found_count": 2,
                "matched_count": 2, "occurrence_recall": 1.0,
                "occurrence_precision": 1.0, "barcode_accuracy": 1.0,
                "elapsed_ms": 100, "audit_latency_ms": 0,
            },
        ]
        agg = regression._aggregate(results)
        bp = tmp_path / "baseline.json"
        regression.write_baseline(results, agg, baseline_path=bp)
        assert bp.exists()
        data = json.loads(bp.read_text())
        assert data["aggregate"]["n"] == 1
        assert data["cases"][0]["id"] == "case1"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_no_results(self, tmp_path: Path, monkeypatch) -> None:
        # Empty dataset → returns 1.
        monkeypatch.setattr(regression, "load_dataset", lambda: [])
        rc = regression.main([])
        assert rc == 1

    def test_main_write_baseline(self, tmp_path: Path, monkeypatch) -> None:
        # One fake result, write-baseline mode.
        bp = tmp_path / "baseline.json"
        monkeypatch.setattr(regression, "BASELINE_SCANNER_PATH", bp)

        def _fake_run(*, full_pipeline: bool = False):
            return [{
                "id": "case1", "image_name": "x.png", "expected_count": 1,
                "found_count": 1, "matched_count": 1,
                "occurrence_recall": 1.0, "occurrence_precision": 1.0,
                "barcode_accuracy": 1.0, "elapsed_ms": 10, "audit_latency_ms": 0,
                "raw_scanner_count": 1, "policy_rejected_count": 0,
                "comment_recall": "ok", "comment_precision": "ok",
            }]

        monkeypatch.setattr(regression, "run_regression", _fake_run)
        rc = regression.main(["--write-baseline"])
        assert rc == 0
        assert bp.exists()
