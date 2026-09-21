"""Tests for src/api/feedback.py and src/observability/versions.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.api.feedback import submit_upload_feedback
from src.observability.versions import RunVersions, collect_versions

# ---------------------------------------------------------------------------
# api/feedback.py
# ---------------------------------------------------------------------------


def test_submit_upload_feedback_correct():
    """Submitting correct=True creates feedback with score=1."""
    trace_id = uuid4()
    fake_client = MagicMock()
    with patch("src.api.feedback.Client", return_value=fake_client):
        score = submit_upload_feedback(trace_id=trace_id, correct=True)
    assert score == 1
    fake_client.create_feedback.assert_called_once_with(
        key="user_correct",
        score=1,
        trace_id=trace_id,
        comment=None,
    )


def test_submit_upload_feedback_incorrect():
    """Submitting correct=False creates feedback with score=0."""
    trace_id = uuid4()
    fake_client = MagicMock()
    with patch("src.api.feedback.Client", return_value=fake_client):
        score = submit_upload_feedback(trace_id=trace_id, correct=False)
    assert score == 0
    fake_client.create_feedback.assert_called_once_with(
        key="user_correct",
        score=0,
        trace_id=trace_id,
        comment=None,
    )


def test_submit_upload_feedback_with_comment():
    """Submitting with a comment passes it through."""
    trace_id = uuid4()
    fake_client = MagicMock()
    with patch("src.api.feedback.Client", return_value=fake_client):
        score = submit_upload_feedback(
            trace_id=trace_id, correct=True, comment="great"
        )
    assert score == 1
    fake_client.create_feedback.assert_called_once_with(
        key="user_correct",
        score=1,
        trace_id=trace_id,
        comment="great",
    )


# ---------------------------------------------------------------------------
# observability/versions.py
# ---------------------------------------------------------------------------


def test_collect_versions_defaults():
    versions = collect_versions()
    assert isinstance(versions, RunVersions)
    assert versions.pipeline_version
    assert versions.scanner_version
    assert versions.vision_prompt_version
    assert versions.vision_model
    assert versions.recovery_version


def test_collect_versions_with_model_override():
    versions = collect_versions(model="gemini-custom")
    assert versions.vision_model == "gemini-custom"


def test_collect_versions_with_env_model(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-from-env")
    versions = collect_versions()
    assert versions.vision_model == "gemini-from-env"


def test_collect_versions_model_override_takes_precedence(monkeypatch):
    """Explicit model arg takes precedence over env var."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-from-env")
    versions = collect_versions(model="gemini-explicit")
    assert versions.vision_model == "gemini-explicit"


def test_run_versions_model_dump():
    versions = RunVersions(
        pipeline_version="v1",
        scanner_version="v2",
        vision_prompt_version="v3",
        vision_model="v4",
        recovery_version="v5",
    )
    dumped = versions.model_dump()
    assert dumped["pipeline_version"] == "v1"
    assert dumped["scanner_version"] == "v2"
    assert dumped["vision_prompt_version"] == "v3"
    assert dumped["vision_model"] == "v4"
    assert dumped["recovery_version"] == "v5"
