"""Tests for ExecutionPolicy validation and product-specific presets."""

from __future__ import annotations

import pytest

from src.runtime.policy import (
    EXTERNAL_READ,
    EXTERNAL_WRITE,
    NO_RETRY,
    SCAN_COMPUTE,
    ExecutionPolicy,
)


def test_policy_rejects_max_attempts_zero():
    with pytest.raises(ValueError, match="max_attempts"):
        ExecutionPolicy(max_attempts=0)


def test_policy_rejects_negative_timeout():
    with pytest.raises(ValueError, match="timeout_seconds"):
        ExecutionPolicy(timeout_seconds=-5)


def test_policy_rejects_zero_timeout():
    with pytest.raises(ValueError, match="timeout_seconds"):
        ExecutionPolicy(timeout_seconds=0)


def test_policy_rejects_negative_retry_delay():
    with pytest.raises(ValueError, match="retry_delay_seconds"):
        ExecutionPolicy(retry_delay_seconds=-1)


def test_external_write_is_irreversible():
    assert EXTERNAL_WRITE.irreversible_write is True


def test_external_write_no_retry():
    """Irreversible writes must not retry — safety is structural."""
    assert EXTERNAL_WRITE.max_attempts == 1


def test_scan_compute_no_retry():
    """Scanner is deterministic — retrying the same image is pointless."""
    assert SCAN_COMPUTE.max_attempts == 1


def test_external_read_retries():
    """Gemini audit is retryable."""
    assert EXTERNAL_READ.max_attempts == 3


def test_no_retry_is_no_retry():
    assert NO_RETRY.max_attempts == 1
    assert NO_RETRY.timeout_seconds is None
    assert NO_RETRY.retry_delay_seconds == 0.0
    assert NO_RETRY.irreversible_write is False


def test_policy_accepts_none_timeout():
    p = ExecutionPolicy(max_attempts=1, timeout_seconds=None)
    assert p.timeout_seconds is None


def test_policy_accepts_zero_retry_delay():
    p = ExecutionPolicy(max_attempts=1, retry_delay_seconds=0.0)
    assert p.retry_delay_seconds == 0.0


def test_policy_is_frozen():
    """ExecutionPolicy is frozen — cannot be mutated after creation."""
    p = ExecutionPolicy(max_attempts=1)
    with pytest.raises(AttributeError):
        p.max_attempts = 2  # type: ignore[misc]
