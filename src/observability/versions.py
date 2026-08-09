"""Run versioning — central registry of all component versions affecting behavior.

Every production run carries these versions in trace metadata so runs can
be compared across releases. When a component's algorithm, prompt, or
model changes, bump its version constant here (or in the component module).

The version constants live in their respective modules so they're co-located
with the code they describe. This module collects them into a single
``RunVersions`` model for stamping on traces.
"""

from __future__ import annotations

import os

from pydantic import BaseModel


class RunVersions(BaseModel):
    """All version tags for one run — stamped on trace metadata."""

    pipeline_version: str
    scanner_version: str
    vision_prompt_version: str
    vision_model: str
    recovery_version: str


def collect_versions(model: str | None = None) -> RunVersions:
    """Collect all component versions for the current run.

    Imports are lazy to avoid a circular dependency: ``pipeline.py``
    imports from ``src.observability.tracing``, which triggers this
    module's package ``__init__``. If this module imported
    ``pipeline.py`` at module level, the cycle would break.

    Args:
        model: Override Gemini model name. If None, resolves from
            ``GEMINI_MODEL`` env var or ``DEFAULT_MODEL``.
    """
    from src.ingest.scanner import SCANNER_VERSION
    from src.ingest.vision import DEFAULT_MODEL, VISION_PROMPT_VERSION
    from src.ingest.pipeline import RECOVERY_VERSION
    from src.observability.tracing import PIPELINE_VERSION

    vision_model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    return RunVersions(
        pipeline_version=PIPELINE_VERSION,
        scanner_version=SCANNER_VERSION,
        vision_prompt_version=VISION_PROMPT_VERSION,
        vision_model=vision_model,
        recovery_version=RECOVERY_VERSION,
    )
