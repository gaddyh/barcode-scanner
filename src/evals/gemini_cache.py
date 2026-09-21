"""Gemini audit cache for deterministic full-pipeline evaluation.

Gemini is nondeterministic — the same image can produce different label
detections across calls. This makes full-pipeline A/B comparison
impossible: the variance from Gemini dominates any scanner improvement.

This module provides a file-backed cache that records real Gemini audit
results keyed by (image content hash, prompt version, model), then
replays them on subsequent runs. This makes full-pipeline evaluation
deterministic and reproducible.

The cache key includes the prompt version and model name so that
changing the prompt or model invalidates stale entries — a cache miss
forces a fresh Gemini call rather than silently replaying outdated
results.

Usage::

    # Capture mode: call Gemini, save results to cache file
    python -m src.evals.regression --full-pipeline --cache-gemini

    # Replay mode: use cached results, no Gemini calls
    python -m src.evals.regression --full-pipeline --replay-gemini

The cache file lives at ``tests/eval/gemini_audit_cache.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src.ingest.vision import VISION_PROMPT_VERSION

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_PATH = _REPO_ROOT / "tests" / "eval" / "gemini_audit_cache.json"


def _cache_key(image_path: str | Path, model: str) -> str:
    """Compute cache key from image hash, prompt version, and model.

    Including the prompt version and model in the key ensures that
    changing the prompt or model invalidates stale cache entries —
    a cache miss forces a fresh Gemini call rather than silently
    replaying outdated results.
    """
    with open(image_path, "rb") as f:
        image_hash = hashlib.sha256(f.read()).hexdigest()
    key_str = f"{image_hash}:{VISION_PROMPT_VERSION}:{model}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


class GeminiAuditCache:
    """File-backed cache for Gemini audit results.

    Keyed by (image content hash, prompt version, model). Values are the
    JSON-serialized ``SpatialLabelAuditPixels`` dict (the ``spatial`` field
    of the audit result).
    """

    def __init__(self, cache_path: Path = DEFAULT_CACHE_PATH) -> None:
        self.cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                self._cache = json.load(f)
            logger.info(
                "Gemini audit cache loaded: %d entries from %s "
                "(prompt_version=%s)",
                len(self._cache),
                self.cache_path,
                VISION_PROMPT_VERSION,
            )
        self._loaded = True

    def get(self, image_path: str | Path, model: str = "") -> dict[str, Any] | None:
        """Return cached audit result for the image, or None if not cached."""
        self._load()
        key = _cache_key(image_path, model)
        return self._cache.get(key)

    def put(
        self, image_path: str | Path, spatial: dict[str, Any], model: str = ""
    ) -> None:
        """Record an audit result for the image."""
        self._load()
        key = _cache_key(image_path, model)
        self._cache[key] = spatial

    def save(self) -> None:
        """Write the cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w") as f:
            json.dump(self._cache, f, indent=2, sort_keys=True)
            f.write("\n")
        logger.info(
            "Gemini audit cache saved: %d entries to %s "
            "(prompt_version=%s)",
            len(self._cache),
            self.cache_path,
            VISION_PROMPT_VERSION,
        )

    def __len__(self) -> int:
        self._load()
        return len(self._cache)
