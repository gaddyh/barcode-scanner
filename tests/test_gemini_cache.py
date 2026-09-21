"""Tests for src.evals.gemini_cache — file-backed Gemini audit cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evals.gemini_cache import GeminiAuditCache, _cache_key


class TestCacheKey:
    def test_key_is_stable(self, tmp_path: Path) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"same-bytes")
        k1 = _cache_key(img, "gemini-1.5")
        k2 = _cache_key(img, "gemini-1.5")
        assert k1 == k2
        assert len(k1) == 16

    def test_different_bytes_different_key(self, tmp_path: Path) -> None:
        a = tmp_path / "a.png"
        a.write_bytes(b"aaa")
        b = tmp_path / "b.png"
        b.write_bytes(b"bbb")
        assert _cache_key(a, "m") != _cache_key(b, "m")

    def test_different_model_different_key(self, tmp_path: Path) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"bytes")
        assert _cache_key(img, "model-a") != _cache_key(img, "model-b")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            _cache_key(tmp_path / "missing.png", "m")


class TestGeminiAuditCache:
    def test_get_miss_no_file(self, tmp_path: Path) -> None:
        # Image file must exist for _cache_key to compute (it reads bytes).
        # But the cache file doesn't exist → cache miss returns None.
        img = tmp_path / "x.png"
        img.write_bytes(b"img-bytes")
        cache = GeminiAuditCache(cache_path=tmp_path / "missing.json")
        assert cache.get(img, "m") is None

    def test_put_and_get(self, tmp_path: Path) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"img-bytes")
        cache = GeminiAuditCache(cache_path=tmp_path / "cache.json")
        spatial = {"image_width": 100, "labels": []}
        cache.put(img, spatial, "m")
        got = cache.get(img, "m")
        assert got == spatial

    def test_save_and_reload(self, tmp_path: Path) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"img-bytes")
        cp = tmp_path / "cache.json"
        cache = GeminiAuditCache(cache_path=cp)
        cache.put(img, {"labels": [1]}, "m")
        cache.save()
        assert cp.exists()

        # New instance loads from disk.
        cache2 = GeminiAuditCache(cache_path=cp)
        got = cache2.get(img, "m")
        assert got == {"labels": [1]}

    def test_len_empty(self, tmp_path: Path) -> None:
        cache = GeminiAuditCache(cache_path=tmp_path / "missing.json")
        assert len(cache) == 0

    def test_len_after_put(self, tmp_path: Path) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"b")
        cache = GeminiAuditCache(cache_path=tmp_path / "c.json")
        cache.put(img, {}, "m")
        assert len(cache) == 1

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"b")
        cp = tmp_path / "sub" / "dir" / "cache.json"
        cache = GeminiAuditCache(cache_path=cp)
        cache.put(img, {}, "m")
        cache.save()
        assert cp.exists()

    def test_corrupt_cache_file(self, tmp_path: Path) -> None:
        """Corrupt JSON → raises on load."""
        cp = tmp_path / "cache.json"
        cp.write_text("{not valid json")
        cache = GeminiAuditCache(cache_path=cp)
        with pytest.raises(json.JSONDecodeError):
            cache.get(tmp_path / "x.png", "m")

    def test_empty_cache_file(self, tmp_path: Path) -> None:
        """Empty JSON object {} → no entries, no error."""
        cp = tmp_path / "cache.json"
        cp.write_text("{}")
        cache = GeminiAuditCache(cache_path=cp)
        assert len(cache) == 0

    def test_different_model_no_collision(self, tmp_path: Path) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"b")
        cache = GeminiAuditCache(cache_path=tmp_path / "c.json")
        cache.put(img, {"model": "a"}, "model-a")
        cache.put(img, {"model": "b"}, "model-b")
        assert cache.get(img, "model-a") == {"model": "a"}
        assert cache.get(img, "model-b") == {"model": "b"}
        assert len(cache) == 2
