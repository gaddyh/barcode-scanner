"""SQLite annotation store — captures interesting pipeline failures for review.

The ``AnnotationCandidateSink`` writes here when ``ingest_one()`` emits an
``INGEST_COMPLETED`` event that matches "interesting failure" rules. A
human reviews pending candidates via the admin API, provides ground-truth
barcodes, and then explicitly exports reviewed candidates to
``tests/eval/dataset.json`` so the next eval run picks them up as
regression tests.

Design decisions:
- ``UNIQUE(run_id)`` + ``INSERT OR IGNORE`` — candidate creation is
  idempotent. Retries or duplicate event delivery cannot create duplicate
  annotation work.
- WAL mode for concurrent read/write safety.
- No ORM — raw ``sqlite3`` with parameterized queries (stdlib only).
- Export is transaction-like: ``added_to_dataset = 1`` only after image
  copy + JSON construction + atomic rename all succeed.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default DB path — configurable via env var.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB_PATH = _REPO_ROOT / "data" / "annotations.db"
DATASET_PATH = _REPO_ROOT / "tests" / "eval" / "dataset.json"
SAMPLES_DIR = _REPO_ROOT / "samples"


def _db_path() -> str:
    return os.getenv("ANNOTATION_DB_PATH", str(_DEFAULT_DB_PATH))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AnnotationCandidate(BaseModel):
    """One interesting failure captured for human review."""

    id: str
    run_id: str
    session_id: str
    source: str
    image_ref: str | None = None
    status: str
    scanner_count: int = 0
    vision_count: int = 0
    recovery_attempted: bool = False
    recovery_labels_resolved: int = 0
    issues: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    expected_barcodes: list[dict[str, Any]] | None = None
    expected_outcome: str | None = None
    added_to_dataset: bool = False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotation_candidates (
    id                       TEXT PRIMARY KEY,
    run_id                   TEXT NOT NULL UNIQUE,
    session_id               TEXT NOT NULL,
    source                   TEXT NOT NULL,
    image_ref                TEXT,
    status                   TEXT NOT NULL,
    scanner_count            INTEGER NOT NULL DEFAULT 0,
    vision_count             INTEGER NOT NULL DEFAULT 0,
    recovery_attempted       INTEGER NOT NULL DEFAULT 0,
    recovery_labels_resolved INTEGER NOT NULL DEFAULT 0,
    issues                   TEXT NOT NULL DEFAULT '[]',
    created_at               TEXT NOT NULL,
    reviewed_at              TEXT,
    reviewed_by              TEXT,
    expected_barcodes        TEXT,
    expected_outcome         TEXT,
    added_to_dataset         INTEGER NOT NULL DEFAULT 0
);
"""


def init_db(db_path: str | None = None) -> None:
    """Create the table if it doesn't exist. Sets WAL mode."""
    path = db_path or _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Row <-> model conversion
# ---------------------------------------------------------------------------


def _row_to_candidate(row: sqlite3.Row) -> AnnotationCandidate:
    return AnnotationCandidate(
        id=row["id"],
        run_id=row["run_id"],
        session_id=row["session_id"],
        source=row["source"],
        image_ref=row["image_ref"],
        status=row["status"],
        scanner_count=row["scanner_count"],
        vision_count=row["vision_count"],
        recovery_attempted=bool(row["recovery_attempted"]),
        recovery_labels_resolved=row["recovery_labels_resolved"],
        issues=json.loads(row["issues"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        reviewed_at=(
            datetime.fromisoformat(row["reviewed_at"]) if row["reviewed_at"] else None
        ),
        reviewed_by=row["reviewed_by"],
        expected_barcodes=(
            json.loads(row["expected_barcodes"]) if row["expected_barcodes"] else None
        ),
        expected_outcome=row["expected_outcome"],
        added_to_dataset=bool(row["added_to_dataset"]),
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def create_candidate(candidate: AnnotationCandidate, db_path: str | None = None) -> bool:
    """Insert a candidate. Idempotent via UNIQUE(run_id) + INSERT OR IGNORE.

    Returns True if a new row was inserted, False if the run_id already
    existed (duplicate event).
    """
    path = db_path or _db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO annotation_candidates
                (id, run_id, session_id, source, image_ref, status,
                 scanner_count, vision_count, recovery_attempted,
                 recovery_labels_resolved, issues, created_at,
                 reviewed_at, reviewed_by, expected_barcodes,
                 expected_outcome, added_to_dataset)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 0)
            """,
            (
                candidate.id,
                candidate.run_id,
                candidate.session_id,
                candidate.source,
                candidate.image_ref,
                candidate.status,
                candidate.scanner_count,
                candidate.vision_count,
                int(candidate.recovery_attempted),
                candidate.recovery_labels_resolved,
                json.dumps(candidate.issues),
                candidate.created_at.isoformat(),
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def list_pending(limit: int = 50, db_path: str | None = None) -> list[AnnotationCandidate]:
    """List unreviewed candidates (reviewed_at IS NULL), newest first."""
    path = db_path or _db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT * FROM annotation_candidates
            WHERE reviewed_at IS NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_candidate(r) for r in rows]
    finally:
        conn.close()


def list_reviewed(limit: int = 50, db_path: str | None = None) -> list[AnnotationCandidate]:
    """List reviewed but not-yet-exported candidates."""
    path = db_path or _db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT * FROM annotation_candidates
            WHERE reviewed_at IS NOT NULL AND added_to_dataset = 0
            ORDER BY reviewed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_candidate(r) for r in rows]
    finally:
        conn.close()


def get_candidate(candidate_id: str, db_path: str | None = None) -> AnnotationCandidate | None:
    path = db_path or _db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM annotation_candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
        return _row_to_candidate(row) if row else None
    finally:
        conn.close()


def submit_review(
    candidate_id: str,
    expected_barcodes: list[dict[str, Any]],
    expected_outcome: str,
    reviewer: str,
    db_path: str | None = None,
) -> bool:
    """Record a human review. Returns True if the candidate was found and updated."""
    path = db_path or _db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            """
            UPDATE annotation_candidates
            SET reviewed_at = ?, reviewed_by = ?,
                expected_barcodes = ?, expected_outcome = ?
            WHERE id = ? AND reviewed_at IS NULL
            """,
            (
                datetime.now(UTC).isoformat(),
                reviewer,
                json.dumps(expected_barcodes),
                expected_outcome,
                candidate_id,
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def mark_exported(candidate_ids: list[str], db_path: str | None = None) -> None:
    """Mark candidates as added_to_dataset. Called only after export succeeds."""
    if not candidate_ids:
        return
    path = db_path or _db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        placeholders = ",".join("?" for _ in candidate_ids)
        conn.execute(
            f"UPDATE annotation_candidates SET added_to_dataset = 1 WHERE id IN ({placeholders})",
            candidate_ids,
        )
        conn.commit()
    finally:
        conn.close()


def get_stats(db_path: str | None = None) -> dict[str, int]:
    """Return counts: pending, reviewed, exported, total."""
    path = db_path or _db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        pending = conn.execute(
            "SELECT COUNT(*) FROM annotation_candidates WHERE reviewed_at IS NULL"
        ).fetchone()[0]
        reviewed = conn.execute(
            "SELECT COUNT(*) FROM annotation_candidates "
            "WHERE reviewed_at IS NOT NULL AND added_to_dataset = 0"
        ).fetchone()[0]
        exported = conn.execute(
            "SELECT COUNT(*) FROM annotation_candidates WHERE added_to_dataset = 1"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM annotation_candidates").fetchone()[0]
    finally:
        conn.close()
    return {"pending": pending, "reviewed": reviewed, "exported": exported, "total": total}


# ---------------------------------------------------------------------------
# Export to dataset.json — transaction-like
# ---------------------------------------------------------------------------


def export_to_dataset_json(
    db_path: str | None = None,
    dataset_path: Path | None = None,
    samples_dir: Path | None = None,
) -> int:
    """Export reviewed candidates to dataset.json. Returns count added.

    Transaction-like: ``added_to_dataset = 1`` only after image copy +
    JSON construction + atomic rename all succeed. If any step fails for
    a candidate, that candidate is skipped and remains eligible for retry.

    Candidates with ``image_ref = None`` or missing image files are
    skipped with a warning.
    """
    path = db_path or _db_path()
    dpath = Path(dataset_path) if dataset_path else DATASET_PATH
    sdir = Path(samples_dir) if samples_dir else SAMPLES_DIR

    candidates = list_reviewed(limit=1000, db_path=path)
    if not candidates:
        return 0

    # Load existing dataset.
    with dpath.open() as f:
        dataset = json.load(f)

    new_entries: list[dict[str, Any]] = []
    exported_ids: list[str] = []

    for cand in candidates:
        if not cand.image_ref:
            logger.warning(
                "Skipping export for candidate %s — no image_ref (run_id=%s)",
                cand.id, cand.run_id,
            )
            continue

        src = Path(cand.image_ref)
        if not src.exists():
            logger.warning(
                "Skipping export for candidate %s — image not found: %s (run_id=%s)",
                cand.id, src, cand.run_id,
            )
            continue

        # Copy image to samples/ with a unique name.
        ext = src.suffix or ".jpg"
        dest_name = f"annotation_{cand.run_id}{ext}"
        dest = sdir / dest_name
        try:
            dest.write_bytes(src.read_bytes())
        except OSError:
            logger.exception("Failed to copy image %s → %s", src, dest)
            continue

        # Build the dataset entry.
        boxes = [
            {
                "status": "decoded",
                "value": b.get("value", ""),
                "format": b.get("format", "Code128"),
            }
            for b in (cand.expected_barcodes or [])
        ]
        entry = {
            "image": dest_name,
            "expected_barcode_symbol_count": len(boxes),
            "boxes": boxes,
            "source": "annotation",
            "annotation_run_id": cand.run_id,
        }
        new_entries.append(entry)
        exported_ids.append(cand.id)

    if not new_entries:
        return 0

    # Construct new dataset content.
    dataset["images"].extend(new_entries)

    # Recalculate summary.
    all_images = dataset["images"]
    total_symbols = sum(img["expected_barcode_symbol_count"] for img in all_images)
    total_decoded = sum(
        len([b for b in img["boxes"] if b.get("status") == "decoded"])
        for img in all_images
    )
    dataset["summary"]["image_count"] = len(all_images)
    dataset["summary"]["expected_barcode_symbol_count"] = total_symbols
    dataset["summary"]["expected_decoded_count"] = total_decoded
    dataset["summary"]["images"] = [img["image"] for img in all_images]

    # Atomic write: temp file + rename.
    dpath.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dpath.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(dataset, f, indent=2)
        os.rename(tmp, str(dpath))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    # Only mark exported after the rename succeeded.
    mark_exported(exported_ids, db_path=path)

    logger.info("Exported %d annotation candidates to %s", len(exported_ids), dpath)
    return len(exported_ids)
