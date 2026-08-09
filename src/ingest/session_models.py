"""Typed domain models for multi-image ingest sessions.

A session represents the real user workflow:

    Photo #1 → 12 visible, 11 found, 1 missing
    → "send another photo of the missing box"
    Photo #2 → scans the missing box
    → merge into session → 12/12 → session complete

This is NOT human-in-the-loop approval. It's a multi-turn session with
missing information. The user provides more input; the system accumulates
results until all expected boxes are resolved.

Key design decisions:

- **Identity by barcode value**: boxes are tracked across images by their
  decoded barcode value. A box found in photo #1 and again in photo #2 is
  the same box — deduplicated, not double-counted.

- **Missing boxes tracked by label metadata**: when a box has no readable
  barcode, we track its Gemini label metadata (label_index, label_bbox,
  barcode_bbox) so we can ask the user to photograph that region.

- **Expected count from first audit**: the Gemini audit of the first image
  tells us how many boxes are visible. This becomes the session's expected
  count. Subsequent photos may show fewer boxes (close-ups) — we don't
  update the expected count, we just try to resolve the missing ones.

- **Don't assume photo #2 is a close-up**: the user might photograph 3 boxes
  that include the missing one. We scan all of them and merge by barcode
  value. Already-known barcodes are deduplicated; new barcodes resolve
  missing entries.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SessionStatus(StrEnum):
    """Lifecycle state of an ingest session."""

    ACTIVE = "active"              # session is accepting more images
    COMPLETE = "complete"          # all expected boxes have barcodes
    EXPIRED = "expired"            # session timed out without completing
    CLOSED = "closed"              # session explicitly closed by client
    FAILED = "failed"              # unrecoverable error (e.g. Gemini down)
    NEEDS_USER_SELECTION = "needs_user_selection"  # ambiguous: user must pick


class SessionItem(BaseModel):
    """A barcode confirmed in the session, accumulated across images.

    Identified by ``barcode_value`` — this is the deduplication key.
    If the same barcode appears in multiple photos, it's the same item.
    """

    barcode_value: str
    barcode_format: str | None = None
    barcode_bbox: dict[str, Any] | None = None
    label_bbox: dict[str, Any] | None = None
    label_index: int | None = None          # from the image that first found it
    match_basis: str | None = None
    source_image: int = 0                   # 0-based image index in the session


class MissingItem(BaseModel):
    """A visible box with no decoded barcode, tracked across the session.

    Identified by label metadata from the Gemini audit. When a subsequent
    photo decodes a barcode in this region, the missing item is resolved
    and promoted to a ``SessionItem``.
    """

    label_index: int | None = None
    label_bbox: dict[str, Any] | None = None
    barcode_bbox: dict[str, Any] | None = None
    status: str = "not_visible"             # Gemini label status
    source_image: int = 0                   # which image reported it missing
    resolved: bool = False                  # set True when a later photo decodes it


class ImageResult(BaseModel):
    """The result of running ScanGraph on one image within a session.

    This is a slimmed-down version of ``IngestResult`` — only the fields
    needed for session-level merge. The full ``IngestResult`` is still
    persisted in the ``runs`` table for operational/debugging purposes.
    """

    image_index: int = 0
    run_id: str | None = None               # links to the runs table
    status: str = "complete"                # IngestStatus value
    found: list[SessionItem] = Field(default_factory=list)
    missing: list[MissingItem] = Field(default_factory=list)
    unassigned: list[dict[str, Any]] = Field(default_factory=list)
    visible_label_count: int = 0
    found_count: int = 0
    missing_count: int = 0
    elapsed_ms: int = 0
    audit_available: bool = False
    error: dict[str, Any] | None = None


class SessionResult(BaseModel):
    """The API response for ``ingest_session(session_id, image)``.

    This is what the client sees after each image submission. The client
    checks ``status`` to decide whether to ask for another photo or show
    the final result.
    """

    session_id: str
    status: SessionStatus
    expected_count: int = 0                 # total visible boxes (from first audit)
    found_count: int = 0                    # unique barcodes confirmed so far
    missing_count: int = 0                  # boxes still without barcodes
    items: list[SessionItem] = Field(default_factory=list)
    missing: list[MissingItem] = Field(default_factory=list)
    image_count: int = 0                    # how many images submitted so far
    message: str | None = None              # human-readable prompt for next photo

    # The result of the most recent image (for immediate display)
    latest_image: ImageResult | None = None

    # When status == NEEDS_USER_SELECTION: the candidate barcodes from the
    # latest image that the user must choose from to resolve missing labels.
    candidates: list[SessionItem] = Field(default_factory=list)
