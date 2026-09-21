"""Domain layer for the receiving flow — the product-level model of what
happens when a warehouse receives shoeboxes and creates a draft order.

This module is intentionally pure: no I/O, no FastAPI, no asyncpg. It
operates on plain dataclasses and ``collections.Counter``. The API
layer (``src/api/receiving.py``) and the session repository
(``src/session_repository.py``) translate between these domain types
and their own storage shapes.

Key semantics (from AGENTS.md):

- Physical boxes are occurrences (multiset), not unique barcode values.
  Duplicate barcode values represent separate physical boxes and count
  separately.
- On entering ``SUBMITTING``, session contents are FROZEN (immutable).
  Retrying uses the same ``priority:draft:{session_id}`` key and the
  same frozen payload.
- ``session_id`` is the logical idempotency identity — same session →
  same key → double-submit returns the cached order ID, never a
  duplicate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum


class ReceivingSessionStatus(StrEnum):
    """Lifecycle state of a receiving session's submission flow.

    This is distinct from the ingest session's ``SessionStatus`` (which
    tracks photo accumulation). The receiving submission state machine
    is:

        ACTIVE
           ↓ freeze payload
        SUBMITTING
           ↓ success
        SUBMITTED

        SUBMITTING
           ↓ unknown external outcome
        SUBMISSION_UNKNOWN

    - ``ACTIVE`` — session is accepting images / edits. The draft order
      has NOT been created yet.
    - ``SUBMITTING`` — contents are frozen, the draft-order request is
      in flight (or being retried). Edits require a new session.
    - ``SUBMITTED`` — the draft order was created successfully.
      ``external_order_id`` is populated.
    - ``SUBMISSION_UNKNOWN`` — the draft-order request returned an
      indeterminate outcome (timeout / disconnect after submit). The
      idempotency store replays the outcome on retry; the user must NOT
      create a new session/order blindly.
    """

    ACTIVE = "active"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    SUBMISSION_UNKNOWN = "submission_unknown"


@dataclass(frozen=True)
class PhysicalBox:
    """One physical shoebox detected by the scanner.

    A physical box is an occurrence, not a unique barcode value. If the
    same barcode value appears on two boxes, those are two separate
    ``PhysicalBox`` instances with the same ``barcode_value``.
    """

    barcode_value: str
    barcode_format: str = ""
    label_index: int | None = None

    def __post_init__(self) -> None:
        if not self.barcode_value:
            raise ValueError("barcode_value must be non-empty")


@dataclass(frozen=True)
class Discrepancy:
    """A mismatch between the expected count (from the Gemini audit) and
    the actual decoded barcodes.

    ``expected`` is the number of visible boxes the audit reported.
    ``found`` is the number of barcodes decoded. ``missing`` is the
    number of boxes without barcodes (``expected - found``).
    """

    expected: int
    found: int

    @property
    def missing(self) -> int:
        return max(0, self.expected - self.found)

    @property
    def is_complete(self) -> bool:
        return self.expected > 0 and self.found >= self.expected


@dataclass
class ReceivingSession:
    """A receiving session — the product-level model of one customer +
    branch + set of photos + draft order.

    ``boxes`` is a list of physical box occurrences (multiset). The same
    barcode value may appear multiple times — each occurrence is a
    separate physical box.

    ``status`` tracks the submission state machine (ACTIVE → SUBMITTING
    → SUBMITTED / SUBMISSION_UNKNOWN). Once ``status`` leaves ACTIVE,
    ``boxes`` is FROZEN — the ``frozen`` flag enforces this.

    ``external_order_id`` is populated when the draft order is
    successfully created in Priority.
    """

    session_id: str
    customer_id: str
    branch_id: str
    action: str
    boxes: list[PhysicalBox] = field(default_factory=list)
    status: ReceivingSessionStatus = ReceivingSessionStatus.ACTIVE
    external_order_id: int | None = None
    expected_count: int = 0
    frozen: bool = False

    def add_box(self, box: PhysicalBox) -> None:
        """Append a physical box occurrence. Raises if frozen."""
        if self.frozen:
            raise self._frozen_error()
        self.boxes.append(box)

    def add_boxes(self, boxes: list[PhysicalBox]) -> None:
        """Append multiple physical box occurrences. Raises if frozen."""
        if self.frozen:
            raise self._frozen_error()
        self.boxes.extend(boxes)

    def freeze(self) -> None:
        """Transition to SUBMITTING and freeze contents.

        Once frozen, ``add_box`` / ``add_boxes`` raise. This makes
        ``session_id`` a safe logical idempotency key — the payload
        cannot change after submission starts.
        """
        if self.status != ReceivingSessionStatus.ACTIVE:
            raise ValueError(
                f"Cannot freeze session in status {self.status.value}"
            )
        self.status = ReceivingSessionStatus.SUBMITTING
        self.frozen = True

    def mark_submitted(self, external_order_id: int) -> None:
        """Transition to SUBMITTED after a successful draft-order create."""
        if self.status != ReceivingSessionStatus.SUBMITTING:
            raise ValueError(
                f"Cannot mark submitted from status {self.status.value}"
            )
        self.external_order_id = external_order_id
        self.status = ReceivingSessionStatus.SUBMITTED

    def mark_submission_unknown(self) -> None:
        """Transition to SUBMISSION_UNKNOWN after an indeterminate outcome.

        The idempotency store will replay the outcome on retry with the
        same key. The user must NOT create a new session/order blindly.
        Accepts both SUBMITTING and SUBMISSION_UNKNOWN (idempotent
        re-transition on retry — the outcome is still indeterminate).
        """
        if self.status not in (
            ReceivingSessionStatus.SUBMITTING,
            ReceivingSessionStatus.SUBMISSION_UNKNOWN,
        ):
            raise ValueError(
                f"Cannot mark submission_unknown from status "
                f"{self.status.value}"
            )
        self.status = ReceivingSessionStatus.SUBMISSION_UNKNOWN

    def revert_to_active(self) -> None:
        """Revert SUBMITTING → ACTIVE (pre-submit failure).

        If the call fails BEFORE submission (validation, connection
        refused), stay ACTIVE so the user can retry/edit.
        """
        if self.status != ReceivingSessionStatus.SUBMITTING:
            raise ValueError(
                f"Cannot revert to active from status {self.status.value}"
            )
        self.status = ReceivingSessionStatus.ACTIVE
        self.frozen = False

    # --- Aggregation --------------------------------------------------

    def aggregate_quantities(self) -> list[tuple[str, str, int]]:
        """Aggregate barcode occurrences into (value, format, quantity) tuples.

        Physical boxes are a multiset: duplicate barcode values count
        separately. ``quantity`` is the occurrence count per barcode
        value. Returns a sorted list for deterministic output.
        """
        counts: Counter[str] = Counter()
        formats: dict[str, str] = {}
        for box in self.boxes:
            counts[box.barcode_value] += 1
            if box.barcode_format and box.barcode_value not in formats:
                formats[box.barcode_value] = box.barcode_format
        return sorted(
            (value, formats.get(value, ""), count)
            for value, count in counts.items()
        )

    @property
    def discrepancy(self) -> Discrepancy:
        """Discrepancy between expected and found box counts."""
        return Discrepancy(
            expected=self.expected_count,
            found=len(self.boxes),
        )

    @property
    def idempotency_key(self) -> str:
        """The stable logical idempotency key for this session's draft order.

        Same session → same key → double-submit returns the cached
        order ID, never creates a duplicate.
        """
        return f"priority:draft:{self.session_id}"

    def _frozen_error(self) -> ValueError:
        return ValueError(
            f"Session {self.session_id} is frozen (status={self.status.value}); "
            f"cannot add boxes. Create a new session to edit."
        )
