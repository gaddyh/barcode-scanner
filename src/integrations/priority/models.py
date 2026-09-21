"""Domain shapes for the Priority gateway boundary.

These frozen dataclasses are the canonical types that cross the
``PriorityGateway`` protocol. Both the local adapter and any future real
Priority adapter return these exact shapes — callers never see raw
asyncpg rows or HTTP response dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Customer:
    """A Priority customer (account)."""

    id: str
    name: str


@dataclass(frozen=True)
class Branch:
    """A Priority branch (warehouse/store) belonging to a customer."""

    id: str
    name: str
    customer_id: str


@dataclass(frozen=True)
class OrderLineItem:
    """One aggregated line in a draft order.

    ``barcode_value`` is the decoded barcode string. ``quantity`` is the
    occurrence count (multiset) — duplicate physical boxes with the same
    barcode value count separately.
    """

    barcode_value: str
    barcode_format: str
    quantity: int
    label_index: int | None = None


@dataclass(frozen=True)
class CreateDraftOrderRequest:
    """Request to create a draft order in the Priority-compatible store.

    ``session_id`` is the logical idempotency identity — the same session
    must always produce the same order (enforced by the runtime executor
    + idempotency store in the service layer, and by a UNIQUE constraint
    on ``priority_orders.session_id`` as defense-in-depth).
    """

    session_id: str
    customer_id: str
    branch_id: str
    action: str
    items: list[OrderLineItem] = field(default_factory=list)

    def items_as_dicts(self) -> list[dict[str, Any]]:
        """Serialize items to the JSONB shape stored in ``priority_orders``."""
        return [
            {
                "barcode_value": item.barcode_value,
                "barcode_format": item.barcode_format,
                "quantity": item.quantity,
                "label_index": item.label_index,
            }
            for item in self.items
        ]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CreateDraftOrderRequest:
        """Reconstruct a request from a persisted frozen payload dict.

        The frozen payload is the exact dict persisted at ACTIVE →
        SUBMITTING and reused on every retry. Rebuilding the request
        from it (rather than from the live session) guarantees the
        invariant: same idempotency key → exact same payload.
        """
        items = [
            OrderLineItem(
                barcode_value=item["barcode_value"],
                barcode_format=item.get("barcode_format", ""),
                quantity=item["quantity"],
                label_index=item.get("label_index"),
            )
            for item in payload.get("items", [])
        ]
        return cls(
            session_id=payload["session_id"],
            customer_id=payload["customer_id"],
            branch_id=payload["branch_id"],
            action=payload["action"],
            items=items,
        )


@dataclass(frozen=True)
class CreateDraftOrderResult:
    """Result of creating a draft order."""

    order_id: int
    session_id: str
    status: str = "draft"
