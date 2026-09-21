"""PriorityGateway protocol — the boundary between the application and ERP.

The protocol is intentionally small: catalog reads (customers, branches)
and one irreversible write (create_draft_order). The local adapter
implements this against PostgreSQL; a future real Priority adapter will
implement it against the Priority REST API.

Layering (per AGENTS.md):

    application/service
        ↓ runtime.execute(policy=EXTERNAL_WRITE, idempotency_key=...)
    PriorityGateway (protocol)
        ↓
    LocalPriorityGateway / RealPriorityGateway (adapter performs the op
                                              + classifies errors)

The adapter IS responsible for error classification at the integration
boundary (per PR #3): failure before submission → ``RetryableError`` /
``PermanentError``; timeout/disconnect after submission may have occurred
→ ``IndeterminateError``. The executor preserves and persists what the
adapter raises.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.integrations.priority.models import (
    Branch,
    CreateDraftOrderRequest,
    CreateDraftOrderResult,
    Customer,
)


@runtime_checkable
class PriorityGateway(Protocol):
    """Boundary protocol for the Priority-compatible ERP."""

    async def customers(self) -> list[Customer]:
        """Return all active customers."""
        ...

    async def branches(self, customer_id: str) -> list[Branch]:
        """Return active branches for ``customer_id``."""
        ...

    async def create_draft_order(
        self, request: CreateDraftOrderRequest
    ) -> CreateDraftOrderResult:
        """Create a draft order.

        This is an irreversible external write. The service layer wraps
        this call in ``runtime.execute(policy=EXTERNAL_WRITE,
        idempotency_key=...)`` so double-submit/retry cannot create a
        duplicate order. The adapter itself does NOT wrap in
        ``runtime.execute`` — that is the service layer's job.
        """
        ...
