"""Priority gateway package — protocol, domain models, and local adapter.

Public API:
    PriorityGateway — protocol (port).
    LocalPriorityGateway — asyncpg-backed local stand-in.
    PriorityError — legacy error class (kept for callers that catch it).
    Customer, Branch, OrderLineItem, CreateDraftOrderRequest,
    CreateDraftOrderResult — domain shapes.
"""

from __future__ import annotations

from src.integrations.priority.local import (
    LocalPriorityGateway,
    PriorityError,
)
from src.integrations.priority.models import (
    Branch,
    CreateDraftOrderRequest,
    CreateDraftOrderResult,
    Customer,
    OrderLineItem,
)
from src.integrations.priority.port import PriorityGateway

__all__ = [
    "Branch",
    "CreateDraftOrderRequest",
    "CreateDraftOrderResult",
    "Customer",
    "LocalPriorityGateway",
    "OrderLineItem",
    "PriorityError",
    "PriorityGateway",
]
