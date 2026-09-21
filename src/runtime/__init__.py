"""Runtime package — execution context, executor, structured errors, and events."""

from src.runtime.context import RunContext
from src.runtime.errors import (
    ExecutionError,
    IndeterminateError,
    InvalidInputError,
    PermanentError,
    RetryableError,
    TimeoutError,
)
from src.runtime.events import DomainEvent, EventType
from src.runtime.executor import ExecutionResult, execute
from src.runtime.idempotency import (
    IdempotencyStore,
    IndeterminateOutcome,
    InMemoryIdempotencyStore,
    LostOwnershipError,
    PermanentFailureOutcome,
    ReserveResult,
    ReserveStatus,
    StoredOutcome,
    SuccessOutcome,
)
from src.runtime.policy import (
    EXTERNAL_READ,
    EXTERNAL_WRITE,
    NO_RETRY,
    SCAN_COMPUTE,
    ExecutionPolicy,
)

__all__ = [
    "RunContext",
    "execute",
    "ExecutionResult",
    "ExecutionPolicy",
    "ExecutionError",
    "RetryableError",
    "TimeoutError",
    "PermanentError",
    "IndeterminateError",
    "InvalidInputError",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "IndeterminateOutcome",
    "LostOwnershipError",
    "PermanentFailureOutcome",
    "ReserveResult",
    "ReserveStatus",
    "StoredOutcome",
    "SuccessOutcome",
    "NO_RETRY",
    "SCAN_COMPUTE",
    "EXTERNAL_READ",
    "EXTERNAL_WRITE",
    "DomainEvent",
    "EventType",
]
