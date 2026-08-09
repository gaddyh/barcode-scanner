from uuid import UUID

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    trace_id: UUID
    correct: bool
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    status: str
    trace_id: UUID
    score: int
