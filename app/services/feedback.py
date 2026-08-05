from uuid import UUID

from langsmith import Client


def submit_upload_feedback(
    *,
    trace_id: UUID,
    correct: bool,
    comment: str | None = None,
) -> int:
    """Submit user feedback (correct/incorrect) to LangSmith on a trace.

    Uses a single key ``user_correct`` with score 1 (correct) or 0 (incorrect)
    so filtering and queue automation are simple.
    """
    score = 1 if correct else 0

    client = Client()
    client.create_feedback(
        key="user_correct",
        score=score,
        trace_id=trace_id,
        comment=comment,
    )

    return score
