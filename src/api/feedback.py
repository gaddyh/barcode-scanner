from langsmith import Client


def submit_upload_feedback(
    *,
    trace_id: str,
    correct: bool,
    comment: str | None = None,
) -> int:
    """Submit user feedback (correct/incorrect) to LangSmith on a trace.

    Uses a single key ``user_correct`` with score 1 (correct) or 0 (incorrect)
    so filtering and queue automation are simple.

    ``trace_id`` is a string (UUID or ULID) — the receiving flow uses
    session_id (UUID), the legacy scanner flow uses upload_id (ULID).
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
