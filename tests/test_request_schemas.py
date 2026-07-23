from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.schemas.requests import AskRequest


def test_chat_history_rejects_oversized_content_and_message_lists() -> None:
    with pytest.raises(ValidationError):
        AskRequest(
            question="Continue.",
            chat_history=[{"role": "user", "content": "x" * 8001}],
        )

    with pytest.raises(ValidationError):
        AskRequest(
            question="Continue.",
            chat_history=[{"role": "user", "content": "ok"}] * 51,
        )
