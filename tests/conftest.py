from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend import dependencies
from backend.auth_store import UserRecord
from backend.main import app


@pytest.fixture(autouse=True)
def authenticated_api_user(request):
    if request.node.get_closest_marker("real_auth"):
        yield
        return

    user = UserRecord(
        user_id="api-test-user",
        username="api-test-user",
        created_at=datetime.now(UTC),
    )
    def override() -> UserRecord:
        return user

    app.dependency_overrides[dependencies.get_current_user] = override
    try:
        yield
    finally:
        if app.dependency_overrides.get(dependencies.get_current_user) is override:
            app.dependency_overrides.pop(dependencies.get_current_user, None)
