from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from ai.config.settings import settings
from backend.auth_store import UsernameAlreadyExistsError
from backend.dependencies import (
    SESSION_COOKIE_NAME,
    AuthStoreDep,
    CurrentUserDep,
    SessionTokenDep,
)
from backend.schemas.requests import AuthRequest
from backend.schemas.responses import UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: AuthRequest, response: Response, auth_store: AuthStoreDep) -> UserResponse:
    try:
        user = auth_store.register(payload.username, payload.password.get_secret_value())
    except UsernameAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    _set_session_cookie(response, auth_store.create_session(user.user_id))
    return UserResponse.model_validate(user)


@router.post("/login", response_model=UserResponse)
def login(payload: AuthRequest, response: Response, auth_store: AuthStoreDep) -> UserResponse:
    user = auth_store.authenticate(payload.username, payload.password.get_secret_value())
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    _set_session_cookie(response, auth_store.create_session(user.user_id))
    return UserResponse.model_validate(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    response: Response,
    token: SessionTokenDep,
    _current_user: CurrentUserDep,
    auth_store: AuthStoreDep,
) -> None:
    auth_store.revoke_session(token)
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        secure=settings.auth_cookie_secure,
        httponly=True,
        samesite="lax",
    )


@router.get("/me", response_model=UserResponse)
def me(current_user: CurrentUserDep) -> UserResponse:
    return UserResponse.model_validate(current_user)


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=settings.auth_session_ttl_hours * 3600,
        path="/",
        secure=settings.auth_cookie_secure,
        httponly=True,
        samesite="lax",
    )
    response.headers["Cache-Control"] = "no-store"
