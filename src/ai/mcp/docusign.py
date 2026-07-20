"""Read-only Docusign MCP connection and one-time local OAuth setup."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import secrets
import time
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from ai.config.settings import PROJECT_ROOT, settings

DOCUSIGN_MCP_URL = "https://mcp-d.docusign.com/mcp"
DOCUSIGN_AUTH_URL = "https://account-d.docusign.com/oauth/auth"
DOCUSIGN_TOKEN_URL = "https://account-d.docusign.com/oauth/token"
DOCUSIGN_REDIRECT_URI = "http://127.0.0.1:8765/callback"
DOCUSIGN_SCOPES = "adm_store_unified_repo_read signature"
DOCUSIGN_TOOLS = ("getUserInfo", "getAllAgreements", "getAgreementDetails")
_CALLBACK_TIMEOUT_SECONDS = 300


def _token_path(path: Path) -> Path:
    candidate = path.expanduser()
    return (PROJECT_ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _user_token_path(path: Path, user_id: str) -> Path:
    resolved = _token_path(path)
    cleaned_user_id = user_id.strip()
    if not cleaned_user_id:
        raise ValueError("user_id is required for Docusign token isolation")
    user_key = hashlib.sha256(cleaned_user_id.encode()).hexdigest()[:16]
    return resolved.with_name(f"{resolved.stem}.{user_key}{resolved.suffix}")


def _load_token(path: Path) -> dict[str, Any]:
    resolved = _token_path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("Docusign 未授权，请先运行 docusign-mcp-auth。") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Docusign OAuth token 文件无效。") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
        raise RuntimeError("Docusign OAuth token 文件缺少 access_token。")
    return payload


def _save_token(path: Path, payload: dict[str, Any]) -> None:
    resolved = _token_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(f"{resolved.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(resolved)


def _normalized_token(
    payload: dict[str, Any],
    client_id: str,
    fallback_refresh_token: str | None = None,
) -> dict[str, Any]:
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("Docusign OAuth 响应缺少 access_token。")
    expires_in = payload.get("expires_in", 3600)
    if not isinstance(expires_in, int | float) or expires_in <= 0:
        raise RuntimeError("Docusign OAuth 响应包含无效 expires_in。")
    refresh_token = payload.get("refresh_token") or fallback_refresh_token
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": payload.get("token_type", "Bearer"),
        "scope": payload.get("scope", DOCUSIGN_SCOPES),
        "expires_at": time.time() + float(expires_in),
        "client_id": client_id,
    }


async def _access_token(
    client: httpx.AsyncClient,
    *,
    client_id: str,
    client_secret: str,
    token_path: Path,
) -> str:
    token = _load_token(token_path)
    if token.get("client_id") != client_id:
        raise RuntimeError("Docusign token 与当前 Client ID 不匹配，请重新授权。")
    if float(token.get("expires_at") or 0) > time.time() + 60:
        return str(token["access_token"])
    refresh_token = token.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise RuntimeError("Docusign token 已过期且无法刷新，请重新运行 docusign-mcp-auth。")
    response = await client.post(
        DOCUSIGN_TOKEN_URL,
        auth=httpx.BasicAuth(client_id, client_secret),
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "resource": DOCUSIGN_MCP_URL,
        },
    )
    response.raise_for_status()
    refreshed = _normalized_token(response.json(), client_id, refresh_token)
    _save_token(token_path, refreshed)
    return str(refreshed["access_token"])


@asynccontextmanager
async def open_docusign_session(
    client_id: str,
    client_secret: str,
    token_path: Path,
    user_id: str,
) -> AsyncIterator[ClientSession]:
    """Open one authenticated Streamable HTTP MCP session."""
    timeout = httpx.Timeout(30.0, read=300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        token = await _access_token(
            client,
            client_id=client_id,
            client_secret=client_secret,
            token_path=_user_token_path(token_path, user_id),
        )
        client.headers["Authorization"] = f"Bearer {token}"
        async with (
            streamable_http_client(DOCUSIGN_MCP_URL, http_client=client) as streams,
            ClientSession(streams[0], streams[1]) as session,
        ):
            await session.initialize()
            yield session


def _authorization_url(client_id: str, state: str, verifier: str) -> str:
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    parameters = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": DOCUSIGN_REDIRECT_URI,
        "scope": DOCUSIGN_SCOPES,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": DOCUSIGN_MCP_URL,
    }
    return f"{DOCUSIGN_AUTH_URL}?{urlencode(parameters)}"


def _receive_authorization_code(url: str, expected_state: str) -> str:
    result: dict[str, str] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = parse_qs(urlparse(self.path).query)
            result["code"] = query.get("code", [""])[0]
            result["state"] = query.get("state", [""])[0]
            result["error"] = query.get("error", [""])[0]
            body = "Docusign 授权已返回，可以关闭此页面。".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    with HTTPServer(("127.0.0.1", 8765), CallbackHandler) as server:
        server.timeout = _CALLBACK_TIMEOUT_SECONDS
        print(f"请在浏览器完成 Docusign 授权：\n{url}")
        webbrowser.open(url)
        server.handle_request()

    if result.get("error"):
        raise RuntimeError(f"Docusign 授权失败：{result['error']}")
    if not secrets.compare_digest(result.get("state", ""), expected_state):
        raise RuntimeError("Docusign OAuth state 校验失败。")
    if not result.get("code"):
        raise RuntimeError("等待 Docusign OAuth 回调超时。")
    return result["code"]


async def _exchange_authorization_code(
    code: str,
    verifier: str,
    client_id: str,
    client_secret: str,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            DOCUSIGN_TOKEN_URL,
            auth=httpx.BasicAuth(client_id, client_secret),
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": DOCUSIGN_REDIRECT_URI,
                "code_verifier": verifier,
                "resource": DOCUSIGN_MCP_URL,
            },
        )
        response.raise_for_status()
        return _normalized_token(response.json(), client_id)


def main() -> None:
    """Authorize a developer account once and save the refreshable token locally."""
    parser = argparse.ArgumentParser(description="Authorize Docusign MCP for one app user.")
    parser.add_argument("--user-id", required=True, help="user_id returned by GET /api/v1/auth/me")
    args = parser.parse_args()
    client_id = settings.docusign_client_id.strip()
    client_secret = settings.docusign_client_secret.strip()
    if not client_id or not client_secret:
        raise SystemExit("请先配置 DOC_ASSISTANT_DOCUSIGN_CLIENT_ID 和 CLIENT_SECRET。")
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    code = _receive_authorization_code(
        _authorization_url(client_id, state, verifier),
        state,
    )
    token = asyncio.run(_exchange_authorization_code(code, verifier, client_id, client_secret))
    token_path = _user_token_path(settings.docusign_token_path, args.user_id)
    _save_token(token_path, token)
    print(f"Docusign MCP 授权完成：{token_path}")


if __name__ == "__main__":
    main()
