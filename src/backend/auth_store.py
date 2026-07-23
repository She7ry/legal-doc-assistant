from __future__ import annotations

import base64
import hashlib
import re
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

_USERNAME_PATTERN = re.compile(r"^[\w.@+-]{3,64}$", re.UNICODE)
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_LENGTH = 32


class UsernameAlreadyExistsError(ValueError):
    pass


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    username: str
    created_at: datetime


class AuthStore:
    def __init__(self, db_path: Path, *, session_ttl_hours: int = 24 * 7) -> None:
        self.db_path = Path(db_path)
        self.session_ttl = timedelta(hours=max(1, session_ttl_hours))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def register(self, username: str, password: str) -> UserRecord:
        username = normalize_username(username)
        validate_password(password)
        user = UserRecord(
            user_id=uuid4().hex,
            username=username,
            created_at=datetime.now(UTC),
        )
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO users (user_id, username, username_key, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user.user_id,
                        user.username,
                        user.username.casefold(),
                        hash_password(password),
                        user.created_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise UsernameAlreadyExistsError("Username is already registered.") from exc
        return user

    def authenticate(self, username: str, password: str) -> UserRecord | None:
        try:
            username_key = normalize_username(username).casefold()
        except ValueError:
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, username, password_hash, created_at FROM users WHERE username_key = ?",
                (username_key,),
            ).fetchone()
        stored_hash = str(row["password_hash"]) if row is not None else _DUMMY_PASSWORD_HASH
        valid_password = verify_password(password, stored_hash)
        if row is None or not valid_password:
            return None
        return _user_from_row(row)

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = datetime.now(UTC)
        expires_at = now + self.session_ttl
        with self._connection() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now.isoformat(),))
            connection.execute(
                """
                INSERT INTO sessions (token_hash, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (_token_hash(token), user_id, now.isoformat(), expires_at.isoformat()),
            )
        return token

    def resolve_session(self, token: str) -> UserRecord | None:
        if not 20 <= len(token) <= 256:
            return None
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT users.user_id, users.username, users.created_at, sessions.expires_at
                FROM sessions
                JOIN users ON users.user_id = sessions.user_id
                WHERE sessions.token_hash = ?
                """,
                (_token_hash(token),),
            ).fetchone()
            if row is None:
                return None
            if datetime.fromisoformat(str(row["expires_at"])) <= datetime.now(UTC):
                connection.execute(
                    "DELETE FROM sessions WHERE token_hash = ?",
                    (_token_hash(token),),
                )
                return None
        return _user_from_row(row)

    def revoke_session(self, token: str) -> None:
        if not token:
            return
        with self._connection() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    username_key TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
                """
            )


def normalize_username(value: str) -> str:
    username = value.strip()
    if not _USERNAME_PATTERN.fullmatch(username):
        raise ValueError(
            "Username must be 3-64 characters and contain only letters, numbers, _, ., @, +, or -."
        )
    return username


def validate_password(password: str) -> None:
    if not 8 <= len(password) <= 128:
        raise ValueError("Password must be 8-128 characters.")


def hash_password(password: str) -> str:
    validate_password(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_LENGTH,
    )
    return "$".join(
        (
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            _encode(salt),
            _encode(digest),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$", 5)
        if (
            algorithm != "scrypt"
            or int(n) != _SCRYPT_N
            or int(r) != _SCRYPT_R
            or int(p) != _SCRYPT_P
        ):
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(_decode(expected)),
        )
        return secrets.compare_digest(actual, _decode(expected))
    except (TypeError, ValueError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))


def _user_from_row(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        user_id=str(row["user_id"]),
        username=str(row["username"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


_DUMMY_PASSWORD_HASH = hash_password("dummy-password-not-used-for-login")
