"""SQLite connection and transaction management for matter persistence."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from doc_assistant.matter import _sql as sql


class MatterDatabase:
    """Own SQLite lifecycle and guarantee that every opened connection is closed."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = Lock()
        self._initialize()

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection, with an explicit transaction for writes."""
        if write:
            with self._write_lock, self._managed_connection(write=True) as connection:
                yield connection
            return

        with self._managed_connection(write=False) as connection:
            yield connection

    @contextmanager
    def _managed_connection(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("BEGIN IMMEDIATE")
            for statement in sql.SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(sql.BACKFILL_ARTIFACT_VERSIONS)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
