"""On-disk TTL cache.

Two backends, because the two payload shapes have very different needs:

* **SQLite** for small JSON payloads (quotes, news listings, fundamentals,
  sentiment results). One file, transactional, no server process.
* **Parquet** for OHLCV frames. Columnar, typed, ~10x smaller and far faster
  to reload than JSON, and it round-trips the DatetimeIndex without the
  string-parsing dance JSON would force.

Every entry records the timestamp it was fetched at, so the API can tell the
user how fresh the data actually is instead of implying it is live.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CachedValue:
    """A cache hit, with the provenance the UI needs to label freshness."""

    value: Any
    fetched_at: float
    """Unix timestamp at which the underlying source was actually queried."""

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.fetched_at)


def cache_key(*parts: Any) -> str:
    """Build a stable cache key from arbitrary parts.

    Long or unhashable-looking parts (e.g. an article body used to key a
    sentiment result) are hashed so keys stay bounded in length.
    """
    rendered: list[str] = []
    for part in parts:
        text = part if isinstance(part, str) else json.dumps(part, sort_keys=True, default=str)
        if len(text) > 64:
            text = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        rendered.append(text)
    return "|".join(rendered)


class CacheStore:
    """Thread-safe TTL cache over SQLite (JSON) and Parquet (DataFrames)."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings()
        self._root = root or settings.resolved_cache_dir
        self._root.mkdir(parents=True, exist_ok=True)
        self._frames_dir = self._root / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._root / "cache.sqlite3"
        self._lock = threading.Lock()
        self._init_db()

    # -- setup --------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace  TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    payload    TEXT NOT NULL,
                    fetched_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expires ON cache_entries(expires_at)"
            )

    # -- JSON payloads ------------------------------------------------------
    def get_json(self, namespace: str, key: str) -> CachedValue | None:
        """Return a live entry, or None if absent/expired."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload, fetched_at, expires_at FROM cache_entries "
                "WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()

        if row is None:
            return None

        payload, fetched_at, expires_at = row
        if time.time() >= expires_at:
            logger.debug("Cache expired: %s/%s", namespace, key)
            return None

        try:
            return CachedValue(value=json.loads(payload), fetched_at=fetched_at)
        except json.JSONDecodeError:
            logger.warning("Corrupt cache payload at %s/%s; discarding", namespace, key)
            return None

    def set_json(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        ttl_seconds: int,
        fetched_at: float | None = None,
    ) -> None:
        now = fetched_at if fetched_at is not None else time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache_entries "
                "(namespace, key, payload, fetched_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                (namespace, key, json.dumps(value, default=str), now, now + ttl_seconds),
            )

    # -- DataFrame payloads -------------------------------------------------
    def _frame_path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(f"{namespace}|{key}".encode("utf-8")).hexdigest()[:40]
        return self._frames_dir / f"{namespace}_{digest}.parquet"

    def get_frame(self, namespace: str, key: str) -> tuple[pd.DataFrame, float] | None:
        """Return (frame, fetched_at), or None if absent/expired/unreadable."""
        meta = self.get_json(f"{namespace}::meta", key)
        if meta is None:
            return None

        path = self._frame_path(namespace, key)
        if not path.exists():
            return None

        try:
            return pd.read_parquet(path), meta.fetched_at
        except Exception as exc:
            # A truncated parquet file (e.g. killed mid-write) must degrade to
            # a cache miss, not an application error.
            logger.warning("Unreadable cached frame %s: %s", path, exc)
            return None

    def set_frame(
        self,
        namespace: str,
        key: str,
        frame: pd.DataFrame,
        *,
        ttl_seconds: int,
        fetched_at: float | None = None,
    ) -> None:
        now = fetched_at if fetched_at is not None else time.time()
        path = self._frame_path(namespace, key)
        temp_path = path.with_suffix(".parquet.tmp")
        try:
            # Write-then-rename so a crash can never leave a half-written file
            # visible under the real name.
            frame.to_parquet(temp_path)
            temp_path.replace(path)
        except Exception as exc:
            logger.warning("Failed to cache frame %s/%s: %s", namespace, key, exc)
            temp_path.unlink(missing_ok=True)
            return

        self.set_json(f"{namespace}::meta", key, {"rows": len(frame)},
                      ttl_seconds=ttl_seconds, fetched_at=now)

    # -- maintenance --------------------------------------------------------
    def purge_expired(self) -> int:
        """Delete expired SQLite entries. Returns the number removed."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM cache_entries WHERE expires_at < ?", (time.time(),)
            )
            return cursor.rowcount or 0


_store: CacheStore | None = None
_store_lock = threading.Lock()


def get_cache() -> CacheStore:
    """Process-wide cache singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = CacheStore()
    return _store
