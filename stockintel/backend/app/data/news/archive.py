"""Local archive of historical news articles.

Backfilling news is slow and quota-limited, so fetched articles must persist.
This is a SQLite table rather than the TTL cache: TTL entries expire, and an
article published in March 2024 does not become less true with age.

Deduplication is by URL, since the same story fetched in overlapping windows
must not be counted twice in a sentiment aggregate.

FinBERT scores are stored alongside the article. Scoring is the expensive part
of the pipeline (~50ms per article on CPU), and a backfill of tens of thousands
of articles would otherwise be re-scored on every experiment run.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ArchivedArticle:
    """One stored article, with sentiment if it has been scored."""

    url: str
    symbol: str
    title: str
    description: str
    source: str
    published_at: datetime
    sentiment_label: str | None = None
    sentiment_positive: float | None = None
    sentiment_negative: float | None = None
    sentiment_neutral: float | None = None

    @property
    def is_scored(self) -> bool:
        return self.sentiment_label is not None

    @property
    def signed_score(self) -> float | None:
        """P(positive) - P(negative), or None if unscored."""
        if self.sentiment_positive is None or self.sentiment_negative is None:
            return None
        return self.sentiment_positive - self.sentiment_negative


class NewsArchive:
    """Persistent store of historical articles."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self._path = path or (settings.resolved_cache_dir / "news_archive.sqlite3")
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=20.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    url                TEXT NOT NULL,
                    symbol             TEXT NOT NULL,
                    title              TEXT NOT NULL,
                    description        TEXT NOT NULL DEFAULT '',
                    source             TEXT NOT NULL DEFAULT '',
                    published_at       TEXT NOT NULL,
                    sentiment_label    TEXT,
                    sentiment_positive REAL,
                    sentiment_negative REAL,
                    sentiment_neutral  REAL,
                    PRIMARY KEY (url, symbol)
                )
                """
            )
            # Queries are always "articles for symbol X in date range Y".
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbol_date "
                "ON articles(symbol, published_at)"
            )
            # Fetch coverage, so a resumed backfill knows which windows are done.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS coverage (
                    symbol     TEXT NOT NULL,
                    day        TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    n_articles INTEGER NOT NULL,
                    complete   INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (symbol, day)
                )
                """
            )

    # -- writing ------------------------------------------------------------
    def add_articles(self, symbol: str, articles: list[ArchivedArticle]) -> int:
        """Insert articles, ignoring ones already stored. Returns new rows."""
        if not articles:
            return 0

        rows = [
            (
                a.url, symbol, a.title, a.description, a.source,
                a.published_at.astimezone(timezone.utc).isoformat(),
                a.sentiment_label, a.sentiment_positive,
                a.sentiment_negative, a.sentiment_neutral,
            )
            for a in articles
        ]

        with self._lock, self._connect() as conn:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO articles "
                "(url, symbol, title, description, source, published_at, "
                " sentiment_label, sentiment_positive, sentiment_negative, sentiment_neutral) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return conn.total_changes - before

    def mark_day_complete(self, symbol: str, day: str, n_articles: int) -> None:
        """Record that a calendar day has been fully fetched."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO coverage (symbol, day, fetched_at, n_articles, complete) "
                "VALUES (?, ?, ?, ?, 1)",
                (symbol, day, datetime.now(timezone.utc).isoformat(), n_articles),
            )

    def completed_days(self, symbol: str) -> set[str]:
        """Days already fetched, so a resumed backfill can skip them."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT day FROM coverage WHERE symbol = ? AND complete = 1", (symbol,)
            ).fetchall()
        return {row[0] for row in rows}

    def store_sentiment(
        self, url: str, symbol: str, label: str, scores: dict[str, float]
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE articles SET sentiment_label = ?, sentiment_positive = ?, "
                "sentiment_negative = ?, sentiment_neutral = ? WHERE url = ? AND symbol = ?",
                (
                    label,
                    scores.get("positive"),
                    scores.get("negative"),
                    scores.get("neutral"),
                    url,
                    symbol,
                ),
            )

    # -- reading ------------------------------------------------------------
    def _row_to_article(self, row: tuple) -> ArchivedArticle:
        return ArchivedArticle(
            url=row[0], symbol=row[1], title=row[2], description=row[3],
            source=row[4], published_at=datetime.fromisoformat(row[5]),
            sentiment_label=row[6], sentiment_positive=row[7],
            sentiment_negative=row[8], sentiment_neutral=row[9],
        )

    def get_articles(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        scored_only: bool = False,
    ) -> list[ArchivedArticle]:
        """Articles for a symbol, oldest first."""
        query = "SELECT * FROM articles WHERE symbol = ?"
        params: list[object] = [symbol]

        if start is not None:
            query += " AND published_at >= ?"
            params.append(start.astimezone(timezone.utc).isoformat())
        if end is not None:
            query += " AND published_at < ?"
            params.append(end.astimezone(timezone.utc).isoformat())
        if scored_only:
            query += " AND sentiment_label IS NOT NULL"

        query += " ORDER BY published_at ASC"

        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_article(row) for row in rows]

    def unscored_articles(self, symbol: str, limit: int = 5000) -> list[ArchivedArticle]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM articles WHERE symbol = ? AND sentiment_label IS NULL "
                "ORDER BY published_at ASC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [self._row_to_article(row) for row in rows]

    def stats(self, symbol: str) -> dict[str, object]:
        """Coverage summary, for deciding whether an experiment is viable."""
        with self._lock, self._connect() as conn:
            total, scored, first, last = conn.execute(
                "SELECT COUNT(*), COUNT(sentiment_label), MIN(published_at), "
                "MAX(published_at) FROM articles WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            days = conn.execute(
                "SELECT COUNT(*) FROM coverage WHERE symbol = ? AND complete = 1", (symbol,)
            ).fetchone()[0]

        return {
            "symbol": symbol,
            "articles": total or 0,
            "scored": scored or 0,
            "days_covered": days or 0,
            "first_article": first,
            "last_article": last,
        }


_archive: NewsArchive | None = None
_archive_lock = threading.Lock()


def get_archive() -> NewsArchive:
    """Process-wide archive singleton."""
    global _archive
    if _archive is None:
        with _archive_lock:
            if _archive is None:
                _archive = NewsArchive()
    return _archive
