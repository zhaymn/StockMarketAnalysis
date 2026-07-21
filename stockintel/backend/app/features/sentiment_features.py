"""Point-in-time sentiment features.

This is where sentiment either becomes a legitimate predictive feature or a
leak, so the timing rule is stated precisely:

    The feature vector for session D is computed at D's close, and may use
    only articles published at or before D's close.

That sounds obvious and is easy to violate. Two ways it goes wrong:

* **Naive date joins.** Grouping articles by calendar date and joining on date
  puts an article published at 18:00 into the same bucket as a session that
  closed at 16:00. The model then sees this evening's news while predicting
  from this afternoon's close -- a two-hour peek into the future, which on a
  1-day horizon is a large fraction of the thing being predicted.

* **Timezone drift.** Marketaux timestamps are UTC. A 16:00 ET close is 20:00
  or 21:00 UTC depending on daylight saving, and a 15:30 IST close is 10:00
  UTC. Comparing a UTC article time against a naive local session date silently
  shifts the boundary by hours.

So articles are assigned to sessions by comparing UTC instants against each
session's actual close instant, derived from the market's own conventions. An
article published after D's close belongs to D+1, even if its calendar date is
D. `test_sentiment_features.py` pins this with an article published one minute
either side of the close.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.data.market.base import MarketConventions
from app.data.news.archive import ArchivedArticle

logger = get_logger(__name__)

#: Half-life for the recency-weighted sentiment average, in trading sessions.
#: Two sessions: news is largely priced within a couple of days, and a longer
#: half-life mostly smooths the signal away.
DEFAULT_HALF_LIFE_SESSIONS = 2.0

SENTIMENT_FEATURE_COLUMNS = (
    "sentiment_score",
    "sentiment_score_5d",
    "sentiment_dispersion",
    "news_volume",
    "news_volume_ratio",
    "sessions_since_news",
    "has_news_coverage",
)


@dataclass
class SentimentCoverage:
    """How much of the price series actually has news behind it.

    Reported so a result computed over sparse coverage is not mistaken for one
    computed over dense coverage.
    """

    sessions: int
    sessions_with_news: int
    total_articles: int
    first_article: datetime | None
    last_article: datetime | None

    @property
    def coverage_fraction(self) -> float:
        return self.sessions_with_news / self.sessions if self.sessions else 0.0

    @property
    def is_usable(self) -> bool:
        """Whether coverage supports a defensible experiment.

        Below 30% of sessions carrying news, a sentiment feature is mostly its
        own missing-data indicator: the model learns "was there news today",
        not "what did the news say".
        """
        return self.sessions >= 250 and self.coverage_fraction >= 0.30

    def to_dict(self) -> dict[str, object]:
        return {
            "sessions": self.sessions,
            "sessions_with_news": self.sessions_with_news,
            "coverage_fraction": round(self.coverage_fraction, 4),
            "total_articles": self.total_articles,
            "articles_per_covered_session": round(
                self.total_articles / self.sessions_with_news, 2
            ) if self.sessions_with_news else 0.0,
            "first_article": self.first_article.isoformat() if self.first_article else None,
            "last_article": self.last_article.isoformat() if self.last_article else None,
            "is_usable": self.is_usable,
        }


def session_close_instants(
    sessions: pd.DatetimeIndex,
    conventions: MarketConventions,
) -> pd.Series:
    """UTC instant of each session's close.

    Uses the exchange's own timezone and close time, so daylight saving is
    handled by the tz database rather than by a hardcoded offset.
    """
    tz = ZoneInfo(conventions.timezone)
    closes = []

    for session in sessions:
        local_close = datetime(
            session.year, session.month, session.day,
            conventions.close_time.hour, conventions.close_time.minute,
            tzinfo=tz,
        )
        closes.append(local_close.astimezone(timezone.utc))

    # Kept tz-aware. Converting to numpy datetime64 here would silently drop
    # the offset -- numpy has no timezone representation -- which is exactly
    # the drift this module exists to avoid.
    return pd.Series(
        pd.to_datetime(closes, utc=True), index=sessions, name="close_utc"
    )


def assign_articles_to_sessions(
    articles: list[ArchivedArticle],
    sessions: pd.DatetimeIndex,
    conventions: MarketConventions,
) -> pd.Series:
    """Map each article to the first session whose close is at or after it.

    An article published while the market is open, or overnight, is attributed
    to the next close -- the first moment a model could legitimately have acted
    on it. Articles published after the final session's close are dropped, as
    there is no session at which they were knowable.

    Returns:
        A Series indexed like `articles`, holding the assigned session
        timestamp or NaT.
    """
    if not articles:
        return pd.Series([], dtype="datetime64[ns]")

    closes = session_close_instants(sessions, conventions)

    # Compare through tz-aware DatetimeIndexes and let pandas handle the
    # arithmetic. Doing it by hand via astype("int64") is a trap: pandas 3
    # stores these as datetime64[us] while Timestamp.value returns
    # nanoseconds, so the two sides differ by 1000x and every article silently
    # sorts past every close. Index.searchsorted normalises units and
    # timezones itself.
    close_index = pd.DatetimeIndex(closes)
    published_index = pd.DatetimeIndex(
        [pd.Timestamp(a.published_at) for a in articles]
    )
    if published_index.tz is None:
        published_index = published_index.tz_localize("UTC")
    else:
        published_index = published_index.tz_convert("UTC")

    # side="left" gives the first close >= published time, so an article
    # published exactly at the close belongs to that session.
    positions = close_index.searchsorted(published_index, side="left")

    assigned: list[pd.Timestamp | pd.NaT] = []
    for position in positions:
        assigned.append(sessions[position] if position < len(sessions) else pd.NaT)

    return pd.Series(assigned, dtype="datetime64[ns]")


def build_sentiment_features(
    articles: list[ArchivedArticle],
    sessions: pd.DatetimeIndex,
    conventions: MarketConventions,
    *,
    half_life_sessions: float = DEFAULT_HALF_LIFE_SESSIONS,
) -> tuple[pd.DataFrame, SentimentCoverage]:
    """Build causal sentiment features aligned to trading sessions.

    Every value at session D derives only from articles published at or before
    D's close.

    Returns:
        `(features, coverage)`. Features are NaN-free; sessions with no news
        carry a neutral 0.0 score plus `has_news_coverage = 0`, so a model can
        distinguish "no news" from "neutral news" rather than conflating them.
    """
    features = pd.DataFrame(index=sessions)

    scored = [a for a in articles if a.is_scored and a.signed_score is not None]
    assignment = assign_articles_to_sessions(scored, sessions, conventions)

    per_session_scores: dict[pd.Timestamp, list[float]] = {}
    for article, session in zip(scored, assignment):
        if pd.isna(session):
            continue
        per_session_scores.setdefault(session, []).append(article.signed_score)

    raw_score = pd.Series(0.0, index=sessions)
    raw_count = pd.Series(0.0, index=sessions)
    dispersion = pd.Series(0.0, index=sessions)

    for session, values in per_session_scores.items():
        raw_score[session] = float(np.mean(values))
        raw_count[session] = float(len(values))
        dispersion[session] = float(np.std(values)) if len(values) > 1 else 0.0

    has_news = (raw_count > 0).astype(float)

    # Recency-weighted sentiment. ewm is causal by construction -- the value at
    # D depends only on D and earlier.
    alpha = 1.0 - 0.5 ** (1.0 / half_life_sessions)
    decayed = raw_score.ewm(alpha=alpha, adjust=False).mean()

    features["sentiment_score"] = raw_score
    features["sentiment_score_5d"] = decayed
    features["sentiment_dispersion"] = dispersion
    features["news_volume"] = raw_count

    # Volume relative to the trailing norm: an unusual burst of coverage is a
    # signal independent of its polarity. shift(1) so the current session's own
    # count is excluded from its own baseline.
    trailing_mean = raw_count.shift(1).rolling(21, min_periods=5).mean()
    features["news_volume_ratio"] = (
        (raw_count / trailing_mean.replace(0.0, np.nan)).fillna(1.0).clip(upper=10.0)
    )

    # Sessions since the last covered session, capped so a long dry spell does
    # not dominate the feature's scale.
    since = []
    counter = 0
    for value in has_news:
        if value > 0:
            counter = 0
        else:
            counter = min(counter + 1, 21)
        since.append(float(counter))
    features["sessions_since_news"] = since

    features["has_news_coverage"] = has_news

    coverage = SentimentCoverage(
        sessions=len(sessions),
        sessions_with_news=int(has_news.sum()),
        total_articles=len(scored),
        first_article=min((a.published_at for a in scored), default=None),
        last_article=max((a.published_at for a in scored), default=None),
    )

    logger.info(
        "Sentiment features: %d/%d sessions covered (%.1f%%), %d scored articles",
        coverage.sessions_with_news, coverage.sessions,
        coverage.coverage_fraction * 100, coverage.total_articles,
    )
    return features.fillna(0.0), coverage


def join_sentiment_features(
    technical: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join sentiment onto the technical feature matrix.

    Left join on the technical index: the price series defines which sessions
    exist. Sentiment sessions absent from it are dropped, and technical
    sessions without sentiment get zeros plus `has_news_coverage = 0`.
    """
    joined = technical.join(sentiment, how="left")
    for column in SENTIMENT_FEATURE_COLUMNS:
        if column in joined.columns:
            joined[column] = joined[column].fillna(0.0)
    return joined
